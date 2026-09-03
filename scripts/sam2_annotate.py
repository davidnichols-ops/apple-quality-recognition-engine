#!/usr/bin/env python3
"""SAM 2 -> YOLO-Seg annotation pipeline (Teacher-in-the-Loop).

Keeps SAM 2 frozen. Its zero-shot visual prompting natively segments organic
skin blemishes. The workflow:

    1. Bootstrap 100-200 images with quick bounding box prompts.
    2. Feed box prompts into frozen SAM 2 -> high-precision polygon masks.
    3. (Optional) Exploit SAM 2 video memory across continuous-turn frames to
       propagate masks around the fruit.
    4. Convert polygon masks into YOLO segmentation labels.
    5. Train/distill directly into an edge-optimized student model
       (yolo11-seg / yolo26-seg).

This script performs steps 1-4. Step 5 is a separate training run.

Runs locally on Apple Silicon (MPS) or on a remote GPU host (vast.ai). SAM 2
is loaded via the ``sam2`` package; if it is not installed the script prints a
clear install hint instead of crashing at import time.

Usage (local, single image directory with pre-drawn box prompts):

    python scripts/sam2_annotate.py \\
        --images dataset/raw_ingest \\
        --prompts dataset/raw_ingest/prompts \\
        --output apple_dataset/labels \\
        --device mps

The prompts directory mirrors the image directory structure. Each
``<stem>.txt`` file contains YOLO-format box prompts (class_id cx cy w h,
normalized). If no prompt file exists for an image, the image is skipped.

Output is YOLO segmentation format: one ``<stem>.txt`` per image with
``class_id x1 y1 x2 y2 ... xn yn`` (normalized polygon coordinates).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

# Class IDs must match data.yaml exactly.
CLASS_NAMES = {
    0: "apple",
    1: "stem_calyx",
    2: "defect_surface",
    3: "defect_critical",
}


def parse_yolo_boxes(prompt_path: Path, img_w: int, img_h: int) -> list[tuple[int, list[float]]]:
    """Read YOLO-format normalized box prompts and convert to pixel coords.

    Each line: ``class_id cx cy w h`` (all normalized 0-1).
    Returns a list of (class_id, [x1, y1, x2, y2]) in pixel coordinates.
    """
    prompts: list[tuple[int, list[float]]] = []
    if not prompt_path.exists():
        return prompts
    for line in prompt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        prompts.append((cls_id, [x1, y1, x2, y2]))
    return prompts


def mask_to_polygon(mask: Any, max_points: int = 50) -> list[tuple[float, float]] | None:
    """Extract the largest contour from a binary mask as a polygon.

    Returns a list of (x, y) pixel coordinates, or None if no contour is found.
    Simplified to ``max_points`` via Douglas-Peucker to keep label files small.
    """
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None
    epsilon = 0.005 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    points = [(float(p[0][0]), float(p[0][1])) for p in approx]
    if len(points) < 3:
        return None
    if len(points) > max_points:
        step = len(points) // max_points
        points = points[::step]
    return points


def polygon_to_yolo_seg(
    polygon: list[tuple[float, float]], class_id: int, img_w: int, img_h: int
) -> str:
    """Convert a pixel-space polygon to a YOLO segmentation label line.

    Format: ``class_id x1 y1 x2 y2 ... xn yn`` (all normalized 0-1).
    """
    coords = []
    for x, y in polygon:
        nx = max(0.0, min(1.0, x / img_w))
        ny = max(0.0, min(1.0, y / img_h))
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    return f"{class_id} {' '.join(coords)}"


def load_sam2_predictor(device: str) -> Any:
    """Load a frozen SAM 2 predictor.

    Requires the ``sam2`` package and a model checkpoint. Prints a clear
    install hint if the package is missing.
    """
    try:
        from sam2.build_sam import build_sam2_model
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        print("[ERROR] sam2 package not installed.")
        print("        Install: pip install git+https://github.com/facebookresearch/sam2")
        print(f"        Detail: {exc}")
        raise

    # Use the default tiny model for edge/local runs; swap for large on GPU hosts.
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    checkpoint = "sam2.1_hiera_tiny.pt"
    model = build_sam2_model(
        config_file=model_cfg, ckpt_path=checkpoint, device=device
    )
    return SAM2ImagePredictor(model)


def annotate_image(
    predictor: Any,
    image_path: Path,
    prompts: list[tuple[int, list[float]]],
    output_path: Path,
) -> int:
    """Run SAM 2 on one image with box prompts and write YOLO-Seg labels.

    Returns the number of polygon labels written.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [SKIP] cannot read: {image_path}")
        return 0
    img_h, img_w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(rgb)

    labels: list[str] = []
    for cls_id, box in prompts:
        box_arr = np.array(box, dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=box_arr[None, :],
            multimask_output=False,
        )
        mask = masks[0]
        polygon = mask_to_polygon(mask)
        if polygon is None:
            print(f"  [WARN] no contour for class {cls_id} in {image_path.name}")
            continue
        labels.append(polygon_to_yolo_seg(polygon, cls_id, img_w, img_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{line}\n" for line in labels), encoding="utf-8"
    )
    return len(labels)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 2 -> YOLO-Seg annotation pipeline (frozen teacher)."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory of source images (JPG/PNG).",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Directory of YOLO-format box prompts mirroring the image layout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for YOLO segmentation labels.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Inference device: 'mps' (Apple Silicon), 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=[".jpg", ".jpeg", ".png"],
        help="Image file extensions to process.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    images_dir = args.images
    prompts_dir = args.prompts
    output_dir = args.output

    if not images_dir.is_dir():
        print(f"[ERROR] images directory not found: {images_dir}")
        return 1

    image_paths: list[Path] = []
    for ext in args.extensions:
        image_paths.extend(sorted(images_dir.rglob(f"*{ext}")))
        image_paths.extend(sorted(images_dir.rglob(f"*{ext.upper()}")))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[ERROR] no images found in {images_dir}")
        return 1

    print(f"[SYSTEM] Found {len(image_paths)} images")
    print(f"[SYSTEM] Device: {args.device}")
    print("[SYSTEM] Loading frozen SAM 2 predictor...")

    try:
        predictor = load_sam2_predictor(args.device)
    except ImportError:
        return 1

    total_labels = 0
    processed = 0
    for image_path in image_paths:
        rel = image_path.relative_to(images_dir)
        prompt_path = prompts_dir / rel.with_suffix(".txt")
        prompts = parse_yolo_boxes(prompt_path, 0, 0)
        if not prompts:
            continue

        # Re-read image dimensions for prompt conversion
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        prompts = parse_yolo_boxes(prompt_path, img_w, img_h)

        output_path = output_dir / rel.with_suffix(".txt")
        count = annotate_image(predictor, image_path, prompts, output_path)
        total_labels += count
        processed += 1
        if processed % 50 == 0:
            print(f"  [PROGRESS] {processed}/{len(image_paths)} images, {total_labels} labels")

    print(f"[DONE] {processed} images processed, {total_labels} polygon labels written")
    print("[NEXT] Train a YOLO-Seg student on the output labels:")
    print("  yolo segment train data=data.yaml model=yolo11n-seg.pt epochs=100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
