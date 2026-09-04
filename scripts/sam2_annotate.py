#!/usr/bin/env python3
"""SAM 2 -> YOLO-Seg annotation pipeline (Teacher-in-the-Loop).

Keeps SAM 2 frozen. Its zero-shot visual prompting natively segments organic
skin blemishes. The workflow:

    1. Bootstrap 100-200 images with quick bounding box prompts.
       (Use --mode draw to interactively draw coarse boxes.)
    2. Feed box prompts into frozen SAM 2 -> high-precision polygon masks.
    3. (Optional) Exploit SAM 2 video memory across continuous-turn frames to
       propagate masks around the fruit. (--mode video)
    4. Convert polygon masks into YOLO segmentation labels.
    5. Train/distill directly into an edge-optimized student model
       (yolo26s-seg).

This script performs steps 1-4. Step 5 is a separate training run.

Runs locally on Apple Silicon (MPS) or on a remote GPU host (vast.ai). SAM 2
is loaded via the ``sam2`` package; if it is not installed the script prints a
clear install hint instead of crashing at import time.

Usage:

    # Step 1: Draw coarse bounding boxes interactively
    python scripts/sam2_annotate.py --mode draw \\
        --images dataset/raw_ingest \\
        --prompts dataset/prompts

    # Step 2: Run SAM 2 on all images with box prompts -> YOLO-Seg labels
    python scripts/sam2_annotate.py --mode image \\
        --images dataset/raw_ingest \\
        --prompts dataset/prompts \\
        --output dataset/labels \\
        --device mps \\
        --model-size large

    # Step 3 (optional): Video mode for turntable sequences
    python scripts/sam2_annotate.py --mode video \\
        --images dataset/raw_ingest \\
        --prompts dataset/prompts \\
        --output dataset/labels \\
        --device mps \\
        --model-size large

    # Step 5: Train the student model
    yolo segment train data=data.yaml model=yolo26s-seg.pt epochs=100 \\
        imgsz=640 batch=16 device=mps

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

# SAM 2 model configs and checkpoints by size.
SAM2_MODELS = {
    "tiny": {
        "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_tiny.pt",
    },
    "small": {
        "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_small.pt",
    },
    "base": {
        "cfg": "configs/sam2.1/sam2.1_hiera_b.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_base.pt",
    },
    "large": {
        "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "ckpt": "checkpoints/sam2.1_hiera_large.pt",
    },
}


# ---------------------------------------------------------------------------
# Prompt parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mask -> YOLO-Seg polygon conversion
# ---------------------------------------------------------------------------

def mask_to_yolo_seg_labels(
    binary_mask: Any,
    class_id: int,
    img_w: int,
    img_h: int,
    min_area: int = 20,
    epsilon_factor: float = 0.002,
) -> list[str]:
    """Convert a binary mask to YOLO-Seg polygon label lines.

    Handles multiple contours (e.g., scattered defect regions) and filters
    micro-noise specks below ``min_area`` pixels. Polygon vertices are
    simplified via Douglas-Peucker with ``epsilon = epsilon_factor * arcLength``.

    Returns a list of label strings, one per contour:
        ``class_id x1 y1 x2 y2 ... xn yn`` (normalized 0-1)
    """
    import cv2
    import numpy as np

    mask_uint8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    labels: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        coords = approx.reshape(-1, 2)
        norm_coords: list[str] = []
        for x, y in coords:
            nx = max(0.0, min(1.0, x / img_w))
            ny = max(0.0, min(1.0, y / img_h))
            norm_coords.append(f"{nx:.6f}")
            norm_coords.append(f"{ny:.6f}")

        if len(norm_coords) >= 6:  # at least 3 points (triangle)
            label_str = f"{class_id} {' '.join(norm_coords)}"
            labels.append(label_str)

    return labels


# ---------------------------------------------------------------------------
# SAM 2 model loading
# ---------------------------------------------------------------------------

def load_sam2_image_predictor(device: str, model_size: str = "large") -> Any:
    """Load a frozen SAM 2 image predictor.

    Requires the ``sam2`` package and a model checkpoint. Prints a clear
    install hint if the package is missing.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        print("[ERROR] sam2 package not installed.")
        print("        Install: pip install git+https://github.com/facebookresearch/sam2")
        print(f"        Detail: {exc}")
        raise

    model_info = SAM2_MODELS.get(model_size, SAM2_MODELS["large"])
    print(f"[SAM2] Loading {model_size} model: {model_info['ckpt']}")
    model = build_sam2(
        config_file=model_info["cfg"],
        ckpt_path=model_info["ckpt"],
        device=device,
    )
    return SAM2ImagePredictor(model)


def load_sam2_video_predictor(device: str, model_size: str = "large") -> Any:
    """Load a frozen SAM 2 video predictor for turntable sequence propagation."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as exc:
        print("[ERROR] sam2 package not installed.")
        print("        Install: pip install git+https://github.com/facebookresearch/sam2")
        print(f"        Detail: {exc}")
        raise

    model_info = SAM2_MODELS.get(model_size, SAM2_MODELS["large"])
    print(f"[SAM2] Loading {model_size} video model: {model_info['ckpt']}")
    model = build_sam2(
        config_file=model_info["cfg"],
        ckpt_path=model_info["ckpt"],
        device=device,
        vae_only=False,
    )
    return SAM2VideoPredictor(model)


# ---------------------------------------------------------------------------
# Image mode: single-frame box-prompted segmentation
# ---------------------------------------------------------------------------

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

    all_labels: list[str] = []
    for cls_id, box in prompts:
        box_arr = np.array(box, dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=box_arr[None, :],
            multimask_output=False,
        )
        mask = masks[0]
        labels = mask_to_yolo_seg_labels(mask, cls_id, img_w, img_h)
        if not labels:
            print(f"  [WARN] no contour for class {cls_id} in {image_path.name}")
            continue
        all_labels.extend(labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{line}\n" for line in all_labels), encoding="utf-8"
    )
    return len(all_labels)


def run_image_mode(args: argparse.Namespace) -> int:
    """Process all images with box prompts -> YOLO-Seg labels."""
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
    print(f"[SYSTEM] Device: {args.device} | Model: {args.model_size}")
    print("[SYSTEM] Loading frozen SAM 2 image predictor...")

    try:
        predictor = load_sam2_image_predictor(args.device, args.model_size)
    except ImportError:
        return 1

    import cv2

    total_labels = 0
    processed = 0
    skipped = 0
    for image_path in image_paths:
        rel = image_path.relative_to(images_dir)
        prompt_path = prompts_dir / rel.with_suffix(".txt")

        img = cv2.imread(str(image_path))
        if img is None:
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]
        prompts = parse_yolo_boxes(prompt_path, img_w, img_h)
        if not prompts:
            skipped += 1
            continue

        output_path = output_dir / rel.with_suffix(".txt")
        count = annotate_image(predictor, image_path, prompts, output_path)
        total_labels += count
        processed += 1
        if processed % 50 == 0:
            print(f"  [PROGRESS] {processed}/{len(image_paths)} images, {total_labels} labels")

    print(f"[DONE] {processed} images processed, {skipped} skipped, {total_labels} polygon labels written")
    print(f"[OUTPUT] Labels in: {output_dir}")
    print("[NEXT] Train a YOLO-Seg student on the output labels:")
    print("  yolo segment train data=data.yaml model=yolo26s-seg.pt epochs=100 imgsz=640 batch=16 device=mps")
    return 0


# ---------------------------------------------------------------------------
# Video mode: turntable sequence propagation
# ---------------------------------------------------------------------------

def group_turntable_sequences(
    image_paths: list[Path], images_dir: Path
) -> list[list[Path]]:
    """Group images by profile_id (same apple, different views).

    Images from the same batch/profile share a prefix like
    ``batch-20260903-discard-00000``. We group by that prefix.
    """
    groups: dict[str, list[Path]] = {}
    for p in image_paths:
        rel = p.relative_to(images_dir)
        stem = rel.stem
        # Extract profile_id: everything up to the last _equatorial_N or _calyx
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].startswith("view"):
            profile = parts[0].rsplit("_", 1)[0]  # strip view_type
        else:
            profile = stem
        groups.setdefault(profile, []).append(p)

    # Sort each group by view index
    for profile in groups:
        groups[profile].sort()
    return list(groups.values())


def run_video_mode(args: argparse.Namespace) -> int:
    """Process turntable sequences with SAM 2 video predictor.

    Prompts on frame 0 only; SAM 2 propagates masks across all frames
    in the sequence via its streaming memory bank.
    """
    import cv2
    import numpy as np

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

    sequences = group_turntable_sequences(image_paths, images_dir)
    print(f"[SYSTEM] Found {len(image_paths)} images in {len(sequences)} turntable sequences")
    print(f"[SYSTEM] Device: {args.device} | Model: {args.model_size}")
    print("[SYSTEM] Loading frozen SAM 2 video predictor...")

    try:
        predictor = load_sam2_video_predictor(args.device, args.model_size)
    except ImportError:
        return 1

    total_labels = 0
    processed_seqs = 0

    for seq_idx, sequence in enumerate(sequences):
        if len(sequence) < 2:
            continue

        # Load all frames in the sequence
        frames: list[np.ndarray] = []
        frame_h, frame_w = 0, 0
        for fp in sequence:
            img = cv2.imread(str(fp))
            if img is None:
                print(f"  [SKIP] cannot read: {fp}")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            frame_h, frame_w = img.shape[:2]

        if len(frames) < 2:
            continue

        # Get prompts for frame 0 only
        rel0 = sequence[0].relative_to(images_dir)
        prompt_path = prompts_dir / rel0.with_suffix(".txt")
        prompts = parse_yolo_boxes(prompt_path, frame_w, frame_h)
        if not prompts:
            print(f"  [SKIP] no prompts for sequence {seq_idx}: {rel0.stem}")
            continue

        print(f"  [SEQ {seq_idx + 1}/{len(sequences)}] {len(frames)} frames, {len(prompts)} prompts")

        # Initialize video predictor state
        inference_state = predictor.init_state(
            video_path=np.stack(frames),
            offload_video_to_cpu=True,
        )

        # Add box prompts on frame 0
        for obj_id, (cls_id, box) in enumerate(prompts, start=1):
            box_arr = np.array(box, dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_box_prompt(
                inference_state, frame_idx=0, box=box_arr, obj_id=obj_id
            )

        # Propagate through the video
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(out_obj_ids)
            }

        # Write YOLO-Seg labels for each frame
        for frame_idx, frame_path in enumerate(sequence):
            if frame_idx not in video_segments:
                continue
            rel = frame_path.relative_to(images_dir)
            output_path = output_dir / rel.with_suffix(".txt")

            all_labels: list[str] = []
            for obj_id, mask in video_segments[frame_idx].items():
                cls_id = prompts[obj_id - 1][0]  # map back to class_id
                labels = mask_to_yolo_seg_labels(mask, cls_id, frame_w, frame_h)
                all_labels.extend(labels)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "".join(f"{line}\n" for line in all_labels), encoding="utf-8"
            )
            total_labels += len(all_labels)

        processed_seqs += 1
        if processed_seqs % 10 == 0:
            print(f"  [PROGRESS] {processed_seqs}/{len(sequences)} sequences, {total_labels} labels")

    print(f"[DONE] {processed_seqs} sequences processed, {total_labels} polygon labels written")
    print(f"[OUTPUT] Labels in: {output_dir}")
    print("[NEXT] Train a YOLO-Seg student on the output labels:")
    print("  yolo segment train data=data.yaml model=yolo26s-seg.pt epochs=100 imgsz=640 batch=16 device=mps")
    return 0


# ---------------------------------------------------------------------------
# Draw mode: interactive box annotation tool
# ---------------------------------------------------------------------------

def run_draw_mode(args: argparse.Namespace) -> int:
    """Interactive bounding box drawing tool for creating coarse prompts.

    Click and drag to draw boxes. Press a number key to set the class ID
    before drawing. Press SPACE to save and advance, ESC to quit without
    saving the current image.
    """
    import cv2

    images_dir = args.images
    prompts_dir = args.prompts

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

    prompts_dir.mkdir(parents=True, exist_ok=True)

    current_class = 0
    drawing = False
    box_start: tuple[int, int] = (0, 0)
    box_end: tuple[int, int] = (0, 0)
    boxes: list[tuple[int, list[float]]] = []
    display = None

    print(f"[DRAW] {len(image_paths)} images")
    print("[DRAW] Controls:")
    print("  0=apple  1=stem_calyx  2=defect_surface  3=defect_critical")
    print("  Click+drag = draw box  SPACE = save+next  ESC = skip  BACKSPACE = undo last box")
    print()

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, box_start, box_end
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            box_start = (x, y)
            box_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            box_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            box_end = (x, y)
            x1, y1 = min(box_start[0], box_end[0]), min(box_start[1], box_end[1])
            x2, y2 = max(box_start[0], box_end[0]), max(box_start[1], box_end[1])
            if x2 - x1 > 5 and y2 - y1 > 5:
                boxes.append((current_class, [float(x1), float(y1), float(x2), float(y2)]))
                print(f"  box #{len(boxes)}: class={current_class} ({CLASS_NAMES[current_class]}) "
                      f"[{x1},{y1},{x2},{y2}]")

    cv2.namedWindow("DRAW", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("DRAW", mouse_callback)

    idx = 0
    while idx < len(image_paths):
        image_path = image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            idx += 1
            continue
        img_h, img_w = image.shape[:2]
        boxes = []
        rel = image_path.relative_to(images_dir)
        prompt_path = prompts_dir / rel.with_suffix(".txt")

        # Load existing prompts if any
        existing = parse_yolo_boxes(prompt_path, img_w, img_h)
        for cls_id, box in existing:
            boxes.append((cls_id, box))

        print(f"[DRAW] Image {idx + 1}/{len(image_paths)}: {rel.name}  "
              f"({len(boxes)} existing boxes)")

        while True:
            display = image.copy()
            # Draw all boxes
            for cls_id, box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                colors = [(0, 255, 255), (255, 0, 255), (0, 0, 255), (0, 0, 128)]
                color = colors[cls_id % len(colors)]
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"{cls_id}:{CLASS_NAMES[cls_id]}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw current box being drawn
            if drawing:
                cv2.rectangle(display, box_start, box_end, (0, 255, 0), 2)

            # Status overlay
            cv2.putText(display, f"Class: {current_class} ({CLASS_NAMES[current_class]})",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Boxes: {len(boxes)} | SPACE=save ESC=skip BKSP=undo",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("DRAW", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC - skip without saving
                print(f"  [SKIP] {rel.name}")
                break
            elif key == 32:  # SPACE - save and advance
                if boxes:
                    lines = []
                    for cls_id, box in boxes:
                        x1, y1, x2, y2 = box
                        cx = ((x1 + x2) / 2) / img_w
                        cy = ((y1 + y2) / 2) / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h
                        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    prompt_path.parent.mkdir(parents=True, exist_ok=True)
                    prompt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    print(f"  [SAVED] {len(boxes)} boxes -> {prompt_path.name}")
                break
            elif key == 8:  # BACKSPACE - undo last box
                if boxes:
                    removed = boxes.pop()
                    print(f"  [UNDO] removed box: class={removed[0]}")
            elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
                current_class = int(chr(key))
                print(f"  [CLASS] set to {current_class} ({CLASS_NAMES[current_class]})")

        idx += 1

    cv2.destroyAllWindows()
    print(f"[DONE] Drew prompts for {len(image_paths)} images")
    print("[NEXT] Run SAM 2 annotation:")
    print("  python scripts/sam2_annotate.py --mode image --images dataset/raw_ingest \\")
    print("    --prompts dataset/prompts --output dataset/labels --device mps --model-size large")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM 2 -> YOLO-Seg annotation pipeline (frozen teacher)."
    )
    parser.add_argument(
        "--mode",
        choices=["draw", "image", "video"],
        default="image",
        help="draw: interactive box annotation. image: single-frame SAM 2. video: turntable propagation.",
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
        default=Path("dataset/labels"),
        help="Output directory for YOLO segmentation labels.",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Inference device: 'mps' (Apple Silicon), 'cuda', or 'cpu'.",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "base", "large"],
        default="large",
        help="SAM 2 model size. large = best quality, tiny = fastest.",
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

    if args.mode == "draw":
        return run_draw_mode(args)
    elif args.mode == "video":
        return run_video_mode(args)
    else:
        return run_image_mode(args)


if __name__ == "__main__":
    sys.exit(main())
