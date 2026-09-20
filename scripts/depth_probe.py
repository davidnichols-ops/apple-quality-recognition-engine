#!/usr/bin/env python3
"""Probe depth at reviewed defect masks; never assign a grade.

Supports YOLO26 Nano depth and Depth Anything V2 Small from third_party.
Checkpoints live under checkpoints. Output is diagnostic evidence, not a
calibrated depth measurement or an automatic surface/critical classifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))

from scripts.apple_surface_geometry import (
    exclude_calyx_candidate,
    fit_surface,
    region_deviation,
)


def cavity_contrast(depth: np.ndarray, apple: np.ndarray, defect: np.ndarray) -> dict:
    """Compare a defect with a fitted patch of nearby intact apple skin."""
    apple = apple.astype(bool)
    defect = defect.astype(bool) & apple
    if depth.shape != apple.shape or defect.sum() < 25:
        return {"status": "insufficient_mask"}
    kernel = np.ones((31, 31), np.uint8)
    ring = cv2.dilate(defect.astype(np.uint8), kernel).astype(bool) & apple & ~defect
    if ring.sum() < 100:
        return {"status": "insufficient_surrounding_skin"}

    ys, xs = np.where(defect | ring)
    cx, cy = float(xs.mean()), float(ys.mean())
    scale = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    yy, xx = np.indices(depth.shape)
    x = (xx - cx) / scale
    y = (yy - cy) / scale
    basis = np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)
    coefficients, *_ = np.linalg.lstsq(basis[ring], depth[ring], rcond=None)
    fitted = basis @ coefficients
    residual = depth - fitted
    ring_noise = float(np.median(np.abs(residual[ring] - np.median(residual[ring]))))
    defect_delta = float(np.median(residual[defect]))
    return {
        "status": "measured",
        "defect_pixels": int(defect.sum()),
        "surrounding_pixels": int(ring.sum()),
        "relative_depth_residual": defect_delta,
        "surrounding_mad": ring_noise,
        "absolute_contrast_to_noise": abs(defect_delta) / max(ring_noise, 1e-6),
        "note": "Model output units only; sign and threshold need measured calibration.",
    }


def masks_from_yolo_seg(path: Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Rasterize the segmentation contract used by the future YOLO-Seg model."""
    h, w = shape
    masks = {name: np.zeros(shape, np.uint8) for name in
             ("apple", "stem_calyx", "defect_surface", "defect_critical")}
    classes = {0: "apple", 1: "stem_calyx", 2: "defect_surface", 3: "defect_critical"}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        class_id = int(fields[0])
        if class_id not in classes or len(fields) < 7 or len(fields) % 2 == 0:
            continue
        xy = np.asarray([float(v) for v in fields[1:]], np.float32).reshape(-1, 2)
        xy[:, 0] *= w
        xy[:, 1] *= h
        poly = np.rint(xy).astype(np.int32)
        cv2.fillPoly(masks[classes[class_id]], [poly], 1)
    return masks


def apple_aoi(apple: np.ndarray, margin_fraction: float = 0.1) -> tuple[int, int, int, int]:
    """Return a bounded (x0, y0, x1, y1) crop around the segmented fruit."""
    ys, xs = np.where(apple > 0)
    if len(xs) < 25:
        raise ValueError("apple mask is empty or too small for an AOI")
    h, w = apple.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin = round(max(x1 - x0, y1 - y0) * margin_fraction)
    return (max(0, x0 - margin), max(0, y0 - margin),
            min(w, x1 + margin), min(h, y1 + margin))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="Substring of the profile ID")
    parser.add_argument("--backend", choices=("depth-anything-v2", "yolo26n", "yolo26l"),
                        default="yolo26n")
    parser.add_argument("--mask-source", choices=("yolo-seg", "sam-store"),
                        default="yolo-seg")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--aoi", choices=("full", "apple"), default="full")
    args = parser.parse_args()
    if args.checkpoint is None:
        filename = (f"{args.backend}-depth.pt" if args.backend.startswith("yolo26")
                    else "depth_anything_v2_vits.pth")
        args.checkpoint = ROOT / "checkpoints" / filename
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint missing: {args.checkpoint}")

    if args.backend.startswith("yolo26"):
        from ultralytics import YOLO
        model = YOLO(str(args.checkpoint))

        def infer(image: np.ndarray) -> np.ndarray:
            result = model.predict(source=image, imgsz=768, device=args.device,
                                   verbose=False)[0]
            if result.depth is None:
                raise RuntimeError("YOLO26 depth model returned no depth map")
            return result.depth.data.cpu().numpy()
    else:
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder="vits", features=64,
                               out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        model = model.to(args.device).eval()
        infer = model.infer_image

    output = ROOT / "dataset/depth_probes" / args.sequence / args.backend
    if args.aoi == "apple":
        output = output / "aoi-apple"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for frame in sorted((ROOT / "dataset/raw_ingest").glob(f"*{args.sequence}*.jpg")):
        image = cv2.imread(str(frame))
        mask_file = (ROOT / "dataset/labels" / f"{frame.stem}.txt" if
                     args.mask_source == "yolo-seg" else
                     ROOT / "dataset/masks" / f"{frame.stem}.npz")
        if image is None or not mask_file.is_file():
            records.append({"frame": frame.name, "status": "missing_image_or_mask"})
            continue
        if args.mask_source == "yolo-seg":
            masks = masks_from_yolo_seg(mask_file, image.shape[:2])
            apple = masks["apple"]
            calyx = masks["stem_calyx"]
            defects = {name: masks[name] for name in ("defect_surface", "defect_critical")}
        else:
            with np.load(mask_file) as masks:
                apple = masks["apple"]
                calyx = masks["stem_calyx"]
                defects = {name: masks[name] for name in ("defect_surface", "defect_critical")}
        scored_defects = {}
        calyx_filter = {}
        for name, mask in defects.items():
            scored_defects[name], calyx_filter[name] = exclude_calyx_candidate(mask, calyx)
        bbox = apple_aoi(apple) if args.aoi == "apple" else (0, 0, image.shape[1], image.shape[0])
        x0, y0, x1, y1 = bbox
        started = time.perf_counter()
        crop_depth = infer(image[y0:y1, x0:x1])
        if args.backend == "depth-anything-v2":
            # DA-V2 Small emits a near-high relative map. The geometry mapper
            # uses farther-high values, so reverse polarity (not metric scale).
            crop_depth = -crop_depth
        elapsed_ms = (time.perf_counter() - started) * 1000
        if crop_depth.shape != (y1 - y0, x1 - x0) or not np.isfinite(crop_depth).all():
            raise RuntimeError(f"invalid depth map for {frame.name}: {crop_depth.shape}")
        depth = np.full(image.shape[:2], np.nan, np.float32)
        depth[y0:y1, x0:x1] = crop_depth
        if not np.isfinite(depth[apple > 0]).all():
            raise RuntimeError(f"apple depth is incomplete for {frame.name}")
        np.save(output / f"{frame.stem}_depth.npy", depth)
        view = cv2.normalize(crop_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_crop = cv2.applyColorMap(view, cv2.COLORMAP_TURBO)
        colored = np.zeros_like(image)
        colored[y0:y1, x0:x1] = colored_crop
        cv2.imwrite(str(output / f"{frame.stem}_depth.png"), colored)
        rgb_evidence = image[y0:y1, x0:x1].copy()
        depth_evidence = colored_crop.copy()
        for name, mask in defects.items():
            color = (255, 255, 255) if name == "defect_critical" else (255, 255, 0)
            contours, _ = cv2.findContours(mask[y0:y1, x0:x1].astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_evidence, contours, -1, color, 2)
            cv2.drawContours(depth_evidence, contours, -1, color, 2)
        evidence = np.concatenate((rgb_evidence, depth_evidence), axis=1)
        cv2.imwrite(str(output / f"{frame.stem}_evidence.jpg"), evidence)
        excluded = calyx.copy()
        for mask in defects.values():
            excluded |= mask
        try:
            residual, fit = fit_surface(depth, apple, excluded)
            noise = fit["intact_residual_mad"]
            fit["calyx_control"] = region_deviation(residual, calyx, noise)
            fit["defects"] = {name: region_deviation(residual, mask, noise)
                              for name, mask in scored_defects.items() if defects[name].any()}
            np.save(output / f"{frame.stem}_surface_residual.npy", residual)
            scale = max(6 * noise, 1e-6)
            contrast = np.clip(np.nan_to_num(residual / scale), -1, 1)
            intensity = (255 * (1 - np.abs(contrast))).astype(np.uint8)
            residual_view = np.zeros_like(image)
            valid = (apple > 0) & np.isfinite(residual)
            residual_view[valid] = np.stack((
                np.where(contrast > 0, intensity, 255),
                intensity,
                np.where(contrast < 0, intensity, 255),
            ), axis=-1)[valid]
            cv2.imwrite(str(output / f"{frame.stem}_surface_residual.png"), residual_view)
        except ValueError as exc:
            fit = {"status": "unavailable", "reason": str(exc)}
        records.append({
            "frame": frame.name,
            "status": "measured",
            "inference_ms": round(elapsed_ms, 1),
            "aoi_xyxy": list(bbox),
            "aoi_pixels": [x1 - x0, y1 - y0],
            "depth_min": float(crop_depth.min()),
            "depth_max": float(crop_depth.max()),
            "surface_fit": fit,
            "calyx_filter": {name: result for name, result in calyx_filter.items()
                             if defects[name].any()},
            "defects": {name: cavity_contrast(depth, apple, mask)
                        for name, mask in scored_defects.items() if defects[name].any()},
        })
    report = {
        "sequence": args.sequence,
        "model": args.backend,
        "depth_transform": ("negated near-high relative output" if
                            args.backend == "depth-anything-v2" else
                            "native predicted distance output"),
        "aoi": args.aoi,
        "mask_source": args.mask_source,
        "mask_provenance": "reviewed SAM 2 export; future YOLO-Seg output uses the same polygon format",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "policy_effect": "none; human reviewed labels and grading policy are unchanged",
        "interpretation": (
            "The probe describes depth output around reviewed masks. Residual "
            "is observed model output minus a smooth fit to nearby skin. "
            "It is not validated physical cavity depth. Cropping changes the "
            "model's global distance scale, so compare only local contrast."
        ),
        "frames": records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
