#!/usr/bin/env python3
"""POC: SAM 2 -> YOLO-Seg distillation on the last captured frames.

Pipeline:
  1. Classical threshold bootstraps the parent `apple` mask + bbox
     (pure black backdrop, gray<20 separates cleanly).
  2. SAM 2 video predictor propagates the apple box + an auto-picked
     stem_calyx point prompt across each turntable sequence.
  3. Binary masks -> normalized YOLO-Seg polygon .txt per frame.
  4. supervision overlays rendered for visual sanity check.

Outputs:
  dataset/poc_labels/<frame>.txt   YOLO-Seg labels
  dataset/poc_preview/<frame>.jpg  mask overlay renders
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw_ingest"
LABELS = ROOT / "dataset" / "poc_labels"
PREVIEW = ROOT / "dataset" / "poc_preview"

# YOLO class IDs (data.yaml): 0 apple, 1 stem_calyx, 2 defect_surface, 3 defect_critical
OBJID_TO_CLASS = {1: 0, 2: 1}
TS_RE = re.compile(r"_(\d{8})_(\d{6})_")


def group_sequences(files: list[Path], gap_s: int = 60) -> list[list[Path]]:
    """Group frames into sequences by timestamp gap."""
    def key(p: Path) -> str:
        m = TS_RE.search(p.name)
        return f"{m.group(1)}{m.group(2)}" if m else p.name

    seqs: list[list[Path]] = []
    for f in sorted(files, key=key):
        if seqs:
            prev = int(key(seqs[-1][-1])[-6:])
            cur = int(key(f)[-6:])
            # crude HHMMSS diff; batches are minutes apart so 60s is plenty
            if abs(cur - prev) > gap_s:
                seqs.append([])
        seqs[-1].append(f) if seqs else seqs.append([f])
    return [s for s in seqs if s]


def apple_box_and_mask(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Step 1: Otsu-style threshold on the black stage -> apple bbox + mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(big)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [big], -1, 255, cv2.FILLED)
    return np.array([x, y, x + w, y + h], np.float32), mask


def pick_calyx_point(img: np.ndarray, apple_mask: np.ndarray) -> np.ndarray | None:
    """Heuristic stem/calyx point: greenest spot inside lower-center of the apple.

    Stem nubs are pale-green; skin is red/yellow (R-dominant). Score = G - R.
    The mask is eroded so backdrop-bleed edge pixels can't win, and the
    search is restricted to the central-lower region where the calyx basin
    sits in these captures.
    """
    eroded = cv2.erode(apple_mask, np.ones((41, 41), np.uint8))
    if eroded.sum() == 0:
        return None
    ys, xs = np.where(eroded > 0)
    cx, cy, bw = xs.mean(), ys.mean(), xs.max() - xs.min()

    region = np.zeros_like(eroded)
    region[eroded > 0] = 1
    region[: int(cy), :] = 0                       # lower half only
    region[:, : int(cx - 0.30 * bw)] = 0           # central band
    region[:, int(cx + 0.30 * bw):] = 0
    if region.sum() < 50:
        return None

    b, g, r = cv2.split(img.astype(np.float32))
    score = g - r
    score[region == 0] = -1e9
    thresh = np.percentile(score[region > 0], 99)
    sel = (score >= thresh) & (region > 0)
    if sel.sum() < 5:
        return None
    py, px = np.where(sel)
    return np.array([[px.mean(), py.mean()]], np.float32)


def mask_to_yolo(mask: np.ndarray, class_id: int, w: int, h: int) -> list[str]:
    """Step 3: binary mask -> normalized YOLO-Seg polygon lines.

    The `apple` class keeps only the largest contour (one fruit per frame);
    other classes keep all contours above the noise floor.
    """
    cnts, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if class_id == 0 and cnts:
        cnts = [max(cnts, key=cv2.contourArea)]
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < 25:
            continue
        poly = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
        pts = [f"{p[0] / w:.6f} {p[1] / h:.6f}" for p in poly]
        if len(pts) >= 3:
            lines.append(f"{class_id} " + " ".join(pts))
    return lines


def run_sequence(predictor, seq: list[Path], seq_name: str) -> int:
    """Steps 2-4 on one turntable sequence. Returns total label lines written."""
    import supervision as sv

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for i, src in enumerate(seq):
            shutil.copy(src, tdir / f"{i:05d}.jpg")

        img0 = cv2.imread(str(seq[0]))
        H, W = img0.shape[:2]
        box, apple_mask = apple_box_and_mask(img0)
        calyx_pt = pick_calyx_point(img0, apple_mask)

        state = predictor.init_state(video_path=str(tdir))
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
        if calyx_pt is not None:
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=2,
                points=calyx_pt, labels=np.array([1], np.int32),
            )
            print(f"  [{seq_name}] calyx prompt @ ({calyx_pt[0,0]:.0f},{calyx_pt[0,1]:.0f})")

        n_lines = 0
        for fidx, obj_ids, logits in predictor.propagate_in_video(state):
            src = seq[fidx]
            img = cv2.imread(str(src))
            masks, cls_ids, boxes, lines = [], [], [], []
            for i, oid in enumerate(obj_ids):
                m = (logits[i] > 0).squeeze().cpu().numpy().astype(bool)
                if m.sum() < 25:
                    continue
                masks.append(m)
                cls_ids.append(OBJID_TO_CLASS.get(int(oid), 0))
                ys, xs = np.where(m)
                boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
                lines += mask_to_yolo(m, OBJID_TO_CLASS.get(int(oid), 0), W, H)

            (LABELS / f"{src.stem}.txt").write_text("\n".join(lines) + "\n")
            n_lines += len(lines)

            det = sv.Detections(
                xyxy=np.array(boxes, np.float32),
                mask=np.stack(masks),
                class_id=np.array(cls_ids),
            )
            ann = sv.MaskAnnotator(opacity=0.5).annotate(img.copy(), det)
            ann = sv.LabelAnnotator().annotate(
                ann, det, labels=[["apple", "stem_calyx"][c] for c in cls_ids]
            )
            if fidx == 0 and calyx_pt is not None:
                cv2.drawMarker(
                    ann, (int(calyx_pt[0, 0]), int(calyx_pt[0, 1])),
                    (255, 255, 0), cv2.MARKER_CROSS, 30, 3,
                )
            cv2.imwrite(str(PREVIEW / f"{src.stem}_overlay.jpg"), ann)
            print(f"    {src.name}: {len(lines)} polys "
                  f"({[int(c) for c in cls_ids]})")
        return n_lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=11, help="use last N raw frames")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"))
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    args = ap.parse_args()

    files = sorted(RAW.glob("*.jpg"), key=lambda p: p.stat().st_mtime)[-args.last:]
    seqs = group_sequences(files)
    print(f"[POC] {len(files)} frames -> {len(seqs)} sequence(s): "
          f"{[len(s) for s in seqs]}")

    LABELS.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        config_file=args.cfg, ckpt_path=args.ckpt, device=args.device,
    )
    print(f"[SAM2] video predictor ready on {args.device} (tiny)")

    total = 0
    for i, seq in enumerate(seqs):
        print(f"[SEQ {i}] {len(seq)} frames: {seq[0].name} .. {seq[-1].name}")
        total += run_sequence(predictor, seq, f"seq{i}")

    print(f"\n[POC DONE] {total} polygon labels -> {LABELS}")
    print(f"           overlays -> {PREVIEW}")


if __name__ == "__main__":
    sys.exit(main())
