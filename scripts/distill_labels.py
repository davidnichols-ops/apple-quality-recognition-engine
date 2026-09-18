#!/usr/bin/env python3
"""Production SAM 2 -> YOLO-Seg distillation pipeline.

Subcommands (run in order):

  pass-a     Auto-label apple + stem_calyx on every sequence.
             Classical threshold bootstraps the apple box; a greenest-point
             heuristic seeds the calyx basin; SAM2VideoPredictor propagates
             both across the turntable rotation.
  pass-b     Interactive defect prompting. Step through frames (a/d), pick
             class (2=surface, 3=critical), drag loose boxes. Handles
             mid-orbit disocclusion: prompts may be dropped on ANY frame.
  pass-c     Propagate defect prompts with multi-frame conditioning and
             merge into the mask store.
  export     Apply priority hierarchy (3 > 1 > 2, all clipped to apple
             interior), vectorize to YOLO-Seg .txt, render overlays.
  split      Sequence-level train/val split (80/20 by key hash) - all
             views of one apple stay in one partition. Negatives (empty
             labels) go to train only.
  benchmark  Time model load + per-frame propagation; extrapolate to a
             target frame count to decide local-vs-offload.
  run        pass-a + pass-c + export (use after pass-b, or standalone
             for auto-only labeling).

Layout:
  dataset/raw_ingest/*.jpg        source frames (production + negatives)
  dataset/masks/<frame>.npz       per-frame mask channels (intermediate)
  dataset/prompts/<seq>.json      defect prompts from pass-b
  dataset/labels/<frame>.txt      final YOLO-Seg labels
  dataset/preview/<frame>.jpg     overlay renders
  dataset/splits/{train,val}/     images/ + labels/ for yolo train
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw_ingest"
MASKS = ROOT / "dataset" / "masks"
PROMPTS = ROOT / "dataset" / "prompts"
LABELS = ROOT / "dataset" / "labels"
PREVIEW = ROOT / "dataset" / "preview"
SPLITS = ROOT / "dataset" / "splits"

CLASS_NAMES = {0: "apple", 1: "stem_calyx", 2: "defect_surface", 3: "defect_critical"}
# export priority: higher number wins; surface loses to calyx and critical
PRIORITY = {3: 3, 1: 2, 2: 1}

CKPTS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base": ("configs/sam2.1/sam2.1_hiera_b.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}

RAW_RE = re.compile(
    r"^raw_\d{8}_\d{6}_\d{6}_(.+)_(g1|g2|g3|cider|discard|unknown)"
    r"_(equatorial|stem|negative)_view_\d+\.jpg$"
)
NEG_RE = re.compile(r"^negative_(\d{8})_(\d{6})_\d+\.jpg$")


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------

def discover_sequences(raw_dir: Path, gap_s: int = 60) -> dict[str, list[Path]]:
    """Group frames into per-apple sequences.

    Production captures share a profile_id in the filename; negatives and
    ad-hoc shots fall back to timestamp-gap grouping.
    """
    seqs: dict[str, list[Path]] = {}
    orphans: list[Path] = []
    for f in sorted(raw_dir.glob("*.jpg")):
        m = RAW_RE.match(f.name)
        if m:
            seqs.setdefault(m.group(1), []).append(f)
        else:
            orphans.append(f)

    # timestamp-gap grouping for non-profile files (negatives, POC shots)
    def ts(p: Path) -> str:
        m = NEG_RE.match(p.name)
        return m.group(1) + m.group(2) if m else "0"

    bucket: list[Path] = []
    for f in sorted(orphans, key=ts):
        if bucket and abs(int(ts(f)[-6:]) - int(ts(bucket[-1])[-6:])) > gap_s:
            seqs[f"adhoc-{ts(bucket[0])}"] = bucket
            bucket = []
        bucket.append(f)
    if bucket:
        seqs[f"adhoc-{ts(bucket[0])}"] = bucket
    return seqs


# ---------------------------------------------------------------------------
# Classical bootstraps
# ---------------------------------------------------------------------------

def apple_box_and_mask(img: np.ndarray, min_area: int = 5000) -> tuple[np.ndarray, np.ndarray] | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(big) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(big)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [big], -1, 255, cv2.FILLED)
    return np.array([x, y, x + w, y + h], np.float32), mask


def pick_calyx_point(img: np.ndarray, apple_mask: np.ndarray) -> np.ndarray | None:
    """Greenest spot in the eroded central-lower region (the stem basin)."""
    eroded = cv2.erode(apple_mask, np.ones((41, 41), np.uint8))
    if eroded.sum() == 0:
        return None
    ys, xs = np.where(eroded > 0)
    cx, cy, bw = xs.mean(), ys.mean(), xs.max() - xs.min()
    region = np.zeros_like(eroded)
    region[eroded > 0] = 1
    region[: int(cy), :] = 0
    region[:, : int(cx - 0.30 * bw)] = 0
    region[:, int(cx + 0.30 * bw):] = 0
    if region.sum() < 50:
        return None
    b, g, r = cv2.split(img.astype(np.float32))
    score = g - r
    score[region == 0] = -1e9
    sel = (score >= np.percentile(score[region > 0], 99)) & (region > 0)
    if sel.sum() < 5:
        return None
    py, px = np.where(sel)
    return np.array([[px.mean(), py.mean()]], np.float32)


# ---------------------------------------------------------------------------
# Mask store
# ---------------------------------------------------------------------------

CHANNELS = {0: "apple", 1: "stem_calyx", 2: "defect_surface", 3: "defect_critical"}


def load_mask_store(frame_stem: str, shape: tuple[int, int]) -> dict[int, np.ndarray]:
    path = MASKS / f"{frame_stem}.npz"
    out = {c: np.zeros(shape, np.uint8) for c in CHANNELS}
    if path.exists():
        z = np.load(path)
        for c, name in CHANNELS.items():
            if name in z:
                out[c] = z[name]
    return out


def save_mask_store(frame_stem: str, store: dict[int, np.ndarray]) -> None:
    np.savez_compressed(
        MASKS / f"{frame_stem}.npz",
        **{CHANNELS[c]: m for c, m in store.items()},
    )


# ---------------------------------------------------------------------------
# SAM 2 video propagation
# ---------------------------------------------------------------------------

def build_video_predictor(device: str, model_size: str):
    from sam2.build_sam import build_sam2_video_predictor
    cfg, ckpt = CKPTS[model_size]
    ckpt_path = ROOT / "checkpoints" / ckpt
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] missing checkpoint: {ckpt_path}\n"
                 f"  curl -L -o {ckpt_path} https://dl.fbaipublicfiles.com/"
                 f"segment_anything_2/092824/{ckpt}")
    t0 = time.time()
    predictor = build_sam2_video_predictor(cfg, str(ckpt_path), device=device)
    print(f"[SAM2] {model_size} video predictor on {device} "
          f"(load {time.time() - t0:.1f}s)")
    return predictor


def propagate(predictor, seq_dir: Path, prompts: list[dict]) -> dict[int, dict[int, np.ndarray]]:
    """Condition on all prompted frames, propagate, return {obj_id: {fidx: mask}}."""
    state = predictor.init_state(video_path=str(seq_dir))
    for i, p in enumerate(prompts):
        kwargs = {"box": np.array(p["box"], np.float32)} if "box" in p else {
            "points": np.array(p["points"], np.float32),
            "labels": np.array(p["labels"], np.int32),
        }
        predictor.add_new_points_or_box(
            state, frame_idx=p["frame_idx"], obj_id=p["obj_id"], **kwargs)
    out: dict[int, dict[int, np.ndarray]] = {}
    for fidx, obj_ids, logits in predictor.propagate_in_video(state):
        for i, oid in enumerate(obj_ids):
            m = (logits[i] > 0).squeeze().cpu().numpy()
            if m.sum() >= 25:
                out.setdefault(int(oid), {})[fidx] = m.astype(np.uint8)
    return out


def stage_frames(seq: list[Path]) -> tempfile.TemporaryDirectory:
    td = tempfile.TemporaryDirectory()
    for i, src in enumerate(seq):
        shutil.copy(src, Path(td.name) / f"{i:05d}.jpg")
    return td


# ---------------------------------------------------------------------------
# pass-a: auto apple + calyx
# ---------------------------------------------------------------------------

def cmd_pass_a(args) -> None:
    seqs = discover_sequences(RAW)
    print(f"[PASS-A] {len(seqs)} sequences")
    predictor = build_video_predictor(args.device, args.model_size)
    MASKS.mkdir(parents=True, exist_ok=True)
    skipped = 0

    for si, (key, seq) in enumerate(sorted(seqs.items())):
        img0 = cv2.imread(str(seq[0]))
        res = apple_box_and_mask(img0)
        if res is None:
            skipped += 1
            print(f"  [{si}] {key}: no apple blob -> empty labels (negative)")
            for f in seq:
                save_mask_store(f.stem, load_mask_store(f.stem, img0.shape[:2]))
            continue
        box, apple_mask = res
        calyx_pt = pick_calyx_point(img0, apple_mask)

        prompts = [{"frame_idx": 0, "obj_id": 0, "box": box.tolist()}]
        if calyx_pt is not None:
            prompts.append({"frame_idx": 0, "obj_id": 1,
                            "points": calyx_pt.tolist(), "labels": [1]})

        with stage_frames(seq) as td:
            masks = propagate(predictor, Path(td), prompts)

        for fidx, f in enumerate(seq):
            store = load_mask_store(f.stem, img0.shape[:2])
            if 0 in masks and fidx in masks[0]:
                store[0] = masks[0][fidx]
            if 1 in masks and fidx in masks[1]:
                store[1] = masks[1][fidx]
            save_mask_store(f.stem, store)
        print(f"  [{si}] {key}: {len(seq)}f apple+calyx "
              f"(calyx {'yes' if calyx_pt is not None else 'NO - review'})")

    print(f"[PASS-A DONE] {len(seqs) - skipped} labeled, {skipped} empty/negative")


# ---------------------------------------------------------------------------
# pass-b: interactive defect prompting
# ---------------------------------------------------------------------------

def cmd_pass_b(args) -> None:
    """Step frames (a/d), pick class (2/3), drag boxes. SPACE=next seq, q=quit."""
    seqs = discover_sequences(RAW)
    PROMPTS.mkdir(parents=True, exist_ok=True)
    win = "DEFECT PROMPTING  |  a/d=frame  2/3=class  z=undo  SPACE=next  s=skip  q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {"fidx": 0, "cls": 2, "prompts": [], "drag": None, "img": None, "oid": 10}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0 = state["drag"]
            state["drag"] = None
            box = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            if box[2] - box[0] > 6 and box[3] - box[1] > 6:
                state["prompts"].append({
                    "frame_idx": state["fidx"], "class_id": state["cls"],
                    "obj_id": state["oid"], "box": box})
                state["oid"] += 1

    cv2.setMouseCallback(win, on_mouse)
    seq_items = sorted(seqs.items())
    si = args.start_seq
    while si < len(seq_items):
        key, seq = seq_items[si]
        if (PROMPTS / f"{key}.json").exists() and not args.redo:
            si += 1
            continue
        state["prompts"] = []
        state["fidx"] = 0
        state["oid"] = 10
        while True:
            img = cv2.imread(str(seq[state["fidx"]]))
            view = img.copy()
            for p in state["prompts"]:
                if p["frame_idx"] == state["fidx"]:
                    x0, y0, x1, y1 = [int(v) for v in p["box"]]
                    col = (0, 165, 255) if p["class_id"] == 2 else (0, 0, 255)
                    cv2.rectangle(view, (x0, y0), (x1, y1), col, 2)
            hud = (f"{key}  frame {state['fidx'] + 1}/{len(seq)}  "
                   f"class={state['cls']}({CLASS_NAMES[state['cls']]})  "
                   f"prompts={len(state['prompts'])}")
            cv2.putText(view, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow(win, view)
            k = cv2.waitKey(50) & 0xFF
            if k in (ord("a"),) and state["fidx"] > 0:
                state["fidx"] -= 1
            elif k in (ord("d"),) and state["fidx"] < len(seq) - 1:
                state["fidx"] += 1
            elif k == ord("2"):
                state["cls"] = 2
            elif k == ord("3"):
                state["cls"] = 3
            elif k in (ord("z"), 8) and state["prompts"]:
                state["prompts"].pop()
            elif k == 32:  # SPACE -> save & next
                if state["prompts"]:
                    (PROMPTS / f"{key}.json").write_text(json.dumps({
                        "sequence": key,
                        "frames": [f.name for f in seq],
                        "prompts": state["prompts"],
                    }, indent=2))
                    print(f"  [SAVED] {key}: {len(state['prompts'])} prompts")
                break
            elif k == ord("s"):
                break
            elif k in (ord("q"), 27):
                if state["prompts"]:
                    (PROMPTS / f"{key}.json").write_text(json.dumps({
                        "sequence": key,
                        "frames": [f.name for f in seq],
                        "prompts": state["prompts"],
                    }, indent=2))
                cv2.destroyAllWindows()
                print("[PASS-B] quit")
                return
        si += 1
    cv2.destroyAllWindows()
    print("[PASS-B DONE]")


# ---------------------------------------------------------------------------
# pass-c: defect propagation + merge
# ---------------------------------------------------------------------------

def cmd_pass_c(args) -> None:
    seqs = discover_sequences(RAW)
    prompt_files = {p.stem: p for p in PROMPTS.glob("*.json")}
    if not prompt_files:
        print("[PASS-C] no prompt files -> nothing to propagate")
        return
    predictor = build_video_predictor(args.device, args.model_size)

    for key, pf in sorted(prompt_files.items()):
        if key not in seqs:
            print(f"  [SKIP] {key}: no matching frames")
            continue
        seq = seqs[key]
        spec = json.loads(pf.read_text())
        prompts = [dict(p) for p in spec["prompts"]]
        img0 = cv2.imread(str(seq[0]))
        with stage_frames(seq) as td:
            masks = propagate(predictor, Path(td), prompts)

        n = 0
        for fidx, f in enumerate(seq):
            store = load_mask_store(f.stem, img0.shape[:2])
            for p in prompts:
                oid = p["obj_id"]
                if oid in masks and fidx in masks[oid]:
                    store[p["class_id"]] |= masks[oid][fidx]
                    n += 1
            save_mask_store(f.stem, store)
        print(f"  {key}: {len(prompts)} prompts -> {n} defect mask-frames")

    print("[PASS-C DONE]")


# ---------------------------------------------------------------------------
# export: priority merge -> YOLO-Seg + overlays
# ---------------------------------------------------------------------------

def mask_to_yolo(mask: np.ndarray, class_id: int, w: int, h: int) -> list[str]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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


def cmd_export(args) -> None:
    LABELS.mkdir(parents=True, exist_ok=True)
    if args.overlays:
        PREVIEW.mkdir(parents=True, exist_ok=True)
        import supervision as sv
    files = sorted(MASKS.glob("*.npz"))
    total = 0
    for mp in files:
        z = np.load(mp)
        ch = {c: z[name] for c, name in CHANNELS.items() if name in z}
        if not ch:
            continue
        h, w = next(iter(ch.values())).shape

        # clip everything to apple interior, then priority-subtract
        apple = ch.get(0, np.zeros((h, w), np.uint8))
        ordered = sorted(ch, key=lambda c: -PRIORITY.get(c, 0))
        final = {}
        occupied = np.zeros((h, w), np.uint8)
        for c in ordered:
            if c == 0:
                final[c] = ch[c]
                continue
            m = ch[c] & apple
            m = m & ~occupied
            final[c] = m
            occupied |= m

        lines = []
        for c, m in final.items():
            lines += mask_to_yolo(m, c, w, h)
        (LABELS / f"{mp.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)

        if args.overlays and any(m.sum() for m in final.values()):
            img = cv2.imread(str(RAW / f"{mp.stem}.jpg"))
            if img is None:
                continue
            masks_l, cls_l, boxes_l = [], [], []
            for c, m in final.items():
                for cnt in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                    if cv2.contourArea(cnt) < 25:
                        continue
                    mm = np.zeros((h, w), np.uint8)
                    cv2.drawContours(mm, [cnt], -1, 1, cv2.FILLED)
                    masks_l.append(mm.astype(bool))
                    cls_l.append(c)
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    boxes_l.append([x, y, x + bw, y + bh])
            if masks_l:
                det = sv.Detections(
                    xyxy=np.array(boxes_l, np.float32),
                    mask=np.stack(masks_l),
                    class_id=np.array(cls_l),
                )
                ann = sv.MaskAnnotator(opacity=0.5).annotate(img.copy(), det)
                ann = sv.LabelAnnotator().annotate(
                    ann, det, labels=[CLASS_NAMES[c] for c in cls_l])
                cv2.imwrite(str(PREVIEW / f"{mp.stem}_overlay.jpg"), ann)
    print(f"[EXPORT] {total} polys -> {LABELS} ({len(files)} frames)")


# ---------------------------------------------------------------------------
# split: sequence-level train/val
# ---------------------------------------------------------------------------

def cmd_split(args) -> None:
    seqs = discover_sequences(RAW)
    n_train = n_val = 0
    for key, seq in seqs.items():
        bucket = zlib.crc32(key.encode()) % 5
        split = "val" if bucket == 0 else "train"
        for f in seq:
            lbl = LABELS / f"{f.stem}.txt"
            has_labels = lbl.exists() and lbl.stat().st_size > 1
            dst = "train" if not has_labels else split  # negatives -> train only
            for sub in ("images", "labels"):
                (SPLITS / dst / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy(f, SPLITS / dst / "images" / f.name)
            shutil.copy(lbl if lbl.exists() else Path(os.devnull),
                        SPLITS / dst / "labels" / f"{f.stem}.txt")
            if dst == "val":
                n_val += 1
            else:
                n_train += 1
    print(f"[SPLIT] train={n_train} val={n_val} -> {SPLITS}")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args) -> None:
    seqs = discover_sequences(RAW)
    keys = sorted(seqs)[: args.seqs]
    predictor = build_video_predictor(args.device, args.model_size)
    per_frame: list[float] = []
    for key in keys:
        seq = seqs[key]
        img0 = cv2.imread(str(seq[0]))
        res = apple_box_and_mask(img0)
        if res is None:
            continue
        box, _ = res
        with stage_frames(seq) as td:
            t0 = time.time()
            propagate(predictor, Path(td),
                      [{"frame_idx": 0, "obj_id": 0, "box": box.tolist()}])
            dt = time.time() - t0
        per_frame.append(dt / len(seq))
        print(f"  {key}: {len(seq)}f in {dt:.1f}s ({dt / len(seq):.2f}s/frame)")
    if not per_frame:
        print("[BENCH] no usable sequences")
        return
    avg = sum(per_frame) / len(per_frame)
    target = args.target
    total_s = avg * target
    print(f"\n[BENCH] model={args.model_size} device={args.device}")
    print(f"  avg {avg:.2f}s/frame over {len(per_frame)} seqs")
    print(f"  {target} frames ~= {total_s / 60:.1f} min propagation")
    verdict = "LOCAL OK" if total_s < 3600 else "CONSIDER OFFLOAD"
    print(f"  verdict: {verdict} (<60min threshold)")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pass-a", "pass-b", "pass-c", "export",
                                    "split", "benchmark", "run"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-size", default="large", choices=list(CKPTS))
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--redo", action="store_true", help="pass-b: re-prompt sequences that already have prompts")
    ap.add_argument("--start-seq", type=int, default=0)
    ap.add_argument("--seqs", type=int, default=3, help="benchmark: sequences to time")
    ap.add_argument("--target", type=int, default=500, help="benchmark: extrapolate to N frames")
    args = ap.parse_args()

    if args.cmd == "pass-a":
        cmd_pass_a(args)
    elif args.cmd == "pass-b":
        cmd_pass_b(args)
    elif args.cmd == "pass-c":
        cmd_pass_c(args)
    elif args.cmd == "export":
        cmd_export(args)
    elif args.cmd == "split":
        cmd_split(args)
    elif args.cmd == "benchmark":
        cmd_benchmark(args)
    elif args.cmd == "run":
        cmd_pass_a(args)
        cmd_pass_c(args)
        cmd_export(args)


if __name__ == "__main__":
    main()
