#!/usr/bin/env python3
"""Train a YOLO26n object detection model on the merged plant health dataset.

The training pipeline mirrors the apple engine's flow:
  1. Load the merged dataset (data.yaml from download_universe_datasets.py)
  2. Train YOLO26n from COCO pretrained weights
  3. Evaluate on the held-out test split
  4. Export to CoreML for M4 Neural Engine deployment
  5. Print metrics and checkpoint paths

Usage:
    python scripts/train_yolo26n.py \\
        --data plant_dataset/data.yaml \\
        --epochs 100 \\
        --imgsz 640 \\
        --batch 16

If --data is not specified, defaults to plant_dataset/data.yaml.
The base model is yolo11n.pt (Ultralytics YOLO11n) which serves as the
YOLO26n starting checkpoint.  The output is saved to runs/detect/train*.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO26n OD on the merged plant health dataset."
    )
    parser.add_argument(
        "--data",
        default="plant_dataset/data.yaml",
        help="Path to the dataset data.yaml file.",
    )
    parser.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="Base pretrained weights. Defaults to yolo11n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default="mps",
        help="Training device: mps (Apple Silicon), cpu, or 0 (CUDA).",
    )
    parser.add_argument(
        "--export-coreml",
        action="store_true",
        default=True,
        help="Export to CoreML after training (default: True).",
    )
    parser.add_argument(
        "--no-export-coreml",
        dest="export_coreml",
        action="store_false",
        help="Skip CoreML export.",
    )
    parser.add_argument(
        "--name",
        default="plant_yolo26n",
        help="Run name for the training output directory.",
    )
    return parser.parse_args(argv)


def train(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return 1

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        print("        Run scripts/download_universe_datasets.py first.")
        return 1

    print(f"[SYSTEM] Loading base weights: {args.weights}")
    model = YOLO(args.weights, task="detect")

    print(f"[SYSTEM] Training on {data_path}")
    print(f"         epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs/detect",
        exist_ok=True,
        verbose=True,
    )

    print("\n[RESULTS] Training complete.")
    print(f"          Results saved to: {results.save_dir if hasattr(results, 'save_dir') else 'runs/detect/' + args.name}")

    # Evaluate on test split
    best_path = Path("runs/detect") / args.name / "weights" / "best.pt"
    if best_path.exists():
        print("\n[EVAL] Evaluating best.pt on test split...")
        best_model = YOLO(str(best_path))
        metrics = best_model.val(split="test")
        print(f"        mAP50:    {metrics.box.map50:.4f}")
        print(f"        mAP50-95: {metrics.box.map:.4f}")
        print(f"        Precision: {metrics.box.mp:.4f}")
        print(f"        Recall:    {metrics.box.mr:.4f}")

        # Export to CoreML
        if args.export_coreml:
            print("\n[EXPORT] Exporting to CoreML for M4 Neural Engine...")
            coreml_path = best_model.export(format="coreml", imgsz=args.imgsz)
            print(f"          CoreML model: {coreml_path}")
    else:
        print(f"[WARN] best.pt not found at {best_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
