#!/usr/bin/env python3
"""Download and merge multiple Roboflow Universe plant datasets into a
unified 4-class YOLO dataset.

Target schema (data.yaml):
    0: plant            (parent bounding box)
    1: leaf             (individual leaf instance)
    2: unfit_discard    (discard trigger)
    3: class_defect     (generic defect on leaf)

Source datasets on Roboflow Universe are downloaded via the Roboflow Python
SDK.  Each source dataset's native class names are mapped to our 4-class
schema via a configurable mapping table.  Datasets that lack ``plant``
parent boxes or ``leaf`` instance boxes will have them computed from the
bounding boxes of their existing annotations (leaf = original annotation,
plant = union of all leaf boxes in the image).

Usage:
    python scripts/download_universe_datasets.py \\
        --output-dir plant_dataset \\
        --api-key "$ROBOFLOW_API_KEY"

The script is idempotent: if a dataset has already been downloaded and
merged, it is skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ── Dataset registry ────────────────────────────────────────────────
#
# Each entry maps a Universe dataset (workspace/project) to our 4-class
# schema.  The ``class_map`` maps source class names to target class IDs.
#   - Source classes not listed are mapped to ``class_defect`` (3) by
#     default, since most unrecognised disease names are defects.
#   - ``"healthy"`` variants map to ``leaf`` (1) — they are leaf instances
#     without defects.
#   - ``"plant"`` variants map to ``plant`` (0) — the parent box.
#   - Classes listed under ``"discard"`` map to ``unfit_discard`` (2).

DATASETS = [
    {
        "name": "eggplant_diseases",
        "workspace": "jomans-workspace-rlmua",
        "project": "all_eggplant_diseases",
        "version": 3,
        "format": "yolov8",
        "class_map": {
            "Healthy Plant": 0,      # plant parent
            "Healthy Leaf": 1,       # leaf instance
            "Fruit_Rot": 3,
            "Fruit_borer": 3,
            "Insect-Pest": 3,
            "Leaf-Spot": 3,
            "Melon_Thrips": 3,
            "Mosaic": 3,
            "White-Mold": 3,
            "Wilt": 3,
        },
    },
    {
        "name": "strawberry_diseases",
        "workspace": "research-proj",
        "project": "strawberry-diseases-detection",
        "version": 1,
        "format": "yolov8",
        "class_map": {
            "Healthy Leaf": 1,
            "Healthy Flower": 1,
            "Healthy Fruit": 1,
            "Angular Leafspot": 3,
            "Anthracnose Fruit Rot": 3,
            "Blossom Blight": 3,
            "Gray Mold": 3,
            "Leaf Spot": 3,
            "Powdery Mildew Fruit": 3,
            "Powdery Mildew Leaf": 3,
        },
    },
    {
        "name": "plant_disease_floragenic",
        "workspace": "floragenic-9v9os",
        "project": "plant-disease-detection-3anip",
        "version": 2,
        "format": "yolov8",
        "class_map": {
            # Leaf-type classes → leaf instance
            "Apple leaf": 1,
            "grape leaf": 1,
            "Bell_pepper leaf": 1,
            "Blueberry leaf": 1,
            "Cherry leaf": 1,
            "Peach leaf": 1,
            "Potato leaf": 1,
            "Raspberry leaf": 1,
            "Soyabean leaf": 1,
            "Soybean leaf": 1,
            "Strawberry leaf": 1,
            # Disease classes → class_defect
            "Apple Scab Leaf": 3,
            "Apple rust leaf": 3,
            "Bell_pepper leaf spot": 3,
            "Corn Gray leaf spot": 3,
            "Corn leaf blight": 3,
            "Corn rust leaf": 3,
            "Potato leaf early blight": 3,
            "Potato leaf late blight": 3,
            "Squash Powdery mildew leaf": 3,
        },
    },
    {
        "name": "plant_disease_89qrx",
        "workspace": "plant-disease-detection-89qrx",
        "project": "plant-disease-detection-znzrh",
        "version": 10,
        "format": "yolov8",
        "class_map": {
            # Healthy variants → leaf
            "apple healthy": 1,
            "bell pepper healthy": 1,
            "cherry healthy": 1,
            "corn healthy": 1,
            "grape healthy": 1,
            "healthy cassava": 1,
            # Disease variants → class_defect
            "apple black rot": 3,
            "apple scab": 3,
            "bell pepper bacterial spot": 3,
            "cassava Brown Streak Disease ": 3,
            "cassava bacterial blight": 3,
            "cassava green mottle": 3,
            "cedar apple rust": 3,
            "cherry powdery mildew": 3,
            "corn cerespora leaf spot": 3,
            "corn common rust": 3,
            "grape black rot": 3,
            "grape esca": 3,
            "grape leaf blight": 3,
            "mosaic  cassava": 3,
        },
    },
]

# Default fallback for unmapped classes
DEFAULT_CLASS_ID = 3  # class_defect

# Our target class names
TARGET_NAMES = {0: "plant", 1: "leaf", 2: "unfit_discard", 3: "class_defect"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and merge Roboflow Universe plant datasets."
    )
    parser.add_argument(
        "--output-dir",
        default="plant_dataset",
        help="Root directory for the merged YOLO dataset.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key. Defaults to $ROBOFLOW_API_KEY.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a dataset is already present.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of dataset names to download. Defaults to all.",
    )
    return parser.parse_args(argv)


def download_one(entry: dict, raw_dir: Path, api_key: str, force: bool) -> Path | None:
    """Download a single Universe dataset via the Roboflow SDK."""
    dest = raw_dir / entry["name"]
    if dest.exists() and not force:
        print(f"[SKIP] {entry['name']} already downloaded at {dest}")
        return dest

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed. Run: pip install roboflow")
        return None

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(entry["workspace"])
    project = workspace.project(entry["project"])
    version = project.version(entry["version"])
    dataset = version.download(entry["format"], location=str(dest))

    print(f"[OK] Downloaded {entry['name']} to {dest}")
    _ = dataset  # download object not needed further
    return dest


def remap_label_file(
    label_path: Path,
    class_map: dict[str, int],
    image_size: tuple[int, int],
) -> list[str]:
    """Remap a YOLO label file from source classes to our 4-class schema.

    YOLO label format: ``class_id cx cy w h`` (normalised 0-1).
    Returns a list of remapped label lines.
    """
    # We need the source data.yaml to map source class IDs to names.
    # This is handled by the caller which passes the names list.
    raise NotImplementedError("Use remap_label_file_with_names instead")


def remap_label_file_with_names(
    label_path: Path,
    source_names: list[str],
    class_map: dict[str, int],
) -> list[str]:
    """Remap a YOLO label file using source class names.

    Args:
        label_path: Path to the source .txt label file.
        source_names: List of source class names (index = class_id).
        class_map: Mapping from source class name → target class ID.

    Returns:
        List of remapped label lines (YOLO format).
    """
    remapped = []
    if not label_path.exists():
        return remapped

    with label_path.open("r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            src_class_id = int(parts[0])
            coords = parts[1:5]

            if src_class_id < 0 or src_class_id >= len(source_names):
                continue

            src_name = source_names[src_class_id]
            target_id = class_map.get(src_name, DEFAULT_CLASS_ID)
            remapped.append(f"{target_id} {' '.join(coords)}")

    return remapped


def compute_plant_box_from_leaves(
    label_lines: list[str],
    img_w: int,
    img_h: int,
) -> str | None:
    """Compute a parent plant box as the union of all leaf boxes.

    If there are leaf boxes (class 1) but no plant box (class 0),
    compute the bounding box that encompasses all leaves and add it
    as a plant box.

    Returns a YOLO label line for the plant box, or None if no leaves.
    """
    has_plant = False
    leaf_boxes = []

    for line in label_lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        if cls == 0:
            has_plant = True
        elif cls == 1:
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            leaf_boxes.append((x1, y1, x2, y2))

    if has_plant or not leaf_boxes:
        return None

    x1 = min(b[0] for b in leaf_boxes)
    y1 = min(b[1] for b in leaf_boxes)
    x2 = max(b[2] for b in leaf_boxes)
    y2 = max(b[3] for b in leaf_boxes)

    # Add small padding
    pad = 0.02
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(1.0, x2 + pad)
    y2 = min(1.0, y2 + pad)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def find_source_names(dataset_dir: Path) -> list[str]:
    """Read the source data.yaml to get class names."""
    for yaml_path in dataset_dir.rglob("data.yaml"):
        try:
            import yaml
            with yaml_path.open("r") as fh:
                data = yaml.safe_load(fh)
            names = data.get("names")
            if isinstance(names, dict):
                max_id = max(names.keys())
                return [names.get(i, f"unknown_{i}") for i in range(max_id + 1)]
            if isinstance(names, list):
                return names
        except Exception:
            continue
    return []


def merge_one_dataset(
    entry: dict,
    raw_dir: Path,
    output_dir: Path,
    split: str,
) -> int:
    """Merge one downloaded dataset into the output directory.

    Returns the number of images merged.
    """
    src_dir = raw_dir / entry["name"]
    if not src_dir.exists():
        return 0

    source_names = find_source_names(src_dir)
    if not source_names:
        print(f"[WARN] Could not find source class names for {entry['name']}")
        return 0

    class_map = entry["class_map"]

    # Find image and label directories for this split
    # Roboflow YOLOv8 format: train/images, train/labels (or data/train/images)
    img_dirs = list(src_dir.rglob(f"{split}/images"))
    lbl_dirs = list(src_dir.rglob(f"{split}/labels"))

    if not img_dirs:
        # Try alternate layout
        img_dirs = list(src_dir.rglob(f"{split}"))
        img_dirs = [d for d in img_dirs if d.is_dir() and any(d.iterdir())]

    if not img_dirs:
        print(f"[WARN] No {split} split found in {entry['name']}")
        return 0

    out_img = output_dir / "images" / split
    out_lbl = output_dir / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_dir, lbl_dir in zip(img_dirs, lbl_dirs or [img_dirs[0]]):
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue

            # Find corresponding label
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Remap labels
            remapped = remap_label_file_with_names(lbl_path, source_names, class_map)
            if not remapped:
                continue

            # Compute plant box if missing
            plant_line = compute_plant_box_from_leaves(remapped, 0, 0)
            if plant_line:
                remapped.insert(0, plant_line)

            # Write with prefixed filename to avoid collisions
            prefix = entry["name"] + "_"
            out_img_path = out_img / (prefix + img_path.name)
            out_lbl_path = out_lbl / (prefix + img_path.stem + ".txt")

            shutil.copy2(img_path, out_img_path)
            with out_lbl_path.open("w") as fh:
                fh.write("\n".join(remapped) + "\n")
            count += 1

    print(f"[MERGED] {entry['name']} {split}: {count} images")
    return count


def write_merged_data_yaml(output_dir: Path) -> None:
    """Write the merged dataset's data.yaml."""
    yaml_path = output_dir / "data.yaml"
    content = f"""# Merged plant health dataset — 4-class schema
# Generated by scripts/download_universe_datasets.py

path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: plant
  1: leaf
  2: unfit_discard
  3: class_defect
"""
    yaml_path.write_text(content)
    print(f"[OK] Wrote {yaml_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    api_key = args.api_key
    if not api_key:
        import os
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("[ERROR] No API key. Pass --api-key or set $ROBOFLOW_API_KEY.")
        return 1

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS
    if args.datasets:
        datasets = [d for d in DATASETS if d["name"] in args.datasets]

    # Download
    for entry in datasets:
        download_one(entry, raw_dir, api_key, args.force)

    # Merge into train/val/test
    total = 0
    for split in ("train", "valid", "test"):
        split_out = "val" if split == "valid" else split
        for entry in datasets:
            total += merge_one_dataset(entry, raw_dir, output_dir, split_out)

    write_merged_data_yaml(output_dir)
    print(f"\n[DONE] Merged {total} images into {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
