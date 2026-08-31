#!/usr/bin/env python3
"""Deterministic profile-level dataset splitting.

A profile is one physical plant and all five views.  All views of one
plant must remain in the same partition to prevent data leakage.

Reads the capture manifest (JSONL) and writes YOLO-format split files
(``splits/train.txt``, ``splits/val.txt``, ``splits/test.txt``) listing
absolute image paths.

Usage:
    python scripts/split_profiles.py \\
        --manifest dataset/raw_ingest/manifest.jsonl \\
        --output-dir splits \\
        --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split capture profiles into train/val/test partitions."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-views", type=int, default=5,
        help="Expected views per profile. Profiles with fewer are rejected."
    )
    return parser.parse_args(argv)


def load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_profile(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for rec in records:
        pid = rec["profile_id"]
        groups.setdefault(pid, []).append(rec)
    return groups


def validate_profile(profile_id: str, records: list[dict], expected_views: int) -> bool:
    grades = {r["reference_grade"] for r in records}
    view_indexes = [int(r["view_index"]) for r in records]
    if len(grades) != 1:
        print(f"[WARN] profile {profile_id} has inconsistent reference grades: {grades}")
        return False
    if len(view_indexes) != expected_views or len(set(view_indexes)) != expected_views:
        print(f"[WARN] profile {profile_id} must contain {expected_views} unique views, got {len(view_indexes)}")
        return False
    return True


def deterministic_bucket(profile_id: str, seed: int, num_buckets: int = 100) -> int:
    """Map a profile_id to a deterministic bucket in [0, num_buckets)."""
    digest = hashlib.sha256(f"{seed}:{profile_id}".encode()).hexdigest()
    return int(digest[:8], 16) % num_buckets


def split_profiles(
    profile_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    total = sum([train_ratio, val_ratio, test_ratio])
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    train_cutoff = int(train_ratio * 100)
    val_cutoff = train_cutoff + int(val_ratio * 100)

    train, val, test = [], [], []
    for pid in sorted(profile_ids):
        bucket = deterministic_bucket(pid, seed)
        if bucket < train_cutoff:
            train.append(pid)
        elif bucket < val_cutoff:
            val.append(pid)
        else:
            test.append(pid)

    return train, val, test


def write_split(split_name: str, profile_ids: list[str],
                profile_groups: dict[str, list[dict]], output_dir: Path) -> int:
    split_file = output_dir / f"{split_name}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with split_file.open("w", encoding="utf-8") as fh:
        for pid in profile_ids:
            for rec in profile_groups[pid]:
                fh.write(f"{Path(rec['image_path']).resolve()}\n")
                count += 1
    print(f"[OK] {split_name}: {len(profile_ids)} profiles, {count} images → {split_file}")
    return count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 1

    records = load_manifest(manifest_path)
    groups = group_by_profile(records)

    # Validate profiles
    valid_ids = []
    for pid, prof_records in groups.items():
        if validate_profile(pid, prof_records, args.expected_views):
            valid_ids.append(pid)

    if not valid_ids:
        print("[ERROR] No valid profiles found.")
        return 1

    print(f"[INFO] {len(valid_ids)} valid profiles out of {len(groups)} total.")

    train_ids, val_ids, test_ids = split_profiles(
        valid_ids, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    output_dir = Path(args.output_dir)
    write_split("train", train_ids, groups, output_dir)
    write_split("val", val_ids, groups, output_dir)
    write_split("test", test_ids, groups, output_dir)

    print(f"\n[DONE] Splits written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
