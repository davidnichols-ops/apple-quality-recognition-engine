#!/usr/bin/env python3
"""Create deterministic train/validation/test lists without profile leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

SPLITS = ("train", "val", "test")


def stable_profile_split(
    profile_id: str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: str = "apple-quality-v1",
) -> str:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    digest = hashlib.sha256(f"{seed}:{profile_id}".encode()).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"


def read_manifest(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"profile_id", "reference_grade", "image_path", "view_index"}
            missing = required - set(record)
            if missing:
                raise ValueError(
                    f"manifest line {line_number} missing fields: {sorted(missing)}"
                )
            records.append(record)
    return records


def validate_profiles(records: Iterable[dict], expected_views: int = 5) -> None:
    profiles: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        profiles[str(record["profile_id"])].append(record)
    for profile_id, profile_records in profiles.items():
        grades = {record["reference_grade"] for record in profile_records}
        view_indexes = [int(record["view_index"]) for record in profile_records]
        if len(grades) != 1:
            raise ValueError(f"profile {profile_id} has inconsistent reference grades")
        if (
            len(view_indexes) != expected_views
            or len(set(view_indexes)) != expected_views
        ):
            raise ValueError(
                f"profile {profile_id} must contain {expected_views} unique views"
            )


def assign_records(
    records: Iterable[dict],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: str = "apple-quality-v1",
) -> dict[str, list[dict]]:
    assignments = {split: [] for split in SPLITS}
    profile_splits: dict[str, str] = {}
    for record in records:
        profile_id = str(record["profile_id"])
        split = stable_profile_split(
            profile_id,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        previous = profile_splits.setdefault(profile_id, split)
        if previous != split:
            raise AssertionError(f"profile {profile_id} crossed split boundaries")
        assignments[split].append(record)
    return assignments


def write_splits(assignments: dict[str, list[dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        paths = sorted(str(record["image_path"]) for record in assignments[split])
        (output_dir / f"{split}.txt").write_text(
            "".join(f"{path}\n" for path in paths),
            encoding="utf-8",
        )

    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        profiles = {record["profile_id"] for record in assignments[split]}
        grades = Counter(record["reference_grade"] for record in assignments[split])
        counts[split] = {
            "images": len(assignments[split]),
            "profiles": len(profiles),
            **{f"grade_{grade}": count for grade, count in sorted(grades.items())},
        }
    (output_dir / "split_summary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_no_profile_leakage(assignments: dict[str, list[dict]]) -> None:
    locations: dict[str, set[str]] = defaultdict(set)
    for split, records in assignments.items():
        for record in records:
            locations[str(record["profile_id"])].add(split)
    leaked = {
        profile: splits for profile, splits in locations.items() if len(splits) > 1
    }
    if leaked:
        raise ValueError(f"profile leakage detected: {leaked}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset/raw_ingest/capture_manifest.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("apple_dataset/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", default="apple-quality-v1")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_manifest(args.manifest)
    validate_profiles(records)
    assignments = assign_records(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    assert_no_profile_leakage(assignments)
    write_splits(assignments, args.output_dir)
    profile_count = len({record["profile_id"] for record in records})
    print(
        f"Wrote {len(records)} images from {profile_count} profiles to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
