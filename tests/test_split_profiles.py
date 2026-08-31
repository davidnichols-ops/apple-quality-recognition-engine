import json

import pytest

from scripts.split_profiles import (
    assert_no_profile_leakage,
    assign_records,
    read_manifest,
    stable_profile_split,
    validate_profiles,
    write_splits,
)


def records() -> list[dict]:
    return [
        {
            "profile_id": f"profile-{profile}",
            "reference_grade": grade,
            "image_path": f"images/profile-{profile}-view-{view}.jpg",
            "view_index": view,
        }
        for profile, grade in enumerate(("G1", "G2", "G3", "DISCARD"))
        for view in range(5)
    ]


def test_stable_split_is_deterministic() -> None:
    assert stable_profile_split("profile-1") == stable_profile_split("profile-1")


def test_assignment_keeps_all_views_of_profile_together() -> None:
    assignments = assign_records(records())
    assert_no_profile_leakage(assignments)
    for profile in range(4):
        containing = [
            split
            for split, split_records in assignments.items()
            if any(
                record["profile_id"] == f"profile-{profile}" for record in split_records
            )
        ]
        assert len(containing) == 1


def test_invalid_ratios_are_rejected() -> None:
    with pytest.raises(ValueError):
        stable_profile_split("profile", train_ratio=0.9, val_ratio=0.2)


def test_incomplete_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="5 unique views"):
        validate_profiles(records()[:-1])


def test_duplicate_view_is_rejected() -> None:
    duplicated = records()
    duplicated[-1] = {**duplicated[-1], "view_index": 3}
    with pytest.raises(ValueError, match="5 unique views"):
        validate_profiles(duplicated)


def test_manifest_requires_profile_grade_and_path(tmp_path) -> None:
    manifest = tmp_path / "capture_manifest.jsonl"
    manifest.write_text('{"profile_id": "missing-fields"}\n')
    with pytest.raises(ValueError, match="missing fields"):
        read_manifest(manifest)


def test_write_splits_emits_lists_and_summary(tmp_path) -> None:
    assignments = assign_records(records())
    write_splits(assignments, tmp_path)

    listed_paths = {
        line
        for split in ("train", "val", "test")
        for line in (tmp_path / f"{split}.txt").read_text().splitlines()
    }
    summary = json.loads((tmp_path / "split_summary.json").read_text())
    assert listed_paths == {record["image_path"] for record in records()}
    assert sum(split["images"] for split in summary.values()) == 20
    assert sum(split["profiles"] for split in summary.values()) == 4
