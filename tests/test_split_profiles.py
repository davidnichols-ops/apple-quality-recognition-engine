import json

import pytest

from scripts.split_profiles import (
    assert_no_profile_leakage,
    assign_records,
    partition_cohorts,
    read_manifest,
    stable_profile_split,
    validate_profiles,
    write_calyx_inventory,
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
        for profile, grade in enumerate(("G1", "G2", "G3", "CIDER", "DISCARD"))
        for view in range(4)
    ]


def test_stable_split_is_deterministic() -> None:
    assert stable_profile_split("profile-1") == stable_profile_split("profile-1")


def test_assignment_keeps_all_views_of_profile_together() -> None:
    assignments = assign_records(records())
    assert_no_profile_leakage(assignments)
    for profile in range(5):
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
    with pytest.raises(ValueError, match=r"\[0, 1, 2, 3\] exactly"):
        validate_profiles(records()[:-1])


def test_duplicate_view_is_rejected() -> None:
    duplicated = records()
    duplicated[-1] = {**duplicated[-1], "view_index": 2}
    with pytest.raises(ValueError, match=r"\[0, 1, 2, 3\] exactly"):
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
    assert sum(split["profiles"] for split in summary.values()) == 5


def test_standalone_calyx_is_inventoried_not_split(tmp_path) -> None:
    equatorial = [{**record, "view_type": "equatorial"} for record in records()]
    calyx = {
        "profile_id": "batch-stem-00000",
        "reference_grade": "unknown",
        "image_path": "images/calyx.jpg",
        "view_index": 0,
        "view_type": "calyx",
    }
    grading, references = partition_cohorts([*equatorial, calyx])
    assert len(grading) == 20
    assert references == [calyx]

    assignments = assign_records(grading)
    assert all(calyx not in split_records for split_records in assignments.values())
    write_calyx_inventory(references, tmp_path)
    payload = json.loads((tmp_path / "standalone_calyx_inventory.jsonl").read_text())
    assert payload["image_path"] == "images/calyx.jpg"


def test_equatorial_indexes_must_be_exactly_zero_through_three() -> None:
    shifted = records()
    shifted[:4] = [
        {**record, "view_index": index}
        for record, index in zip(shifted[:4], (1, 2, 3, 4), strict=True)
    ]
    with pytest.raises(ValueError, match=r"\[0, 1, 2, 3\] exactly"):
        validate_profiles(shifted)
