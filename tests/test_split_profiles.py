"""Unit tests for scripts/split_profiles.py."""

import json
import sys
from pathlib import Path

import pytest

# Add scripts dir to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from split_profiles import (
    deterministic_bucket,
    group_by_profile,
    load_manifest,
    split_profiles,
    validate_profile,
    write_split,
)


def make_record(profile_id, view_index, grade="HEALTHY", view_type="equatorial"):
    return {
        "schema_version": 1,
        "batch_id": "batch-001",
        "profile_id": profile_id,
        "reference_grade": grade,
        "view_index": view_index,
        "view_type": view_type,
        "captured_at": "2026-08-31T12:00:00+00:00",
        "image_path": f"/tmp/{profile_id}_view_{view_index}.jpg",
        "width": 1280,
        "height": 720,
        "camera_index": 0,
        "facility_id": "TEST",
        "grader_id": "david",
        "species": "monstera",
        "location": "living-room",
    }


@pytest.fixture
def manifest_file(tmp_path):
    """Create a manifest with 10 valid profiles (5 views each)."""
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w") as fh:
        for i in range(10):
            pid = f"batch-001-{i:05d}"
            for v in range(5):
                rec = make_record(pid, v)
                fh.write(json.dumps(rec) + "\n")
    return manifest


class TestLoadManifest:
    def test_load(self, manifest_file):
        records = load_manifest(manifest_file)
        assert len(records) == 50  # 10 profiles × 5 views

    def test_empty(self, tmp_path):
        manifest = tmp_path / "empty.jsonl"
        manifest.write_text("")
        assert load_manifest(manifest) == []


class TestGroupByProfile:
    def test_grouping(self, manifest_file):
        records = load_manifest(manifest_file)
        groups = group_by_profile(records)
        assert len(groups) == 10
        for pid, recs in groups.items():
            assert len(recs) == 5


class TestValidateProfile:
    def test_valid(self):
        records = [make_record("p1", i) for i in range(5)]
        assert validate_profile("p1", records, 5) is True

    def test_missing_view(self):
        records = [make_record("p1", i) for i in range(4)]  # only 4 views
        assert validate_profile("p1", records, 5) is False

    def test_inconsistent_grade(self):
        records = [make_record("p1", i, "HEALTHY" if i < 3 else "POOR") for i in range(5)]
        assert validate_profile("p1", records, 5) is False


class TestDeterministicBucket:
    def test_deterministic(self):
        b1 = deterministic_bucket("profile-001", 42)
        b2 = deterministic_bucket("profile-001", 42)
        assert b1 == b2

    def test_different_ids_different_buckets(self):
        buckets = set()
        for i in range(100):
            buckets.add(deterministic_bucket(f"profile-{i:03d}", 42))
        # With 100 profiles and 100 buckets, we should get many distinct buckets
        assert len(buckets) > 50

    def test_range(self):
        for i in range(1000):
            b = deterministic_bucket(f"profile-{i}", 42)
            assert 0 <= b < 100


class TestSplitProfiles:
    def test_ratios(self):
        ids = [f"profile-{i:03d}" for i in range(100)]
        train, val, test = split_profiles(ids, 0.7, 0.2, 0.1, 42)
        total = len(train) + len(val) + len(test)
        assert total == 100
        assert len(train) >= 60
        assert len(test) >= 5

    def test_no_overlap(self):
        ids = [f"profile-{i:03d}" for i in range(100)]
        train, val, test = split_profiles(ids, 0.7, 0.2, 0.1, 42)
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()

    def test_deterministic(self):
        ids = [f"profile-{i:03d}" for i in range(100)]
        t1, v1, te1 = split_profiles(ids, 0.7, 0.2, 0.1, 42)
        t2, v2, te2 = split_profiles(ids, 0.7, 0.2, 0.1, 42)
        assert t1 == t2
        assert v1 == v2
        assert te1 == te2

    def test_invalid_ratios(self):
        ids = ["p1", "p2"]
        with pytest.raises(ValueError, match="ratios must sum to 1.0"):
            split_profiles(ids, 0.5, 0.2, 0.1, 42)


class TestWriteSplit:
    def test_write(self, tmp_path):
        groups = {
            "p1": [make_record("p1", 0), make_record("p1", 1)],
            "p2": [make_record("p2", 0), make_record("p2", 1)],
        }
        count = write_split("train", ["p1", "p2"], groups, tmp_path)
        assert count == 4
        split_file = tmp_path / "train.txt"
        lines = split_file.read_text().strip().split("\n")
        assert len(lines) == 4
