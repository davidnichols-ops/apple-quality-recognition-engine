"""Unit tests for capture_dataset.py."""

import json
from datetime import datetime, timezone

import pytest

from capture_dataset import (
    CaptureRecord,
    append_manifest,
    build_filename,
    build_profile_id,
    normalize_grade,
    normalize_identifier,
    parse_args,
    resolve_capture_inputs,
)


class TestNormalizeGrade:
    def test_valid_grades(self):
        assert normalize_grade("healthy") == "HEALTHY"
        assert normalize_grade("MODERATE") == "MODERATE"
        assert normalize_grade("  poor  ") == "POOR"
        assert normalize_grade("DISCARD") == "DISCARD"

    def test_invalid_grade(self):
        with pytest.raises(ValueError, match="grade must be one of"):
            normalize_grade("G1")


class TestNormalizeIdentifier:
    def test_valid(self):
        assert normalize_identifier("batch-001", "batch_id") == "batch-001"
        assert normalize_identifier("monstera.lot_7", "lot_id") == "monstera.lot_7"

    def test_empty(self):
        with pytest.raises(ValueError):
            normalize_identifier("", "batch_id")

    def test_starts_with_special(self):
        with pytest.raises(ValueError):
            normalize_identifier("-bad", "batch_id")

    def test_too_long(self):
        with pytest.raises(ValueError):
            normalize_identifier("a" * 65, "batch_id")


class TestBuildProfileId:
    def test_basic(self):
        pid = build_profile_id("batch-20260831", 5)
        assert pid == "batch-20260831-00005"

    def test_zero(self):
        pid = build_profile_id("batch001", 0)
        assert pid == "batch001-00000"

    def test_negative(self):
        with pytest.raises(ValueError):
            build_profile_id("batch001", -1)


class TestBuildFilename:
    def test_filename_format(self):
        ts = datetime(2026, 8, 31, 12, 30, 0, 123456, tzinfo=timezone.utc)
        name = build_filename(ts, "batch-001-00001", "healthy", 2, "equatorial")
        assert "raw_20260831_123000_123456" in name
        assert "batch-001-00001" in name
        assert "healthy" in name
        assert "equatorial" in name
        assert "view_2" in name
        assert name.endswith(".jpg")


class TestAppendManifest:
    def test_append(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        rec = CaptureRecord(
            schema_version=1,
            batch_id="batch-001",
            profile_id="batch-001-00000",
            reference_grade="HEALTHY",
            view_index=0,
            view_type="equatorial",
            captured_at="2026-08-31T12:00:00+00:00",
            image_path="/tmp/img.jpg",
            width=1280,
            height=720,
            camera_index=0,
            facility_id="TEST",
            grader_id="david",
            species="monstera",
            location="living-room",
        )
        append_manifest(manifest, rec)
        append_manifest(manifest, rec)

        lines = manifest.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert data["profile_id"] == "batch-001-00000"
            assert data["reference_grade"] == "HEALTHY"


class TestParseArgs:
    def test_defaults(self):
        args = parse_args(["--grade", "HEALTHY", "--count", "5"])
        assert args.grade == "HEALTHY"
        assert args.count == 5
        assert args.output_dir == "dataset/raw_ingest"

    def test_custom_output(self):
        args = parse_args(["--grade", "POOR", "--count", "1", "--output-dir", "/tmp/test"])
        assert args.output_dir == "/tmp/test"


class TestResolveInputs:
    def test_with_args(self):
        args = parse_args(["--grade", "MODERATE", "--count", "3"])
        grade, count = resolve_capture_inputs(args)
        assert grade == "MODERATE"
        assert count == 3

    def test_zero_count(self):
        args = parse_args(["--grade", "HEALTHY", "--count", "0"])
        with pytest.raises(ValueError, match="positive"):
            resolve_capture_inputs(args)
