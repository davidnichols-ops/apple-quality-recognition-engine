import argparse
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
    resolve_capture_inputs,
)


def test_normalize_grade_accepts_case_and_whitespace() -> None:
    assert normalize_grade(" g2 ") == "G2"


def test_normalize_grade_rejects_unknown_grade() -> None:
    with pytest.raises(ValueError, match="grade must be one of"):
        normalize_grade("premium")


def test_profile_id_is_stable_within_batch() -> None:
    assert build_profile_id("batch-20260831", 7) == "batch-20260831-00007"


def test_identifier_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="batch_id"):
        normalize_identifier("../../outside", "batch_id")


def test_filename_preserves_profile_grade_and_view() -> None:
    captured_at = datetime(2026, 8, 31, 12, 30, tzinfo=timezone.utc)
    filename = build_filename(captured_at, "batch-a-00001", "G3", 4, "calyx")
    assert filename == ("raw_20260831_123000_000000_batch-a-00001_g3_calyx_view_4.jpg")


def test_resolve_capture_inputs_rejects_nonpositive_count() -> None:
    args = argparse.Namespace(grade="G1", count=0)
    with pytest.raises(ValueError, match="positive"):
        resolve_capture_inputs(args)


def test_append_manifest_writes_machine_readable_profile_metadata(tmp_path) -> None:
    record = CaptureRecord(
        schema_version=1,
        batch_id="batch-a",
        profile_id="batch-a-00000",
        reference_grade="G1",
        view_index=0,
        view_type="equatorial",
        captured_at="2026-08-31T12:30:00+00:00",
        image_path="dataset/raw_ingest/example.jpg",
        width=1280,
        height=720,
        camera_index=0,
        facility_id="facility",
        grader_id="grader",
        lot_id="lot-1",
        cultivar="gala",
    )
    manifest = tmp_path / "capture_manifest.jsonl"
    append_manifest(manifest, record)

    payload = json.loads(manifest.read_text())
    assert payload["profile_id"] == "batch-a-00000"
    assert payload["reference_grade"] == "G1"
    assert payload["view_type"] == "equatorial"
