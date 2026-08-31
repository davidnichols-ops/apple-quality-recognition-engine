import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from edge_harvest_schema import validate_telemetry, write_telemetry


class Frame:
    def tobytes(self) -> bytes:
        return b"frame-bytes"


def install_fake_cv2(monkeypatch) -> None:
    def imwrite(path, frame):
        Path(path).write_bytes(frame.tobytes())
        return True

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imwrite=imwrite))


def detections() -> list[dict]:
    return [
        {"id": 0, "name": "apple", "box": [0, 0, 100, 100], "conf": 0.95},
        {"id": 2, "name": "class_defect", "box": [10, 10, 20, 20], "conf": 0.55},
    ]


def test_write_telemetry_records_full_context_and_grading(
    monkeypatch, tmp_path
) -> None:
    install_fake_cv2(monkeypatch)
    path = write_telemetry(
        Frame(),
        detections(),
        str(tmp_path),
        grading_results=[{"grade": "G2", "coverage_pct": 7.0}],
        model_id="candidate.mlpackage",
        policy_version="v2",
    )

    payload = json.loads(Path(path).read_text())
    assert validate_telemetry(payload)
    assert len(payload["bounding_boxes"]) == 2
    assert payload["grading_results"][0]["grade"] == "G2"
    assert payload["review_status"] == "pending_human_review"


def test_force_review_persists_high_confidence_frame(monkeypatch, tmp_path) -> None:
    install_fake_cv2(monkeypatch)
    high_confidence = [
        {"id": 0, "name": "apple", "box": [0, 0, 100, 100], "conf": 0.95}
    ]
    path = write_telemetry(
        Frame(),
        high_confidence,
        str(tmp_path),
        force_review=True,
        review_reason="coverage_near_grade_boundary",
    )
    assert path is not None


def test_high_confidence_frame_is_not_harvested_without_reason(tmp_path) -> None:
    high_confidence = [
        {"id": 0, "name": "apple", "box": [0, 0, 100, 100], "conf": 0.95}
    ]
    assert write_telemetry(Frame(), high_confidence, str(tmp_path)) is None


def test_resolved_review_requires_human_identity() -> None:
    payload = {
        "timestamp": "2026-08-31T12:00:00",
        "frame_hash": "a" * 64,
        "bounding_boxes": [],
        "operator_override": False,
        "source_camera": "test",
        "review_status": "approved",
    }
    with pytest.raises(ValueError, match="reviewed_by"):
        validate_telemetry(payload)
