import hashlib
import json

from scripts.reingest_harvest import _scan_harvest_dir, promote_frames


def payload(status: str, frame_hash: str) -> dict:
    return {
        "timestamp": "2026-08-31T12:00:00",
        "frame_hash": frame_hash,
        "bounding_boxes": [],
        "operator_override": False,
        "source_camera": "test",
        "review_status": status,
        "reviewed_by": "human-reviewer" if status != "pending_human_review" else None,
        "reviewed_at": "2026-08-31T12:30:00"
        if status != "pending_human_review"
        else None,
    }


def write_pair(root, day: str, stem: str, status: str) -> None:
    directory = root / day
    directory.mkdir(parents=True, exist_ok=True)
    frame_bytes = f"image-{stem}".encode()
    frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    (directory / f"telemetry_{stem}.json").write_text(
        json.dumps(payload(status, frame_hash))
    )
    (directory / f"frame_{stem}.jpg").write_bytes(frame_bytes)


def test_only_human_approved_harvest_is_eligible_for_reingest(tmp_path) -> None:
    write_pair(tmp_path, "2026-08-31", "approved", "approved")
    write_pair(tmp_path, "2026-08-31", "pending", "pending_human_review")
    write_pair(tmp_path, "2026-08-31", "rejected", "rejected")

    entries = _scan_harvest_dir(str(tmp_path))
    assert [entry["review_status"] for entry in entries] == ["approved"]


def test_tampered_frame_is_not_eligible_for_reingest(tmp_path) -> None:
    write_pair(tmp_path, "2026-08-31", "approved", "approved")
    (tmp_path / "2026-08-31" / "frame_approved.jpg").write_bytes(b"tampered")
    assert _scan_harvest_dir(str(tmp_path)) == []


def test_promoted_manifest_keeps_review_audit_fields(tmp_path) -> None:
    harvest = tmp_path / "harvest"
    destination = tmp_path / "raw"
    write_pair(harvest, "2026-08-31", "approved", "approved")
    entries = _scan_harvest_dir(str(harvest))

    manifest = promote_frames(entries, str(destination))
    contents = open(manifest, encoding="utf-8").read()
    assert "human-reviewer" in contents
    assert "2026-08-31T12:30:00" in contents
