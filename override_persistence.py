#!/usr/bin/env python3
"""
Operator Override Persistence Layer
Apple Quality Recognition Engine

Dedicated persistence module for operator grading disagreements.
When an operator presses 'g' in local_inference.py, the computed grade
for the current frame is logged here — separate from the general
edge_harvest directory — so the CEO gets a clean view of disagreement
hotspots for active-learning retraining cycles.

Storage layout:
    dataset/operator_overrides/<YYYY-MM-DD>/
        override_<timestamp>.json   # OperatorOverride payload
        override_<timestamp>.jpg    # frame snapshot
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

import yaml


# ---------------------------------------------------------------------------
# Facility identity (loaded lazily from grading_policy.yaml)
# ---------------------------------------------------------------------------

_DEFAULT_FACILITY_ID = "UNKNOWN_FACILITY"


def _load_facility_id(policy_path: str = "grading_policy.yaml") -> str:
    """Load the facility_id from the grading policy YAML.

    Args:
        policy_path: Path to grading_policy.yaml.

    Returns:
        The facility_id string, or a fallback sentinel if the file
        cannot be read.
    """
    try:
        with open(policy_path, "r") as fh:
            policy = yaml.safe_load(fh)
        return policy.get("facility_id", _DEFAULT_FACILITY_ID)
    except Exception:
        # Non-fatal: persistence should still work without the policy file.
        return _DEFAULT_FACILITY_ID


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class OperatorOverride:
    """Typed record of a single operator grading disagreement.

    Attributes:
        timestamp: ISO-8601 timestamp of the override event.
        frame_path: Relative path to the saved frame snapshot JPG.
        detections: Raw detection dicts (id, name, box, conf) for the frame.
        grading_result: The computed grade(s) for each apple in the frame.
            Stored as a list of dicts, e.g. [{"grade": "G2", "defects": [...}].
        operator_note: Free-text note entered by the operator (optional).
        facility_id: Facility identifier sourced from grading_policy.yaml.
    """

    timestamp: str
    frame_path: str
    detections: List[Dict[str, Any]] = field(default_factory=list)
    grading_result: List[Dict[str, Any]] = field(default_factory=list)
    operator_note: str = ""
    facility_id: str = _DEFAULT_FACILITY_ID


# ---------------------------------------------------------------------------
# Persistence API
# ---------------------------------------------------------------------------


def persist_override(
    frame,
    detections: List[Dict[str, Any]],
    grading_results: List[Dict[str, Any]],
    output_dir: str = "dataset/operator_overrides",
    operator_note: str = "",
    policy_path: str = "grading_policy.yaml",
) -> OperatorOverride:
    """Persist a single operator override event to disk.

    Saves a timestamped JSON payload and a frame snapshot JPG to
    ``<output_dir>/<YYYY-MM-DD>/``. The JSON is the canonical record;
    the JPG is referenced by ``frame_path`` inside it.

    Args:
        frame: OpenCV BGR frame (numpy array) to save as a JPG snapshot.
        detections: List of raw detection dicts for the frame, each with
            keys ``id``, ``name``, ``box``, ``conf``.
        grading_results: The computed grade for each apple in the frame.
            Expected as a list of dicts, e.g. ``[{"grade": "G2", ...}]``.
        output_dir: Root directory for override storage.
        operator_note: Optional free-text note from the operator.
        policy_path: Path to grading_policy.yaml for facility_id lookup.

    Returns:
        The persisted :class:`OperatorOverride` record.
    """
    import cv2  # local import to keep module import side-effect free

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    file_stem = now.strftime("override_%Y%m%d_%H%M%S_%f")

    day_dir = os.path.join(output_dir, date_str)
    os.makedirs(day_dir, exist_ok=True)

    frame_path = os.path.join(day_dir, f"{file_stem}.jpg")
    json_path = os.path.join(day_dir, f"{file_stem}.json")

    # Save frame snapshot
    cv2.imwrite(frame_path, frame)

    facility_id = _load_facility_id(policy_path)

    record = OperatorOverride(
        timestamp=now.isoformat(),
        frame_path=frame_path,
        detections=detections,
        grading_result=grading_results,
        operator_note=operator_note,
        facility_id=facility_id,
    )

    with open(json_path, "w") as fh:
        json.dump(asdict(record), fh, indent=2)

    return record


def load_overrides(
    date_str: Optional[str] = None,
    output_dir: str = "dataset/operator_overrides",
) -> List[OperatorOverride]:
    """Load all operator overrides for a given date.

    Scans ``<output_dir>/<date_str>/`` for ``override_*.json`` files and
    reconstructs them into :class:`OperatorOverride` instances.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format. If ``None``, defaults to
            today's date.
        output_dir: Root directory for override storage.

    Returns:
        List of :class:`OperatorOverride` records, sorted by timestamp
        ascending. Returns an empty list if the directory does not exist
        or contains no override files.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    day_dir = os.path.join(output_dir, date_str)
    if not os.path.isdir(day_dir):
        return []

    records: List[OperatorOverride] = []
    for entry in sorted(os.listdir(day_dir)):
        if not (entry.startswith("override_") and entry.endswith(".json")):
            continue
        json_path = os.path.join(day_dir, entry)
        try:
            with open(json_path, "r") as fh:
                payload = json.load(fh)
            records.append(OperatorOverride(
                timestamp=payload.get("timestamp", ""),
                frame_path=payload.get("frame_path", ""),
                detections=payload.get("detections", []),
                grading_result=payload.get("grading_result", []),
                operator_note=payload.get("operator_note", ""),
                facility_id=payload.get("facility_id", _DEFAULT_FACILITY_ID),
            ))
        except (json.JSONDecodeError, KeyError) as exc:
            # Skip corrupt records but keep going.
            print(f"[override_persistence] WARNING: skipping malformed "
                  f"override file {json_path}: {exc}")
            continue

    # Sort by timestamp ascending (filename sort already handles this,
    # but ISO timestamps are the canonical ordering).
    records.sort(key=lambda r: r.timestamp)
    return records
