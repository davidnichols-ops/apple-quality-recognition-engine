#!/usr/bin/env python3
"""
Edge Harvest Telemetry Schema
=============================

Formal schema definition for the edge_harvest telemetry payload produced
by the M4 Edge Sorting Pipeline. Low-confidence detections (conf 0.40-0.65)
and operator overrides are captured as JSON + JPG pairs under
``dataset/edge_harvest/<YYYY-MM-DD>/`` for later re-annotation and active
learning loops.

This module is import-safe: importing it produces no side effects.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

# Default source camera identifier for the Arducam OV9782 sensor.
DEFAULT_SOURCE_CAMERA = "arducam_ov9782"

# Required top-level keys in every telemetry payload.
REQUIRED_FIELDS = {
    "timestamp",
    "frame_hash",
    "bounding_boxes",
    "operator_override",
    "source_camera",
}


@dataclass
class BoundingBox:
    """A single detection bounding box.

    Attributes:
        class_id: Integer class index from the model schema.
        class_name: Flat class name string (e.g. "apple", "z_bruise").
        box: Pixel coordinates as [x1, y1, x2, y2].
        confidence: Detection confidence in the range [0.0, 1.0].
    """

    class_id: int
    class_name: str
    box: List[int]
    confidence: float


@dataclass
class TelemetryPayload:
    """Formal telemetry payload for an edge-harvested frame.

    Attributes:
        timestamp: ISO-8601 timestamp of the harvest event.
        frame_hash: MD5 hex digest of the raw frame bytes.
        bounding_boxes: List of bounding box detections on the frame.
        operator_override: True if captured via operator override key.
        lighting: Optional ambient lighting descriptor (e.g. "daylight").
        source_camera: Identifier of the originating camera sensor.
    """

    timestamp: str
    frame_hash: str
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    operator_override: bool = False
    lighting: Optional[str] = None
    source_camera: str = DEFAULT_SOURCE_CAMERA


def compute_frame_hash(frame_np: Any) -> str:
    """Compute the MD5 hash of a numpy frame array.

    Args:
        frame_np: A numpy array (BGR frame as produced by OpenCV). The
            array is converted to contiguous bytes before hashing so the
            digest is stable regardless of view/layout.

    Returns:
        A 32-character lowercase hex string MD5 digest.
    """
    # ``tobytes()`` on a contiguous array yields the raw pixel buffer.
    buf = bytes(frame_np) if not hasattr(frame_np, "tobytes") else frame_np.tobytes()
    return hashlib.md5(buf).hexdigest()


def validate_telemetry(json_dict: Dict[str, Any]) -> bool:
    """Validate that a dictionary conforms to the telemetry schema.

    Checks for the presence and types of all required fields and validates
    the structure of each entry in ``bounding_boxes``.

    Args:
        json_dict: Parsed JSON dictionary to validate.

    Returns:
        True if the payload is valid.

    Raises:
        ValueError: If a required field is missing or a field has an
            incorrect type, or if a bounding box entry is malformed.
    """
    if not isinstance(json_dict, dict):
        raise ValueError("Telemetry payload must be a dict.")

    missing = REQUIRED_FIELDS - set(json_dict.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    # Type checks for top-level fields.
    if not isinstance(json_dict["timestamp"], str):
        raise ValueError("'timestamp' must be a string.")
    if not isinstance(json_dict["frame_hash"], str):
        raise ValueError("'frame_hash' must be a string.")
    if not isinstance(json_dict["bounding_boxes"], list):
        raise ValueError("'bounding_boxes' must be a list.")
    if not isinstance(json_dict["operator_override"], bool):
        raise ValueError("'operator_override' must be a bool.")
    if not isinstance(json_dict["source_camera"], str):
        raise ValueError("'source_camera' must be a string.")

    # Optional lighting field.
    if "lighting" in json_dict and json_dict["lighting"] is not None:
        if not isinstance(json_dict["lighting"], str):
            raise ValueError("'lighting' must be a string or None.")

    # Validate each bounding box entry.
    for idx, bb in enumerate(json_dict["bounding_boxes"]):
        if not isinstance(bb, dict):
            raise ValueError(f"bounding_boxes[{idx}] must be a dict.")
        for req in ("class_id", "class_name", "box", "confidence"):
            if req not in bb:
                raise ValueError(f"bounding_boxes[{idx}] missing '{req}'.")
        if not isinstance(bb["class_id"], int):
            raise ValueError(f"bounding_boxes[{idx}].class_id must be int.")
        if not isinstance(bb["class_name"], str):
            raise ValueError(f"bounding_boxes[{idx}].class_name must be str.")
        if not isinstance(bb["box"], list) or len(bb["box"]) != 4:
            raise ValueError(
                f"bounding_boxes[{idx}].box must be a list of 4 numbers."
            )
        if not all(isinstance(v, (int, float)) for v in bb["box"]):
            raise ValueError(
                f"bounding_boxes[{idx}].box values must be numbers."
            )
        if not isinstance(bb["confidence"], (int, float)):
            raise ValueError(
                f"bounding_boxes[{idx}].confidence must be a number."
            )

    return True


def write_telemetry(
    frame: Any,
    detections: List[Dict[str, Any]],
    harvest_dir: str,
    operator_override: bool = False,
    lighting: Optional[str] = None,
    source_camera: str = DEFAULT_SOURCE_CAMERA,
) -> Optional[str]:
    """Build, validate, and persist a telemetry payload + JPG pair.

    The frame and its telemetry JSON are written to a date-organized
    subdirectory ``<harvest_dir>/<YYYY-MM-DD>/``. Filenames are derived
    from a high-resolution timestamp to avoid collisions.

    Args:
        frame: OpenCV BGR frame (numpy array) to persist as JPG.
        detections: List of detection dicts, each with keys ``id`` (int),
            ``name`` (str), ``box`` ([x1, y1, x2, y2]), and ``conf``
            (float). Detections outside the volatile 0.40-0.65 band are
            still recorded when ``operator_override`` is True.
        harvest_dir: Root directory for harvested frames (e.g.
            ``dataset/edge_harvest``).
        operator_override: Whether this harvest was triggered by an
            operator override rather than volatile confidence.
        lighting: Optional ambient lighting descriptor.
        source_camera: Identifier of the originating camera sensor.

    Returns:
        The path to the written telemetry JSON file, or ``None`` if no
        detections qualified for harvesting.
    """
    # Import cv2 lazily so the module remains import-safe in environments
    # without OpenCV installed (e.g. CI schema checks).
    import cv2

    # Determine whether any detections fall in the volatile band.
    volatile_detections = [d for d in detections if 0.40 <= d["conf"] <= 0.65]

    # Only save if volatile detections exist OR operator override fired.
    if not volatile_detections and not operator_override:
        return None

    # When an override fires without volatile detections, record all
    # detections for context. Otherwise record only volatile ones.
    recorded = volatile_detections if volatile_detections else detections

    # Build bounding box entries.
    bounding_boxes = [
        BoundingBox(
            class_id=int(d["id"]),
            class_name=str(d["name"]),
            box=[int(v) for v in d["box"]],
            confidence=float(d["conf"]),
        )
        for d in recorded
    ]

    # Compute frame hash and ISO-8601 timestamp.
    frame_hash = compute_frame_hash(frame)
    iso_timestamp = datetime.now().isoformat(timespec="microseconds")

    payload = TelemetryPayload(
        timestamp=iso_timestamp,
        frame_hash=frame_hash,
        bounding_boxes=bounding_boxes,
        operator_override=operator_override,
        lighting=lighting,
        source_camera=source_camera,
    )

    # Serialize dataclass to a plain dict for validation + JSON output.
    payload_dict = asdict(payload)
    validate_telemetry(payload_dict)

    # Organize by date subdirectory.
    date_str = datetime.now().strftime("%Y-%m-%d")
    day_dir = os.path.join(harvest_dir, date_str)
    os.makedirs(day_dir, exist_ok=True)

    # High-resolution filename stem to avoid collisions.
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = os.path.join(day_dir, f"frame_{stem}.jpg")
    telemetry_path = os.path.join(day_dir, f"telemetry_{stem}.json")

    cv2.imwrite(frame_path, frame)
    with open(telemetry_path, "w") as f:
        json.dump(payload_dict, f, indent=2)

    if operator_override:
        print("[OVERRIDE LOGGED] Manual mismatch recorded to harvest cache.")
    else:
        print(f"[EDGE HARVEST]: Saved volatile frame to {frame_path}")

    return telemetry_path


if __name__ == "__main__":
    # Quick self-test when run directly.
    sample = {
        "timestamp": datetime.now().isoformat(),
        "frame_hash": "0" * 32,
        "bounding_boxes": [
            {
                "class_id": 0,
                "class_name": "apple",
                "box": [10, 20, 30, 40],
                "confidence": 0.55,
            }
        ],
        "operator_override": False,
        "source_camera": DEFAULT_SOURCE_CAMERA,
    }
    assert validate_telemetry(sample) is True
    print("[schema self-test] OK")
