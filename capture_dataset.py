#!/usr/bin/env python3
"""Capture five known-grade views per plant with profile-level metadata.

Each plant is photographed from 5 angles (4 equatorial + 1 top-down)
with its known health grade (HEALTHY / MODERATE / POOR / DISCARD) and
associated metadata (batch ID, profile ID, facility ID, grader ID, etc.).

This produces ground-truth data for training and evaluation.  The
profile-level metadata ensures that all five views of one plant stay
in the same train/val/test partition.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from plant_grading_engine import VALID_GRADES

WIDTH = 1280
HEIGHT = 720
EQUATORIAL_INTERVAL_SECONDS = 2.5
EQUATORIAL_SHOTS = 4
WINDOW_NAME = "Plant Health Capture Engine"
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class CaptureRecord:
    schema_version: int
    batch_id: str
    profile_id: str
    reference_grade: str
    view_index: int
    view_type: str
    captured_at: str
    image_path: str
    width: int
    height: int
    camera_index: int
    facility_id: str
    grader_id: str
    species: str | None
    location: str | None


def normalize_grade(value: str) -> str:
    grade = value.strip().upper()
    if grade not in VALID_GRADES:
        raise ValueError(f"grade must be one of: {', '.join(VALID_GRADES)}")
    return grade


def normalize_identifier(value: str, field_name: str) -> str:
    identifier = value.strip()
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(
            f"{field_name} must start with an alphanumeric character and contain only "
            "letters, numbers, dots, underscores, or hyphens (maximum 64 characters)"
        )
    return identifier


def build_profile_id(batch_id: str, plant_index: int) -> str:
    if plant_index < 0:
        raise ValueError("plant_index must be non-negative")
    return f"{normalize_identifier(batch_id, 'batch_id')}-{plant_index:05d}"


def build_filename(
    captured_at: datetime,
    profile_id: str,
    reference_grade: str,
    view_index: int,
    view_type: str,
) -> str:
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")
    return (
        f"raw_{timestamp}_{profile_id}_{reference_grade.lower()}_"
        f"{view_type}_view_{view_index}.jpg"
    )


def append_manifest(manifest_path: Path, record: CaptureRecord) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        json.dump(asdict(record), handle, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture four equatorial views and one top-down view per known-grade plant."
    )
    parser.add_argument(
        "--grade", choices=VALID_GRADES, help="Reference health grade for this batch."
    )
    parser.add_argument("--count", type=int, help="Number of plants to capture.")
    parser.add_argument(
        "--batch-id",
        default=datetime.now().strftime("batch-%Y%m%d-%H%M%S"),
        help="Stable batch identifier used to group profiles.",
    )
    parser.add_argument("--facility-id", default="HOUSEHOLD_BOTANICAL_CHECK")
    parser.add_argument("--grader-id", default="operator")
    parser.add_argument("--species", help="Plant species (e.g. monstera, pothos).")
    parser.add_argument("--location", help="Location of the plant (e.g. living-room).")
    parser.add_argument("--output-dir", default="dataset/raw_ingest")
    parser.add_argument(
        "--allow-camera-fallback",
        action="store_true",
        help="Allow the built-in camera for a non-production test.",
    )
    return parser.parse_args(argv)


def resolve_capture_inputs(args: argparse.Namespace) -> tuple[str, int]:
    grade = normalize_grade(
        args.grade or input("Reference grade (HEALTHY/MODERATE/POOR/DISCARD): ")
    )
    if args.count is None:
        try:
            count = int(input("Number of plants to capture: "))
        except ValueError as exc:
            raise ValueError("plant count must be an integer") from exc
    else:
        count = args.count
    if count < 1:
        raise ValueError("plant count must be positive")
    return grade, count


def _read_frame(cap, max_failures: int = 120):
    failures = 0
    while failures < max_failures:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        failures += 1
    raise RuntimeError("camera stopped producing frames")


def _wait_for_space(cv2, cap, lines: Sequence[str]) -> None:
    while True:
        frame = _read_frame(cap)
        overlay = frame.copy()
        for index, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (20, 40 + index * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        cv2.imshow(WINDOW_NAME, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            return
        if key == 27:  # ESC
            raise KeyboardInterrupt


def _capture_burst(cv2, cap, profile_id: str, grade: str, output_dir: Path,
                   manifest_path: Path, record_template: dict) -> int:
    """Capture 4 equatorial + 1 top-down = 5 views.  Returns view count."""
    views = [
        ("equatorial", 0),
        ("equatorial", 1),
        ("equatorial", 2),
        ("equatorial", 3),
        ("topdown", 4),
    ]

    for view_type, view_index in views:
        if view_type == "equatorial" and view_index > 0:
            _wait_for_space(
                cv2, cap,
                [f"Rotate plant ~90 degrees (view {view_index + 1}/4)",
                 "Press SPACE when ready."],
            )
        elif view_type == "topdown":
            _wait_for_space(
                cv2, cap,
                ["Position camera above the plant (top-down view)",
                 "Press SPACE when ready."],
            )

        # Countdown
        for countdown in range(3, 0, -1):
            frame = _read_frame(cap)
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Capturing in {countdown}...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, overlay)
            cv2.waitKey(1000)

        # Capture
        frame = _read_frame(cap)
        captured_at = datetime.now(timezone.utc)
        filename = build_filename(
            captured_at, profile_id, grade, view_index, view_type
        )
        image_path = output_dir / filename
        cv2.imwrite(str(image_path), frame)

        record = CaptureRecord(
            schema_version=1,
            batch_id=record_template["batch_id"],
            profile_id=profile_id,
            reference_grade=grade,
            view_index=view_index,
            view_type=view_type,
            captured_at=captured_at.isoformat(),
            image_path=str(image_path),
            width=WIDTH,
            height=HEIGHT,
            camera_index=record_template["camera_index"],
            facility_id=record_template["facility_id"],
            grader_id=record_template["grader_id"],
            species=record_template.get("species"),
            location=record_template.get("location"),
        )
        append_manifest(manifest_path, record)

        # Show captured frame
        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"CAPTURED view {view_index + 1}/5 ({view_type})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow(WINDOW_NAME, overlay)
        cv2.waitKey(500)

    return len(views)


def main(argv: Sequence[str] | None = None) -> int:
    import cv2

    args = parse_args(argv)
    grade, count = resolve_capture_inputs(args)

    batch_id = normalize_identifier(args.batch_id, "batch_id")
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.jsonl"

    # Camera selection
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        if args.allow_camera_fallback:
            print("[WARN] Primary camera unavailable, trying fallback indices...")
            for idx in range(1, 5):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    camera_index = idx
                    break
        if not cap.isOpened():
            print("[ERROR] No camera available.")
            return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print("[SYSTEM] Plant Health Capture Engine")
    print(f"         Batch: {batch_id}")
    print(f"         Grade: {grade}")
    print(f"         Plants: {count}")
    print(f"         Output: {output_dir}")

    record_template = {
        "batch_id": batch_id,
        "camera_index": camera_index,
        "facility_id": args.facility_id,
        "grader_id": args.grader_id,
        "species": args.species,
        "location": args.location,
    }

    try:
        for plant_index in range(count):
            profile_id = build_profile_id(batch_id, plant_index)
            print(f"\n[READY] Plant {plant_index + 1}/{count} — profile {profile_id}")
            _wait_for_space(
                cv2, cap,
                [f"Place plant {plant_index + 1}/{count} (profile {profile_id})",
                 f"Grade: {grade}",
                 "Press SPACE to start 5-view capture."],
            )
            views = _capture_burst(
                cv2, cap, profile_id, grade, output_dir, manifest_path, record_template
            )
            print(f"[DONE] Captured {views} views for {profile_id}")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Capture stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[COMPLETE] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
