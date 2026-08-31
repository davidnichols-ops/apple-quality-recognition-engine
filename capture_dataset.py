#!/usr/bin/env python3
"""Capture five known-grade views per apple with profile-level metadata."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from camera_utils import detect_arducam_index
from grading_engine import VALID_GRADES

WIDTH = 1280
HEIGHT = 720
EQUATORIAL_INTERVAL_SECONDS = 2.5
EQUATORIAL_SHOTS = 4
WINDOW_NAME = "Raw Feature Harvesting Engine"
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
    lot_id: str | None
    cultivar: str | None


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


def build_profile_id(batch_id: str, fruit_index: int) -> str:
    if fruit_index < 0:
        raise ValueError("fruit_index must be non-negative")
    return f"{normalize_identifier(batch_id, 'batch_id')}-{fruit_index:05d}"


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
        description="Capture four equatorial views and one calyx view per known-grade apple."
    )
    parser.add_argument(
        "--grade", choices=VALID_GRADES, help="Reference grade for this batch."
    )
    parser.add_argument("--count", type=int, help="Number of apples to capture.")
    parser.add_argument(
        "--batch-id",
        default=datetime.now().strftime("batch-%Y%m%d-%H%M%S"),
        help="Stable batch identifier used to group profiles.",
    )
    parser.add_argument("--facility-id", default="DSM_COLD_STORAGE_01")
    parser.add_argument("--grader-id", default="operator")
    parser.add_argument("--lot-id")
    parser.add_argument("--cultivar")
    parser.add_argument("--output-dir", default="dataset/raw_ingest")
    parser.add_argument(
        "--allow-camera-fallback",
        action="store_true",
        help="Allow the built-in camera for a non-production test.",
    )
    return parser.parse_args(argv)


def resolve_capture_inputs(args: argparse.Namespace) -> tuple[str, int]:
    grade = normalize_grade(args.grade or input("Reference grade (G1/G2/G3/DISCARD): "))
    if args.count is None:
        try:
            count = int(input("Number of apples to capture: "))
        except ValueError as exc:
            raise ValueError("apple count must be an integer") from exc
    else:
        count = args.count
    if count < 1:
        raise ValueError("apple count must be positive")
    return grade, count


def _read_frame(cap, max_failures: int = 120):
    failures = 0
    while failures < max_failures:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        failures += 1
    raise RuntimeError("camera stopped producing frames")


def _wait_for_space(cv2, cap, lines: Sequence[str]) -> object:
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
        if key == 32:
            return frame
        if key == 27:
            raise KeyboardInterrupt


def _save_frame(cv2, path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"failed to write image: {path}")


def _record_frame(
    cv2,
    frame,
    output_dir: Path,
    manifest_path: Path,
    args: argparse.Namespace,
    profile_id: str,
    reference_grade: str,
    view_index: int,
    view_type: str,
    camera_index: int,
) -> str:
    captured_at = datetime.now(timezone.utc)
    filename = build_filename(
        captured_at,
        profile_id,
        reference_grade,
        view_index,
        view_type,
    )
    image_path = output_dir / filename
    _save_frame(cv2, image_path, frame)
    append_manifest(
        manifest_path,
        CaptureRecord(
            schema_version=1,
            batch_id=args.batch_id,
            profile_id=profile_id,
            reference_grade=reference_grade,
            view_index=view_index,
            view_type=view_type,
            captured_at=captured_at.isoformat(),
            image_path=image_path.as_posix(),
            width=WIDTH,
            height=HEIGHT,
            camera_index=camera_index,
            facility_id=args.facility_id,
            grader_id=args.grader_id,
            lot_id=args.lot_id,
            cultivar=args.cultivar,
        ),
    )
    print(f"  [SAVED] {filename}")
    return filename


def run_capture(args: argparse.Namespace) -> int:
    import cv2

    reference_grade, count = resolve_capture_inputs(args)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "capture_manifest.jsonl"
    camera_index = detect_arducam_index(
        allow_builtin_fallback=args.allow_camera_fallback
    )
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Arducam hardware pipeline failed to initialize")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    print("[SYSTEM] Five-view known-grade capture initialized")
    print(f"[INFO] Batch: {args.batch_id} | Grade: {reference_grade}")
    print(f"[INFO] Target: {output_dir} | Resolution: {WIDTH}x{HEIGHT} MJPG")

    try:
        for fruit_index in range(count):
            profile_id = build_profile_id(args.batch_id, fruit_index)
            _wait_for_space(
                cv2,
                cap,
                (
                    f"APPLE {fruit_index + 1}/{count} | {reference_grade} | STEM UP",
                    "START TURNTABLE, THEN PRESS SPACE",
                ),
            )
            print(f"[CAPTURE] {profile_id}: four equatorial views")
            sequence_start = time.monotonic()
            next_capture_at = sequence_start
            for view_index in range(EQUATORIAL_SHOTS):
                while True:
                    frame = _read_frame(cap)
                    remaining = max(0.0, next_capture_at - time.monotonic())
                    overlay = frame.copy()
                    cv2.putText(
                        overlay,
                        f"EQUATORIAL {view_index + 1}/{EQUATORIAL_SHOTS} | {remaining:.1f}s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow(WINDOW_NAME, overlay)
                    if cv2.waitKey(1) & 0xFF == 27:
                        raise KeyboardInterrupt
                    if time.monotonic() >= next_capture_at:
                        break
                _record_frame(
                    cv2,
                    frame,
                    output_dir,
                    manifest_path,
                    args,
                    profile_id,
                    reference_grade,
                    view_index,
                    "equatorial",
                    camera_index,
                )
                next_capture_at = sequence_start + (
                    (view_index + 1) * EQUATORIAL_INTERVAL_SECONDS
                )

            calyx_frame = _wait_for_space(
                cv2,
                cap,
                (
                    f"{profile_id} | INVERT: STEM DOWN / CALYX UP",
                    "PRESS SPACE TO CAPTURE CALYX VIEW",
                ),
            )
            _record_frame(
                cv2,
                calyx_frame,
                output_dir,
                manifest_path,
                args,
                profile_id,
                reference_grade,
                EQUATORIAL_SHOTS,
                "calyx",
                camera_index,
            )
            print(f"[COMPLETE] {profile_id}: 5/5 views")
    except KeyboardInterrupt:
        print("[INTERRUPT] Capture stopped safely.")
        return 130
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[COMPLETE] {count} profile(s) captured; manifest: {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_capture(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
