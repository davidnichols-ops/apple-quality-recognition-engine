#!/usr/bin/env python3
"""Capture equatorial views per apple, plus a separate stem/calyx pass."""

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
        description="Capture equatorial views per apple, or a separate stem/calyx pass."
    )
    parser.add_argument(
        "--mode",
        choices=["equatorial", "stem"],
        default="equatorial",
        help="equatorial: 4 turntable views per apple. stem: standalone calyx/stem photos at end of session.",
    )
    parser.add_argument(
        "--grade", choices=VALID_GRADES, help="Reference grade for this batch."
    )
    parser.add_argument("--count", type=int, help="Number of apples (equatorial) or photos (stem) to capture.")
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


def resolve_capture_inputs(args: argparse.Namespace) -> tuple[str | None, int]:
    if args.mode == "stem":
        grade = args.grade  # optional for stem — may be None
        if grade is not None:
            grade = normalize_grade(grade)
    else:
        grade = normalize_grade(args.grade or input("Reference grade (G1/G2/G3/CIDER/DISCARD): "))
    if args.count is None:
        label = "photos" if args.mode == "stem" else "apples"
        try:
            count = int(input(f"Number of {label} to capture: "))
        except ValueError as exc:
            raise ValueError("count must be an integer") from exc
    else:
        count = args.count
    if count < 1:
        raise ValueError("count must be positive")
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


def _init_camera(args: argparse.Namespace):
    """Initialize the camera with WB lock and warmup. Returns (cv2, cap, camera_index)."""
    import cv2

    camera_index = detect_arducam_index(
        allow_builtin_fallback=args.allow_camera_fallback
    )
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Arducam hardware pipeline failed to initialize")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # Lock white balance to manual 5600K (daylight). The Arducam OV9782 does
    # not expose CAP_PROP_WB_TEMPERATURE as a standard UVC control, so we
    # disable auto WB and set the per-channel gains directly. 5600K on this
    # sensor maps approximately to R=1.0, G=1.0, B=1.32 (slight blue boost to
    # match daylight under diffused LED lighting).
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # disable auto white balance
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual exposure mode
    cap.set(cv2.CAP_PROP_GAIN, 0)  # master gain neutral
    wb_set = cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5600)
    auto_wb = cap.get(cv2.CAP_PROP_AUTO_WB)
    wb_actual = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    if wb_set and wb_actual > 0:
        print(f"[CAMERA] White balance: auto={auto_wb}, temp={wb_actual}K (UVC)")
    else:
        print(
            f"[CAMERA] White balance: auto={auto_wb} (manual mode, "
            "factory color matrix — WB_TEMPERATURE not exposed by this sensor)"
        )

    # Explicitly create the preview window up front. On macOS with OpenCV 5,
    # cv2.imshow alone often fails to create a visible window; namedWindow
    # forces the GUI backend to initialize before the first frame is shown.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    # Warm up the camera with a few discarded reads so the first preview
    # frame is not a stale buffer.
    for _ in range(5):
        cap.read()

    return cv2, cap, camera_index


def run_equatorial(args: argparse.Namespace, cv2, cap, camera_index: int) -> int:
    """Capture 4 equatorial turntable views per apple. No calyx shot."""
    reference_grade, count = resolve_capture_inputs(args)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "capture_manifest.jsonl"

    print("[SYSTEM] Equatorial capture initialized (4 views per apple, no calyx)")
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
            print(f"[COMPLETE] {profile_id}: 4/4 equatorial views")
    except KeyboardInterrupt:
        print("[INTERRUPT] Capture stopped safely.")
        return 130

    print(f"[COMPLETE] {count} profile(s) captured; manifest: {manifest_path}")
    print("[TIP] Run --mode stem at the end of the session to capture calyx/stem photos.")
    return 0


def run_stem(args: argparse.Namespace, cv2, cap, camera_index: int) -> int:
    """Capture standalone stem/calyx photos. Not tied to specific profiles."""
    reference_grade, count = resolve_capture_inputs(args)
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "capture_manifest.jsonl"
    grade_str = reference_grade or "unknown"

    print("[SYSTEM] Stem/calyx capture initialized (standalone, no turntable)")
    print(f"[INFO] Batch: {args.batch_id} | Grade: {grade_str}")
    print(f"[INFO] Target: {output_dir} | Resolution: {WIDTH}x{HEIGHT} MJPG")

    try:
        for photo_index in range(count):
            stem_id = f"{args.batch_id}-stem-{photo_index:05d}"
            frame = _wait_for_space(
                cv2,
                cap,
                (
                    f"STEM/CALYX {photo_index + 1}/{count}",
                    "Position apple: STEM DOWN / CALYX UP",
                    "No rush — press SPACE when ready",
                ),
            )
            _record_frame(
                cv2,
                frame,
                output_dir,
                manifest_path,
                args,
                stem_id,
                grade_str,
                0,
                "calyx",
                camera_index,
            )
            print(f"[COMPLETE] {stem_id}: stem/calyx photo {photo_index + 1}/{count}")
    except KeyboardInterrupt:
        print("[INTERRUPT] Stem capture stopped safely.")
        return 130

    print(f"[COMPLETE] {count} stem/calyx photo(s) captured; manifest: {manifest_path}")
    return 0


def run_capture(args: argparse.Namespace) -> int:
    cv2, cap, camera_index = _init_camera(args)
    try:
        if args.mode == "stem":
            return run_stem(args, cv2, cap, camera_index)
        else:
            return run_equatorial(args, cv2, cap, camera_index)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    return run_capture(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
