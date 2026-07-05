"""
Arducam auto-detection helper for macOS / AVFoundation.

All three production scripts (capture_dataset.py, baseline_verify.py,
local_inference.py) previously hardcoded CAM_INDEX=0, which on this
MacBook Air M4 maps to the built-in FaceTime camera — NOT the Arducam
OV9782 USB Global Shutter, which enumerates at cv2 index 1.

This module probes system_profiler for an Arducam by name, then maps
to the correct cv2.VideoCapture index by matching the Arducam's native
resolution (1920x1080). Falls back gracefully if detection fails.

No external dependencies beyond cv2 and the macOS system_profiler tool.
"""

from __future__ import annotations

import subprocess

import cv2

ARDUCAM_KEYWORD = "Arducam"
ARDUCAM_NATIVE_W = 1920
ARDUCAM_NATIVE_H = 1080
FALLBACK_INDEX = 1
BUILTIN_FALLBACK_INDEX = 0
MAX_PROBE = 4


def arducam_connected() -> bool:
    """Return True if an Arducam camera appears in system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return ARDUCAM_KEYWORD in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def detect_arducam_index(max_probe: int = MAX_PROBE) -> int:
    """Return the cv2 camera index for the Arducam.

    Detection strategy:
      1. Check system_profiler for an Arducam by name.
      2. If connected, probe cv2 indices 0..max_probe-1 and match by
         native resolution (1920x1080).
      3. If connected but not matched by resolution, fall back to
         FALLBACK_INDEX (1) with a warning.
      4. If not connected at all, fall back to BUILTIN_FALLBACK_INDEX (0)
         with a warning so the script still runs on the built-in webcam.
    """
    if not arducam_connected():
        print(
            "[CAMERA] WARNING: Arducam not detected via system_profiler. "
            "Falling back to built-in camera at index "
            f"{BUILTIN_FALLBACK_INDEX}."
        )
        return BUILTIN_FALLBACK_INDEX

    for i in range(max_probe):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == ARDUCAM_NATIVE_W and h == ARDUCAM_NATIVE_H:
            print(f"[CAMERA] Arducam detected at cv2 index {i} ({w}x{h}).")
            return i

    print(
        "[CAMERA] WARNING: Arducam is connected but could not be matched "
        f"by resolution at cv2 indices 0-{max_probe - 1}. "
        f"Falling back to index {FALLBACK_INDEX}."
    )
    return FALLBACK_INDEX
