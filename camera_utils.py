"""
Arducam auto-detection helper for macOS / AVFoundation.

All three production scripts (capture_dataset.py, baseline_verify.py,
local_inference.py) previously hardcoded CAM_INDEX=0, which on this
MacBook Air M4 maps to the built-in FaceTime camera — NOT the Arducam
OV9782 USB Global Shutter, which enumerates at cv2 index 1.

This module checks system_profiler for an Arducam by name and returns
the correct cv2.VideoCapture index WITHOUT opening any camera during
detection. Opening cameras during detection (e.g. to probe resolution)
causes macOS AVFoundation to grab the FaceTime camera and light its
indicator LED, which can interfere with the subsequent production
camera open. We avoid that entirely.

No external dependencies beyond the macOS system_profiler tool.
"""

from __future__ import annotations

import subprocess

ARDUCAM_KEYWORD = "Arducam"
ARDUCAM_INDEX = 0
BUILTIN_FALLBACK_INDEX = 0


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


def detect_arducam_index() -> int:
    """Return the cv2 camera index for the Arducam.

    Detection strategy:
      1. Check system_profiler for an Arducam by name.
      2. If connected, return ARDUCAM_INDEX (0) — on this MacBook Air
         M4 the Arducam OV9782 enumerates at cv2 index 0.
      3. If not connected, fall back to BUILTIN_FALLBACK_INDEX (0)
         with a warning so the script still runs on the built-in webcam.

    We deliberately do NOT probe cv2.VideoCapture indices during
    detection. Opening cameras during detection causes macOS
    AVFoundation to grab the FaceTime camera and light its indicator
    LED, which can interfere with the subsequent production camera open.
    """
    if not arducam_connected():
        print(
            "[CAMERA] WARNING: Arducam not detected via system_profiler. "
            "Falling back to built-in camera at index "
            f"{BUILTIN_FALLBACK_INDEX}."
        )
        return BUILTIN_FALLBACK_INDEX

    print(f"[CAMERA] Arducam detected. Using cv2 index {ARDUCAM_INDEX}.")
    return ARDUCAM_INDEX
