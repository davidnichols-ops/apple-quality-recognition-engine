"""Arducam presence check and explicit OpenCV camera-index selection on macOS."""

from __future__ import annotations

import os
import subprocess

ARDUCAM_KEYWORD = "Arducam"
DEFAULT_ARDUCAM_INDEX = 0
BUILTIN_FALLBACK_INDEX = 0


def arducam_connected() -> bool:
    try:
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0 and ARDUCAM_KEYWORD in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def configured_arducam_index() -> int:
    value = os.environ.get("ARDUCAM_CAMERA_INDEX", str(DEFAULT_ARDUCAM_INDEX))
    try:
        index = int(value)
    except ValueError as exc:
        raise ValueError("ARDUCAM_CAMERA_INDEX must be an integer") from exc
    if index < 0:
        raise ValueError("ARDUCAM_CAMERA_INDEX must be non-negative")
    return index


def detect_arducam_index(*, allow_builtin_fallback: bool = True) -> int:
    """Return the configured OpenCV index after checking Arducam presence.

    macOS does not expose a stable mapping from ``system_profiler`` device names
    to OpenCV AVFoundation indices. Set ``ARDUCAM_CAMERA_INDEX`` when the external
    camera does not enumerate at index 0. Production capture should pass
    ``allow_builtin_fallback=False`` so a missing Arducam fails closed.
    """
    if arducam_connected():
        index = configured_arducam_index()
        print(f"[CAMERA] Arducam present; using configured cv2 index {index}.")
        return index
    if not allow_builtin_fallback:
        raise RuntimeError(
            "Arducam not detected. Connect it or explicitly enable the built-in "
            "camera fallback for a non-production test."
        )
    print(
        "[CAMERA] WARNING: Arducam not detected; using built-in camera at "
        f"cv2 index {BUILTIN_FALLBACK_INDEX}."
    )
    return BUILTIN_FALLBACK_INDEX
