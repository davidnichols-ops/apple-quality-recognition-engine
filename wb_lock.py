#!/usr/bin/env python3
"""Software white balance lock for the Arducam OV9782.

The OV9782's AVFoundation DAL device does not support hardware WB lock.
This module provides a software alternative:

1. CALIBRATION: User holds a white/gray reference in front of the camera.
   The script samples the average R/G/B values and computes per-channel
   gain multipliers that normalize the reference to neutral white.
   Gains are saved to wb_calibration.json.

2. APPLICATION: Every captured frame is multiplied by the calibrated gains
   before saving. This produces consistent 5600K-equivalent rendering
   across all frames regardless of the camera's auto WB drift.

Usage:
    # Calibrate (do this once at the start of the session)
    .venv/bin/python wb_lock.py --calibrate

    # The capture script loads wb_calibration.json automatically
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CALIBRATION_FILE = Path(__file__).parent / "wb_calibration.json"


def calibrate_white_balance(cv2, cap) -> dict:
    """Interactive calibration: user holds white reference, we sample gains."""
    print("\n[WB CALIBRATION] Software white balance lock")
    print("[WB CALIBRATION] Place a WHITE or GRAY reference card in front of the camera.")
    print("[WB CALIBRATION] Fill as much of the frame as possible with the card.")
    print("[WB CALIBRATION] Press SPACE to sample, ESC to cancel.\n")

    cv2.namedWindow("WB CALIBRATION", cv2.WINDOW_AUTOSIZE)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Draw a center ROI rectangle (25% of frame) to show sampling area
        h, w = frame.shape[:2]
        roi_w, roi_h = w // 4, h // 4
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, "Fill center box with WHITE/GRAY card",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(overlay, "SPACE=sample  ESC=cancel",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("WB CALIBRATION", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyWindow("WB CALIBRATION")
            print("[WB CALIBRATION] Cancelled.")
            return None
        elif key == 32:
            # Sample the center ROI
            roi = frame[y1:y2, x1:x2]
            avg_b = float(np.mean(roi[:, :, 0]))
            avg_g = float(np.mean(roi[:, :, 1]))
            avg_r = float(np.mean(roi[:, :, 2]))

            print(f"[WB CALIBRATION] Sampled ROI: R={avg_r:.1f} G={avg_g:.1f} B={avg_b:.1f}")

            # Compute gains to normalize to neutral white (all channels equal)
            # Use green as reference (sensor is most sensitive there)
            ref = (avg_r + avg_g + avg_b) / 3.0  # average as target
            r_gain = ref / avg_r if avg_r > 0 else 1.0
            g_gain = ref / avg_g if avg_g > 0 else 1.0
            b_gain = ref / avg_b if avg_b > 0 else 1.0

            # If we want to target 5600K specifically, we can apply a
            # daylight correction on top. 5600K is slightly cool vs the
            # neutral gray-world assumption, so we boost blue slightly.
            # The correction factor is small (~3-5% blue boost).
            daylight_blue_boost = 1.03
            b_gain *= daylight_blue_boost

            gains = {
                "r_gain": round(r_gain, 4),
                "g_gain": round(g_gain, 4),
                "b_gain": round(b_gain, 4),
                "calibrated_r": round(avg_r, 1),
                "calibrated_g": round(avg_g, 1),
                "calibrated_b": round(avg_b, 1),
                "target_temp_k": 5600,
                "method": "gray_world + daylight_blue_boost",
            }

            print(f"[WB CALIBRATION] Computed gains:")
            print(f"  R gain: {r_gain:.4f}")
            print(f"  G gain: {g_gain:.4f}")
            print(f"  B gain: {b_gain:.4f} (includes 3% daylight blue boost)")

            # Save
            CALIBRATION_FILE.write_text(json.dumps(gains, indent=2))
            print(f"[WB CALIBRATION] Saved to {CALIBRATION_FILE}")

            # Show corrected preview for verification
            corrected = apply_wb_gains(frame, gains)
            cv2.imshow("WB CALIBRATION", corrected)
            cv2.putText(corrected, "CORRECTED — press SPACE to confirm, ESC to redo",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("WB CALIBRATION", corrected)

            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == 27:
                    break  # redo
                elif key2 == 32:
                    cv2.destroyWindow("WB CALIBRATION")
                    print("[WB CALIBRATION] White balance locked. All captures will use these gains.")
                    return gains

    return None


def load_calibration() -> dict | None:
    """Load saved WB calibration if it exists."""
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def apply_wb_gains(frame: np.ndarray, gains: dict) -> np.ndarray:
    """Apply per-channel WB gains to a frame.

    Args:
        frame: BGR uint8 frame from OpenCV
        gains: dict with r_gain, g_gain, b_gain keys

    Returns:
        Corrected BGR uint8 frame
    """
    if gains is None:
        return frame

    r_gain = gains.get("r_gain", 1.0)
    g_gain = gains.get("g_gain", 1.0)
    b_gain = gains.get("b_gain", 1.0)

    # Convert to float32, multiply per-channel, clip back to uint8
    # OpenCV uses BGR order
    out = frame.astype(np.float32)
    out[:, :, 0] *= b_gain  # B
    out[:, :, 1] *= g_gain  # G
    out[:, :, 2] *= r_gain  # R
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def main():
    import cv2

    parser = argparse.ArgumentParser(description="Software white balance lock for Arducam OV9782")
    parser.add_argument("--calibrate", action="store_true", help="Run interactive calibration")
    parser.add_argument("--show", action="store_true", help="Show current calibration")
    args = parser.parse_args()

    if args.show:
        cal = load_calibration()
        if cal:
            print(f"Calibration file: {CALIBRATION_FILE}")
            print(json.dumps(cal, indent=2))
        else:
            print("No calibration found. Run --calibrate first.")
        return

    if args.calibrate:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: camera not available")
            return
        # Warmup
        for _ in range(10):
            cap.read()
        calibrate_white_balance(cv2, cap)
        cap.release()
        cv2.destroyAllWindows()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
