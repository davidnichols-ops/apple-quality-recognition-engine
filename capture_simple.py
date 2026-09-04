#!/usr/bin/env python3
"""Minimal capture script — just camera + preview + spacebar save.
Loads WB calibration from wb_calibration.json if available."""
import cv2
import os
import time
from datetime import datetime

from wb_lock import apply_wb_gains, load_calibration

OUT = "dataset/raw_ingest"
os.makedirs(OUT, exist_ok=True)

# Load WB calibration
wb_gains = load_calibration()
if wb_gains is not None:
    print(f"[CAMERA] Software WB lock loaded: "
          f"R={wb_gains['r_gain']:.3f} G={wb_gains['g_gain']:.3f} "
          f"B={wb_gains['b_gain']:.3f} (target {wb_gains.get('target_temp_k', '?')}K)")
else:
    print("[CAMERA] No WB calibration found. Run: python wb_lock.py --calibrate")
    print("[CAMERA] Captures will use camera auto WB (not recommended)")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: camera 0 won't open")
    exit(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Warmup
for i in range(10):
    ret, frame = cap.read()
    if ret:
        print(f"warmup {i+1}/10 OK  shape={frame.shape}")
    else:
        print(f"warmup {i+1}/10 FAIL")

cv2.namedWindow("CAPTURE", cv2.WINDOW_AUTOSIZE)

count = 0
print("SPACE=capture  ESC=quit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("WARN: dropped frame, retrying...")
        time.sleep(0.1)
        continue

    # Overlay
    cv2.putText(frame, f"Frame {count} | SPACE=capture ESC=quit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("CAPTURE", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(OUT, f"negative_{ts}.jpg")
        # Apply WB gains before saving
        save_frame = apply_wb_gains(frame, wb_gains) if wb_gains else frame
        cv2.imwrite(path, save_frame)
        count += 1
        print(f"  SAVED {path}  ({count} total)")
        # Flash effect
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
        cv2.imshow("CAPTURE", frame)
        cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} frames captured.")
