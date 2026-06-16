#!/usr/bin/env python3
"""
Production Raw Data Capture Script
Arducam USB Global Shutter Data Acquisition
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
Pure Feature Harvesting Pipeline - No Classification Logic
"""

import cv2
import os
from datetime import datetime

# =========================
# SYSTEM CONFIGURATION
# =========================
RAW_INGEST_DIR = "dataset/raw_ingest"
SHOTS_PER_FRUIT = 4
CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720

# Create raw ingest directory
os.makedirs(RAW_INGEST_DIR, exist_ok=True)

print("[SYSTEM] Raw Feature Harvesting Pipeline Initialized")
print(f"[INFO] Target directory: {RAW_INGEST_DIR}")
print(f"[INFO] Resolution: {WIDTH}x{HEIGHT} MJPG")

NUM_FRUITS = int(input("Number of fruits to capture: "))

# =========================
# HARDWARE OPTICAL CORE
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Arducam hardware pipeline failed to initialize")

print(f"\n[INFO] Core initialized. Streaming from camera index {CAM_INDEX}")

fruit_id = 0
while fruit_id < NUM_FRUITS:
    print(f"\n[READY] Position fruit {fruit_id+1}/{NUM_FRUITS} on fabric backdrop")
    shot_id = 0
    
    while shot_id < SHOTS_PER_FRUIT:
        ret, frame = cap.read()
        if not ret:
            continue

        overlay = frame.copy()
        display_text = f"RAW INGEST | Fruit {fruit_id+1} | View {shot_id+1}/4"
        cv2.putText(overlay, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Raw Feature Harvesting Engine", overlay)

        key = cv2.waitKey(1) & 0xFF
        
        # SPACEBAR = Capture Exposure
        if key == 32:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"raw_{timestamp}_{fruit_id:04d}_view_{shot_id}.jpg"
            path = os.path.join(RAW_INGEST_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"[SAVED] -> {path}")
            shot_id += 1

        # ESC = Safe System Interrupt
        if key == 27:
            print("[INTERRUPT] Graceful shutdown executed.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    fruit_id += 1
    print(f"[SUCCESS] Unit sequence complete for fruit item {fruit_id}.")
    
    # --- NO-CLICK WINDOW FOCUS FIX ---
    if fruit_id < NUM_FRUITS:
        print("[WAIT] Camera loop paused. Press ENTER inside the CAMERA WINDOW to unblock...")
        
        # Keep updating the display frame to tell the user to swap fruits and hit Enter
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            overlay = frame.copy()
            prompt_text1 = f"FRUIT {fruit_id} COMPLETE."
            prompt_text2 = "SWAP FRUIT & PRESS ENTER (IN THIS WINDOW) TO CONTINUE..."
            
            cv2.putText(overlay, prompt_text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(overlay, prompt_text2, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Raw Feature Harvesting Engine", overlay)
            
            next_key = cv2.waitKey(1) & 0xFF
            
            # 13 is the ASCII value for the Carriage Return / ENTER key
            if next_key == 13: 
                break
            
            # Allow ESC interrupt even during the swap phase
            if next_key == 27:
                print("[INTERRUPT] Graceful shutdown executed.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

cap.release()
cv2.destroyAllWindows()
print("[METRIC] Raw batch acquisition sequence completed cleanly.")
print(f"[INFO] Total raw frames saved to: {RAW_INGEST_DIR}")
