#!/usr/bin/env python3
"""
Production Data Capture Script
Arducam USB Global Shutter Data Acquisition
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
"""

import cv2
import os
from datetime import datetime

# =========================
# SYSTEM CONFIGURATION
# =========================
DATASET_ROOT = "dataset"
SHOTS_PER_APPLE = 4
CAM_INDEX = 0
WIDTH, HEIGHT = 1280, 720

# Structured Prefix Mapping to Force Order Symmetry
VARIETY_MAP = {
    "zestar": "a_zestar",
    "redfree": "b_redfree",
    "grand_gala": "c_grand_gala",
    "priscilla": "d_priscilla",
    "freedom": "e_freedom",
    "sweet_16": "f_sweet_16",
    "crimson_crisp": "g_crimson_crisp",
    "spartan": "h_spartan",
    "macoun": "i_macoun",
    "snowsweet": "j_snowsweet",
    "liberty": "k_liberty",
    "pink_lady": "l_pink_lady",
    "chieftain": "m_chieftain",
    "winecrisp": "n_winecrisp",
    "ludacrisp": "o_ludacrisp",
    "enterprise": "p_enterprise",
    "rosalee": "q_rosalee",
    "evercrisp": "r_evercrisp"
}

print("Available varieties:", ", ".join(VARIETY_MAP.keys()))
user_input = input("\nEnter target variety label: ").strip().lower()

if user_input not in VARIETY_MAP:
    raise ValueError(f"Invalid variety. Must be one of {list(VARIETY_MAP.keys())}")

PREFIXED_VARIETY = VARIETY_MAP[user_input]
NUM_APPLES = int(input("Number of apples to capture: "))

save_dir = os.path.join(DATASET_ROOT, PREFIXED_VARIETY)
os.makedirs(save_dir, exist_ok=True)

# =========================
# HARDWARE OPTICAL CORE
# =========================
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    raise RuntimeError("Arducam hardware pipeline failed to initialize")

print(f"\n[INFO] Core initialized. Streaming from camera index {CAM_INDEX} at {WIDTH}x{HEIGHT}")

apple_id = 0
while apple_id < NUM_APPLES:
    print(f"\n[READY] Position apple {apple_id+1}/{NUM_APPLES} on fabric backdrop")
    shot_id = 0
    
    while shot_id < SHOTS_PER_APPLE:
        ret, frame = cap.read()
        if not ret:
            continue

        overlay = frame.copy()
        display_text = f"{PREFIXED_VARIETY.upper()} | Apple {apple_id+1} | View {shot_id+1}/4"
        cv2.putText(overlay, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Production Data Acquisition Engine", overlay)

        key = cv2.waitKey(1) & 0xFF
        
        # SPACEBAR = Capture Exposure
        if key == 32:
            # Output name template matching dataset tracking requirements
            filename = f"{PREFIXED_VARIETY}_{apple_id:04d}_view_{shot_id}.jpg"
            path = os.path.join(save_dir, filename)
            cv2.imwrite(path, frame)
            print(f"[SAVED INTERMEDIATE] -> {path}")
            shot_id += 1

        # ESC = Safe System Interrupt
        if key == 27:
            print("[INTERRUPT] Graceful shutdown executed.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    apple_id += 1
    print(f"[SUCCESS] Unit sequence complete for apple item {apple_id}.")
    if apple_id < NUM_APPLES:
        input("Physically change apple item and press ENTER to unblock camera bus...")

cap.release()
cv2.destroyAllWindows()
print("[METRIC] Batch acquisition sequence completed cleanly.")
