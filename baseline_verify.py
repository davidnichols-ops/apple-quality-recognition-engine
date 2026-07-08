#!/usr/bin/env python3
"""
Baseline Camera Verification Script
Arducam USB Global Shutter Camera + YOLO Inference Test
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)

Modes:
  --pytorch   : YOLO26n PyTorch MPS baseline (default)
  --coreml    : YOLO26x CoreML FP16 ANE (6.2x faster, zero accuracy loss)
  --compare   : Run both back-to-back and print FPS comparison
"""

import argparse
import os
import cv2
import time
from ultralytics import YOLO

from camera_utils import detect_arducam_index


def capture_frame_hardened(cap, camera_index=0):
    """Captures a frame with automatic hardware reconnection logic."""
    ret, frame = cap.read()
    
    # If hardware disconnects or drops a frame
    if not ret or frame is None:
        print("[CRITICAL ERROR]: Arducam dropped connection. Initiating hardware reset...")
        cap.release()
        
        while True:
            time.sleep(2.0)  # Wait 2 seconds before retrying hardware bus
            print("[RETRYING]: Attempting to re-bind Arducam sensor...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Re-apply camera settings after reconnection
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print("[SUCCESS]: Arducam hardware link re-established.")
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, frame
    return cap, frame


def run_backend(cap, cam_index, model_path, label, window_title, imgsz=640):
    """Run inference loop for one backend, return FPS samples."""
    if not os.path.exists(model_path):
        print(f"[ERROR]: Model file '{model_path}' not found. Skipping {label}.")
        return cap, []

    print(f"[SYSTEM]: Loading {label} model '{model_path}'...")
    model = YOLO(model_path, task="detect")
    print(f"[SYSTEM]: {label} backend active. Press 'q' to exit.")

    fps_samples = []

    while True:
        start_time = time.time()

        cap, frame = capture_frame_hardened(cap, camera_index=cam_index)
        results = model(frame, imgsz=imgsz, verbose=False)
        annotated_frame = results[0].plot()

        fps = 1.0 / (time.time() - start_time)
        fps_samples.append(fps)

        cv2.putText(
            annotated_frame,
            f"{label} FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window_title, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return cap, fps_samples


def main():
    parser = argparse.ArgumentParser(description="Baseline Camera Verification")
    parser.add_argument("--pytorch", action="store_true", help="YOLO26n PyTorch MPS baseline")
    parser.add_argument("--coreml", action="store_true", help="YOLO26x CoreML FP16 ANE")
    parser.add_argument("--compare", action="store_true", help="Run both back-to-back")
    args = parser.parse_args()

    if not (args.pytorch or args.coreml or args.compare):
        args.pytorch = True  # default

    cam_index = detect_arducam_index()
    print(f"[SYSTEM]: Initializing Arducam USB Global Shutter camera on index {cam_index}...")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[ERROR]: Failed to open camera. Check index or macOS permissions.")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[SYSTEM]: Camera configured at {actual_width}x{actual_height} with MJPG encoding")

    if args.compare:
        # PyTorch first
        cap, pt_fps = run_backend(cap, cam_index, "yolo26n.pt", "PyTorch MPS", "Baseline - PyTorch MPS")
        cv2.destroyAllWindows()

        # CoreML second
        cap, cml_fps = run_backend(cap, cam_index, "yolo26x_640.mlpackage", "CoreML ANE", "Baseline - CoreML ANE")
        cv2.destroyAllWindows()

        # Print comparison (discard first 10 samples as warmup, then median)
        pt_warm = pt_fps[10:] if len(pt_fps) > 10 else pt_fps
        cml_warm = cml_fps[10:] if len(cml_fps) > 10 else cml_fps
        pt_med = sorted(pt_warm)[len(pt_warm) // 2] if pt_warm else 0
        cml_med = sorted(cml_warm)[len(cml_warm) // 2] if cml_warm else 0
        print(f"\n{'='*50}")
        print("BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"PyTorch MPS (yolo26n): {pt_med:.1f} FPS")
        print(f"CoreML ANE  (yolo26x): {cml_med:.1f} FPS")
        if pt_med > 0:
            print(f"Speedup: {cml_med/pt_med:.1f}x")
        print(f"{'='*50}")
    elif args.coreml:
        cap, _ = run_backend(cap, cam_index, "yolo26x_640.mlpackage", "CoreML ANE", "Baseline - CoreML ANE")
    else:
        cap, _ = run_backend(cap, cam_index, "yolo26n.pt", "PyTorch MPS", "Baseline - PyTorch MPS")

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM]: Camera and window resources released.")


if __name__ == "__main__":
    main()
