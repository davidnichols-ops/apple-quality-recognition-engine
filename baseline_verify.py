#!/usr/bin/env python3
"""
Baseline Camera Verification Script
Arducam USB Global Shutter Camera + YOLO11n Inference Test
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
"""

import cv2
import time
from ultralytics import YOLO


def main():
    print("[SYSTEM]: Initializing Arducam USB Global Shutter camera on index 0...")
    
    # Initialize camera on index 0
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR]: Failed to open camera. Check index or macOS permissions.")
        return
    
    # Force raw uncompressed streaming with MJPG fourcc encoding
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[SYSTEM]: Camera configured at {actual_width}x{actual_height} with MJPG encoding")
    
    # Load lightweight YOLO11n benchmark model
    print("[SYSTEM]: Loading YOLO11n baseline model...")
    model = YOLO("yolo11n.pt")
    
    print("[SYSTEM]: Baseline inference engine active. Press 'q' to exit.")
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR]: Could not read frame from camera.")
            break
        
        # Run inference at imgsz=640
        results = model(frame, imgsz=640, verbose=False)
        
        # Annotate frame with predictions
        annotated_frame = results[0].plot()
        
        # Compute live pipeline FPS
        fps = 1.0 / (time.time() - start_time)
        
        # Add FPS annotation to frame
        cv2.putText(
            annotated_frame,
            f"Pipeline FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Baseline Verification - YOLO11n", annotated_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SYSTEM]: Exiting baseline verification...")
            break
    
    # Cleanly release hardware hooks
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM]: Camera and window resources released.")


if __name__ == "__main__":
    main()
