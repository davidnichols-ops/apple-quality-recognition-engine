#!/usr/bin/env python3
"""
Production Local Inference Script
Apple CoreML Deployment on M4 Neural Engine
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
"""

import cv2
import time
from ultralytics import YOLO


def parse_class_token(class_name):
    """
    Parse class token to separate Variety from Grade.
    Expected format: 'variety_grade' or 'anomaly'
    Returns: (variety, grade) tuple or (anomaly, None)
    """
    if '_' in class_name:
        parts = class_name.split('_')
        if len(parts) == 2:
            return parts[0], parts[1]
        # Handle anomalies (single token with underscores like 'sooty_blotch_flyspeck')
        return class_name, None
    return class_name, None


def format_display_text(class_name, confidence):
    """
    Format upscale display string for variety-grade predictions.
    Example: "ENTERPRISE - VERY FANCY [0.89]"
    """
    variety, grade = parse_class_token(class_name)
    
    if grade:
        # Format variety and grade with uppercase
        variety_upper = variety.replace('_', ' ').upper()
        grade_upper = grade.upper()
        return f"{variety_upper} - {grade_upper} [{confidence:.2f}]"
    else:
        # Anomaly class
        return f"{variety.upper()} [{confidence:.2f}]"


def main():
    print("[SYSTEM]: Initializing M4 Edge Sorting Pipeline Engine...")
    
    # Initialize camera with same profile as baseline_verify.py
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR]: Failed to open camera. Check index or macOS permissions.")
        return
    
    # Force raw uncompressed streaming with MJPG fourcc encoding
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[SYSTEM]: Camera configured at {actual_width}x{actual_height} with MJPG encoding")
    
    # Load local Apple CoreML compiled package
    print("[SYSTEM]: Loading CoreML model 'best.mlpackage' with task='detect'...")
    try:
        model = YOLO("best.mlpackage", task="detect")
    except Exception as e:
        print(f"[ERROR]: Failed to load CoreML model: {e}")
        print("[INFO]: Ensure 'best.mlpackage' exists in the project directory.")
        cap.release()
        return
    
    print("[SYSTEM]: M4 Neural Engine backend active. Press 'q' to exit.")
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR]: Could not read frame from camera.")
            break
        
        # Run inference at imgsz=1024 on Apple Silicon/Neural Engine backend
        results = model(frame, imgsz=1024, verbose=False)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Loop through predicted bounding box coordinates
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class index and confidence
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get class name from model
                    class_name = model.names[class_id]
                    
                    # Format display text
                    display_text = format_display_text(class_name, confidence)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame,
                        display_text,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
        
        # Calculate frame timing benchmarks
        frame_time = time.time() - start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        # Add FPS annotation
        cv2.putText(
            annotated_frame,
            f"M4 Neural Engine FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Project output to window
        cv2.imshow("M4 Edge Sorting Pipeline Engine", annotated_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SYSTEM]: Shutting down M4 Edge Sorting Pipeline Engine...")
            break
    
    # Cleanly release hardware hooks
    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM]: Camera and window resources released.")


if __name__ == "__main__":
    main()
