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
    Expected format: 'prefix_variety_grade' or 'prefix_defect'
    Uses rsplit to handle multi-word varieties correctly.
    Returns: (variety, grade) tuple or (defect, None)
    """
    # Strip the sorting prefix from the front
    _, clean_name = class_name.split("_", 1)
    
    # Split from the RIGHT to isolate the grade (handles multi-word varieties)
    if clean_name.startswith("z_"):
        # Defect class
        _, defect_name = clean_name.split("_", 1)
        return defect_name, None
    else:
        # Variety class - split from right to get grade
        variety, grade = clean_name.rsplit("_", 1)
        return variety, grade


def format_display_text(class_name, confidence):
    """
    Format upscale display string for variety-grade predictions.
    Example: "CRIMSON CRISP (G2) [0.89]"
    """
    variety, grade = parse_class_token(class_name)
    
    if grade:
        # Format variety and grade with uppercase
        variety_upper = variety.replace('_', ' ').upper()
        grade_upper = grade.upper()
        return f"{variety_upper} ({grade_upper}) [{confidence:.2f}]"
    else:
        # Defect class
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
        
        # Run core inference through Apple Neural Engine
        results = model(frame, conf=0.35, imgsz=1024, verbose=False)
        
        macro_apples = []
        micro_anomalies = []

        # --- STAGE 1 & 2: INSTANCE PARSING ---
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            coords = list(map(int, box.xyxy[0])) # [x1, y1, x2, y2]
            conf = float(box.conf[0])
            flat_name = model.names[cls_id]

            if cls_id < 54: # New Variety Bound (18 varieties × 3 tiers)
                macro_apples.append({"id": cls_id, "name": flat_name, "box": coords, "conf": conf, "defects": []})
            else:
                micro_anomalies.append({"id": cls_id, "name": flat_name, "box": coords, "conf": conf})

        # --- STAGE 3: SPATIAL BINDING LAYER ---
        for anomaly in micro_anomalies:
            ax1, ay1, ax2, ay2 = anomaly["box"]
            acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2 # Centroid coordinates
            
            best_parent = None
            min_distance = float('inf')

            for apple in macro_apples:
                mx1, my1, mx2, my2 = apple["box"]
                
                # Intersection test: centroid containment check
                if mx1 <= acx <= mx2 and my1 <= acy <= my2:
                    mcx, mcy = (mx1 + mx2) / 2, (my1 + my2) / 2
                    distance = ((acx - mcx)**2 + (acy - mcy)**2)**0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_parent = apple
                        
            if best_parent:
                best_parent["defects"].append(anomaly)

        # --- STAGE 4: OUTPUT RENDERING ENGINE ---
        # Draw apples first (Green layer)
        for apple in macro_apples:
            x1, y1, x2, y2 = apple["box"]
            display_text = format_display_text(apple["name"], apple["conf"])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

            # Draw bounded child blemishes belonging to this instance (Red layer)
            for defect in apple["defects"]:
                dx1, dy1, dx2, dy2 = defect["box"]
                defect_text = format_display_text(defect["name"], defect["conf"])
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                cv2.putText(frame, defect_text, (dx1, dy1 - 5), cv2.FONT_HERSHEY_MINI, 0.4, (0, 0, 255), 1)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"M4 Edge Engine: {fps:.1f} FPS", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("M4 Edge Sorting Pipeline Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM]: Camera and window resources released.")


if __name__ == "__main__":
    main()
