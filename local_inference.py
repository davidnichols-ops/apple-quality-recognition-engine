#!/usr/bin/env python3
"""
Production Local Inference Script
Apple CoreML Deployment on M4 Neural Engine
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
Feature Detector Pipeline: 13-Class Architecture
"""

import cv2
import time
import json
import os
from datetime import datetime
from ultralytics import YOLO


def calculate_defect_area(box):
    """Calculate pixel area of a bounding box."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def calculate_intersection_area(box1, box2):
    """
    Calculate the intersection area between two bounding boxes.
    Returns 0 if boxes do not intersect.
    """
    x1, y1, x2, y2 = box1
    px1, py1, px2, py2 = box2
    
    # Calculate intersection coordinates
    ix1 = max(x1, px1)
    iy1 = max(y1, py1)
    ix2 = min(x2, px2)
    iy2 = min(y2, py2)
    
    # If no intersection, return 0
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    
    return (ix2 - ix1) * (iy2 - iy1)


def calculate_ioa(defect_box, parent_box):
    """
    Calculate Intersection-over-Area (IoA) ratio.
    IoA = intersection_area / defect_box_area
    """
    intersection = calculate_intersection_area(defect_box, parent_box)
    defect_area = calculate_defect_area(defect_box)
    
    if defect_area == 0:
        return 0.0
    
    return intersection / defect_area


def compute_grade(defects, parent_area):
    """
    Deterministic scoring function based on defect count, type, and area coverage.
    Returns: Grade string ('G1', 'G2', 'G3') or 'DISCARD'
    """
    if not defects:
        return 'G1'
    
    # Defect severity mapping
    mild_defects = {'z_bruise', 'z_russeting'}
    structural_defects = {'z_split_crack', 'z_misshapen', 'z_rot', 'z_insect_damage'}
    moderate_defects = {'z_scarf_skin', 'z_sunburn', 'z_stem_puncture', 'z_scab', 'z_sooty_blotch_flyspeck'}
    
    total_defect_area = sum(calculate_defect_area(d['box']) for d in defects)
    area_ratio = total_defect_area / parent_area if parent_area > 0 else 0
    
    # Count by severity
    mild_count = sum(1 for d in defects if d['name'] in mild_defects)
    structural_count = sum(1 for d in defects if d['name'] in structural_defects)
    moderate_count = sum(1 for d in defects if d['name'] in moderate_defects)
    
    # Grade determination logic
    if structural_count > 0 or area_ratio > 0.15:
        return 'G3'
    elif mild_count > 2 or moderate_count > 1 or area_ratio > 0.05:
        return 'G2'
    else:
        return 'G1'


def format_display_text(class_name, confidence, grade=None):
    """
    Format display string for predictions.
    Example: "APPLE (G2) [0.89]" or "BRUISE [0.76]"
    """
    if class_name == 'apple':
        grade_str = f" ({grade})" if grade else ""
        return f"APPLE{grade_str} [{confidence:.2f}]"
    elif class_name == 'unfit_bin_discard':
        return f"DISCARD [{confidence:.2f}]"
    else:
        # Defect class - strip z_ prefix
        defect_name = class_name.replace('z_', '').upper()
        return f"{defect_name} [{confidence:.2f}]"


def save_edge_harvest_frame(frame, detections, harvest_dir, operator_override=False):
    """
    Save frame and telemetry if any detection confidence falls in volatile threshold (0.40-0.65)
    or if operator override is triggered.
    """
    volatile_detections = [d for d in detections if 0.40 <= d['conf'] <= 0.65]
    
    # Only save if volatile detections exist OR operator override is triggered
    if not volatile_detections and not operator_override:
        return
    
    # Create harvest directory if it doesn't exist
    os.makedirs(harvest_dir, exist_ok=True)
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = os.path.join(harvest_dir, f"frame_{timestamp}.jpg")
    telemetry_path = os.path.join(harvest_dir, f"telemetry_{timestamp}.json")
    
    # Save frame
    cv2.imwrite(frame_path, frame)
    
    # Save telemetry
    telemetry = {
        "timestamp": timestamp,
        "frame_path": frame_path,
        "operator_override": operator_override,
        "volatile_detections": [
            {
                "class_id": d['id'],
                "class_name": d['name'],
                "confidence": d['conf'],
                "box": d['box']
            }
            for d in volatile_detections
        ]
    }
    
    with open(telemetry_path, 'w') as f:
        json.dump(telemetry, f, indent=2)
    
    if operator_override:
        print("[OVERRIDE LOGGED] Manual mismatch recorded to harvest cache.")
    else:
        print(f"[EDGE HARVEST]: Saved volatile frame to {frame_path}")


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
    
    # Edge harvest directory
    harvest_dir = "dataset/edge_harvest"
    
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR]: Could not read frame from camera.")
            break
        
        # Run core inference through Apple Neural Engine
        results = model(frame, conf=0.35, imgsz=1024, verbose=False)
        
        parent_boxes = []
        discard_triggers = []
        defect_boxes = []
        all_detections = []

        # --- STAGE 1: INSTANCE PARSING (13-Class Paradigm) ---
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            coords = list(map(int, box.xyxy[0]))
            conf = float(box.conf[0])
            flat_name = model.names[cls_id]
            
            detection = {"id": cls_id, "name": flat_name, "box": coords, "conf": conf}
            all_detections.append(detection)
            
            if cls_id == 0:  # apple - Universal Macro Parent Box
                parent_boxes.append({"id": cls_id, "name": flat_name, "box": coords, "conf": conf, "defects": []})
            elif cls_id == 1:  # unfit_bin_discard - Discard Trigger
                discard_triggers.append(detection)
            elif cls_id >= 2:  # Defect classes (Indices 2-12)
                defect_boxes.append(detection)

        # --- STAGE 2: DISCARD SEQUENCE CHECK ---
        # If discard trigger detected inside or near any parent box cluster, mark for discard
        discard_mode = False
        if discard_triggers:
            for trigger in discard_triggers:
                tx1, ty1, tx2, ty2 = trigger["box"]
                tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
                
                for parent in parent_boxes:
                    px1, py1, px2, py2 = parent["box"]
                    # Check if trigger centroid is inside parent box or within 50px proximity
                    if (px1 <= tcx <= px2 and py1 <= tcy <= py2) or \
                       ((px1 - 50) <= tcx <= (px2 + 50) and (py1 - 50) <= tcy <= (py2 + 50)):
                        discard_mode = True
                        break
                if discard_mode:
                    break

        # --- STAGE 3: SPATIAL BINDING LAYER (IoA Override) ---
        for defect in defect_boxes:
            defect_box = defect["box"]
            
            best_parent = None
            max_ioa = 0.0

            for parent in parent_boxes:
                parent_box = parent["box"]
                
                # Calculate Intersection-over-Area (IoA) ratio
                ioa = calculate_ioa(defect_box, parent_box)
                
                # Bind if IoA >= 0.10 (10% threshold)
                if ioa >= 0.10:
                    # Track parent with highest intersection density
                    if ioa > max_ioa:
                        max_ioa = ioa
                        best_parent = parent
                        
            if best_parent:
                best_parent["defects"].append(defect)

        # --- STAGE 4: DETERMINISTIC GRADING ---
        for parent in parent_boxes:
            parent_area = calculate_defect_area(parent["box"])
            parent["grade"] = compute_grade(parent["defects"], parent_area)
            
            # Override grade if discard mode active
            if discard_mode:
                parent["grade"] = "DISCARD"

        # --- STAGE 5: EDGE HARVESTING (Active Learning) ---
        save_edge_harvest_frame(frame, all_detections, harvest_dir)

        # --- STAGE 6: OUTPUT RENDERING ENGINE ---
        # Draw parent boxes first
        for parent in parent_boxes:
            x1, y1, x2, y2 = parent["box"]
            grade = parent["grade"]
            display_text = format_display_text(parent["name"], parent["conf"], grade)
            
            # Color coding by grade
            if grade == "DISCARD":
                color = (0, 0, 255)  # Red
            elif grade == "G1":
                color = (0, 255, 0)  # Green
            elif grade == "G2":
                color = (0, 165, 255)  # Orange
            else:  # G3
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

            # Draw bounded child defects (Red layer)
            for defect in parent["defects"]:
                dx1, dy1, dx2, dy2 = defect["box"]
                defect_text = format_display_text(defect["name"], defect["conf"])
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                cv2.putText(frame, defect_text, (dx1, dy1 - 5), cv2.FONT_HERSHEY_MINI, 0.4, (0, 0, 255), 1)

        # Draw discard triggers (Magenta)
        for trigger in discard_triggers:
            tx1, ty1, tx2, ty2 = trigger["box"]
            trigger_text = format_display_text(trigger["name"], trigger["conf"])
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
            cv2.putText(frame, trigger_text, (tx1, ty1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"M4 Edge Engine: {fps:.1f} FPS", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("M4 Edge Sorting Pipeline Engine", frame)

        # Keyboard input handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            # Operator discrepancy override - trigger edge harvest event
            save_edge_harvest_frame(frame, all_detections, harvest_dir, operator_override=True)

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM]: Camera and window resources released.")


if __name__ == "__main__":
    main()
