#!/usr/bin/env python3
"""
Agritech Edge Sandbox Demo Inference Script
Interactive Plant Health Monitoring with YOLO11x COCO Model
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
Feature Detector Pipeline: Dynamic Schema Architecture
"""

import cv2
import time
import json
import os
import yaml
from datetime import datetime
from ultralytics import YOLO

# COCO Class 58 = potted plant (our parent container)
PARENT_CLASS_ID = 58

# COCO class names that roughly map to plant health severity categories
COCO_SEVERITY_MAPPING = {
    # Small objects that could be considered mild leaf spots
    'mild': ['sports ball', 'apple', 'orange', 'banana', 'coin', 'simulated_leaf_rot'],
    # Medium objects that could be moderate canopy issues
    'moderate': ['cup', 'book', 'cell phone'],
    # Large/dangerous objects that could be severe threats
    'severe': ['scissors', 'bird', 'mouse', 'knife']
}

# --- GLOBAL VARIABLES FOR MANUAL TESTING OVERRIDE ---
manual_test_boxes = []

def mouse_click_handler(event, x, y, flags, param):
    """Handles manual clicking to simulate leaf defects on healthy plants."""
    global manual_test_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        # Create a mock 40x40 pixel child defect box centered on mouse coordinates
        manual_test_boxes.append([x - 20, y - 20, x + 20, y + 20])
        print(f"[MANUAL BREAKOUT]: Simulated defect injected at ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        manual_test_boxes.clear()
        print("[MANUAL BREAKOUT]: Cleared all simulated leaf defects.")


def map_coco_to_severity(class_name, mild_defects, moderate_defects, severe_defects):
    """
    Dynamically map COCO class names to severity categories.
    Falls back to 'mild' for any unmapped class as generic impurity.
    """
    if class_name in mild_defects:
        return 'mild'
    elif class_name in moderate_defects:
        return 'moderate'
    elif class_name in severe_defects:
        return 'severe'
    
    class_lower = class_name.lower()
    if class_lower in COCO_SEVERITY_MAPPING['mild']:
        return 'mild'
    elif class_lower in COCO_SEVERITY_MAPPING['moderate']:
        return 'moderate'
    elif class_lower in COCO_SEVERITY_MAPPING['severe']:
        return 'severe'
    
    return 'mild'


def capture_frame_hardened(cap, camera_index=0):
    """Captures a frame with automatic hardware reconnection logic."""
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[CRITICAL ERROR]: Camera dropped connection. Initiating hardware reset...")
        cap.release()
        while True:
            time.sleep(2.0)
            print("[RETRYING]: Attempting to re-bind camera sensor...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print("[SUCCESS]: Camera hardware link re-established.")
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, frame
    return cap, frame


def load_demo_policy(policy_path="demo_grading_policy.yaml"):
    """Load demo grading policy from YAML configuration file."""
    try:
        with open(policy_path, 'r') as file:
            policy = yaml.safe_load(file)
        
        mild = set(policy['severity_mapping']['mild_defects'])
        moderate = set(policy['severity_mapping']['moderate_defects'])
        severe = set(policy['severity_mapping']['severe_defects'])
        rules = policy['rules']
        
        print(f"[SANDBOX ACTIVE]: Successfully loaded demo grading rules from {policy_path}")
        print(f"[SANDBOX ACTIVE]: Monitoring Parent Class ID {PARENT_CLASS_ID} via IoA Binding Engine.")
        return mild, moderate, severe, rules
    except Exception as e:
        print(f"[ERROR]: Failed to load demo grading policy: {e}")
        print("[SYSTEM]: Falling back to default demo grading rules")
        return {"stray_crumb", "dust_speck"}, \
               {"smudge_mark", "surface_scratch"}, \
               {"structural_crack", "spill_liquid"}, \
               {'max_mild_for_g1': 0, 'area_threshold_g3_pct': 1.0, 'ioa_binding_threshold': 0.10}


def calculate_defect_area(box):
    """Calculate pixel area of a bounding box."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def calculate_intersection_area(box1, box2):
    """Calculate the intersection area between two bounding boxes."""
    x1, y1, x2, y2 = box1
    px1, py1, px2, py2 = box2
    
    ix1 = max(x1, px1)
    iy1 = max(y1, py1)
    ix2 = min(x2, px2)
    iy2 = min(y2, py2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    
    return (ix2 - ix1) * (iy2 - iy1)


def calculate_ioa(defect_box, parent_box):
    """Calculate Intersection-over-Area (IoA) ratio."""
    intersection = calculate_intersection_area(defect_box, parent_box)
    defect_area = calculate_defect_area(defect_box)
    if defect_area == 0:
        return 0.0
    return intersection / defect_area


def compute_demo_grade(defects, parent_area, rules):
    """Deterministic scoring function for container health validation."""
    if not defects:
        return 'GRADE_1'
    
    mild_count = sum(1 for d in defects if d['severity'] == 'mild')
    moderate_count = sum(1 for d in defects if d['severity'] == 'moderate')
    severe_count = sum(1 for d in defects if d['severity'] == 'severe')
    
    total_impurities = mild_count + moderate_count + severe_count
    max_mild_for_g1 = rules.get('max_mild_for_g1', 0)
    area_threshold_g3 = rules.get('area_threshold_g3_pct', 10.0) / 100.0
    
    total_defect_area = sum(calculate_defect_area(d['box']) for d in defects)
    area_ratio = total_defect_area / parent_area if parent_area > 0 else 0
    
    if severe_count > 0 or area_ratio > area_threshold_g3:
        return 'CRITICAL'
    elif total_impurities > max_mild_for_g1:
        return 'GRADE_2'
    else:
        return 'GRADE_1'


def format_demo_display_text(class_name, confidence, grade=None):
    """Format display string for demo predictions."""
    if grade:
        if grade == 'GRADE_1':
            return f"GRADE 1: HEALTHY CANOPY [{confidence:.2f}]"
        elif grade == 'GRADE_2':
            return f"GRADE 2: MINOR ANOMALIES [{confidence:.2f}]"
        elif grade == 'CRITICAL':
            return f"CRITICAL: AUDIT REQUIRED / CANOPY DAMAGE [{confidence:.2f}]"
    return f"{class_name.upper()} [{confidence:.2f}]"


def save_demo_telemetry(frame, parent_containers, telemetry_path="demo_telemetry.json"):
    """Save telemetry data for demo sessions when operator override is triggered."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    telemetry = {
        "timestamp": timestamp,
        "facility_id": "LIVING_ROOM_BOTANICAL_CHECK",
        "parent_containers": [
            {
                "class_id": p["id"],
                "class_name": p["name"],
                "confidence": p["conf"],
                "box": p["box"],
                "grade": p["grade"],
                "impurity_count": len(p["defects"]),
                "impurities": [
                    {"class_name": d["name"], "confidence": d["conf"], "box": d["box"], "severity": d["severity"]}
                    for d in p["defects"]
                ]
            }
            for p in parent_containers
        ]
    }
    try:
        if os.path.exists(telemetry_path):
            with open(telemetry_path, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(telemetry)
            else:
                existing_data = [existing_data, telemetry]
        else:
            existing_data = [telemetry]
        
        with open(telemetry_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
        print("[DEMO TELEMETRY]: Session data logged to demo_telemetry.json")
    except Exception as e:
        print(f"[ERROR]: Failed to save telemetry: {e}")


def main():
    global manual_test_boxes
    print("[SANDBOX ACTIVE]: Initializing Agritech Plant Canopy Audit Demo...")
    
    MILD_DEFECTS, MODERATE_DEFECTS, SEVERE_DEFECTS, DEMO_RULES = load_demo_policy()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR]: Failed to open camera. Check index or macOS permissions.")
        return
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize high-speed OpenCV named window and bind mouse event callback
    window_name = "Agritech Edge Sandbox - Plant Canopy Audit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click_handler)
    
    print(f"[SYSTEM]: Camera configured with MJPG encoding")
    
    try:
        model = YOLO("yolo11x.mlpackage")
    except Exception as e:
        print(f"[ERROR]: Failed to load YOLO11x model: {e}")
        cap.release()
        return
    
    print(f"[SANDBOX ACTIVE]: YOLO11x COCO model loaded with {len(model.names)} classes")
    print(f"[SANDBOX ACTIVE]: Parent Class ID {PARENT_CLASS_ID} = '{model.names[PARENT_CLASS_ID]}'")
    print("[SANDBOX ACTIVE]: Left-Click Window to add Simulated Anomalies | Right-Click to clear.")
    print("[SANDBOX ACTIVE]: Press 'q' to exit, 'g' to log telemetry event")
    
    while True:
        start_time = time.time()
        cap, frame = capture_frame_hardened(cap, camera_index=0)
        
        results = model(frame, conf=0.35, imgsz=640, verbose=False)
        
        parent_boxes = []
        child_boxes = []

        # --- STAGE 1: INSTANCE PARSING (COCO Schema) ---
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            coords = list(map(int, box.xyxy[0]))
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            detection = {"id": cls_id, "name": class_name, "box": coords, "conf": conf}
            
            if cls_id == PARENT_CLASS_ID:
                parent_boxes.append({"id": cls_id, "name": class_name, "box": coords, "conf": conf, "defects": []})
            else:
                severity = map_coco_to_severity(class_name, MILD_DEFECTS, MODERATE_DEFECTS, SEVERE_DEFECTS)
                child_boxes.append({**detection, "severity": severity})

        # --- INTERACTIVE SIMULATION OVERRIDE INJECTION (Option 2) ---
        for mock_box in manual_test_boxes:
            mock_child = {
                "id": 999,
                "name": "simulated_leaf_rot",
                "box": mock_box,
                "conf": 1.00,
                "severity": "mild"
            }
            child_boxes.append(mock_child)

        # --- STAGE 2: SPATIAL BINDING LAYER (IoA Override) ---
        ioa_threshold = DEMO_RULES.get('ioa_binding_threshold', 0.10)
        
        for child in child_boxes:
            child_box = child["box"]
            best_parent = None
            max_ioa = 0.0

            for parent in parent_boxes:
                parent_box = parent["box"]
                ioa = calculate_ioa(child_box, parent_box)
                if ioa >= ioa_threshold and ioa > max_ioa:
                    max_ioa = ioa
                    best_parent = parent
                        
            if best_parent:
                best_parent["defects"].append(child)

        # --- STAGE 3: DETERMINISTIC GRADING ---
        for parent in parent_boxes:
            parent_area = calculate_defect_area(parent["box"])
            parent["grade"] = compute_demo_grade(parent["defects"], parent_area, DEMO_RULES)

        # --- STAGE 4: OUTPUT RENDERING ENGINE ---
        for parent in parent_boxes:
            x1, y1, x2, y2 = parent["box"]
            grade = parent["grade"]
            display_text = format_demo_display_text(parent["name"], parent["conf"], grade)
            
            color = (0, 255, 0) if grade == "GRADE_1" else ((0, 165, 255) if grade == "GRADE_2" else (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

            for child in parent["defects"]:
                dx1, dy1, dx2, dy2 = child["box"]
                # Color code manual vs neural network targets
                is_mock = child["id"] == 999
                child_color = (0, 255, 255) if is_mock else (0, 0, 255)
                child_text = child["name"].upper() if is_mock else format_demo_display_text(child["name"], child["conf"])
                
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), child_color, 2)
                cv2.putText(frame, child_text, (dx1, dy1 - 5), cv2.FONT_HERSHEY_PLAIN, 0.5, child_color, 1)

        # Draw unbound child boxes
        for child in child_boxes:
            is_bound = any(child in parent["defects"] for parent in parent_boxes)
            if not is_bound:
                cx1, cy1, cx2, cy2 = child["box"]
                if child["id"] == 999:
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 1)
                    cv2.putText(frame, "SIMULATED_LEAF_ROT", (cx1, cy1 - 5), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255), 1)
                else:
                    child_text = format_demo_display_text(child["name"], child["conf"])
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (128, 128, 128), 1)
                    cv2.putText(frame, child_text, (cx1, cy1 - 5), cv2.FONT_HERSHEY_PLAIN, 0.3, (128, 128, 128), 1)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"Agritech Plant Audit: {fps:.1f} FPS", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Parent: {model.names[PARENT_CLASS_ID]} | IoA Threshold: {ioa_threshold:.2f}", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            save_demo_telemetry(frame, parent_boxes)

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM]: Camera and window resources released.")


if __name__ == "__main__":
    main()