#!/usr/bin/env python3
"""
Plant Health Recognition Engine — Local Inference
Apple Silicon M4 Neural Engine Deployment
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)

3-Tier Feature Detector Pipeline:
  plant (parent) → leaf (child) → class_defect (grandchild)

The YOLO26n model detects all four classes.  The deterministic grading
engine walks the tree bottom-up:
  1. Bind defects to leaves via IoA
  2. Grade each leaf (HEALTHY / MODERATE / POOR / DISCARD)
  3. Bind leaves to plants via IoA
  4. Grade each plant from aggregate leaf grades

Models observe.  Deterministic policy decides.  Humans authorize learning.
"""

import argparse
import json
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from plant_grading_engine import (
    CLASS_DEFECT_ID,
    LEAF_CLASS_ID,
    PLANT_CLASS_ID,
    UNFIT_DISCARD_CLASS_ID,
    Detection,
    grade_detections,
    load_grading_policy,
    model_names_match_expected_schema,
)


# ── Camera hardening ────────────────────────────────────────────────

def capture_frame_hardened(cap, camera_index=0, max_retries=5):
    """Capture a frame with bounded hardware reconnection attempts."""
    ret, frame = cap.read()
    if ret and frame is not None:
        return cap, frame

    print("[CRITICAL]: Camera dropped connection. Initiating hardware reset...")
    cap.release()
    for attempt in range(1, max_retries + 1):
        time.sleep(2.0)
        print(f"[RETRY] Attempt {attempt}/{max_retries}")
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Warmup: discard first few frames
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                print("[SUCCESS]: Camera link re-established.")
                return cap, frame
            time.sleep(0.3)
        cap.release()
    return cap, None


# ── Display helpers ─────────────────────────────────────────────────

GRADE_COLORS = {
    "HEALTHY": (0, 255, 0),    # Green
    "MODERATE": (0, 165, 255),  # Orange
    "POOR": (0, 0, 255),       # Red
    "DISCARD": (128, 0, 128),  # Purple
}


def format_plant_label(grade: str, conf: float, leaf_count: int) -> str:
    return f"PLANT ({grade}) [{conf:.2f}] leaves={leaf_count}"


def format_leaf_label(grade: str, conf: float) -> str:
    return f"LEAF ({grade}) [{conf:.2f}]"


def format_defect_label(conf: float) -> str:
    return f"DEFECT [{conf:.2f}]"


# ── Edge harvest ────────────────────────────────────────────────────

def save_edge_harvest_frame(
    frame,
    detections_raw,
    plant_grades,
    harvest_dir,
    operator_override=False,
):
    """Save frame and telemetry for low-confidence or operator-override cases."""
    volatile = [d for d in detections_raw if 0.40 <= d.confidence <= 0.65]
    needs_refine = any(pg.needs_refinement for pg in plant_grades)

    if not volatile and not operator_override and not needs_refine:
        return

    os.makedirs(harvest_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = os.path.join(harvest_dir, f"frame_{timestamp}.jpg")
    telemetry_path = os.path.join(harvest_dir, f"telemetry_{timestamp}.json")

    cv2.imwrite(frame_path, frame)

    telemetry = {
        "timestamp": timestamp,
        "frame_path": frame_path,
        "operator_override": operator_override,
        "needs_refinement": needs_refine,
        "volatile_detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "box": list(d.box),
            }
            for d in volatile
        ],
        "plant_grades": [
            {
                "grade": pg.grade,
                "leaf_count": pg.leaf_count,
                "healthy_leaves": pg.healthy_leaves,
                "moderate_leaves": pg.moderate_leaves,
                "poor_leaves": pg.poor_leaves,
                "discard_triggered": pg.discard_triggered,
            }
            for pg in plant_grades
        ],
    }

    with open(telemetry_path, "w") as fh:
        json.dump(telemetry, fh, indent=2)

    if operator_override:
        print("[OVERRIDE] Manual mismatch recorded to harvest cache.")
    elif needs_refine:
        print(f"[REFINE] Boundary case saved to {frame_path}")
    else:
        print(f"[HARVEST] Volatile frame saved to {frame_path}")


# ── Main inference loop ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plant Health Recognition Engine — local inference"
    )
    parser.add_argument(
        "--model",
        default="best.mlpackage",
        help="Path to CoreML or PyTorch model. Defaults to best.mlpackage.",
    )
    parser.add_argument(
        "--policy",
        default="grading_policy.yaml",
        help="Path to grading policy YAML.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--harvest-dir",
        default="dataset/edge_harvest",
        help="Directory for edge-harvested frames.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="FPS-only mode — no grading, no display annotations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("[SYSTEM] Plant Health Recognition Engine — initializing...")

    # Load grading policy
    rules = load_grading_policy(args.policy)
    print(f"[SYSTEM] Grading policy loaded from {args.policy}")

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera index {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[SYSTEM] Camera: {w:.0f}x{h:.0f} MJPG")

    # Load model
    print(f"[SYSTEM] Loading model: {args.model}")
    try:
        model = YOLO(args.model, task="detect")
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        cap.release()
        return

    # Validate schema
    if not model_names_match_expected_schema(list(model.names.values())):
        print("[WARN] Model class names do not match expected 4-class schema:")
        print("       Expected: plant, leaf, unfit_discard, class_defect")
        print(f"       Got:      {list(model.names.values())}")
        print("       Grading will still attempt to run with available classes.")

    print(f"[SYSTEM] Model classes: {model.names}")
    print("[SYSTEM] Press 'q' to quit, 'g' to log operator override.")

    while True:
        start = time.time()

        cap, frame = capture_frame_hardened(cap, camera_index=args.camera)
        if frame is None:
            print("[ERROR] Camera reconnect limit reached. Stopping.")
            break

        results = model(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        if args.benchmark:
            fps = 1.0 / (time.time() - start)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            cv2.imshow("Plant Health Engine — Benchmark", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Parse detections into Detection objects
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            coords = tuple(float(v) for v in box.xyxy[0])
            conf = float(box.conf[0])
            name = model.names.get(cls_id, f"unknown_{cls_id}")
            detections.append(Detection(cls_id, name, conf, coords))

        # Grade using the deterministic engine
        plant_grades = grade_detections(detections, rules)

        # Edge harvest
        save_edge_harvest_frame(frame, detections, plant_grades, args.harvest_dir)

        # Render
        _render(frame, detections, plant_grades, model.names)

        fps = 1.0 / (time.time() - start)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Plant Health Recognition Engine", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g"):
            save_edge_harvest_frame(
                frame, detections, plant_grades, args.harvest_dir,
                operator_override=True,
            )

    cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Resources released.")


def _render(frame, detections, plant_grades, model_names):
    """Draw bounding boxes and grade labels on the frame."""
    # Build a lookup from plant index to PlantGrade
    plant_boxes = [d for d in detections if d.class_id == PLANT_CLASS_ID]
    leaf_boxes = [d for d in detections if d.class_id == LEAF_CLASS_ID]
    defect_boxes = [d for d in detections if d.class_id == CLASS_DEFECT_ID]
    discard_boxes = [d for d in detections if d.class_id == UNFIT_DISCARD_CLASS_ID]

    # Draw defects (small red boxes)
    for d in defect_boxes:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(
            frame,
            format_defect_label(d.confidence),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
        )

    # Draw leaves (blue boxes with grade)
    for i, d in enumerate(leaf_boxes):
        x1, y1, x2, y2 = (int(v) for v in d.box)
        # Find this leaf's grade from plant_grades
        leaf_grade = "HEALTHY"
        for pg in plant_grades:
            if i < len(pg.leaf_grades):
                lg = pg.leaf_grades[i]
                leaf_grade = lg.grade
                break
        color = GRADE_COLORS.get(leaf_grade, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            format_leaf_label(leaf_grade, d.confidence),
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
        )

    # Draw plants (thick boxes with grade)
    for i, d in enumerate(plant_boxes):
        x1, y1, x2, y2 = (int(v) for v in d.box)
        grade = plant_grades[i].grade if i < len(plant_grades) else "UNKNOWN"
        color = GRADE_COLORS.get(grade, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = format_plant_label(
            grade, d.confidence,
            plant_grades[i].leaf_count if i < len(plant_grades) else 0,
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            color,
            2,
        )

    # Draw discard triggers (magenta)
    for d in discard_boxes:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            frame,
            f"DISCARD [{d.confidence:.2f}]",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (255, 0, 255),
            2,
        )


if __name__ == "__main__":
    main()
