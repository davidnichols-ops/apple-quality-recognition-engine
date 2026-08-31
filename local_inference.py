#!/usr/bin/env python3
"""
Production Local Inference Script
Apple CoreML Deployment on M4 Neural Engine
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
Feature Detector Pipeline: Three-Class Schema Architecture
Candidate model: YOLO26 CoreML export at 640x640; deployment size selected by benchmark
"""

import argparse
import os
import time
from dataclasses import asdict

import cv2
from ultralytics import YOLO

from camera_utils import detect_arducam_index
from edge_harvest_schema import write_telemetry
from grading_engine import (
    EXPECTED_CLASS_NAMES,
    bind_defects_to_parents,
    discard_parent_indexes,
    grade_apple,
    load_grading_policy,
    model_names_match_expected_schema,
)
from override_persistence import persist_override


def capture_frame_hardened(cap, camera_index=0, max_retries=5):
    """Captures a frame with automatic hardware reconnection logic.

    Args:
        cap: OpenCV VideoCapture object.
        camera_index: Index to re-open the camera at if reconnection is needed.
        max_retries: Maximum reconnection attempts before giving up.

    Returns:
        Tuple of (cap, frame). If all retries fail, returns (cap, None).
    """
    ret, frame = cap.read()

    # Fast path: frame captured successfully
    if ret and frame is not None:
        return cap, frame

    # Hardware disconnect or dropped frame — attempt reconnection
    print("[CRITICAL ERROR]: Arducam dropped connection. Initiating hardware reset...")
    cap.release()
    time.sleep(1.0)

    for attempt in range(1, max_retries + 1):
        print(
            f"[RETRYING]: Attempt {attempt}/{max_retries} — re-binding Arducam sensor..."
        )
        time.sleep(2.0)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            continue

        # Re-apply camera settings after reconnection
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Warmup: discard first few frames — USB cameras on macOS often
        # return empty frames immediately after opening before the sensor
        # finishes initializing.
        warmed = False
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                warmed = True
                break
            time.sleep(0.3)

        if warmed:
            print("[SUCCESS]: Arducam hardware link re-established.")
            return cap, frame

        cap.release()

    print(
        f"[ERROR]: Failed to re-establish Arducam connection after {max_retries} attempts."
    )
    return cap, None


def format_display_text(class_name, confidence, grade=None):
    if class_name == "apple":
        grade_str = f" ({grade})" if grade else ""
        return f"APPLE{grade_str} [{confidence:.2f}]"
    if class_name == "unfit_bin_discard":
        return f"DISCARD [{confidence:.2f}]"
    return f"DEFECT [{confidence:.2f}]"


def main():
    parser = argparse.ArgumentParser(
        description="Apple Quality Recognition Engine - Production Inference"
    )
    parser.add_argument("--policy", default="grading_policy.yaml")
    parser.add_argument("--model", default="yolo26x_640.mlpackage")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--allow-camera-fallback",
        action="store_true",
        help="Allow the built-in camera for a non-production benchmark.",
    )
    parser.add_argument(
        "--benchmark-fallback",
        action="store_true",
        help="Use yolo26x.pt only when the requested model is unavailable.",
    )
    args = parser.parse_args()

    print("[SYSTEM]: Initializing M4 Edge Sorting Pipeline Engine...")
    policy = load_grading_policy(args.policy)
    print(f"[SYSTEM]: Policy {policy.policy_version} for facility {policy.facility_id}")

    # Initialize camera with auto-detected index (matches baseline_verify.py)
    cam_index = detect_arducam_index(allow_builtin_fallback=args.allow_camera_fallback)
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("[ERROR]: Failed to open camera. Check index or macOS permissions.")
        return

    # Force raw uncompressed streaming with MJPG fourcc encoding
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(
        f"[SYSTEM]: Camera configured at {actual_width}x{actual_height} with MJPG encoding"
    )

    # Warmup: discard first few frames — USB cameras on macOS return empty
    # or corrupt frames before the sensor fully initializes.
    for i in range(5):
        ret, _ = cap.read()
        if ret:
            print(f"[SYSTEM]: Camera warmup frame {i + 1}/5 OK")
        else:
            print(f"[SYSTEM]: Camera warmup frame {i + 1}/5 failed (retrying...)")
    print("[SYSTEM]: Camera warmup complete.")

    model_path = args.model
    if not os.path.exists(model_path):
        if not args.benchmark_fallback:
            raise FileNotFoundError(
                f"Model not found: {model_path}. Pass --benchmark-fallback only for "
                "a non-production COCO benchmark."
            )
        fallback = "yolo26x.pt"
        print(f"[WARNING]: {model_path} not found; using {fallback} for benchmarking.")
        model_path = fallback

    print(f"[SYSTEM]: Loading model '{model_path}'...")
    model = YOLO(model_path, task="detect")
    num_classes = len(model.names)
    benchmark_mode = not model_names_match_expected_schema(model.names)
    print(f"[SYSTEM]: Loaded model with {num_classes} classes")
    print(f"[SYSTEM]: Expected schema: {EXPECTED_CLASS_NAMES}")
    if benchmark_mode:
        print(f"[WARNING]: Model schema mismatch: {model.names}")
        print(
            "[WARNING]: BENCHMARK MODE — grades are disabled and detections are not harvested."
        )
    else:
        print("[SYSTEM]: Three-class candidate schema active. Press 'q' to exit.")

    # Edge harvest directory
    harvest_dir = "dataset/edge_harvest"

    frame_count = 0
    try:
        while True:
            frame_count += 1
            start_time = time.time()

            cap, frame = capture_frame_hardened(cap, camera_index=cam_index)

            # If the camera failed to produce a frame after all retries, bail out
            if frame is None:
                print("[ERROR]: No frame available. Exiting inference loop.")
                break

            results = model(frame, conf=0.35, imgsz=640, verbose=False)

            parent_boxes = []
            discard_triggers = []
            defect_boxes = []
            all_detections = []

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                coords = list(map(int, box.xyxy[0]))
                confidence = float(box.conf[0])
                class_name = model.names[cls_id]
                detection = {
                    "id": cls_id,
                    "name": class_name,
                    "box": coords,
                    "conf": confidence,
                }
                all_detections.append(detection)
                if benchmark_mode:
                    continue
                if class_name == "apple":
                    parent_boxes.append({**detection, "defects": []})
                elif class_name == "unfit_bin_discard":
                    discard_triggers.append(detection)
                elif class_name == "class_defect":
                    defect_boxes.append(detection)

            parent_coordinates = [parent["box"] for parent in parent_boxes]
            defect_coordinates = [defect["box"] for defect in defect_boxes]
            bindings = bind_defects_to_parents(
                defect_coordinates,
                parent_coordinates,
                policy.ioa_binding_threshold,
            )
            for parent_index, defect_indexes in enumerate(bindings):
                parent_boxes[parent_index]["defects"] = [
                    defect_boxes[index] for index in defect_indexes
                ]
            bound_defect_indexes = {
                defect_index
                for defect_indexes in bindings
                for defect_index in defect_indexes
            }
            orphan_defect_count = len(defect_boxes) - len(bound_defect_indexes)

            discarded = discard_parent_indexes(
                [trigger["box"] for trigger in discard_triggers],
                parent_coordinates,
                policy.discard_proximity_px,
            )
            for parent_index, parent in enumerate(parent_boxes):
                decision = grade_apple(
                    parent["box"],
                    [defect["box"] for defect in parent["defects"]],
                    policy,
                    discard=parent_index in discarded,
                )
                parent["decision"] = decision
                parent["grade"] = decision.grade

            grading_results = [
                {
                    **asdict(parent["decision"]),
                    "defects": parent["defects"],
                }
                for parent in parent_boxes
            ]
            review_reasons = []
            if any(result["requires_refinement"] for result in grading_results):
                review_reasons.append("coverage_near_grade_boundary")
            if orphan_defect_count:
                review_reasons.append("orphan_defect")
            if not benchmark_mode:
                write_telemetry(
                    frame,
                    all_detections,
                    harvest_dir,
                    grading_results=grading_results,
                    force_review=bool(review_reasons),
                    review_reason=",".join(review_reasons) or None,
                    model_id=os.path.basename(model_path),
                    policy_version=policy.policy_version,
                )

            # --- STAGE 6: OUTPUT RENDERING ENGINE ---
            fps = 1.0 / (time.time() - start_time)

            if not args.no_display:
                # Draw parent boxes first
                for parent in parent_boxes:
                    x1, y1, x2, y2 = parent["box"]
                    grade = parent["grade"]
                    display_text = format_display_text(
                        parent["name"], parent["conf"], grade
                    )

                    # Color coding by grade
                    if grade == "DISCARD":
                        color = (255, 0, 255)  # Magenta
                    elif grade == "G1":
                        color = (0, 255, 0)  # Green
                    elif grade == "G2":
                        color = (0, 165, 255)  # Orange
                    else:  # G3
                        color = (0, 0, 255)  # Red

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        display_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.5,
                        color,
                        1,
                    )

                    # Draw bounded child defects (Red layer)
                    for defect in parent["defects"]:
                        dx1, dy1, dx2, dy2 = defect["box"]
                        defect_text = format_display_text(
                            defect["name"], defect["conf"]
                        )
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            defect_text,
                            (dx1, dy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                        )

                # Draw discard triggers (Magenta)
                for trigger in discard_triggers:
                    tx1, ty1, tx2, ty2 = trigger["box"]
                    trigger_text = format_display_text(trigger["name"], trigger["conf"])
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
                    cv2.putText(
                        frame,
                        trigger_text,
                        (tx1, ty1 - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.5,
                        (255, 0, 255),
                        1,
                    )

                cv2.putText(
                    frame,
                    f"M4 Edge Engine: {fps:.1f} FPS",
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
                cv2.imshow("M4 Edge Sorting Pipeline Engine", frame)

                # Keyboard input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("g"):
                    persist_override(
                        frame,
                        all_detections,
                        grading_results,
                        policy_path=args.policy,
                        facility_id=policy.facility_id,
                    )
                    write_telemetry(
                        frame,
                        all_detections,
                        harvest_dir,
                        operator_override=True,
                        grading_results=grading_results,
                        force_review=True,
                        review_reason="operator_override",
                        model_id=os.path.basename(model_path),
                        policy_version=policy.policy_version,
                    )
            elif frame_count % 30 == 0:
                print(f"\r[FPS] {fps:.1f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[SYSTEM]: Interrupted by user.")
    except Exception as e:
        print(f"[CRITICAL ERROR]: Inference loop crashed: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM]: Camera and window resources released.")


if __name__ == "__main__":
    main()
