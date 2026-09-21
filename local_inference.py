#!/usr/bin/env python3
"""
Production Local Inference Script
Apple CoreML Deployment on M4 Neural Engine
Target: MacBook Air M4 (macOS 26 Tahoe / Darwin 25.5.0)
Feature Detector Pipeline: Four-Class Vision Taxonomy + Five-Grade Policy
Candidate model: YOLO26 CoreML export at 640x640; deployment size selected by benchmark
"""

import argparse
import json
import os
import time

import cv2
from ultralytics import YOLO

from camera_utils import detect_arducam_index
from edge_harvest_schema import compute_frame_hash, write_telemetry
from grading_engine import (
    EXPECTED_CLASS_NAMES,
    load_grading_policy,
    model_names_match_expected_schema,
)
from glove_masking import mask_blue_gloves
from line_runtime import (
    DETECTION_CONFIDENCE_FLOOR,
    FrameAssessment,
    assess_result,
    focus_score,
    frame_event,
    gate_frame,
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
    if class_name == "stem_calyx":
        return f"STEM/CALYX [{confidence:.2f}]"
    if class_name == "defect_surface":
        return f"SURFACE [{confidence:.2f}]"
    if class_name == "defect_critical":
        return f"CRITICAL [{confidence:.2f}]"
    return f"{class_name.upper()} [{confidence:.2f}]"


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
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=25.0,
        help="Candidate Laplacian-variance floor; lower-scoring frames emit no grade.",
    )
    parser.add_argument(
        "--max-consecutive-harvest-frames",
        type=int,
        default=300,
        help="Stop with no grade after this many successive harvested frames.",
    )
    args = parser.parse_args()
    if args.min_sharpness < 0:
        parser.error("--min-sharpness must be non-negative")
    if args.max_consecutive_harvest_frames < 1:
        parser.error("--max-consecutive-harvest-frames must be positive")

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
        print("[SYSTEM]: Four-class candidate schema active. Press 'q' to exit.")

    # Edge harvest directory
    harvest_dir = "dataset/edge_harvest"

    frame_count = 0
    previous_frame_signature = None
    previous_status = None
    consecutive_harvest_frames = 0
    try:
        while True:
            frame_count += 1
            start_time = time.time()

            cap, frame = capture_frame_hardened(cap, camera_index=cam_index)

            # If the camera failed to produce a frame after all retries, bail out
            if frame is None:
                print("[ERROR]: No frame available. Exiting inference loop.")
                print(
                    json.dumps(
                        {
                            "event": "frame_assessment",
                            "frame_id": frame_count,
                            "status": "camera_failure",
                            "image": None,
                            "grades": [],
                            "review_reasons": ["camera_failure"],
                        }
                    ),
                    flush=True,
                )
                break

            height, width = frame.shape[:2]
            frame_hash = None
            try:
                frame_hash = compute_frame_hash(frame)
                sharpness = focus_score(frame)
                assessment = gate_frame(
                    frame,
                    previous_frame_signature,
                    frame_hash,
                    sharpness,
                    args.min_sharpness,
                )
            except Exception as exc:
                print(f"[ERROR]: Frame preprocessing failed: {exc}")
                assessment = FrameAssessment(
                    "preprocess_error_no_grade",
                    height,
                    width,
                    review_reasons=["preprocess_error"],
                )
            if assessment is None:
                # Query below the former 0.35 cutoff so uncertain critical
                # candidates are available to grading and review harvest.
                try:
                    glove_mask, _ = mask_blue_gloves(frame)
                    results = model(
                        frame,
                        conf=DETECTION_CONFIDENCE_FLOOR,
                        imgsz=640,
                        verbose=False,
                    )
                    if len(results) != 1:
                        assessment = FrameAssessment(
                            "invalid_result",
                            height,
                            width,
                            review_reasons=["invalid_result_count"],
                        )
                    else:
                        assessment = assess_result(
                            frame,
                            results[0],
                            model.names,
                            policy,
                            glove_mask=glove_mask,
                            benchmark_mode=benchmark_mode,
                        )
                except Exception as exc:
                    print(f"[ERROR]: Frame inference failed: {exc}")
                    assessment = FrameAssessment(
                        "inference_error_no_grade",
                        height,
                        width,
                        review_reasons=["inference_error"],
                    )
            if frame_hash is not None:
                previous_frame_signature = (height, width, frame_hash)

            parent_boxes = assessment.parents
            stem_calyx_boxes = assessment.stem_calyx
            all_detections = assessment.detections
            grading_results = assessment.grading_results
            review_reasons = list(assessment.review_reasons)
            # The transition off a graded frame is recorded once, not by
            # replaying its last grade or harvesting every empty belt frame.
            if assessment.status == "no_apple" and previous_status == "graded":
                review_reasons.append("empty_after_grade")
            elif assessment.status in {"duplicate_frame", "blur_no_grade"}:
                if previous_status != assessment.status:
                    review_reasons.append(assessment.status)
            previous_status = assessment.status
            try:
                saved_path = None
                if not benchmark_mode:
                    saved_path = write_telemetry(
                        frame,
                        all_detections,
                        harvest_dir,
                        grading_results=grading_results,
                        force_review=bool(review_reasons),
                        review_reason=",".join(review_reasons) or None,
                        model_id=os.path.basename(model_path),
                        policy_version=policy.policy_version,
                        frame_status=assessment.status,
                    )
            except Exception as exc:
                print(f"[ERROR]: Review harvest failed: {exc}")
                failed = FrameAssessment(
                    "harvest_failure_no_grade",
                    height,
                    width,
                    review_reasons=["harvest_failure"],
                )
                print(json.dumps(frame_event(frame_count, failed)), flush=True)
                break
            consecutive_harvest_frames = (
                consecutive_harvest_frames + 1 if saved_path is not None else 0
            )
            if consecutive_harvest_frames >= args.max_consecutive_harvest_frames:
                stopped = FrameAssessment(
                    "harvest_backlog_no_grade",
                    height,
                    width,
                    review_reasons=["harvest_backlog"],
                )
                print(json.dumps(frame_event(frame_count, stopped)), flush=True)
                break
            print(json.dumps(frame_event(frame_count, assessment)), flush=True)

            # --- STAGE 6: OUTPUT RENDERING ENGINE ---
            fps = 1.0 / (time.time() - start_time)

            if not args.no_display:
                # Draw parent boxes only from this frame's accepted result.
                for parent in parent_boxes:
                    x1, y1, x2, y2 = (int(round(v)) for v in parent["box"])
                    grade = parent["grade"]
                    display_text = format_display_text(
                        parent["name"], parent["conf"], grade
                    )

                    # Color coding by grade
                    if grade == "DISCARD":
                        color = (255, 0, 255)  # Magenta
                    elif grade == "CIDER":
                        color = (255, 0, 0)  # Blue
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

                    # Draw bounded surface defects (Red)
                    for defect in parent["defects"]:
                        dx1, dy1, dx2, dy2 = (int(round(v)) for v in defect["box"])
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

                    # Draw bounded critical defects (Magenta)
                    for critical in parent["criticals"]:
                        cx1, cy1, cx2, cy2 = (int(round(v)) for v in critical["box"])
                        critical_text = format_display_text(
                            critical["name"], critical["conf"]
                        )
                        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 0, 255), 2)
                        cv2.putText(
                            frame,
                            critical_text,
                            (cx1, cy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 0, 255),
                            1,
                        )

                # Draw stem/calyx exclusion zones (Yellow)
                for sc in stem_calyx_boxes:
                    sx1, sy1, sx2, sy2 = (int(round(v)) for v in sc["box"])
                    sc_text = format_display_text(sc["name"], sc["conf"])
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
                    cv2.putText(
                        frame,
                        sc_text,
                        (sx1, sy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
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
                        frame_status=assessment.status,
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
