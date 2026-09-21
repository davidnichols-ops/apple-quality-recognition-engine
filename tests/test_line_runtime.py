import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from edge_harvest_schema import write_telemetry
from grading_engine import EXPECTED_CLASS_NAMES, load_grading_policy
from line_runtime import assess_result, focus_score, frame_event, gate_frame


class Frame:
    def __init__(self, token: bytes, height: int = 120, width: int = 160):
        self.data = token
        self.shape = (height, width, 3)

    def tobytes(self):
        return self.data


def detection(class_id, xyxy, confidence):
    return SimpleNamespace(cls=[class_id], xyxy=[xyxy], conf=[confidence])


def result(frame, *boxes, orig_shape=None, orig_img=None):
    return SimpleNamespace(
        orig_shape=orig_shape or frame.shape[:2],
        orig_img=orig_img or frame,
        boxes=boxes,
    )


def policy():
    return load_grading_policy("grading_policy.yaml")


def test_empty_hop_then_new_size_uses_current_frame_coordinates():
    empty = Frame(b"empty", height=120, width=160)
    next_frame = Frame(b"new", height=480, width=640)
    first = assess_result(empty, result(empty), EXPECTED_CLASS_NAMES, policy())
    second = assess_result(
        next_frame,
        result(
            next_frame,
            detection(0, [100, 100, 500, 400], 0.92),
            detection(3, [101, 101, 105, 105], 0.57),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    assert frame_event(1, first)["image"] == {"width": 160, "height": 120}
    assert frame_event(1, first)["grades"] == []
    assert frame_event(2, second)["image"] == {"width": 640, "height": 480}
    assert second.grading_results[0]["grade"] == "DISCARD"
    assert second.grading_results[0]["critical_defect_count"] == 1
    assert second.review_reasons == ["uncertain_critical"]


def test_small_rot_on_large_apple_is_not_diluted_by_surface_area():
    frame = Frame(b"rot", height=1000, width=1000)
    assessment = assess_result(
        frame,
        result(
            frame,
            detection(0, [10, 10, 990, 990], 0.94),
            detection(3, [20, 20, 23, 23], 0.27),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    assert assessment.status == "graded"
    assert assessment.grading_results[0]["grade"] == "DISCARD"
    assert assessment.grading_results[0]["critical_defect_count"] == 1
    assert "uncertain_critical" in assessment.review_reasons
    assert assessment.detections[1]["conf"] == 0.27


def test_lower_detector_floor_does_not_turn_uncertain_surface_into_g1():
    frame = Frame(b"uncertain")
    assessment = assess_result(
        frame,
        result(
            frame,
            detection(0, [0, 0, 100, 100], 0.9),
            detection(2, [10, 10, 13, 13], 0.27),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    assert assessment.status == "uncertain_detection_no_grade"
    assert frame_event(1, assessment)["grades"] == []
    assert "low_confidence_noncritical" in assessment.review_reasons


def test_low_confidence_stem_calyx_does_not_veto_apple_grade():
    frame = Frame(b"stem")
    assessment = assess_result(
        frame,
        result(
            frame,
            detection(0, [0, 0, 100, 100], 0.9),
            detection(1, [10, 10, 20, 20], 0.27),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    assert frame_event(1, assessment)["grades"] == ["G1"]


def test_orphan_critical_cannot_emit_g1():
    frame = Frame(b"orphan")
    assessment = assess_result(
        frame,
        result(
            frame,
            detection(0, [0, 0, 50, 50], 0.9),
            detection(3, [100, 80, 105, 85], 0.50),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    assert assessment.status == "critical_orphan_no_grade"
    assert frame_event(3, assessment)["grades"] == []
    assert "orphan_defect" in assessment.review_reasons


def test_critical_in_two_overlapping_apples_holds_both_grades():
    frame = Frame(b"ambiguous")
    assessment = assess_result(
        frame,
        result(
            frame,
            detection(0, [0, 0, 100, 100], 0.9),
            detection(0, [50, 0, 150, 100], 0.9),
            detection(3, [60, 20, 64, 24], 0.55),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    assert assessment.status == "ambiguous_critical_no_grade"
    assert frame_event(1, assessment)["grades"] == []
    assert "ambiguous_critical_parent" in assessment.review_reasons


def test_blur_clears_previous_grade_without_using_previous_result():
    sharp = Frame(b"sharp")
    blur = Frame(b"blur")
    graded = assess_result(
        sharp,
        result(sharp, detection(0, [0, 0, 100, 100], 0.9)),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    blocked = gate_frame(blur, (120, 160, "sharp-hash"), "blur-hash", 4.0, 25.0)
    assert frame_event(1, graded)["grades"] == ["G1"]
    assert blocked.status == "blur_no_grade"
    assert frame_event(2, blocked)["grades"] == []
    duplicate = gate_frame(sharp, (120, 160, "same"), "same", 80.0, 25.0)
    assert frame_event(3, duplicate)["grades"] == []
    resized = Frame(b"same", height=480, width=640)
    assert gate_frame(resized, (120, 160, "same"), "same", 80.0, 25.0) is None


def test_stale_or_mismatched_detector_result_never_grades():
    current = Frame(b"current", height=480, width=640)
    previous = Frame(b"previous", height=480, width=640)
    box = detection(0, [0, 0, 100, 100], 0.90)
    stale = assess_result(
        current,
        result(current, box, orig_img=previous),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    wrong_size = assess_result(
        current,
        result(current, box, orig_shape=(120, 160)),
        EXPECTED_CLASS_NAMES,
        policy(),
    )
    assert stale.status == "stale_result"
    assert wrong_size.status == "coordinate_mismatch"
    assert frame_event(1, stale)["grades"] == []
    assert frame_event(2, wrong_size)["grades"] == []


def test_mid_confidence_critical_survives_to_harvest(monkeypatch, tmp_path):
    frame = Frame(b"critical")
    assessment = assess_result(
        frame,
        result(
            frame,
            detection(0, [0, 0, 120, 100], 0.9),
            detection(3, [10, 10, 13, 13], 0.27),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
    )

    def imwrite(path, image):
        Path(path).write_bytes(image.tobytes())
        return True

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(imwrite=imwrite))
    path = write_telemetry(
        frame,
        assessment.detections,
        str(tmp_path),
        grading_results=assessment.grading_results,
        force_review=bool(assessment.review_reasons),
        review_reason=",".join(assessment.review_reasons),
        frame_status=assessment.status,
    )
    payload = json.loads(Path(path).read_text())
    assert payload["grading_results"][0]["grade"] == "DISCARD"
    assert payload["bounding_boxes"][1]["class_name"] == "defect_critical"
    assert payload["bounding_boxes"][1]["confidence"] == pytest.approx(0.27)
    assert payload["review_reason"] == "uncertain_critical"


def test_glove_suppression_cannot_turn_critical_candidate_into_g1():
    np = pytest.importorskip("numpy")
    frame = Frame(b"glove")
    glove = np.zeros((120, 160), dtype=np.uint8)
    glove[10:13, 10:13] = 255
    assessment = assess_result(
        frame,
        result(
            frame,
            detection(0, [0, 0, 100, 100], 0.9),
            detection(3, [10, 10, 13, 13], 0.5),
        ),
        EXPECTED_CLASS_NAMES,
        policy(),
        glove_mask=glove,
    )
    assert assessment.status == "suppressed_critical_no_grade"
    assert frame_event(1, assessment)["grades"] == []
    assert "critical_overlaps_glove" in assessment.review_reasons


def test_actual_blurred_pixels_get_lower_focus_score():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    checker = (np.indices((120, 160)).sum(axis=0) % 2 * 255).astype("uint8")
    sharp = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(sharp, (21, 21), 6)
    sharp_score = focus_score(sharp)
    blur_score = focus_score(blurred)
    assert sharp_score > blur_score
    assert (
        gate_frame(
            blurred,
            None,
            "blur",
            blur_score,
            (sharp_score + blur_score) / 2,
        ).status
        == "blur_no_grade"
    )


@pytest.mark.parametrize("no_display", [True, False])
@pytest.mark.parametrize("harvest_limit", [None, 3])
def test_camera_loop_emits_current_frame_status_across_blur_empty_and_rot(
    monkeypatch, capsys, no_display, harvest_limit, tmp_path
):
    sharp = Frame(b"sharp")
    blurry = Frame(b"blurry")
    empty = Frame(b"empty")
    rot = Frame(b"rot", height=480, width=640)
    orphan_a = Frame(b"orphan-a")
    orphan_b = Frame(b"orphan-b")

    class Camera:
        frames = [Frame(f"warmup-{i}".encode()) for i in range(5)] + [
            sharp,
            blurry,
            empty,
            rot,
            orphan_a,
            orphan_b,
            None,
        ]

        def isOpened(self):
            return True

        def set(self, *_args):
            pass

        def get(self, prop):
            return 160 if prop == 1 else 120

        def read(self):
            frame = self.frames.pop(0)
            return frame is not None, frame

        def release(self):
            pass

    camera = Camera()

    drawn = []

    def rectangle(_frame, top_left, bottom_right, *_args):
        assert all(isinstance(value, int) for value in (*top_left, *bottom_right))
        drawn.append((top_left, bottom_right))

    def imwrite(path, frame):
        Path(path).write_bytes(frame.tobytes())
        return True

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda _index: camera,
        VideoWriter_fourcc=lambda *_args: 0,
        CAP_PROP_FOURCC=0,
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
        FONT_HERSHEY_DUPLEX=3,
        FONT_HERSHEY_SIMPLEX=4,
        rectangle=rectangle,
        imwrite=imwrite,
        putText=lambda *_args: None,
        imshow=lambda *_args: None,
        waitKey=lambda _delay: -1,
        destroyAllWindows=lambda: None,
    )

    class Model:
        names = EXPECTED_CLASS_NAMES

        def __init__(self, *_args, **_kwargs):
            self.conf_values = []

        def __call__(self, frame, *, conf, **_kwargs):
            self.conf_values.append(conf)
            boxes = {
                b"sharp": [detection(0, [0, 0, 100, 100], 0.9)],
                b"empty": [],
                b"rot": [
                    detection(0, [100, 100, 500, 400], 0.9),
                    detection(3, [101, 101, 104, 104], 0.27),
                ],
                b"orphan-a": [
                    detection(0, [0, 0, 50, 50], 0.9),
                    detection(3, [100, 80, 105, 85], 0.27),
                ],
                b"orphan-b": [
                    detection(0, [0, 0, 50, 50], 0.9),
                    detection(3, [100, 80, 105, 85], 0.27),
                ],
            }
            return [result(frame, *boxes[frame.data])]

    model = Model()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *_a, **_k: model)
    )
    spec = importlib.util.spec_from_file_location(
        "_isolated_line_loop", Path("local_inference.py")
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(runtime, "detect_arducam_index", lambda **_kwargs: 0)
    monkeypatch.setattr(
        runtime, "capture_frame_hardened", lambda cap, **_k: (cap, cap.read()[1])
    )
    monkeypatch.setattr(
        runtime, "focus_score", lambda frame: 0.0 if frame is blurry else 100.0
    )
    monkeypatch.setattr(runtime, "mask_blue_gloves", lambda frame: (None, None))
    harvested = []

    def capture_telemetry(frame, detections, _harvest_dir, **kwargs):
        path = write_telemetry(frame, detections, str(tmp_path), **kwargs)
        harvested.append((kwargs, path))
        return path

    monkeypatch.setattr(runtime, "write_telemetry", capture_telemetry)
    args = ["local_inference.py", "--model", "grading_policy.yaml"]
    if no_display:
        args.append("--no-display")
    if harvest_limit is not None:
        args.extend(["--max-consecutive-harvest-frames", str(harvest_limit)])
    monkeypatch.setattr(sys, "argv", args)

    runtime.main()
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"event": "frame_assessment"')
    ]
    expected_statuses = [
        "graded",
        "blur_no_grade",
        "no_apple",
        "graded",
        "critical_orphan_no_grade",
        (
            "harvest_backlog_no_grade"
            if harvest_limit is not None
            else "critical_orphan_no_grade"
        ),
    ]
    if harvest_limit is None:
        expected_statuses.append("camera_failure")
    assert [event["status"] for event in events] == expected_statuses
    assert [event["grades"] for event in events] == [
        ["G1"],
        [],
        [],
        ["DISCARD"],
        [],
        [],
    ] + ([[]] if harvest_limit is None else [])
    assert events[2]["image"] == {"width": 160, "height": 120}
    assert events[3]["image"] == {"width": 640, "height": 480}
    assert model.conf_values == [0.20] * 5
    assert harvested[3][0]["force_review"]
    assert harvested[3][0]["review_reason"] == "uncertain_critical"
    assert all(record[0]["force_review"] for record in harvested[-2:])
    assert all(record[1] is not None for record in harvested[-2:])
    assert all(
        json.loads(Path(record[1]).read_text())["bounding_boxes"][1]["confidence"]
        == pytest.approx(0.27)
        for record in harvested[-2:]
    )
    assert bool(drawn) is not no_display


@pytest.mark.parametrize(
    ("failed_stage", "expected_status"),
    [
        ("focus", "preprocess_error_no_grade"),
        ("mask", "inference_error_no_grade"),
        ("harvest", "harvest_failure_no_grade"),
    ],
)
def test_line_failures_clear_grade(monkeypatch, capsys, failed_stage, expected_status):
    frame = Frame(b"apple")

    class Camera:
        frames = [Frame(f"warmup-{i}".encode()) for i in range(5)] + [frame, None]

        def isOpened(self):
            return True

        def set(self, *_args):
            pass

        def get(self, _prop):
            return 160

        def read(self):
            value = self.frames.pop(0)
            return value is not None, value

        def release(self):
            pass

    camera = Camera()
    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda _index: camera,
        VideoWriter_fourcc=lambda *_args: 0,
        CAP_PROP_FOURCC=0,
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
        destroyAllWindows=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    class Model:
        names = EXPECTED_CLASS_NAMES

        def __call__(self, image, **_kwargs):
            return [result(image, detection(0, [0, 0, 100, 100], 0.9))]

    monkeypatch.setitem(
        sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda *_a, **_k: Model())
    )
    spec = importlib.util.spec_from_file_location(
        "_isolated_failure_loop", Path("local_inference.py")
    )
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    monkeypatch.setattr(runtime, "detect_arducam_index", lambda **_kwargs: 0)
    monkeypatch.setattr(
        runtime, "capture_frame_hardened", lambda cap, **_k: (cap, cap.read()[1])
    )

    def focus(_frame):
        if failed_stage == "focus":
            raise RuntimeError("focus unavailable")
        return 100.0

    def mask(_frame):
        if failed_stage == "mask":
            raise RuntimeError("mask unavailable")
        return None, None

    def harvest(*_args, **_kwargs):
        if failed_stage == "harvest":
            raise OSError("disk full")
        return None

    monkeypatch.setattr(runtime, "focus_score", focus)
    monkeypatch.setattr(runtime, "mask_blue_gloves", mask)
    monkeypatch.setattr(runtime, "write_telemetry", harvest)
    monkeypatch.setattr(
        sys,
        "argv",
        ["local_inference.py", "--no-display", "--model", "grading_policy.yaml"],
    )
    runtime.main()
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('{"event": "frame_assessment"')
    ]
    assert events[0]["status"] == expected_status
    assert events[0]["grades"] == []
    assert events[0]["image"] == {"width": 160, "height": 120}
