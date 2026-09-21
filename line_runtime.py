"""Per-frame grading boundary for the camera line.

No state from a previous image is accepted here.  A frame without trustworthy
current detections has no grade; downstream consumers must honor ``status``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from grading_engine import (
    GradingPolicy,
    bind_defects_to_parents,
    calculate_ioa,
    grade_apple,
)
from glove_masking import suppress_glove_defects

# Candidate operating cutoffs, not validated model-performance claims.  The
# lower detector floor keeps uncertain critical candidates visible for review.
DETECTION_CONFIDENCE_FLOOR = 0.20
STANDARD_CONFIDENCE_FLOOR = 0.35
CRITICAL_REVIEW_CONFIDENCE = 0.65


@dataclass
class FrameAssessment:
    status: str
    image_height: int
    image_width: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    parents: list[dict[str, Any]] = field(default_factory=list)
    stem_calyx: list[dict[str, Any]] = field(default_factory=list)
    grading_results: list[dict[str, Any]] = field(default_factory=list)
    review_reasons: list[str] = field(default_factory=list)

    @property
    def has_grade(self) -> bool:
        return self.status == "graded" and bool(self.grading_results)


def gate_frame(
    frame: Any,
    previous_frame_signature: tuple[int, int, str] | None,
    frame_hash: str,
    focus_score: float,
    min_sharpness: float,
) -> FrameAssessment | None:
    """Return a no-grade assessment for frozen or blurry camera frames."""
    height, width = frame.shape[:2]
    if (height, width, frame_hash) == previous_frame_signature:
        return FrameAssessment("duplicate_frame", height, width)
    if not math.isfinite(focus_score) or focus_score < min_sharpness:
        return FrameAssessment("blur_no_grade", height, width)
    return None


def focus_score(frame: Any) -> float:
    """Variance of Laplacian on the actual camera image, before annotation."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_event(frame_id: int, assessment: FrameAssessment) -> dict[str, Any]:
    """An explicit current-frame event; ungradable frames clear output grades."""
    return {
        "event": "frame_assessment",
        "frame_id": frame_id,
        "status": assessment.status,
        "image": {
            "width": assessment.image_width,
            "height": assessment.image_height,
        },
        "grades": (
            [g["grade"] for g in assessment.grading_results]
            if assessment.has_grade
            else []
        ),
        "review_reasons": assessment.review_reasons,
    }


def _image_matches(frame: Any, result: Any) -> bool:
    """Guard against stale model output, including same-sized prior frames."""
    source = getattr(result, "orig_img", None)
    return (
        source is not None
        and getattr(source, "shape", None) == frame.shape
        and (source is frame or source.tobytes() == frame.tobytes())
    )


def assess_result(
    frame: Any,
    result: Any,
    model_names: Mapping[int, str],
    policy: GradingPolicy,
    *,
    glove_mask: Any = None,
    benchmark_mode: bool = False,
) -> FrameAssessment:
    """Grade one detector result only in the current frame's pixel space."""
    height, width = frame.shape[:2]
    assessment = FrameAssessment("no_grade", height, width)
    source_shape = getattr(result, "orig_shape", None)
    if source_shape is None or tuple(source_shape) != (height, width):
        assessment.status = "coordinate_mismatch"
        assessment.review_reasons.append("coordinate_mismatch")
        return assessment
    if not _image_matches(frame, result):
        assessment.status = "stale_result"
        assessment.review_reasons.append("stale_result")
        return assessment

    parents: list[dict[str, Any]] = []
    surface: list[dict[str, Any]] = []
    critical: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id not in model_names:
            assessment.status = "invalid_detection"
            assessment.review_reasons.append("unknown_class")
            return assessment
        coords = [float(value) for value in box.xyxy[0]]
        if (
            len(coords) != 4
            or not all(math.isfinite(value) for value in coords)
            or not (0 <= coords[0] < coords[2] <= width)
            or not (0 <= coords[1] < coords[3] <= height)
        ):
            assessment.status = "invalid_detection"
            assessment.review_reasons.append("invalid_coordinates")
            return assessment
        confidence = float(box.conf[0])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            assessment.status = "invalid_detection"
            assessment.review_reasons.append("invalid_confidence")
            return assessment
        detection = {
            "id": class_id,
            "name": model_names[class_id],
            "box": coords,
            "conf": confidence,
        }
        assessment.detections.append(detection)
        if benchmark_mode:
            continue
        if detection["name"] == "apple":
            parents.append({**detection, "defects": [], "criticals": []})
        elif detection["name"] == "stem_calyx":
            assessment.stem_calyx.append(detection)
        elif detection["name"] == "defect_surface":
            surface.append(detection)
        elif detection["name"] == "defect_critical":
            critical.append(detection)

    if benchmark_mode:
        assessment.status = "benchmark_no_grade"
        return assessment

    if any(
        d["name"] in {"apple", "defect_surface"}
        and d["conf"] < STANDARD_CONFIDENCE_FLOOR
        for d in assessment.detections
    ):
        assessment.status = "uncertain_detection_no_grade"
        assessment.review_reasons.append("low_confidence_noncritical")
        return assessment

    if glove_mask is not None:
        if glove_mask.shape[:2] != (height, width):
            assessment.status = "coordinate_mismatch"
            assessment.review_reasons.append("glove_mask_shape_mismatch")
            return assessment
        kept_surface = suppress_glove_defects([d["box"] for d in surface], glove_mask)
        kept_critical = suppress_glove_defects([d["box"] for d in critical], glove_mask)
        surface = [d for d in surface if d["box"] in kept_surface]
        if len(kept_critical) < len(critical):
            assessment.status = "suppressed_critical_no_grade"
            assessment.review_reasons.append("critical_overlaps_glove")
            return assessment
        critical = [d for d in critical if d["box"] in kept_critical]

    if any(d["conf"] < CRITICAL_REVIEW_CONFIDENCE for d in critical):
        assessment.review_reasons.append("uncertain_critical")

    parent_coords = [d["box"] for d in parents]
    if any(
        sum(
            score > 0 and score >= policy.ioa_binding_threshold
            for score in (
                calculate_ioa(candidate["box"], parent_box)
                for parent_box in parent_coords
            )
        )
        > 1
        for candidate in critical
    ):
        assessment.status = "ambiguous_critical_no_grade"
        assessment.review_reasons.append("ambiguous_critical_parent")
        return assessment
    surface_bindings = bind_defects_to_parents(
        [d["box"] for d in surface], parent_coords, policy.ioa_binding_threshold
    )
    critical_bindings = bind_defects_to_parents(
        [d["box"] for d in critical], parent_coords, policy.ioa_binding_threshold
    )
    for index, bound in enumerate(surface_bindings):
        parents[index]["defects"] = [surface[child] for child in bound]
    for index, bound in enumerate(critical_bindings):
        parents[index]["criticals"] = [critical[child] for child in bound]

    bound_surface = {child for group in surface_bindings for child in group}
    bound_critical = {child for group in critical_bindings for child in group}
    if len(bound_surface) < len(surface) or len(bound_critical) < len(critical):
        assessment.review_reasons.append("orphan_defect")
    if len(bound_critical) < len(critical):
        # Never emit a G1 while an observed critical candidate is unassigned.
        assessment.status = "critical_orphan_no_grade"
        return assessment
    if not parents:
        assessment.status = "no_apple"
        return assessment

    for parent in parents:
        decision = grade_apple(
            parent["box"],
            [d["box"] for d in parent["defects"]],
            [d["box"] for d in parent["criticals"]],
            policy,
        )
        parent["decision"] = decision
        parent["grade"] = decision.grade
        assessment.grading_results.append(
            {
                **asdict(decision),
                "defects": parent["defects"],
                "criticals": parent["criticals"],
            }
        )
    if any(result["requires_refinement"] for result in assessment.grading_results):
        assessment.review_reasons.append("coverage_near_grade_boundary")
    assessment.parents = parents
    assessment.status = "graded"
    return assessment
