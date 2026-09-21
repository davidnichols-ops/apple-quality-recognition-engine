from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

Box = tuple[float, float, float, float]

# Five-bin physical sorting topology aligned to USDA processing standards:
#   G1      — U.S. Extra Fancy (pristine dessert fruit, strict surface limits)
#   G2      — U.S. Fancy / No. 1 (minor blemishes within tolerance)
#   G3      — U.S. Utility / Processing (heavier cosmetic defects, intact skin)
#   CIDER   — Sound fruit failing fresh-market tolerances (heavy hail, russet,
#             scab) but free from internal rot or decay — still pressable.
#   DISCARD — Total cull (open fungal decay, active soft rot, deep punctures,
#             insect boring).
VALID_GRADES = ("G1", "G2", "G3", "CIDER", "DISCARD")

# Four-class vision taxonomy. The neural network observes physical features
# only; it has zero concept of commercial grades. The deterministic policy
# engine maps these observations to the five output grades.
#
#   0 apple            — parent mask (caliper size & area denominator)
#   1 stem_calyx       — anatomical exclusion zones (prevents mistaking
#                        navels/stems for rot)
#   2 defect_surface   — non-decay cosmetic marks (russeting, limb rubs,
#                        healed hail)
#   3 defect_critical  — structural & fungal damage (active rot, wet lesions,
#                        open punctures)
EXPECTED_CLASS_NAMES = {
    0: "apple",
    1: "stem_calyx",
    2: "defect_surface",
    3: "defect_critical",
}
_GRADE_RANK = {grade: rank for rank, grade in enumerate(VALID_GRADES)}


@dataclass(frozen=True)
class GradingPolicy:
    """Versioned facility calibration for the deterministic grade engine.

    The policy maps observed defect coverage ratios to five output grades.
    Thresholds are candidate calibration values, not claims of compliance with
    any USDA standard. Promote changes only after held-out profile evaluation
    and human approval.

    Decision tree (per view):

        any defect_critical bound to apple  ->  DISCARD
        else R_surf = Area(defect_surface union) / Area(apple) * 100
            R_surf < surface_ratio_g1_pct   ->  G1
            R_surf < surface_ratio_g2_pct   ->  G2
            R_surf < surface_ratio_g3_pct   ->  G3
            R_surf >= surface_ratio_g3_pct  ->  CIDER  (sound flesh, no critical)
    """

    facility_id: str
    policy_version: str
    surface_ratio_g1_pct: float
    surface_ratio_g2_pct: float
    surface_ratio_g3_pct: float
    ioa_binding_threshold: float
    refinement_margin_pct: float
    expected_profile_views: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GradingPolicy":
        rules = payload["rules"]
        policy = cls(
            facility_id=str(payload["facility_id"]),
            policy_version=str(payload["policy_version"]),
            surface_ratio_g1_pct=float(rules["surface_ratio_g1_pct"]),
            surface_ratio_g2_pct=float(rules["surface_ratio_g2_pct"]),
            surface_ratio_g3_pct=float(rules["surface_ratio_g3_pct"]),
            ioa_binding_threshold=float(rules["ioa_binding_threshold"]),
            refinement_margin_pct=float(rules["refinement_margin_pct"]),
            expected_profile_views=int(rules["expected_profile_views"]),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not 0 < self.surface_ratio_g1_pct < self.surface_ratio_g2_pct:
            raise ValueError("surface ratios must satisfy 0 < G1 < G2")
        if not self.surface_ratio_g2_pct < self.surface_ratio_g3_pct <= 100:
            raise ValueError("surface ratios must satisfy G2 < G3 <= 100")
        if not 0 <= self.ioa_binding_threshold <= 1:
            raise ValueError("ioa_binding_threshold must be between 0 and 1")
        if self.refinement_margin_pct < 0:
            raise ValueError("refinement_margin_pct must be non-negative")
        if self.expected_profile_views < 1:
            raise ValueError("expected_profile_views must be positive")


@dataclass(frozen=True)
class GradeDecision:
    grade: str
    surface_defect_count: int
    critical_defect_count: int
    coverage_pct: float
    coverage_source: str
    requires_refinement: bool


@dataclass(frozen=True)
class ProfileGradeDecision:
    grade: str
    observed_views: int
    expected_views: int
    is_complete: bool
    requires_review: bool


def load_grading_policy(policy_path: str | Path) -> GradingPolicy:
    path = Path(policy_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Grading policy must be a mapping: {path}")
    return GradingPolicy.from_mapping(payload)


def model_names_match_expected_schema(
    model_names: Mapping[int, str] | Sequence[str],
) -> bool:
    if isinstance(model_names, Mapping):
        normalized = {int(index): name for index, name in model_names.items()}
    else:
        normalized = dict(enumerate(model_names))
    return normalized == EXPECTED_CLASS_NAMES


def normalize_box(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("boxes must contain exactly four coordinates")
    x1, y1, x2, y2 = (float(value) for value in box)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = normalize_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_box(first: Sequence[float], second: Sequence[float]) -> Box | None:
    ax1, ay1, ax2, ay2 = normalize_box(first)
    bx1, by1, bx2, by2 = normalize_box(second)
    intersection = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
        return None
    return intersection


def calculate_ioa(child_box: Sequence[float], parent_box: Sequence[float]) -> float:
    child_area = box_area(child_box)
    if child_area == 0:
        return 0.0
    intersection = intersection_box(child_box, parent_box)
    return box_area(intersection) / child_area if intersection else 0.0


def union_area(
    boxes: Sequence[Sequence[float]], clip_to: Sequence[float] | None = None
) -> float:
    normalized: list[Box] = []
    for box in boxes:
        candidate = normalize_box(box)
        if clip_to is not None:
            clipped = intersection_box(candidate, clip_to)
            if clipped is None:
                continue
            candidate = clipped
        if box_area(candidate) > 0:
            normalized.append(candidate)
    if not normalized:
        return 0.0

    x_edges = sorted({edge for box in normalized for edge in (box[0], box[2])})
    total = 0.0
    for left, right in zip(x_edges, x_edges[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (box[1], box[3]) for box in normalized if box[0] < right and box[2] > left
        )
        if not intervals:
            continue
        covered_y = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                covered_y += end - start
                start, end = next_start, next_end
        covered_y += end - start
        total += (right - left) * covered_y
    return total


def bind_defects_to_parents(
    defect_boxes: Sequence[Sequence[float]],
    parent_boxes: Sequence[Sequence[float]],
    ioa_threshold: float,
) -> list[list[int]]:
    bindings: list[list[int]] = [[] for _ in parent_boxes]
    for defect_index, defect_box in enumerate(defect_boxes):
        scores = [calculate_ioa(defect_box, parent_box) for parent_box in parent_boxes]
        if not scores:
            continue
        best_parent_index = max(range(len(scores)), key=scores.__getitem__)
        if scores[best_parent_index] > 0 and scores[best_parent_index] >= ioa_threshold:
            bindings[best_parent_index].append(defect_index)
    return bindings


def grade_apple(
    parent_box: Sequence[float],
    defect_surface_boxes: Sequence[Sequence[float]],
    defect_critical_boxes: Sequence[Sequence[float]],
    policy: GradingPolicy,
    *,
    refined_coverage_pct: float | None = None,
) -> GradeDecision:
    """Compute the deterministic grade for one apple view.

    The decision tree is:

        any defect_critical bound to apple  ->  DISCARD
        else R_surf = Area(defect_surface union) / Area(apple) * 100
            R_surf < surface_ratio_g1_pct   ->  G1
            R_surf < surface_ratio_g2_pct   ->  G2
            R_surf < surface_ratio_g3_pct   ->  G3
            R_surf >= surface_ratio_g3_pct  ->  CIDER

    ``stem_calyx`` detections are excluded from defect counting — they are
    anatomical structures, not defects. The caller filters them out before
    passing surface/critical boxes.

    Args:
        parent_box: The apple parent bounding box.
        defect_surface_boxes: Non-decay cosmetic defect boxes (russeting, limb
            rub, healed hail).
        defect_critical_boxes: Structural/fungal defect boxes (active rot, wet
            lesions, open punctures). Any bound critical defect forces DISCARD.
        policy: The versioned facility grading policy.
        refined_coverage_pct: Optional externally measured segmentation
            coverage. When supplied it replaces the box-union coverage.

    Returns:
        The deterministic GradeDecision for this view.
    """
    valid_surface = [
        box
        for box in defect_surface_boxes
        if intersection_box(box, parent_box) is not None
    ]
    valid_critical = [
        box
        for box in defect_critical_boxes
        if intersection_box(box, parent_box) is not None
    ]
    parent_area = box_area(parent_box)
    box_coverage_pct = (
        union_area(valid_surface, clip_to=parent_box) / parent_area * 100
        if parent_area > 0
        else 0.0
    )
    coverage_pct = (
        float(refined_coverage_pct)
        if refined_coverage_pct is not None
        else box_coverage_pct
    )
    if not 0 <= coverage_pct <= 100:
        raise ValueError("coverage must be between 0 and 100 percent")

    critical_count = len(valid_critical)
    surface_count = len(valid_surface)

    if critical_count > 0:
        grade = "DISCARD"
    elif coverage_pct >= policy.surface_ratio_g3_pct:
        grade = "CIDER"
    elif coverage_pct >= policy.surface_ratio_g2_pct:
        grade = "G3"
    elif coverage_pct >= policy.surface_ratio_g1_pct:
        grade = "G2"
    else:
        grade = "G1"

    distances = (
        abs(box_coverage_pct - policy.surface_ratio_g1_pct),
        abs(box_coverage_pct - policy.surface_ratio_g2_pct),
        abs(box_coverage_pct - policy.surface_ratio_g3_pct),
    )
    requires_refinement = (
        critical_count == 0
        and refined_coverage_pct is None
        and surface_count > 0
        and min(distances) <= policy.refinement_margin_pct
    )
    return GradeDecision(
        grade=grade,
        surface_defect_count=surface_count,
        critical_defect_count=critical_count,
        coverage_pct=coverage_pct,
        coverage_source="segmentation" if refined_coverage_pct is not None else "boxes",
        requires_refinement=requires_refinement,
    )


def aggregate_profile_grades(
    decisions: Sequence[GradeDecision],
    expected_views: int,
) -> ProfileGradeDecision:
    if not decisions:
        raise ValueError("at least one view decision is required")
    if expected_views < 1:
        raise ValueError("expected_views must be positive")
    grade = max((decision.grade for decision in decisions), key=_GRADE_RANK.__getitem__)
    observed_views = len(decisions)
    is_complete = observed_views >= expected_views
    return ProfileGradeDecision(
        grade=grade,
        observed_views=observed_views,
        expected_views=expected_views,
        is_complete=is_complete,
        requires_review=(
            not is_complete
            or any(decision.requires_refinement for decision in decisions)
        ),
    )
