"""Deterministic plant health grading engine.

3-tier hierarchy:  plant (parent) → leaf (child) → class_defect (grandchild)

The engine is pure: no model inference, no I/O, no side effects.
It takes detection boxes and a policy dict, returns structured grades.

Grading walks bottom-up:
  1. Bind each ``class_defect`` to the leaf with highest IoA above threshold.
  2. Clip defect boxes to the parent leaf; compute clipped union coverage.
  3. Grade each leaf (HEALTHY / MODERATE / POOR / DISCARD) from defect count
     and coverage area.
  4. Bind each leaf to the plant with highest IoA above threshold.
  5. Grade the plant from the aggregate of leaf grades.

``unfit_discard`` applies to the nearby plant, not globally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

# ── Schema constants ────────────────────────────────────────────────

PLANT_CLASS_ID = 0
LEAF_CLASS_ID = 1
UNFIT_DISCARD_CLASS_ID = 2
CLASS_DEFECT_ID = 3

EXPECTED_CLASS_NAMES = ("plant", "leaf", "unfit_discard", "class_defect")

VALID_GRADES = ("HEALTHY", "MODERATE", "POOR", "DISCARD")

_GRADE_RANK = {"HEALTHY": 0, "MODERATE": 1, "POOR": 2, "DISCARD": 3}
_RANK_GRADE = {v: k for k, v in _GRADE_RANK.items()}


# ── Data structures ─────────────────────────────────────────────────

@dataclass(frozen=True)
class Detection:
    """One YOLO detection box."""
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class LeafGrade:
    grade: str
    defect_count: int
    coverage_pct: float
    needs_refinement: bool
    bound_defects: list[Detection] = field(default_factory=list)


@dataclass
class PlantGrade:
    grade: str
    leaf_count: int
    healthy_leaves: int
    moderate_leaves: int
    poor_leaves: int
    discarded_leaves: int
    needs_refinement: bool
    leaf_grades: list[LeafGrade] = field(default_factory=list)
    discard_triggered: bool = False


# ── Geometry helpers ────────────────────────────────────────────────

def _intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _ioa(child_box: tuple[float, float, float, float],
         parent_box: tuple[float, float, float, float]) -> float:
    """Intersection-over-Area: intersection / child_area."""
    intersection = _intersection_area(child_box, parent_box)
    cx1, cy1, cx2, cy2 = child_box
    child_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    if child_area == 0:
        return 0.0
    return intersection / child_area


def _clip_box(
    box: tuple[float, float, float, float],
    parent: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = parent
    return (max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2))


def _union_area(
    boxes: Sequence[tuple[float, float, float, float]],
) -> float:
    """Approximate union area of overlapping boxes via clipping grid.

    For small numbers of boxes this uses a simple pairwise inclusion-exclusion
    approximation.  For production use a sweep-line algorithm; the current
    implementation is sufficient for grading decisions where exact coverage
    is not required — the policy exposes a refinement margin for boundary
    cases.
    """
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        x1, y1, x2, y2 = boxes[0]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    # Grid-based union for accuracy with overlapping boxes.
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    ys = sorted({y for b in boxes for y in (b[1], b[3])})
    total = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx1, cx2 = xs[i], xs[i + 1]
            cy1, cy2 = ys[j], ys[j + 1]
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            cell = (cx1, cy1, cx2, cy2)
            for b in boxes:
                if _intersection_area(cell, b) > 0:
                    total += (cx2 - cx1) * (cy2 - cy1)
                    break
    return total


# ── Binding ─────────────────────────────────────────────────────────

def bind_defects_to_leaves(
    defects: Sequence[Detection],
    leaves: Sequence[Detection],
    ioa_threshold: float,
) -> dict[int, list[Detection]]:
    """Bind each defect to the leaf with the highest IoA above threshold.

    Returns a mapping  leaf_index → list[Detection].
    Unbound defects (no leaf above threshold) are silently dropped —
    they are noise or annotation errors outside any leaf.
    """
    binding: dict[int, list[Detection]] = {i: [] for i in range(len(leaves))}
    for defect in defects:
        best_leaf = -1
        best_ioa = ioa_threshold
        for i, leaf in enumerate(leaves):
            ioa = _ioa(defect.box, leaf.box)
            if ioa > best_ioa:
                best_ioa = ioa
                best_leaf = i
        if best_leaf >= 0:
            binding[best_leaf].append(defect)
    return binding


def bind_leaves_to_plants(
    leaves: Sequence[Detection],
    plants: Sequence[Detection],
    ioa_threshold: float,
) -> dict[int, list[int]]:
    """Bind each leaf to the plant with the highest IoA above threshold.

    Returns a mapping  plant_index → list[leaf_index].
    """
    binding: dict[int, list[int]] = {i: [] for i in range(len(plants))}
    for leaf_idx, leaf in enumerate(leaves):
        best_plant = -1
        best_ioa = ioa_threshold
        for i, plant in enumerate(plants):
            ioa = _ioa(leaf.box, plant.box)
            if ioa > best_ioa:
                best_ioa = ioa
                best_plant = i
        if best_plant >= 0:
            binding[best_plant].append(leaf_idx)
    return binding


def discard_parent_indexes(
    discard_boxes: Sequence[Detection],
    plants: Sequence[Detection],
    proximity_px: float,
) -> set[int]:
    """Return the set of plant indexes that have an unfit_discard box nearby."""
    discarded: set[int] = set()
    for discard in discard_boxes:
        dx1, dy1, dx2, dy2 = discard.box
        for i, plant in enumerate(plants):
            px1, py1, px2, py2 = plant.box
            # Check if discard box is within proximity_px of the plant box
            if (dx2 + proximity_px >= px1 and dx1 - proximity_px <= px2
                    and dy2 + proximity_px >= py1 and dy1 - proximity_px <= py2):
                discarded.add(i)
    return discarded


# ── Grading ─────────────────────────────────────────────────────────

def grade_leaf(
    leaf: Detection,
    bound_defects: Sequence[Detection],
    rules: dict,
) -> LeafGrade:
    """Grade a single leaf from its bound defects.

    Uses defect count and clipped union coverage area relative to leaf area.
    """
    if not bound_defects:
        return LeafGrade(
            grade="HEALTHY",
            defect_count=0,
            coverage_pct=0.0,
            needs_refinement=False,
            bound_defects=[],
        )

    # Clip defect boxes to the leaf and compute union coverage.
    clipped = [_clip_box(d.box, leaf.box) for d in bound_defects]
    union = _union_area(clipped)
    leaf_area = leaf.area
    coverage_pct = (union / leaf_area * 100.0) if leaf_area > 0 else 0.0

    defect_count = len(bound_defects)
    max_healthy = rules.get("max_defects_for_healthy_leaf", 2)
    max_moderate = rules.get("max_defects_for_moderate_leaf", 5)
    mod_area = rules.get("leaf_area_threshold_moderate_pct", 5.0)
    poor_area = rules.get("leaf_area_threshold_poor_pct", 15.0)
    refinement_margin = rules.get("refinement_margin_pct", 2.0)

    # Determine grade
    if defect_count > max_moderate or coverage_pct > poor_area:
        grade = "POOR"
    elif defect_count > max_healthy or coverage_pct > mod_area:
        grade = "MODERATE"
    else:
        grade = "HEALTHY"

    # Refinement flag: coverage near a boundary
    needs_refinement = (
        abs(coverage_pct - mod_area) < refinement_margin
        or abs(coverage_pct - poor_area) < refinement_margin
    )

    return LeafGrade(
        grade=grade,
        defect_count=defect_count,
        coverage_pct=round(coverage_pct, 2),
        needs_refinement=needs_refinement,
        bound_defects=list(bound_defects),
    )


def grade_plant(
    plant: Detection,
    leaf_grades: Sequence[LeafGrade],
    rules: dict,
    discard_triggered: bool = False,
) -> PlantGrade:
    """Grade a plant from the aggregate of its leaf grades."""
    if discard_triggered:
        return PlantGrade(
            grade="DISCARD",
            leaf_count=len(leaf_grades),
            healthy_leaves=sum(1 for lg in leaf_grades if lg.grade == "HEALTHY"),
            moderate_leaves=sum(1 for lg in leaf_grades if lg.grade == "MODERATE"),
            poor_leaves=sum(1 for lg in leaf_grades if lg.grade == "POOR"),
            discarded_leaves=len(leaf_grades),
            needs_refinement=False,
            leaf_grades=list(leaf_grades),
            discard_triggered=True,
        )

    if not leaf_grades:
        # No leaves detected — can't grade meaningfully
        return PlantGrade(
            grade="DISCARD",
            leaf_count=0,
            healthy_leaves=0,
            moderate_leaves=0,
            poor_leaves=0,
            discarded_leaves=0,
            needs_refinement=True,
            leaf_grades=[],
            discard_triggered=False,
        )

    total = len(leaf_grades)
    healthy = sum(1 for lg in leaf_grades if lg.grade == "HEALTHY")
    moderate = sum(1 for lg in leaf_grades if lg.grade == "MODERATE")
    poor = sum(1 for lg in leaf_grades if lg.grade == "POOR")
    discarded = sum(1 for lg in leaf_grades if lg.grade == "DISCARD")

    # Worst leaf grade
    worst_rank = max(_GRADE_RANK[lg.grade] for lg in leaf_grades)
    worst_grade = _RANK_GRADE[worst_rank]

    # Percentage of POOR leaves (not MODERATE — MODERATE alone shouldn't
    # escalate the plant to POOR just because it's the only leaf).
    poor_pct = (poor / total * 100.0) if total > 0 else 0.0

    max_poor_for_moderate = rules.get("max_poor_leaves_for_moderate_plant", 1)
    poor_threshold_poor = rules.get("unhealthy_leaf_pct_threshold_poor", 40.0)
    refinement_margin = rules.get("refinement_margin_pct", 2.0)

    # Plant grade logic
    if worst_grade == "DISCARD":
        grade = "DISCARD"
    elif poor_pct >= poor_threshold_poor:
        grade = "POOR"
    elif poor > max_poor_for_moderate:
        grade = "POOR"
    elif moderate > 0 or poor > 0:
        grade = "MODERATE"
    else:
        grade = "HEALTHY"

    needs_refinement = (
        abs(poor_pct - poor_threshold_poor) < refinement_margin
        or any(lg.needs_refinement for lg in leaf_grades)
    )

    return PlantGrade(
        grade=grade,
        leaf_count=total,
        healthy_leaves=healthy,
        moderate_leaves=moderate,
        poor_leaves=poor,
        discarded_leaves=discarded,
        needs_refinement=needs_refinement,
        leaf_grades=list(leaf_grades),
        discard_triggered=False,
    )


# ── Top-level entry ─────────────────────────────────────────────────

def grade_detections(
    detections: Sequence[Detection],
    rules: dict,
) -> list[PlantGrade]:
    """Grade all plants in a single view's detections.

    This is the main entry point for the inference loop.
    """
    plants = [d for d in detections if d.class_id == PLANT_CLASS_ID]
    leaves = [d for d in detections if d.class_id == LEAF_CLASS_ID]
    defects = [d for d in detections if d.class_id == CLASS_DEFECT_ID]
    discards = [d for d in detections if d.class_id == UNFIT_DISCARD_CLASS_ID]

    if not plants:
        return []

    defect_ioa = rules.get("defect_to_leaf_ioa_threshold", 0.10)
    leaf_ioa = rules.get("leaf_to_plant_ioa_threshold", 0.10)
    proximity = rules.get("discard_proximity_px", 50)

    # Bind defects → leaves
    defect_binding = bind_defects_to_leaves(defects, leaves, defect_ioa)

    # Grade each leaf
    leaf_grades_all = [
        grade_leaf(leaves[i], defect_binding[i], rules)
        for i in range(len(leaves))
    ]

    # Bind leaves → plants
    leaf_binding = bind_leaves_to_plants(leaves, plants, leaf_ioa)

    # Discard proximity
    discarded_plants = discard_parent_indexes(discards, plants, proximity)

    # Grade each plant
    results = []
    for p_idx, plant in enumerate(plants):
        bound_leaf_grades = [leaf_grades_all[li] for li in leaf_binding[p_idx]]
        is_discarded = p_idx in discarded_plants
        results.append(grade_plant(plant, bound_leaf_grades, rules, is_discarded))

    return results


# ── Multi-view aggregation ──────────────────────────────────────────

def aggregate_profile_grades(
    view_grades: Sequence[str],
    expected_views: int,
) -> PlantGrade:
    """Aggregate grades across multiple views of the same plant.

    Uses worst visible grade across complete views.
    """
    if len(view_grades) != expected_views:
        raise ValueError(
            f"profile must contain {expected_views} views, got {len(view_grades)}"
        )

    worst_rank = max(_GRADE_RANK[g] for g in view_grades)
    worst_grade = _RANK_GRADE[worst_rank]

    return PlantGrade(
        grade=worst_grade,
        leaf_count=0,
        healthy_leaves=0,
        moderate_leaves=0,
        poor_leaves=0,
        discarded_leaves=0,
        needs_refinement=False,
        leaf_grades=[],
        discard_triggered=any(g == "DISCARD" for g in view_grades),
    )


# ── Policy loading ──────────────────────────────────────────────────

def load_grading_policy(policy_path: str | Path = "grading_policy.yaml") -> dict:
    """Load grading policy from YAML.  Returns the ``rules`` dict."""
    with open(policy_path, "r", encoding="utf-8") as fh:
        policy = yaml.safe_load(fh)
    return policy["rules"]


def model_names_match_expected_schema(names: Sequence[str]) -> bool:
    """Check that a trained model's class names match the expected 4-class schema."""
    if len(names) < len(EXPECTED_CLASS_NAMES):
        return False
    for i, expected in enumerate(EXPECTED_CLASS_NAMES):
        if names[i] != expected:
            return False
    return True
