import pytest

from grading_engine import (
    GradeDecision,
    GradingPolicy,
    aggregate_profile_grades,
    bind_defects_to_parents,
    calculate_ioa,
    grade_apple,
    model_names_match_expected_schema,
    union_area,
)


def policy(**overrides) -> GradingPolicy:
    values = {
        "facility_id": "test",
        "policy_version": "test-v3",
        "surface_ratio_g1_pct": 2.0,
        "surface_ratio_g2_pct": 10.0,
        "surface_ratio_g3_pct": 25.0,
        "ioa_binding_threshold": 0.10,
        "refinement_margin_pct": 2.0,
        "expected_profile_views": 5,
    }
    values.update(overrides)
    return GradingPolicy(**values)


def decision(grade: str, *, refine: bool = False) -> GradeDecision:
    return GradeDecision(grade, 0, 0, 0.0, "boxes", refine)


def test_union_area_does_not_double_count_overlapping_boxes() -> None:
    assert union_area([(0, 0, 10, 10), (5, 0, 15, 10)]) == 150


def test_union_area_clips_defects_to_parent() -> None:
    assert union_area([(-5, -5, 5, 5), (8, 8, 20, 20)], clip_to=(0, 0, 10, 10)) == 29


def test_calculate_ioa_uses_child_area() -> None:
    assert calculate_ioa((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(0.5)


def test_binding_chooses_parent_with_highest_ioa() -> None:
    parents = [(0, 0, 100, 100), (80, 0, 180, 100)]
    defects = [(85, 10, 95, 20), (150, 10, 160, 20), (300, 0, 310, 10)]
    assert bind_defects_to_parents(defects, parents, 0.10) == [[0], [1]]


def test_expected_four_class_schema_is_exact() -> None:
    assert model_names_match_expected_schema(
        {0: "apple", 1: "stem_calyx", 2: "defect_surface", 3: "defect_critical"}
    )
    assert not model_names_match_expected_schema(
        {0: "apple", 1: "defect_surface", 2: "stem_calyx", 3: "defect_critical"}
    )
    assert not model_names_match_expected_schema(
        {0: "apple", 1: "unfit_bin_discard", 2: "class_defect"}
    )


def test_no_defects_is_g1() -> None:
    result = grade_apple((0, 0, 100, 100), [], [], policy())
    assert result.grade == "G1"
    assert result.coverage_pct == 0
    assert result.critical_defect_count == 0


def test_surface_coverage_below_g1_threshold_is_g1() -> None:
    # 1% coverage — below the 2% G1 threshold
    result = grade_apple((0, 0, 100, 100), [(0, 0, 1, 100)], [], policy())
    assert result.grade == "G1"


def test_exact_g2_coverage_threshold_is_g2() -> None:
    # 2% coverage — at the G1/G2 boundary
    result = grade_apple((0, 0, 100, 100), [(0, 0, 2, 100)], [], policy())
    assert result.grade == "G2"
    assert result.coverage_pct == pytest.approx(2.0)
    assert result.requires_refinement


def test_exact_g3_coverage_threshold_is_g3() -> None:
    # 10% coverage — at the G2/G3 boundary
    result = grade_apple((0, 0, 100, 100), [(0, 0, 10, 100)], [], policy())
    assert result.grade == "G3"
    assert result.coverage_pct == pytest.approx(10.0)
    assert result.requires_refinement


def test_exact_cider_coverage_threshold_is_cider() -> None:
    # 25% coverage — at the G3/CIDER boundary
    result = grade_apple((0, 0, 100, 100), [(0, 0, 25, 100)], [], policy())
    assert result.grade == "CIDER"
    assert result.coverage_pct == pytest.approx(25.0)
    assert result.requires_refinement


def test_heavy_surface_coverage_above_cider_is_cider() -> None:
    # 40% surface coverage, no critical — still pressable
    result = grade_apple((0, 0, 100, 100), [(0, 0, 40, 100)], [], policy())
    assert result.grade == "CIDER"


def test_any_critical_defect_forces_discard() -> None:
    result = grade_apple(
        (0, 0, 100, 100),
        [],  # no surface defects
        [(10, 10, 20, 20)],  # one critical defect
        policy(),
    )
    assert result.grade == "DISCARD"
    assert result.critical_defect_count == 1
    assert not result.requires_refinement


def test_critical_defect_overrides_heavy_surface_coverage() -> None:
    # 40% surface + 1 critical -> DISCARD, not CIDER
    result = grade_apple(
        (0, 0, 100, 100),
        [(0, 0, 40, 100)],
        [(10, 10, 20, 20)],
        policy(),
    )
    assert result.grade == "DISCARD"


def test_segmentation_coverage_can_refine_bbox_grade() -> None:
    # Box says 10% (G3 boundary) but segmentation says 8% (G2)
    result = grade_apple(
        (0, 0, 100, 100),
        [(0, 0, 10, 100)],
        [],
        policy(),
        refined_coverage_pct=8.0,
    )
    assert result.grade == "G2"
    assert result.coverage_source == "segmentation"
    assert not result.requires_refinement


def test_profile_grade_uses_worst_complete_view() -> None:
    result = aggregate_profile_grades(
        [
            decision("G1"),
            decision("G1"),
            decision("G2"),
            decision("G1"),
            decision("G1"),
        ],
        expected_views=5,
    )
    assert result.grade == "G2"
    assert result.is_complete
    assert not result.requires_review


def test_profile_grade_worst_is_discard() -> None:
    result = aggregate_profile_grades(
        [decision("G1"), decision("G2"), decision("DISCARD"), decision("G3"), decision("G1")],
        expected_views=5,
    )
    assert result.grade == "DISCARD"


def test_profile_grade_cider_is_worse_than_g3() -> None:
    result = aggregate_profile_grades(
        [decision("G3"), decision("G3"), decision("CIDER"), decision("G3"), decision("G3")],
        expected_views=5,
    )
    assert result.grade == "CIDER"


def test_incomplete_profile_requires_review() -> None:
    result = aggregate_profile_grades([decision("G1")], expected_views=5)
    assert not result.is_complete
    assert result.requires_review


def test_refinement_flag_propagates_to_profile() -> None:
    result = aggregate_profile_grades(
        [decision("G1", refine=True)] * 5,
        expected_views=5,
    )
    assert result.requires_review
