import pytest

from grading_engine import (
    GradeDecision,
    GradingPolicy,
    aggregate_profile_grades,
    bind_defects_to_parents,
    calculate_ioa,
    discard_parent_indexes,
    grade_apple,
    model_names_match_expected_schema,
    union_area,
)


def policy(**overrides) -> GradingPolicy:
    values = {
        "facility_id": "test",
        "policy_version": "test-v1",
        "max_defects_for_g1": 2,
        "max_defects_for_g2": 5,
        "area_threshold_g2_pct": 5.0,
        "area_threshold_g3_pct": 15.0,
        "ioa_binding_threshold": 0.10,
        "discard_proximity_px": 50.0,
        "refinement_margin_pct": 2.0,
        "expected_profile_views": 5,
    }
    values.update(overrides)
    return GradingPolicy(**values)


def decision(grade: str, *, refine: bool = False) -> GradeDecision:
    return GradeDecision(grade, 0, 0.0, "boxes", refine)


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


def test_discard_trigger_only_marks_nearby_parent() -> None:
    parents = [(0, 0, 100, 100), (500, 0, 600, 100)]
    assert discard_parent_indexes([(20, 20, 30, 30)], parents, 50) == {0}


def test_expected_three_class_schema_is_exact() -> None:
    assert model_names_match_expected_schema(
        {0: "apple", 1: "unfit_bin_discard", 2: "class_defect"}
    )
    assert not model_names_match_expected_schema(
        {0: "apple", 1: "class_defect", 2: "unfit_bin_discard"}
    )


def test_no_defects_is_g1() -> None:
    result = grade_apple((0, 0, 100, 100), [], policy())
    assert result.grade == "G1"
    assert result.coverage_pct == 0


def test_exact_g2_coverage_threshold_is_g2() -> None:
    result = grade_apple((0, 0, 100, 100), [(0, 0, 5, 100)], policy())
    assert result.grade == "G2"
    assert result.coverage_pct == pytest.approx(5.0)
    assert result.requires_refinement


def test_exact_g3_coverage_threshold_is_g3() -> None:
    result = grade_apple((0, 0, 100, 100), [(0, 0, 15, 100)], policy())
    assert result.grade == "G3"
    assert result.coverage_pct == pytest.approx(15.0)
    assert result.requires_refinement


def test_defect_count_can_raise_grade_without_large_coverage() -> None:
    defects = [(index * 2, 0, index * 2 + 1, 1) for index in range(3)]
    assert grade_apple((0, 0, 100, 100), defects, policy()).grade == "G2"


def test_large_defect_count_is_g3() -> None:
    defects = [(index * 2, 0, index * 2 + 1, 1) for index in range(6)]
    assert grade_apple((0, 0, 100, 100), defects, policy()).grade == "G3"


def test_discard_overrides_other_grade_conditions() -> None:
    result = grade_apple(
        (0, 0, 100, 100),
        [(0, 0, 100, 100)],
        policy(),
        discard=True,
    )
    assert result.grade == "DISCARD"
    assert not result.requires_refinement


def test_segmentation_coverage_can_refine_bbox_grade() -> None:
    result = grade_apple(
        (0, 0, 100, 100),
        [(0, 0, 15, 100)],
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
