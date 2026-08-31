"""Unit tests for the 3-tier plant grading engine."""

import pytest

from plant_grading_engine import (
    CLASS_DEFECT_ID,
    LEAF_CLASS_ID,
    PLANT_CLASS_ID,
    UNFIT_DISCARD_CLASS_ID,
    Detection,
    aggregate_profile_grades,
    bind_defects_to_leaves,
    bind_leaves_to_plants,
    discard_parent_indexes,
    grade_detections,
    grade_leaf,
    grade_plant,
    load_grading_policy,
    model_names_match_expected_schema,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def rules():
    return {
        "max_defects_for_healthy_leaf": 2,
        "max_defects_for_moderate_leaf": 5,
        "leaf_area_threshold_moderate_pct": 5.0,
        "leaf_area_threshold_poor_pct": 15.0,
        "max_poor_leaves_for_moderate_plant": 1,
        "unhealthy_leaf_pct_threshold_poor": 40.0,
        "defect_to_leaf_ioa_threshold": 0.10,
        "leaf_to_plant_ioa_threshold": 0.10,
        "discard_proximity_px": 50,
        "refinement_margin_pct": 2.0,
    }


def det(cls_id, name, box, conf=0.9):
    return Detection(cls_id, name, conf, box)


def leaf(box, conf=0.9):
    return det(LEAF_CLASS_ID, "leaf", box, conf)


def defect(box, conf=0.8):
    return det(CLASS_DEFECT_ID, "class_defect", box, conf)


def plant(box, conf=0.95):
    return det(PLANT_CLASS_ID, "plant", box, conf)


def discard(box, conf=0.85):
    return det(UNFIT_DISCARD_CLASS_ID, "unfit_discard", box, conf)


# ── Geometry tests ──────────────────────────────────────────────────

class TestIoA:
    def test_full_overlap(self):
        from plant_grading_engine import _ioa
        assert _ioa((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0

    def test_no_overlap(self):
        from plant_grading_engine import _ioa
        assert _ioa((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0

    def test_half_overlap(self):
        from plant_grading_engine import _ioa
        # child: 100x100 = 10000 area, intersection: 50x100 = 5000
        assert _ioa((50, 0, 150, 100), (0, 0, 100, 100)) == 0.5

    def test_zero_area_child(self):
        from plant_grading_engine import _ioa
        assert _ioa((50, 50, 50, 50), (0, 0, 100, 100)) == 0.0


class TestClipBox:
    def test_clip_inside(self):
        from plant_grading_engine import _clip_box
        assert _clip_box((10, 10, 50, 50), (0, 0, 100, 100)) == (10, 10, 50, 50)

    def test_clip_overflow(self):
        from plant_grading_engine import _clip_box
        assert _clip_box((-10, -10, 110, 110), (0, 0, 100, 100)) == (0, 0, 100, 100)


class TestUnionArea:
    def test_single_box(self):
        from plant_grading_engine import _union_area
        assert _union_area([(0, 0, 100, 100)]) == 10000

    def test_non_overlapping(self):
        from plant_grading_engine import _union_area
        assert _union_area([(0, 0, 50, 50), (100, 100, 150, 150)]) == 5000

    def test_overlapping(self):
        from plant_grading_engine import _union_area
        # Two 100x100 boxes overlapping by 50x50 = 2500
        # Union = 10000 + 10000 - 2500 = 17500
        result = _union_area([(0, 0, 100, 100), (50, 50, 150, 150)])
        assert abs(result - 17500) < 1

    def test_empty(self):
        from plant_grading_engine import _union_area
        assert _union_area([]) == 0.0


# ── Binding tests ───────────────────────────────────────────────────

class TestBindDefectsToLeaves:
    def test_simple_binding(self, rules):
        leaves = [leaf((0, 0, 100, 100)), leaf((200, 200, 300, 300))]
        defects = [defect((10, 10, 30, 30)), defect((210, 210, 230, 230))]
        binding = bind_defects_to_leaves(defects, leaves, 0.10)
        assert len(binding[0]) == 1
        assert len(binding[1]) == 1

    def test_below_threshold(self, rules):
        leaves = [leaf((0, 0, 100, 100))]
        defects = [defect((200, 200, 210, 210))]  # no overlap
        binding = bind_defects_to_leaves(defects, leaves, 0.10)
        assert len(binding[0]) == 0

    def test_best_leaf_selection(self, rules):
        leaves = [leaf((0, 0, 100, 100)), leaf((20, 20, 120, 120))]
        defects = [defect((25, 25, 35, 35))]  # overlaps both, more with leaf 1
        binding = bind_defects_to_leaves(defects, leaves, 0.10)
        # Should bind to leaf with highest IoA
        assert len(binding[0]) + len(binding[1]) == 1


class TestBindLeavesToPlants:
    def test_simple_binding(self, rules):
        plants = [plant((0, 0, 500, 500)), plant((600, 600, 1000, 1000))]
        leaves = [leaf((50, 50, 150, 150)), leaf((650, 650, 750, 750))]
        binding = bind_leaves_to_plants(leaves, plants, 0.10)
        assert len(binding[0]) == 1
        assert len(binding[1]) == 1


class TestDiscardProximity:
    def test_nearby_discard(self, rules):
        plants = [plant((100, 100, 300, 300))]
        discards = [discard((310, 310, 320, 320))]  # within 50px
        result = discard_parent_indexes(discards, plants, 50)
        assert 0 in result

    def test_far_discard(self, rules):
        plants = [plant((100, 100, 300, 300))]
        discards = [discard((500, 500, 510, 510))]  # far away
        result = discard_parent_indexes(discards, plants, 50)
        assert 0 not in result


# ── Leaf grading tests ──────────────────────────────────────────────

class TestGradeLeaf:
    def test_no_defects(self, rules):
        lg = grade_leaf(leaf((0, 0, 100, 100)), [], rules)
        assert lg.grade == "HEALTHY"
        assert lg.defect_count == 0
        assert lg.coverage_pct == 0.0

    def test_few_small_defects(self, rules):
        lf = leaf((0, 0, 100, 100))
        defects = [defect((10, 10, 15, 15)), defect((20, 20, 25, 25))]
        lg = grade_leaf(lf, defects, rules)
        assert lg.grade == "HEALTHY"
        assert lg.defect_count == 2

    def test_many_defects(self, rules):
        lf = leaf((0, 0, 100, 100))
        defects = [defect((i * 10, i * 10, i * 10 + 5, i * 10 + 5)) for i in range(6)]
        lg = grade_leaf(lf, defects, rules)
        assert lg.grade == "POOR"

    def test_high_coverage(self, rules):
        lf = leaf((0, 0, 100, 100))
        defects = [defect((0, 0, 50, 50))]  # 25% coverage
        lg = grade_leaf(lf, defects, rules)
        assert lg.grade == "POOR"
        assert lg.coverage_pct > 15.0


# ── Plant grading tests ─────────────────────────────────────────────

class TestGradePlant:
    def test_all_healthy_leaves(self, rules):
        from plant_grading_engine import LeafGrade
        leaf_grades = [
            LeafGrade("HEALTHY", 0, 0.0, False),
            LeafGrade("HEALTHY", 0, 0.0, False),
            LeafGrade("HEALTHY", 0, 0.0, False),
        ]
        pg = grade_plant(plant((0, 0, 500, 500)), leaf_grades, rules)
        assert pg.grade == "HEALTHY"
        assert pg.healthy_leaves == 3

    def test_one_moderate_leaf(self, rules):
        from plant_grading_engine import LeafGrade
        leaf_grades = [
            LeafGrade("HEALTHY", 0, 0.0, False),
            LeafGrade("MODERATE", 3, 7.0, False),
            LeafGrade("HEALTHY", 0, 0.0, False),
        ]
        pg = grade_plant(plant((0, 0, 500, 500)), leaf_grades, rules)
        assert pg.grade == "MODERATE"

    def test_many_poor_leaves(self, rules):
        from plant_grading_engine import LeafGrade
        leaf_grades = [
            LeafGrade("POOR", 6, 20.0, False),
            LeafGrade("POOR", 7, 25.0, False),
            LeafGrade("HEALTHY", 0, 0.0, False),
        ]
        pg = grade_plant(plant((0, 0, 500, 500)), leaf_grades, rules)
        assert pg.grade == "POOR"

    def test_discard_triggered(self, rules):
        from plant_grading_engine import LeafGrade
        leaf_grades = [LeafGrade("HEALTHY", 0, 0.0, False)]
        pg = grade_plant(plant((0, 0, 500, 500)), leaf_grades, rules, discard_triggered=True)
        assert pg.grade == "DISCARD"
        assert pg.discard_triggered is True

    def test_no_leaves(self, rules):
        pg = grade_plant(plant((0, 0, 500, 500)), [], rules)
        assert pg.grade == "DISCARD"
        assert pg.needs_refinement is True


# ── Integration: grade_detections ───────────────────────────────────

class TestGradeDetections:
    def test_healthy_plant(self, rules):
        detections = [
            plant((0, 0, 500, 500)),
            leaf((50, 50, 150, 200)),
            leaf((200, 100, 300, 250)),
        ]
        results = grade_detections(detections, rules)
        assert len(results) == 1
        assert results[0].grade == "HEALTHY"
        assert results[0].leaf_count == 2

    def test_plant_with_defects(self, rules):
        detections = [
            plant((0, 0, 500, 500)),
            leaf((50, 50, 150, 200)),
            defect((55, 55, 65, 65)),
            defect((70, 70, 80, 80)),
            defect((90, 90, 100, 100)),
        ]
        results = grade_detections(detections, rules)
        assert len(results) == 1
        # 3 defects > max_defects_for_healthy_leaf (2) but <= max_moderate (5)
        assert results[0].grade == "MODERATE"

    def test_discard_trigger(self, rules):
        detections = [
            plant((100, 100, 400, 400)),
            leaf((150, 150, 250, 250)),
            discard((120, 120, 130, 130)),
        ]
        results = grade_detections(detections, rules)
        assert len(results) == 1
        assert results[0].grade == "DISCARD"
        assert results[0].discard_triggered is True

    def test_no_plants(self, rules):
        detections = [leaf((50, 50, 150, 200))]
        results = grade_detections(detections, rules)
        assert len(results) == 0

    def test_two_plants(self, rules):
        detections = [
            plant((0, 0, 300, 300)),
            plant((400, 400, 700, 700)),
            leaf((50, 50, 150, 150)),
            leaf((450, 450, 550, 550)),
        ]
        results = grade_detections(detections, rules)
        assert len(results) == 2
        assert all(r.grade == "HEALTHY" for r in results)


# ── Multi-view aggregation ──────────────────────────────────────────

class TestAggregateProfile:
    def test_all_healthy(self):
        pg = aggregate_profile_grades(
            ["HEALTHY", "HEALTHY", "HEALTHY", "HEALTHY", "HEALTHY"],
            expected_views=5,
        )
        assert pg.grade == "HEALTHY"

    def test_worst_view_propagates(self):
        pg = aggregate_profile_grades(
            ["HEALTHY", "HEALTHY", "MODERATE", "HEALTHY", "HEALTHY"],
            expected_views=5,
        )
        assert pg.grade == "MODERATE"

    def test_discard_in_one_view(self):
        pg = aggregate_profile_grades(
            ["HEALTHY", "DISCARD", "HEALTHY", "HEALTHY", "HEALTHY"],
            expected_views=5,
        )
        assert pg.grade == "DISCARD"

    def test_wrong_view_count(self):
        with pytest.raises(ValueError, match="5 views"):
            aggregate_profile_grades(["HEALTHY", "MODERATE"], expected_views=5)


# ── Schema validation ───────────────────────────────────────────────

class TestSchemaValidation:
    def test_correct_schema(self):
        names = ["plant", "leaf", "unfit_discard", "class_defect"]
        assert model_names_match_expected_schema(names) is True

    def test_wrong_first_class(self):
        names = ["apple", "leaf", "unfit_discard", "class_defect"]
        assert model_names_match_expected_schema(names) is False

    def test_too_few_classes(self):
        names = ["plant", "leaf"]
        assert model_names_match_expected_schema(names) is False

    def test_extra_classes_ok(self):
        names = ["plant", "leaf", "unfit_discard", "class_defect", "extra"]
        assert model_names_match_expected_schema(names) is True


# ── Policy loading ──────────────────────────────────────────────────

class TestLoadPolicy:
    def test_load_grading_policy(self, tmp_path):
        import yaml
        policy = {
            "facility_id": "TEST",
            "rules": {
                "max_defects_for_healthy_leaf": 3,
                "defect_to_leaf_ioa_threshold": 0.15,
            },
        }
        p = tmp_path / "policy.yaml"
        p.write_text(yaml.dump(policy))
        rules = load_grading_policy(p)
        assert rules["max_defects_for_healthy_leaf"] == 3
        assert rules["defect_to_leaf_ioa_threshold"] == 0.15
