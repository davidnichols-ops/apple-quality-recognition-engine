import numpy as np

from scripts.apple_surface_geometry import (
    exclude_calyx_candidate,
    fit_surface,
    region_deviation,
)


def test_surface_fit_finds_recess_after_apple_curve_is_removed():
    yy, xx = np.indices((160, 160))
    radius_sq = ((xx - 80) / 65) ** 2 + ((yy - 80) / 65) ** 2
    apple = radius_sq < 1
    bite = ((xx - 80) ** 2 + (yy - 45) ** 2) < 12 ** 2
    depth = 0.7 + 0.1 * (1 - np.sqrt(np.maximum(1 - radius_sq, 0)))
    depth[bite] += 0.03

    residual, fit = fit_surface(depth, apple, bite)
    deviation = region_deviation(residual, bite, fit["intact_residual_mad"])

    assert fit["status"] == "convex"
    assert fit["curvature_coefficient"] > 0
    assert deviation["median_residual"] > 0.02


def test_calyx_candidate_is_suppressed_but_adjacent_defect_is_retained():
    calyx = np.zeros((100, 100), np.uint8)
    calyx[45:55, 45:55] = 1
    false_positive = calyx.copy()
    real_defect = np.zeros_like(calyx)
    real_defect[40:60, 70:90] = 1

    suppressed, reason = exclude_calyx_candidate(false_positive, calyx)
    retained, retained_reason = exclude_calyx_candidate(real_defect, calyx)

    assert reason["status"] == "suppressed_normal_calyx"
    assert suppressed.sum() == 0
    assert retained_reason["status"] == "retained"
    assert retained.sum() == real_defect.sum()


def test_inverted_depth_curve_is_explicitly_flagged():
    yy, xx = np.indices((160, 160))
    radius_sq = ((xx - 80) / 65) ** 2 + ((yy - 80) / 65) ** 2
    apple = radius_sq < 1
    inverted_depth = 1.0 - 0.1 * (1 - np.sqrt(np.maximum(1 - radius_sq, 0)))

    _, fit = fit_surface(inverted_depth, apple, np.zeros_like(apple))

    assert fit["status"] == "nonconvex_depth_map"
