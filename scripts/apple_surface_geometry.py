"""Fit a smooth convex apple surface to a monocular depth map.

The fit is a diagnostic. It cannot recover a cavity absent from the input
depth map, and its residuals are not physical millimetres without calibration.
"""

from __future__ import annotations

import cv2
import numpy as np


def fit_surface(
    depth: np.ndarray,
    apple_mask: np.ndarray,
    excluded_mask: np.ndarray,
    *,
    max_samples: int = 25000,
) -> tuple[np.ndarray, dict]:
    """Fit tilted ellipsoid depth to intact interior skin using robust least squares.

    At fixed camera distance, a convex fruit should have lower metric depth at
    its centre than near its silhouette. The radial basis models that curve;
    linear x/y terms absorb camera tilt. A negative fitted curvature is kept as
    an explicit sanity failure rather than silently flipping model output.
    """
    if depth.shape != apple_mask.shape or depth.shape != excluded_mask.shape:
        raise ValueError("depth and masks must have the same shape")
    apple = apple_mask.astype(bool)
    ys, xs = np.where(apple)
    if len(xs) < 500:
        raise ValueError("apple mask is too small for a surface fit")
    cx = (float(xs.min()) + float(xs.max())) / 2
    cy = (float(ys.min()) + float(ys.max())) / 2
    rx = max((float(xs.max()) - float(xs.min())) / 2, 1.0)
    ry = max((float(ys.max()) - float(ys.min())) / 2, 1.0)
    yy, xx = np.indices(depth.shape)
    x = (xx - cx) / rx
    y = (yy - cy) / ry
    rho2 = x * x + y * y
    radial = 1 - np.sqrt(np.maximum(1 - np.minimum(rho2, 1), 0))

    excluded = cv2.dilate(excluded_mask.astype(np.uint8),
                          np.ones((21, 21), np.uint8)).astype(bool)
    interior = cv2.erode(apple.astype(np.uint8),
                         np.ones((9, 9), np.uint8)).astype(bool)
    fit_pixels = interior & ~excluded & (rho2 < 0.85) & np.isfinite(depth)
    fit_y, fit_x = np.where(fit_pixels)
    if len(fit_x) < 500:
        raise ValueError("not enough intact apple skin for a surface fit")
    stride = max(1, int(np.ceil(np.sqrt(len(fit_x) / max_samples))))
    fit_y, fit_x = fit_y[::stride], fit_x[::stride]
    design = np.stack((np.ones(len(fit_x)), x[fit_y, fit_x],
                       y[fit_y, fit_x], radial[fit_y, fit_x]), axis=1)
    target = depth[fit_y, fit_x].astype(np.float64)
    weights = np.ones(len(target), np.float64)
    for _ in range(6):
        root_w = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(design * root_w[:, None],
                                           target * root_w, rcond=None)
        errors = target - design @ coefficients
        scale = 1.4826 * np.median(np.abs(errors - np.median(errors)))
        cutoff = max(1.5 * scale, 1e-6)
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(errors), 1e-12))

    expected = (coefficients[0] + coefficients[1] * x +
                coefficients[2] * y + coefficients[3] * radial).astype(np.float32)
    residual = depth - expected
    intact_mad = float(np.median(np.abs(residual[fit_pixels] -
                                         np.median(residual[fit_pixels]))))
    diagnostics = {
        "status": "convex" if coefficients[3] > 0 else "nonconvex_depth_map",
        "intact_pixels": int(fit_pixels.sum()),
        "sampled_pixels": int(len(target)),
        "curvature_coefficient": float(coefficients[3]),
        "intact_residual_mad": intact_mad,
        "center_xy": [cx, cy],
        "radii_xy": [rx, ry],
        "interpretation": "Positive residual means modelled surface is farther than the fitted intact skin.",
    }
    return residual, diagnostics


def region_deviation(residual: np.ndarray, mask: np.ndarray, noise_mad: float) -> dict:
    pixels = residual[(mask > 0) & np.isfinite(residual)]
    if len(pixels) < 25:
        return {"status": "insufficient_mask"}
    median = float(np.median(pixels))
    return {
        "status": "measured",
        "pixels": int(len(pixels)),
        "median_residual": median,
        "signed_contrast_to_noise": median / max(noise_mad, 1e-6),
    }


def exclude_calyx_candidate(
    defect_mask: np.ndarray,
    calyx_mask: np.ndarray,
    *,
    overlap_threshold: float = 0.8,
) -> tuple[np.ndarray, dict]:
    """Suppress normal calyx candidates; retain tissue outside its safety rim."""
    defect = defect_mask.astype(bool)
    zone = cv2.dilate(calyx_mask.astype(np.uint8),
                      np.ones((15, 15), np.uint8)).astype(bool)
    count = int(defect.sum())
    if count == 0:
        return defect.astype(np.uint8), {"status": "empty_candidate"}
    overlap = int((defect & zone).sum())
    fraction = overlap / count
    if fraction >= overlap_threshold:
        return np.zeros_like(defect_mask, np.uint8), {
            "status": "suppressed_normal_calyx",
            "calyx_overlap_fraction": fraction,
        }
    filtered = (defect & ~zone).astype(np.uint8)
    return filtered, {
        "status": "retained",
        "calyx_overlap_fraction": fraction,
        "remaining_pixels": int(filtered.sum()),
    }
