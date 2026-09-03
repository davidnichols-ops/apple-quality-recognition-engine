#!/usr/bin/env python3
"""Blue nitrile glove masking for the apple capture/inference pipeline.

Human skin tones share optical spectrum with yellow, red, and russet apples.
Bright surgical blue nitrile gloves, however, have zero overlap with organic
apple pigments or the matte black backdrop. Converting to HSV and thresholding
via ``cv2.inRange`` is simple array math that runs on the CPU in under 1 ms,
before anything reaches the neural network.

The glove mask is subtracted from the fruit detection mask so fingers never
trigger false bruise or defect detections:

    Visible Fruit Mask = Apple Mask \\ Glove Mask

Defect candidates with high overlap against the glove mask are suppressed as
finger edge artifacts.

This module is import-safe: importing it produces no side effects. The OpenCV
import is deferred to call time so the module can be imported in test
environments without a camera.
"""

from __future__ import annotations

from typing import Any

# HSV bounds for bright surgical blue nitrile in OpenCV's H range [0, 179].
# Hue 100-135 covers pure blue; high saturation separates nitrile from any
# blue-ish apple specular highlights (which rarely exceed S=120).
_LOWER_BLUE_HSV = (100, 120, 50)
_UPPER_BLUE_HSV = (135, 255, 255)
_MORPH_KERNEL_SIZE = (5, 5)

# A defect candidate overlapping the glove mask by more than this fraction is
# suppressed as a finger edge artifact.
GLOVE_OVERLAP_SUPPRESS_THRESHOLD = 0.30


def mask_blue_gloves(frame: Any) -> tuple[Any, Any]:
    """Detect bright blue nitrile gloves in a BGR frame.

    Args:
        frame: OpenCV BGR frame (numpy array).

    Returns:
        A tuple ``(glove_mask, non_glove_mask)`` of single-channel uint8
        masks. ``glove_mask`` is 255 where gloves are detected, 0 elsewhere.
        ``non_glove_mask`` is the bitwise inverse.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(_LOWER_BLUE_HSV, dtype=np.uint8)
    upper = np.array(_UPPER_BLUE_HSV, dtype=np.uint8)
    glove_mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _MORPH_KERNEL_SIZE)
    glove_mask = cv2.morphologyEx(glove_mask, cv2.MORPH_CLOSE, kernel)
    non_glove_mask = cv2.bitwise_not(glove_mask)
    return glove_mask, non_glove_mask


def visible_fruit_mask(apple_mask: Any, glove_mask: Any) -> Any:
    """Subtract the glove mask from the apple mask.

    Visible Fruit Mask = Apple Mask \\ Glove Mask

    Args:
        apple_mask: Single-channel mask of detected apple pixels.
        glove_mask: Single-channel mask of detected glove pixels.

    Returns:
        Single-channel mask of apple pixels not covered by gloves.
    """
    import cv2

    return cv2.bitwise_and(apple_mask, cv2.bitwise_not(glove_mask))


def suppress_glove_defects(
    defect_boxes: list[list[int]],
    glove_mask: Any,
    threshold: float = GLOVE_OVERLAP_SUPPRESS_THRESHOLD,
) -> list[list[int]]:
    """Filter defect candidate boxes that overlap the glove mask.

    A defect candidate whose box has more than ``threshold`` fraction of its
    pixels inside the glove mask is suppressed as a finger edge artifact.

    Args:
        defect_boxes: List of [x1, y1, x2, y2] pixel boxes.
        glove_mask: Single-channel mask of detected glove pixels.
        threshold: Overlap fraction above which a box is suppressed.

    Returns:
        Filtered list of boxes that passed the glove-overlap check.
    """
    import numpy as np

    kept: list[list[int]] = []
    for box in defect_boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(glove_mask.shape[1], x2)
        y2 = min(glove_mask.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            kept.append(box)
            continue
        region = glove_mask[y1:y2, x1:x2]
        box_area = (x2 - x1) * (y2 - y1)
        glove_pixels = int(np.count_nonzero(region))
        if box_area > 0 and glove_pixels / box_area > threshold:
            continue
        kept.append(box)
    return kept
