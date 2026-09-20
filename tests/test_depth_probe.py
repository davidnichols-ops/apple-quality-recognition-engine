import numpy as np

from scripts.depth_probe import apple_aoi, cavity_contrast, masks_from_yolo_seg


def test_cavity_contrast_detects_recess_in_synthetic_depth_map():
    yy, xx = np.indices((160, 160))
    apple = ((xx - 80) ** 2 + (yy - 80) ** 2) < 65 ** 2
    defect = ((xx - 80) ** 2 + (yy - 80) ** 2) < 15 ** 2
    smooth_surface = 5.0 - 0.0002 * ((xx - 80) ** 2 + (yy - 80) ** 2)
    depth = smooth_surface - 0.4 * defect

    result = cavity_contrast(depth, apple, defect)

    assert result["status"] == "measured"
    assert result["relative_depth_residual"] < -0.3
    assert result["absolute_contrast_to_noise"] > 10


def test_yolo_seg_polygons_become_aligned_depth_probe_masks(tmp_path):
    labels = tmp_path / "frame.txt"
    labels.write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
        "3 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n"
    )

    masks = masks_from_yolo_seg(labels, (100, 100))

    assert masks["apple"][50, 50] == 1
    assert masks["defect_critical"][50, 50] == 1
    assert masks["defect_surface"].sum() == 0
    assert masks["defect_critical"][5, 5] == 0


def test_apple_aoi_includes_entire_fruit_and_clamps_to_frame():
    apple = np.zeros((100, 160), np.uint8)
    apple[10:90, 5:85] = 1

    assert apple_aoi(apple) == (0, 2, 93, 98)
