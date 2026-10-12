#!/usr/bin/env python3
"""Run saved Roboflow instance-segmentation predictions through AQRE policy."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grading_engine import (
    EXPECTED_CLASS_NAMES,
    aggregate_profile_grades,
    bind_defects_to_parents,
    grade_apple,
    load_grading_policy,
)

PROFILE_RE = re.compile(r"_(crimson-crisp-lot-01-\d{5})_cider_equatorial_view_(\d)\.jpg$")
CLASS_THRESHOLDS = {
    "apple": 0.18,
    "defect_critical": 0.11,
    "defect_surface": 0.34,
    "stem_calyx": 0.14,
}


def box(prediction: dict) -> list[float]:
    x, y = float(prediction["x"]), float(prediction["y"])
    w, h = float(prediction["width"]), float(prediction["height"])
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def polygon_mask(prediction: dict, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    points = prediction.get("points") or []
    if len(points) >= 3:
        polygon = np.asarray([[round(p["x"]), round(p["y"])] for p in points], np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def mask_coverage(apple: dict, surfaces: list[dict], height: int, width: int) -> float:
    apple_mask = polygon_mask(apple, height, width)
    denominator = int(apple_mask.sum())
    if denominator == 0:
        return 0.0
    surface_mask = np.zeros_like(apple_mask)
    for prediction in surfaces:
        surface_mask |= polygon_mask(prediction, height, width)
    return float((surface_mask & apple_mask).sum() / denominator * 100.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--policy", type=Path, default=Path("grading_policy.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-mode", choices=("class-optimal", "global-035"), default="class-optimal")
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text())
    policy = load_grading_policy(args.policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    profiles = defaultdict(list)
    split_by_profile = defaultdict(set)
    observed_ids = defaultdict(set)

    for record in payload["records"]:
        predictions = []
        for prediction in record["predictions"]:
            observed_ids[prediction["class_name"]].add(int(prediction["class_id"]))
            threshold = 0.35 if args.threshold_mode == "global-035" else CLASS_THRESHOLDS[prediction["class_name"]]
            if float(prediction["confidence"]) >= threshold:
                predictions.append(prediction)

        apples = sorted((p for p in predictions if p["class_name"] == "apple"), key=lambda p: p["confidence"], reverse=True)
        surfaces = [p for p in predictions if p["class_name"] == "defect_surface"]
        criticals = [p for p in predictions if p["class_name"] == "defect_critical"]
        match = PROFILE_RE.search(record["filename"])
        profile_id, view_index = (match.group(1), int(match.group(2))) if match else (None, None)

        frame = {
            "filename": record["filename"], "split": record["split"],
            "profile_id": profile_id, "view_index": view_index,
            "apple_count": len(apples), "surface_count_raw": len(surfaces),
            "critical_count_raw": len(criticals), "grade": "NO_APPLE",
            "surface_coverage_pct": 0.0, "critical_bound": 0,
        }
        if apples:
            apple = apples[0]
            apple_box = box(apple)
            surface_boxes, critical_boxes = [box(p) for p in surfaces], [box(p) for p in criticals]
            surface_idx = bind_defects_to_parents(surface_boxes, [apple_box], policy.ioa_binding_threshold)[0]
            critical_idx = bind_defects_to_parents(critical_boxes, [apple_box], policy.ioa_binding_threshold)[0]
            bound_surfaces = [surfaces[i] for i in surface_idx]
            coverage = mask_coverage(apple, bound_surfaces, int(record["height"]), int(record["width"]))
            decision = grade_apple(
                apple_box, [surface_boxes[i] for i in surface_idx],
                [critical_boxes[i] for i in critical_idx], policy,
                refined_coverage_pct=coverage,
            )
            frame.update({"grade": decision.grade, "surface_coverage_pct": round(coverage, 5),
                          "critical_bound": decision.critical_defect_count,
                          "surface_bound": decision.surface_defect_count,
                          "requires_refinement": decision.requires_refinement})
            if profile_id:
                profiles[profile_id].append(decision)
                split_by_profile[profile_id].add(record["split"])
        frames.append(frame)

    profile_rows = []
    for profile_id, decisions in sorted(profiles.items()):
        aggregate = aggregate_profile_grades(decisions, policy.expected_profile_views)
        profile_rows.append({"profile_id": profile_id, **asdict(aggregate),
                             "splits": sorted(split_by_profile[profile_id])})

    negatives = [row for row in frames if row["profile_id"] is None]
    apples = [row for row in frames if row["profile_id"] is not None]
    id_contract = {name: sorted(ids) for name, ids in observed_ids.items()}
    expected_by_name = {name: class_id for class_id, name in EXPECTED_CLASS_NAMES.items()}
    summary = {
        "model_id": payload["model_id"], "dataset_images": len(frames),
        "apple_frames": len(apples), "negative_frames": len(negatives),
        "threshold_mode": args.threshold_mode, "policy_version": policy.policy_version,
        "class_id_contract_expected": expected_by_name, "class_id_contract_observed": id_contract,
        "class_id_contract_matches": all(id_contract.get(name) == [class_id] for name, class_id in expected_by_name.items()),
        "apple_frames_without_apple": sum(row["apple_count"] == 0 for row in apples),
        "negative_false_positive_frames": sum(row["apple_count"] > 0 for row in negatives),
        "frame_grades": dict(Counter(row["grade"] for row in apples)),
        "profile_grades": dict(Counter(row["grade"] for row in profile_rows)),
        "complete_profiles": sum(row["is_complete"] for row in profile_rows),
        "total_profiles": len(profile_rows),
        "profile_split_leakage": {key: sorted(value) for key, value in split_by_profile.items() if len(value) > 1},
        "frames_with_bound_critical": sum(row["critical_bound"] > 0 for row in apples),
        "frames_with_bound_surface": sum(row.get("surface_bound", 0) > 0 for row in apples),
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "profiles.json").write_text(json.dumps(profile_rows, indent=2) + "\n")
    with (args.output_dir / "frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in frames for key in row}))
        writer.writeheader(); writer.writerows(frames)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
