# Roboflow Small-Dataset Smoke Test

Date: 2026-10-12

## Scope

This test exercised Roboflow model
`davids-workspace-o4qfg/test-demo-7aocj-1-yolo26s-seg-t2` against all 127
images in dataset version 1, then normalized detections by class name and ran
the 120 apple frames through the repository's deterministic grading engine.
The 120 frames represent 30 physical apples with four equatorial views each.
Seven negative frames were treated as a separate no-apple robustness check.

This is a pipeline and resubstitution smoke test. Most evaluated images were
used for training, and the Roboflow partition does not isolate physical apple
profiles.

## Roboflow evaluation

Dataset split: 89 train, 25 validation, 13 test.

| Class | mAP50 | Precision | Recall | F1 | Evaluation threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| apple | 1.000 | 1.000 | 1.000 | 1.000 | 0.18 |
| defect_critical | 0.546 | 0.500 | 0.750 | 0.600 | 0.11 |
| defect_surface | 0.218 | 1.000 | 0.167 | 0.286 | 0.34 |
| stem_calyx | 0.735 | 0.778 | 0.700 | 0.737 | 0.14 |

Aggregate metrics were mAP50 0.625, precision 0.539, recall 0.725, and F1
0.611. Median hosted inference time was 6.51 ms. The 13-image test split is
too small for a reliable deployment estimate.

## End-to-end results

At the per-class evaluation thresholds:

- Apple detection: 120/120 apple frames.
- Negative robustness: 0/7 negative frames produced an apple detection.
- Surface defects bound to an apple: 6/120 frames.
- Critical defects bound to an apple: 24/120 frames.
- Current-policy frame grades: 94 G1, 2 G2, 24 DISCARD.
- Current-policy profile grades: 15 G1, 1 G2, 14 DISCARD.

At the application's current global 0.35 confidence threshold:

- Apple detection: 120/120 apple frames.
- Negative robustness: 0/7 negative frames produced an apple detection.
- Surface defects bound to an apple: 6/120 frames.
- Critical defects bound to an apple: 9/120 frames.
- Current-policy frame grades: 109 G1, 2 G2, 9 DISCARD.
- Current-policy profile grades: 23 G1, 1 G2, 6 DISCARD.

The grade distributions are diagnostic outputs, not accuracy claims. The
capture manifest labels all 30 apples CIDER at profile level and does not
contain per-defect physical ground truth. The repository policy used by this
test is also stale: it sends any detected critical defect to DISCARD, while
the latest owner rule says two critical defects send an apple to CIDER.

## Contract and data failures

1. Roboflow emitted class IDs as `apple=0`, `defect_critical=1`,
   `defect_surface=2`, and `stem_calyx=3`. The runtime contract requires
   `apple=0`, `stem_calyx=1`, `defect_surface=2`, and
   `defect_critical=3`. The smoke adapter safely maps by class name. Direct
   checkpoint deployment must repair or explicitly normalize this mapping.
2. Twenty-two of 30 physical apples have their four views split across more
   than one of train, validation, and test. This violates the repository's
   profile isolation invariant and inflates held-out confidence.
3. Surface-defect recall is 16.7% on the leaked 13-image test split. The model
   therefore fails the grading-readiness gate even before a clean profile-level
   evaluation.
4. Mask localization remains weak: test mAP50-95 is 0.185 for critical
   defects, 0.093 for surface defects, and 0.183 for stem/calyx. Roboflow did
   not report mIoU for this evaluation.

## Decision

The model passes API, segmentation-output, apple-AOI, negative-frame, and
grading-pipeline connectivity checks. It fails the deployment smoke test for
surface-defect sensitivity, profile-isolated evaluation, class-ID compatibility,
and current policy compatibility.

As a narrow fail-safe, the candidate policy enables
`review_zero_defect_profiles`. A complete profile with zero surface and zero
critical detections keeps its provisional calculated grade, but cannot pass
without review. This prevents the tested model's missed defects from silently
certifying G1 fruit. The switch can be disabled after a replacement model
passes held-out sensitivity testing.

Before retraining, assign every four-view apple profile to exactly one split.
After retraining, repeat the same smoke test on a physically characterized,
profile-held-out trial set. Update the deterministic critical-defect policy
only after the one-critical and discard rules are explicitly recorded.

The reusable adapter is
`scripts/smoke_roboflow_predictions.py`. Raw hosted predictions and generated
reports remain under the ignored `dataset/smoke_tests/` evidence directory.
