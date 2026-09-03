# Repository Agent Guide

## Architecture invariants

- Detector classes are exactly `apple`, `stem_calyx`, `defect_surface`, and
  `defect_critical` with IDs 0, 1, 2, and 3.
- The neural network observes physical features only; it has zero concept of
  commercial grades. The deterministic policy engine owns the operational grade.
- Output grades are exactly `G1`, `G2`, `G3`, `CIDER`, and `DISCARD`.
- USDA mapping: G1 = Extra Fancy, G2 = Fancy/No. 1, G3 = Utility/Processing,
  CIDER = sound fruit failing fresh-market tolerances, DISCARD = total cull.
- Any `defect_critical` bound to an apple forces `DISCARD` regardless of
  surface coverage.
- Surface coverage ratio `R_surf = Area(defect_surface union) / Area(apple) *
  100` maps to G1/G2/G3/CIDER via versioned thresholds.
- `stem_calyx` detections are anatomical exclusion zones, never counted as
  defects.
- All five views of one physical apple must stay in one dataset partition.
- Segmentation may refine coverage but does not replace grading policy.
- VLM output is advisory and defaults to `pending_human_review`.
- No live training, automatic annotation acceptance, policy mutation, or
  checkpoint promotion.
- Candidate thresholds and benchmark numbers are not production claims.
- Blue nitrile glove masking (HSV threshold, <1ms CPU) runs before inference
  to suppress finger edge artifacts that share spectrum with apple skin tones.
- Annotation uses a local SAM 2 -> YOLO-Seg pipeline (frozen teacher), not a
  cloud labeling service.

## Verification

Run the hardware-independent suite before committing:

```bash
uv run --with pytest --with pyyaml pytest -q
uv run --with pyflakes pyflakes \
  baseline_verify.py camera_utils.py capture_dataset.py edge_harvest_schema.py \
  glove_masking.py grading_engine.py kernel_apple_coreml.py kernel_dispatch.py \
  local_inference.py override_persistence.py vlm_review_schema.py scripts tests
uv run --with ruff ruff check --select F --ignore F401,F841 .
python3 -m compileall -q .
git diff --check
```

Camera and CoreML checks require physical hardware and model artifacts. Report them separately from unit tests; never infer hardware success from mocks.

## Data safety

- `dataset/` and model artifacts are intentionally ignored by Git.
- Verify SHA-256 before re-ingesting a reviewed frame.
- Resolved review records require reviewer identity and timestamp.
- Do not send customer imagery to a cloud VLM without explicit authorization.
