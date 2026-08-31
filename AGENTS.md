# Repository Agent Guide

## Architecture invariants

- Detector classes are exactly `apple`, `unfit_bin_discard`, and `class_defect` with IDs 0, 1, and 2.
- Reference grade is profile metadata, never a YOLO class.
- All five views of one physical apple must stay in one dataset partition.
- The deterministic grading engine owns the operational grade.
- Segmentation may refine coverage but does not replace grading policy.
- VLM output is advisory and defaults to `pending_human_review`.
- No live training, automatic annotation acceptance, policy mutation, or checkpoint promotion.
- Candidate thresholds and benchmark numbers are not production claims.

## Verification

Run the hardware-independent suite before committing:

```bash
uv run --with pytest --with pyyaml pytest -q
uv run --with pyflakes pyflakes \
  baseline_verify.py camera_utils.py capture_dataset.py edge_harvest_schema.py \
  grading_engine.py kernel_apple_coreml.py kernel_dispatch.py local_inference.py \
  override_persistence.py vlm_review_schema.py scripts tests
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
