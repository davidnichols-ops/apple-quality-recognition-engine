# Apple Quality Recognition Engine

An edge-first system for grading apples on Apple Silicon. A detector locates the fruit and visible anomaly regions; geometry binds those child regions to the correct apple; a versioned policy produces the grade.

**Models observe. Deterministic policy decides. Humans authorize learning and promotion.**

This repository is an engineering prototype, not a validated commercial grader. The current model artifact is a COCO placeholder. Production claims begin only after a custom model, a profile-isolated holdout, and a real line trial pass the gates in `docs/roadmap.md`.

## Product contract

The system separates four kinds of truth:

1. **Reference grade** — G1, G2, G3, or DISCARD assigned before capture by a trusted grader under a named facility policy. It is profile metadata, not a YOLO class.
2. **Visual observations** — three YOLO detection classes: `apple`, `unfit_bin_discard`, and `class_defect`.
3. **Deterministic decision** — grade derived from bound defect count, union coverage, discard geometry, and a versioned YAML policy.
4. **Advisory review** — optional segmentation and Gemini 3.7 Flash review can propose refinements. Neither may silently change a grade, policy, annotation, dataset, or checkpoint.

## Production flow

```text
Known-grade apple
  -> five-view capture (4 equatorial + 1 calyx)
  -> profile manifest
  -> Roboflow annotation / reviewed label assist
  -> profile-level train/val/test split
  -> YOLO26 candidate training
  -> held-out evaluation + M4 benchmark
  -> human checkpoint promotion

Live apple
  -> YOLO26 CoreML candidate
  -> apple + child anomaly boxes
  -> IoA spatial binding
  -> clipped union coverage
  -> deterministic per-view grade
  -> worst visible grade across complete five-view profile
  -> G1 / G2 / G3 / DISCARD
  -> review queue when confidence is volatile or coverage is near a boundary
```

The VLM is deliberately outside the real-time authority path. Gemini 3.7 Flash is the planned offline reviewer because it accepts multiple images and structured output at practical batch cost. Its output is stored as a pending proposal through `vlm_review_schema.py`; a human must approve or reject it.

## Three-class detector schema

| ID | Class | Role |
|---:|---|---|
| 0 | `apple` | Tight macro parent box around one visible fruit |
| 1 | `unfit_bin_discard` | Local discard signal or unfit region; it applies only to a nearby apple |
| 2 | `class_defect` | Tight child box around any visible anomaly region |

Why one generic defect class:

- it concentrates the available examples instead of starving rare defect types;
- it makes annotation and Roboflow label assist easier to review;
- it lets geometry, count, and coverage carry the first production decision;
- it defers taxonomic expansion until confusion data proves a specific split adds grade value.

Annotators still box every visible defect on G1 fruit. Omitting tolerated G1 defects would teach the detector that grade controls whether a defect exists, which is label leakage.

## Deterministic grading

`grading_engine.py` is the pure decision core. It:

- binds each `class_defect` to the apple with the highest child Intersection-over-Area (IoA) above the configured threshold;
- clips defect boxes to the parent apple;
- computes the geometric union of overlapping child boxes so overlap is not counted twice;
- applies count and coverage thresholds from `grading_policy.yaml`;
- applies `unfit_bin_discard` only to the nearby apple rather than globally;
- marks box coverage near a decision threshold as requiring refinement;
- supports an optional externally measured segmentation coverage value;
- aggregates a complete five-view profile using the worst visible grade.

The committed thresholds are candidate calibration values, not claims that 5% and 15% match a customer standard. They remain candidates until held-out known-grade profiles validate them.

## Detection first, segmentation by evidence

Bounding boxes are the v1 annotation format. They are fast to label and sufficient to test whether generic anomaly geometry predicts grade.

Boxes overestimate irregular defect area. The policy therefore exposes a refinement margin around G1/G2 and G2/G3 boundaries. Selective segmentation is a gated experiment, not a current production dependency:

1. establish the box-only confusion matrix on untouched profiles;
2. identify whether boundary errors are materially caused by box-area bias;
3. annotate masks only for those boundary cases;
4. promote a segmentation refiner only if it improves profile grade accuracy enough to justify its latency and labeling cost.

## Dataset protocol

A profile is one physical apple and all five views. The capture script records a stable `profile_id`, reference grade, grader, facility, batch, lot, cultivar, camera, and view metadata in JSONL.

```bash
python capture_dataset.py \
  --grade G2 \
  --count 10 \
  --batch-id gala-lot-17-20260831 \
  --grader-id david \
  --lot-id lot-17 \
  --cultivar gala
```

Generate deterministic 70/20/10 lists after capture or export:

```bash
python scripts/split_profiles.py \
  --manifest dataset/raw_ingest/capture_manifest.jsonl \
  --output-dir apple_dataset/splits
```

All five views of a profile receive the same split. Never let frames from one apple cross train, validation, and test boundaries. Reference grade should be stratified and audited at the profile level before training.

The working scale hypothesis is 3,000-6,000 profiles (15,000-30,000 images), but image count does not prove sufficiency. Advancement depends on held-out per-grade performance, discard recall, calibration stability, cultivar/lot generalization, and edge latency.

## Human-governed improvement loop

```text
low confidence / boundary / operator disagreement
  -> immutable review record
  -> optional VLM proposal
  -> human accept, correct, or reject
  -> Roboflow annotation revision
  -> new immutable dataset version
  -> offline retraining
  -> regression evaluation
  -> human promotion decision
```

There is no live checkpoint writing and no recursive self-training. `scripts/reingest_harvest.py` only admits telemetry marked `review_status: approved`. A VLM proposal defaults to `pending_human_review` and has no auto-apply method.

## Quick start

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.lock.txt

# Pure deterministic tests
pytest -q

# Capture a known-grade batch
python capture_dataset.py --grade G1 --count 5 --batch-id pilot-g1-a

# Run camera/model baseline checks
python baseline_verify.py

# Run live inference with an explicit candidate model
python local_inference.py \
  --model yolo26x_640.mlpackage \
  --policy grading_policy.yaml
```

`capture_dataset.py` and `local_inference.py` fail closed when the Arducam is absent. Use `--allow-camera-fallback` only for an explicit built-in-camera test. If the Arducam does not enumerate at OpenCV index 0, set `ARDUCAM_CAMERA_INDEX` to the verified index.

`local_inference.py` enters benchmark mode if model names do not exactly match the three-class schema. In benchmark mode it renders FPS only; grading and edge harvest remain disabled. A missing candidate model also fails closed unless `--benchmark-fallback` is explicitly supplied.

## Operator controls

| Key | Action |
|---|---|
| `q` | Stop live inference |
| `g` | Persist an operator disagreement and add the frame to human review |
| `space` | Start the equatorial burst or capture the calyx view in capture mode |
| `esc` | Stop capture safely |

## Repository structure

| Path | Responsibility |
|---|---|
| `capture_dataset.py` | Five-view known-grade capture and JSONL profile manifest |
| `grading_engine.py` | Pure geometry, grade decisions, refinement flag, profile aggregation |
| `local_inference.py` | Camera, YOLO inference, binding, rendering, harvest, operator override |
| `edge_harvest_schema.py` | Typed review telemetry contract |
| `vlm_review_schema.py` | Advisory-only VLM proposal contract |
| `grading_policy.yaml` | Versioned facility calibration candidates |
| `data.yaml` | Three-class YOLO dataset schema and profile split files |
| `scripts/split_profiles.py` | Deterministic profile-level split generation |
| `scripts/reingest_harvest.py` | Human-approved review re-ingestion only |
| `docs/annotation_sop.md` | Roboflow labeling and review standard |
| `docs/production_architecture.md` | Authority boundaries and production-line design |
| `docs/roadmap.md` | Staged evidence gates, done criteria, and pivot criteria |
| `tests/` | Hardware-independent regression tests |

## Current status

Implemented and testable now:

- five-view capture metadata contract;
- three-class schema;
- profile-isolated deterministic splitter;
- box union coverage and per-parent discard geometry;
- deterministic grade/refinement/profile aggregation core;
- typed human review and VLM advisory records;
- operator override and typed edge-harvest wiring.

Not yet validated or implemented as production capability:

- a trained apple checkpoint;
- Roboflow dataset/version evidence;
- profile-level accuracy targets on an untouched holdout;
- selective segmentation model;
- Gemini API reviewer execution;
- checkpoint registry and promotion automation;
- sustained physical production-line trial.

See `docs/roadmap.md` for the order of operations. Do not skip from camera demo to production claims.

## Licensing

Ultralytics YOLO26 is AGPL-3.0. Commercial or closed deployment requires a licensing review before customer use. The repository license does not waive upstream obligations.
