# Apple Quality Recognition Engine

An edge-first system for grading apples on Apple Silicon. A detector locates the fruit and visible anomaly regions; geometry binds those child regions to the correct apple; a versioned policy produces the grade.

**Models observe. Deterministic policy decides. Humans authorize learning and promotion.**

This repository is an engineering prototype, not a validated commercial grader. The current model artifact is a COCO placeholder. Production claims begin only after a custom model, a profile-isolated holdout, and a real line trial pass the gates in `docs/roadmap.md`.

## Product contract

The system separates four kinds of truth:

1. **Reference grade** — G1, G2, G3, CIDER, or DISCARD assigned before capture by a trusted grader under a named facility policy. It is profile metadata, not a YOLO class.
2. **Visual observations** — four YOLO detection classes: `apple`, `stem_calyx`, `defect_surface`, and `defect_critical`.
3. **Deterministic decision** — grade derived from bound defect coverage ratios and the presence of any critical defect, via a versioned YAML policy.
4. **Advisory review** — optional segmentation and Gemini 3.7 Flash review can propose refinements. Neither may silently change a grade, policy, annotation, dataset, or checkpoint.

## Production flow

```text
Known-grade apple
  -> five-view capture (4 equatorial + 1 calyx)
  -> profile manifest
  -> local SAM 2 annotation (frozen teacher, box-prompted segmentation)
  -> YOLO-Seg label conversion
  -> profile-level train/val/test split
  -> YOLO26-Seg candidate training
  -> held-out evaluation + M4 benchmark
  -> human checkpoint promotion

Live apple
  -> blue nitrile glove masking (HSV, <1ms CPU)
  -> YOLO26 CoreML candidate
  -> apple + stem_calyx + defect_surface + defect_critical boxes
  -> IoA spatial binding
  -> any defect_critical -> DISCARD
  -> else clipped defect_surface union coverage -> R_surf
  -> deterministic per-view grade
  -> worst visible grade across complete five-view profile
  -> G1 / G2 / G3 / CIDER / DISCARD
  -> review queue when confidence is volatile or coverage is near a boundary
```

The VLM is deliberately outside the real-time authority path. Gemini 3.7 Flash is the planned offline reviewer because it accepts multiple images and structured output at practical batch cost. Its output is stored as a pending proposal through `vlm_review_schema.py`; a human must approve or reject it.

## Four-class detector schema

| ID | Class | Role |
|---:|---|---|
| 0 | `apple` | Tight macro parent box/mask around one visible fruit |
| 1 | `stem_calyx` | Anatomical exclusion zone (stem, calyx) — prevents false rot positives |
| 2 | `defect_surface` | Non-decay cosmetic marks (russeting, limb rub, healed hail) |
| 3 | `defect_critical` | Structural & fungal damage (active rot, wet lesions, open punctures) |

Why a critical/surface split instead of one generic defect class:

- `defect_critical` (any amount) forces `DISCARD` — open rot must never reach fresh or cider bins;
- `defect_surface` coverage drives the G1/G2/G3/CIDER ladder — russeting and hail are cosmetic, not safety;
- `stem_calyx` isolation prevents the dark calyx/stem recess from being misclassified as rot;
- the split makes the deterministic decision tree auditable: a DISCARD is always traceable to a critical defect, not an unexplainable coverage number.

Annotators still box every visible defect on G1 fruit. Omitting tolerated G1 defects would teach the detector that grade controls whether a defect exists, which is label leakage.

## Deterministic grading

`grading_engine.py` is the pure decision core. It:

- binds each `defect_surface` and `defect_critical` to the apple with the highest child Intersection-over-Area (IoA) above the configured threshold;
- forces `DISCARD` when any `defect_critical` is bound to an apple;
- otherwise clips `defect_surface` boxes to the parent apple and computes the geometric union so overlap is not counted twice;
- computes `R_surf = union_coverage / apple_area * 100` and maps it to G1/G2/G3/CIDER via thresholds from `grading_policy.yaml`;
- excludes `stem_calyx` detections from defect counting (anatomical, not damage);
- marks box coverage near a decision threshold as requiring refinement;
- supports an optional externally measured segmentation coverage value;
- aggregates a complete five-view profile using the worst visible grade.

The decision tree:

```text
any defect_critical bound to apple  ->  DISCARD
else R_surf = Area(defect_surface union) / Area(apple) * 100
    R_surf < 2%   ->  G1   (U.S. Extra Fancy)
    R_surf < 10%  ->  G2   (U.S. Fancy / No. 1)
    R_surf < 25%  ->  G3   (U.S. Utility / Processing)
    R_surf >= 25% ->  CIDER (sound, fails fresh market — still pressable)
```

The committed thresholds are candidate calibration values, not claims that 2%, 10%, and 25% match any USDA standard. They remain candidates until held-out known-grade profiles validate them.

## Detection first, segmentation by evidence

Bounding boxes are the v1 annotation format. They are fast to label and sufficient to test whether defect geometry predicts grade. The local SAM 2 -> YOLO-Seg pipeline (`scripts/sam2_annotate.py`) converts box prompts into high-precision polygon masks using a frozen SAM 2 teacher — no SFT on SAM 2 itself.

Boxes overestimate irregular defect area. The policy therefore exposes a refinement margin around G1/G2, G2/G3, and G3/CIDER boundaries. Selective segmentation is a gated experiment, not a current production dependency:

1. establish the box-only confusion matrix on untouched profiles;
2. identify whether boundary errors are materially caused by box-area bias;
3. annotate masks only for those boundary cases via the SAM 2 pipeline;
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
| `local_inference.py` | Camera, glove masking, YOLO inference, binding, rendering, harvest, override |
| `glove_masking.py` | Blue nitrile glove HSV masking and defect suppression |
| `edge_harvest_schema.py` | Typed review telemetry contract |
| `vlm_review_schema.py` | Advisory-only VLM proposal contract |
| `grading_policy.yaml` | Versioned facility calibration candidates |
| `data.yaml` | Four-class YOLO dataset schema and profile split files |
| `scripts/split_profiles.py` | Deterministic profile-level split generation |
| `scripts/reingest_harvest.py` | Human-approved review re-ingestion only |
| `scripts/sam2_annotate.py` | Frozen SAM 2 -> YOLO-Seg annotation pipeline |
| `docs/annotation_sop.md` | Labeling and review standard |
| `docs/production_architecture.md` | Authority boundaries and production-line design |
| `docs/roadmap.md` | Staged evidence gates, done criteria, and pivot criteria |
| `tests/` | Hardware-independent regression tests |

## Current status

Implemented and testable now:

- five-view capture metadata contract;
- four-class vision taxonomy (apple / stem_calyx / defect_surface / defect_critical);
- five-grade output topology (G1 / G2 / G3 / CIDER / DISCARD) with USDA mapping;
- profile-isolated deterministic splitter;
- defect-critical-forces-DISCARD and surface-coverage-ratio grade engine;
- blue nitrile glove masking and defect suppression;
- local SAM 2 -> YOLO-Seg annotation pipeline scaffold;
- typed human review and VLM advisory records;
- operator override and typed edge-harvest wiring.

Not yet validated or implemented as production capability:

- a trained apple checkpoint;
- SAM 2 annotation pipeline execution on real captures;
- profile-level accuracy targets on an untouched holdout;
- selective segmentation model;
- Gemini API reviewer execution;
- checkpoint registry and promotion automation;
- sustained physical production-line trial.

See `docs/roadmap.md` for the order of operations. Do not skip from camera demo to production claims.

## Licensing

Ultralytics YOLO26 is AGPL-3.0. Commercial or closed deployment requires a licensing review before customer use. The repository license does not waive upstream obligations.
