# Apple Quality Recognition Engine

An edge-first system for grading apples on Apple Silicon. The intended primary model is a human-labeled YOLO26-Seg model that locates apples and segments `defect_surface` and `defect_critical`. A versioned policy turns those observations into a grade. The first operational product is a human-readable case review and annotation panel.

**Models observe. Deterministic policy decides. Humans authorize learning and promotion.**

This repository is an engineering prototype, not a validated commercial grader. No trained apple YOLO26-Seg checkpoint or control panel is present yet. Production claims begin only after a custom model, a profile-isolated holdout, and a real line trial pass the gates in `docs/roadmap.md`.

## Current build target

Build the annotation and case-review loop around real captured apples. A human assigns the critical/surface class and corrects SAM 2-assisted masks. Each case should keep all views, original images, predictions, corrected labels, reasons, model and policy versions, and an append-only review history in a format a person or vision-language agent can inspect. The panel should expose and edit those records. Approved records become versioned training data; model or agent suggestions never silently replace the human decision. Logging edge cases is the first use case; future reinforcement learning is a research option, not an implemented training path.

Depth Anything V2 is an optional diagnostic on YOLO26-Seg regions. It joins the decision flow only if a held-out, apple-level comparison demonstrates consistent incremental value on bites, surface marks, and normal stem/calyx anatomy. The local bite probe is exploratory evidence, not validation of that collaboration.

## Product contract

The system separates four kinds of truth:

1. **Reference grade** — G1, G2, G3, CIDER, or DISCARD assigned before capture by a trusted grader under a named facility policy. It is profile metadata, not a YOLO class.
2. **Visual observations** — four YOLO detection classes: `apple`, `stem_calyx`, `defect_surface`, and `defect_critical`.
3. **Deterministic decision** — grade derived from bound defect coverage ratios and the presence of any critical defect, via a versioned YAML policy.
4. **Advisory evidence** — Depth Anything V2 and a future VLM reviewer may propose refinements only after evaluation. Neither may silently change a grade, policy, annotation, dataset, or checkpoint.

## Production flow

```text
Known-grade apple
  -> four equatorial views; separate stem/calyx capture
  -> profile manifest
  -> human class labels and SAM 2-assisted masks, corrected by a human
  -> YOLO-Seg label conversion
  -> profile-level train/val/test split
  -> YOLO26-Seg candidate training
  -> held-out evaluation + M4 benchmark
  -> human checkpoint promotion

Live apple
  -> blue nitrile glove masking (HSV, <1ms CPU)
  -> YOLO26-Seg CoreML candidate
  -> apple + stem_calyx + defect_surface + defect_critical regions
  -> IoA spatial binding
  -> any defect_critical -> DISCARD
  -> else clipped defect_surface union coverage -> R_surf
  -> deterministic per-view grade
  -> worst visible grade across the required captured profile views
  -> G1 / G2 / G3 / CIDER / DISCARD
  -> inspectable case queue for uncertainty, edge cases, and human corrections
```

The VLM is outside the real-time authority path. Gemini 3.7 Flash is a planned offline reviewer, not an integrated service. `vlm_review_schema.py` defines a pending proposal that a human must approve or reject.

## Four-class detector schema

| ID | Class | Role |
|---:|---|---|
| 0 | `apple` | Tight macro parent box/mask around one visible fruit |
| 1 | `stem_calyx` | Anatomical exclusion zone (stem, calyx) — prevents false rot positives |
| 2 | `defect_surface` | Non-decay cosmetic marks (russeting, limb rub, healed hail) |
| 3 | `defect_critical` | Structural & fungal damage (active rot, wet lesions, open punctures) |

Why a critical/surface split instead of one generic defect class:

- `defect_critical` (any amount) forces `DISCARD` under the current candidate policy;
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
- aggregates a complete required-view profile using the worst visible grade.

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

## Segmentation and annotation

The current grading engine uses boxes and can accept externally measured segmentation coverage. The target detector is YOLO26-Seg. The local SAM 2 -> YOLO-Seg pipeline (`scripts/distill_labels.py`) creates candidate polygons from prompts using a frozen SAM 2 teacher. A human must review masks and assign the critical/surface class. The existing distillation run is an experiment on a small local capture, not a production training set.

Boxes overestimate irregular defect area. The policy therefore exposes a refinement margin around G1/G2, G2/G3, and G3/CIDER boundaries. Before using segmentation coverage for grading:

1. establish the box-only confusion matrix on untouched profiles;
2. identify whether boundary errors are materially caused by box-area bias;
3. compare human-reviewed YOLO26-Seg masks against the box baseline;
4. use mask coverage only if it improves profile grade accuracy enough to justify its latency and labeling cost.

## Depth evidence probe

The current Arducam provides RGB frames. `scripts/depth_probe.py` compares
YOLO26 depth variants and Depth Anything V2 as **diagnostics** alongside
segmentation polygons. It writes ignored local depth maps and contrast reports
under `dataset/depth_probes/<sequence>/`. It does not change labels or grades.

```bash
.venv/bin/python scripts/depth_probe.py --sequence test-20260918 \
  --backend yolo26n --mask-source yolo-seg --device mps --aoi apple
```

YOLO26 checkpoints belong in `checkpoints/yolo26n-depth.pt` or
`checkpoints/yolo26l-depth.pt`; choose `--backend yolo26n` or `--backend yolo26l`.
The probe also supports `--backend depth-anything-v2` for comparison; that model's source and
revision are recorded in `third_party/DEPTH_ANYTHING_V2_SOURCE.txt`. A future
trained YOLO-Seg model can feed the same polygon format. A single RGB depth
estimate is not a physical measurement. An open bite, puncture, or rot remains
`defect_critical` when confirmed by a human, regardless of the depth map.
`--aoi apple` crops to the segmented fruit with a 10% margin, saves a comparison
image with defect contours, and maps the result back into the original frame.
The model's global distance scale shifts under this crop; use local contrast
only when comparing AOI and full-frame outputs. Framing can even reverse the
sign of a defect residual on a pretrained monocular model. Choose full-frame
or AOI per model using held-out labeled fruit, not from one example.
The geometry probe then fits a tilted ellipsoid to intact apple skin, excluding
all proposed defects and the normal `stem_calyx` region. It reports residuals
against that fruit curve: `Dfit = a + bx + cy + d(1 - sqrt(1 - r²))`, where
`r²` is the normalized elliptical distance from the apple centre. Positive
`Dmodel - Dfit` means the model sees a possible recess. Depth Anything V2's
near-high relative output is negated before this fit; its units remain relative.
Candidates mostly inside a small calyx safety rim
are flagged as normal anatomy and excluded from defect scoring; partially
overlapping candidates are scored only outside that rim. The calyx residual is
retained as a sanity control for the depth model, not as a defect label.
The one-bite local comparison suggested better visual bite localization from V2
than YOLO26-L depth, but neither has passed a representative held-out test.
Promote collaboration only after measured bite and intact-surface controls show
an incremental gain over YOLO26-Seg alone on unseen fruit.

## Dataset protocol

A profile is one physical apple and its four equatorial views. Stem/calyx views are captured in a separate pass. The capture script records a stable `profile_id`, reference grade, grader, facility, batch, lot, cultivar, camera, and view metadata in JSONL.

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

All views of one physical apple receive the same split, including any separately captured stem/calyx views linked to that apple. Never let frames from one apple cross train, validation, and test boundaries. Reference grade should be stratified and audited at the profile level before training.

The working scale hypothesis is 3,000-6,000 profiles (15,000-30,000 images), but image count does not prove sufficiency. Advancement depends on held-out per-grade performance, discard recall, calibration stability, cultivar/lot generalization, and edge latency.

## Human-governed improvement loop

```text
low confidence / boundary / operator disagreement
  -> inspectable case record with original prediction and append-only edits
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

`local_inference.py` enters benchmark mode if model names do not exactly match the four-class schema. In benchmark mode it renders FPS only; grading and edge harvest remain disabled. A missing candidate model also fails closed unless `--benchmark-fallback` is explicitly supplied.

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
| `capture_dataset.py` | Four-view equatorial capture, separate stem capture, and JSONL profile manifest |
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

- four-view equatorial capture and separate stem/calyx capture;
- four-class vision taxonomy (apple / stem_calyx / defect_surface / defect_critical);
- five-grade output topology (G1 / G2 / G3 / CIDER / DISCARD) with USDA mapping;
- profile-isolated deterministic splitter;
- defect-critical-forces-DISCARD and surface-coverage-ratio grade engine;
- blue nitrile glove masking and defect suppression;
- local SAM 2 -> YOLO-Seg distillation pipeline, exercised on a small real capture;
- experimental depth comparison and apple-curve geometry probe;
- typed human review and VLM advisory records;
- operator override and typed edge-harvest wiring.

Not yet validated or implemented as production capability:

- a trained apple checkpoint;
- a reviewed, sufficiently varied apple dataset and apple-level holdout;
- a unified human-readable/editable annotation and edge-case control panel;
- evidence that depth improves critical/surface decisions over YOLO26-Seg alone;
- profile-level accuracy targets on an untouched holdout;
- a trained apple YOLO26-Seg model and measured mask-coverage benefit;
- Gemini API reviewer execution;
- checkpoint registry and promotion automation;
- sustained physical production-line trial.

See `docs/roadmap.md` for the order of operations. Do not skip from camera demo to production claims.

## Licensing

Ultralytics YOLO26 is AGPL-3.0. Commercial or closed deployment requires a licensing review before customer use. The repository license does not waive upstream obligations.
