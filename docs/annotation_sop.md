# Annotation SOP — Three-Class Apple Grading Dataset

## Scope

This SOP governs annotation of four-view equatorial profiles and separately captured stem/calyx views from `capture_dataset.py`. It separates the trusted reference grade from the visual labels used to train YOLO26-Seg.

A profile contains four equatorial views of one physical apple. Stem/calyx views are captured in a separate pass and must be linked to the same physical apple before joining a profile or split. The reference grade is stored in `capture_manifest.jsonl`; it is never encoded as a detector class.

## Non-negotiable rule

**Annotate what is visible, not what the reference grade suggests should be visible.**

A G1 apple may contain small tolerated anomalies. Box them. A G3 apple may have one large anomaly rather than many small ones. Box what is present. Hiding defects on G1 fruit or inventing defects on G3 fruit leaks grade information into the labels and destroys the validity of the deterministic calibration experiment.

## Class manifest

| ID | Name | Annotation role |
|---:|---|---|
| 0 | `apple` | One tight macro parent box around each visible apple |
| 1 | `stem_calyx` | Tight box around the stem or calyx recess (anatomical exclusion) |
| 2 | `defect_surface` | Tight child box around non-decay cosmetic marks (russeting, limb rub, hail) |
| 3 | `defect_critical` | Tight child box around structural/fungal damage (active rot, wet lesions, punctures) |

The critical/surface split is not a defect taxonomy for its own sake. `defect_critical` (any amount) forces `DISCARD`; `defect_surface` coverage drives the G1/G2/G3/CIDER ladder. `stem_calyx` isolation prevents the dark calyx/stem recess from being misclassified as rot.

## Profile handling

1. Confirm that the four equatorial images share the same `profile_id` and reference grade; verify the physical-apple link for any separate stem/calyx view.
2. Keep all linked views together in annotation jobs when possible.
3. Apply the same interpretation rules across all views, but annotate each image independently.
4. Do not copy a hidden defect into a view where it is not visible.
5. Never split views from one profile across train, validation, or test.
6. Treat duplicate or missing views as a review issue; do not silently substitute an image from another profile.

## Box rules

### `apple`

- Draw one axis-aligned box per visible fruit.
- Fit the visible fruit body tightly; do not include excess turntable or backdrop.
- For partial occlusion, box the visible fruit extent and flag the profile for review if the fruit cannot be graded reliably.
- Never group multiple apples in one box.

### `defect_surface`

- Draw a tight axis-aligned box around each contiguous visible non-decay anomaly region (russeting, limb rub, healed hail, scab patches).
- Separate disconnected regions, even if they appear to be the same defect type.
- Do not draw a defect box around the entire apple.
- Keep the box inside the apple where possible. Edge defects may cross the parent boundary only where the visible anomaly does.
- Do not label normal stem, calyx, specular glare, dust on the lens, backdrop texture, or hard shadow as a defect.

### `defect_critical`

- Draw a tight box around any structural or fungal damage: active rot, wet lesions, open punctures, insect boring, soft decay.
- Any `defect_critical` bound to an apple forces `DISCARD` regardless of surface coverage. Annotate conservatively — a false critical is a false discard.
- If the reference grade is DISCARD but no visible critical defect exists, flag the profile for human adjudication instead of inventing a box.
- Do not use `defect_critical` for cosmetic russeting or healed hail — those are `defect_surface`.

### `stem_calyx`

- Box the stem recess and the calyx (blossom end) dark hole.
- This class exists to prevent the detector from mistaking these anatomical structures for rot. It is an exclusion zone, not a defect.
- Do not label stem_calyx as a defect of any kind.

## Geometry QA

Every `defect_surface` and `defect_critical` must bind to a parent apple with child IoA at or above the candidate policy threshold:

```text
IoA = intersection(defect, apple) / area(defect)
```

The current candidate threshold is 0.10. Aim for full containment; 0.10 is a rejection floor, not an annotation target.

Before completing a job, verify:

- every visible apple has one parent box;
- every visible surface anomaly has one tight `defect_surface` box regardless of reference grade;
- every visible critical defect has one tight `defect_critical` box;
- every child box intersects the correct apple;
- duplicate overlapping boxes do not describe the same anomaly;
- stem/calyx recesses are labeled `stem_calyx`, not as defects;
- class names exactly match `data.yaml`;
- all linked profile views remain grouped.

## Local SAM 2 annotation procedure

### Seed phase (production pipeline: `scripts/distill_labels.py`)

1. Capture profiles with `profile_id`, `reference_grade`, batch, cultivar, lot, and view type metadata.
2. `pass-a` — auto-labels `apple` + `stem_calyx` on every sequence with zero human input (classical threshold bootstraps the apple box; a greenest-point heuristic seeds the calyx; SAM 2 video propagates across the rotation). ~4.2 s/frame at `sam2.1_hiera_large` on M4 MPS.
3. `pass-b` — interactive defect prompting (run in Terminal; OpenCV GUI). Step frames with `a`/`d`, select `2`=`defect_surface` or `3`=`defect_critical`, drag loose boxes. Prompts may be dropped on ANY frame — required for defects that rotate into view mid-orbit.
4. `pass-c` — propagates defect prompts with multi-frame conditioning into the per-frame `.npz` mask store.
5. `export --overlays` — applies the priority merge (`defect_critical` > `stem_calyx` > `defect_surface`, everything clipped inside the apple mask) and writes YOLO-Seg `.txt` plus review renders in `dataset/preview/`.
6. `split` — deterministic 80/20 train/val by `crc32(sequence_key) % 5`; all views of one apple stay in one partition. Negative frames (empty labels) are forced into train only.
7. Review 100% of the seed annotations in `dataset/preview/` before the first training run.

Known edge cases:

- Mid-orbit disocclusion: a defect first visible at frame 3 must be prompted AT frame 3; SAM 2 back-propagates onto earlier frames only if the feature is actually visible there. Check overlays on pre-prompt frames for ghost activations (mitigated by the >0 logit floor and the 25 px contour floor).
- MPS bf16 crash: multi-frame conditioning on MPS aborts in `MPSNDArrayMatrixMultiplication` because `maskmem_features` is hard-cast to bfloat16 in `sam2_video_predictor.py`. `distill_labels.propagate()` shadows `torch.bfloat16`->`float32` when `predictor.device.type == "mps"` (device reports `mps:0`, not `mps`).

A practical first seed is 200 profiles: approximately 40 per reference grade, yielding 1,000 views. This is a planning target, not proof of sufficiency.

### Background (negative) samples

Capture **5–10% of the training set as pure black background frames** — the matte backdrop with no apple present, no gloves, no turntable label. These are YOLO negative samples: empty `.txt` label files (zero annotations) paired with the image.

Why this matters:

- the detector learns that the matte black backdrop is not `defect_surface` or `defect_critical`;
- without negatives, the model can hallucinate defects in empty frame regions or at the apple/backdrop boundary;
- false defects on the backdrop inflate coverage ratios and push sound fruit into G2/G3/CIDER bins.

Procedure:

1. Capture 50–100 frames of the empty matte black backdrop under the same lighting and camera settings as production capture.
2. Include slight camera angle variation and a few frames with the turntable label visible.
3. Create empty label files: `touch apple_dataset/labels/<stem>.txt` (zero bytes = no objects).
4. Add the image paths to the train split list alongside the profile images.
5. Do **not** add background samples to the validation or test splits — those partitions measure apple grading performance, not backdrop rejection.

### Label-assist phase

1. Train the first YOLO-Seg candidate only after seed QA.
2. Use the trained candidate to propose boxes on new profiles, then feed those through the SAM 2 pipeline.
3. Human-review every proposed parent, defect, stem_calyx, and critical box before acceptance.
4. Prioritize corrections on missed small defects, glare, calyx texture, edge defects, and overlapping anomalies.
5. Version the dataset after each approved annotation batch.
6. Never accept an automatically labeled batch without review.

### Split policy

Use a 70/20/10 profile-level split. Run `scripts/split_profiles.py` so all views of one `profile_id` stay in one partition. Audit the exported manifests before training.

Stratify by reference grade and inspect representation by cultivar, lot, capture day, camera, and lighting. The untouched test set must not be used to tune thresholds, prompts, or annotation rules.

## Segmentation policy

Do not polygon-annotate the full dataset by default. First measure box-only grade errors on the untouched profile holdout.

Create a separate mask pilot only when evidence shows that irregular box-area overestimation is a dominant source of G1/G2 or G2/G3 mistakes. Select boundary profiles, annotate masks for the anomaly pixels, and compare the additional accuracy against annotation time and M4 latency. Segmentation remains optional until that experiment passes its promotion gate.

## Reference-grade disagreement

The facility grade is a reference outcome, not permission to falsify visual labels. When visible evidence and the supplied grade disagree:

1. preserve the image and original reference grade;
2. mark the profile for adjudication;
3. record grader, facility policy version, and reason;
4. have a second qualified human resolve the disagreement;
5. never let a VLM resolve it automatically.

## VLM advisory review

Gemini 3.7 Flash may review all linked views together and propose:

- a missed or questionable box;
- a suggested grade for comparison;
- a calibration hypothesis;
- a reason to request segmentation or human review.

Its output is advisory. It must be stored with `pending_human_review` status. It cannot directly edit Roboflow annotations, `grading_policy.yaml`, capture metadata, dataset versions, or model checkpoints.

## Promotion evidence

Each training dataset version must retain:

- capture manifest digest;
- dataset/version identifier (local SAM 2 pipeline output digest);
- exact profile split lists;
- annotation QA record;
- class counts and profiles per reference grade;
- model training configuration;
- held-out detection and profile-grade metrics;
- human promotion decision.

A high image count is not a release gate. Profile-level performance and line behavior are.
