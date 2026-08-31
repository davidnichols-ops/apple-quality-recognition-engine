# Annotation SOP — Three-Class Apple Grading Dataset

## Scope

This SOP governs annotation of the five-view profiles produced by `capture_dataset.py`. It separates the trusted reference grade from the visual labels used to train YOLO26.

A profile contains four equatorial views and one inverted calyx view of the same physical apple. Every view shares one `profile_id`. The reference grade is stored in `capture_manifest.jsonl`; it is never encoded as a detector class.

## Non-negotiable rule

**Annotate what is visible, not what the reference grade suggests should be visible.**

A G1 apple may contain small tolerated anomalies. Box them. A G3 apple may have one large anomaly rather than many small ones. Box what is present. Hiding defects on G1 fruit or inventing defects on G3 fruit leaks grade information into the labels and destroys the validity of the deterministic calibration experiment.

## Class manifest

| ID | Name | Annotation role |
|---:|---|---|
| 0 | `apple` | One tight macro parent box around each visible apple |
| 1 | `unfit_bin_discard` | Tight box around a local discard signal or unfit region |
| 2 | `class_defect` | Tight child box around any visible surface anomaly |

Do not add defect taxonomies during the seed phase. A new class requires evidence that the generic class causes a specific, repeated grade error that count, coverage, or selective segmentation cannot resolve.

## Profile handling

1. Confirm that the five images share the same `profile_id` and reference grade.
2. Keep all five views together in annotation jobs when possible.
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

### `class_defect`

- Draw a tight axis-aligned box around each contiguous visible anomaly region.
- Separate disconnected regions, even if they appear to be the same defect type.
- Do not draw a defect box around the entire apple.
- Keep the box inside the apple where possible. Edge defects may cross the parent boundary only where the visible anomaly does.
- Do not classify bruise, russeting, rot, scab, or other taxonomy. They all use `class_defect` in v1.
- Do not label normal stem, calyx, specular glare, dust on the lens, backdrop texture, or hard shadow as a defect.

### `unfit_bin_discard`

- Use this class for an observable local condition that should override normal grade routing under the facility policy.
- Box the signal or affected region, not the full frame.
- Keep the box near or overlapping the apple it applies to. Runtime discard logic is per parent, not global.
- Do not use this class merely because the capture manifest says `DISCARD`; the visual trigger must be present.
- If the reference grade is DISCARD but no visible discard trigger exists, flag the profile for human adjudication instead of inventing a box.

## Geometry QA

Every `class_defect` must bind to a parent apple with child IoA at or above the candidate policy threshold:

```text
IoA = intersection(defect, apple) / area(defect)
```

The current candidate threshold is 0.10. Aim for full containment; 0.10 is a rejection floor, not an annotation target.

Before completing a job, verify:

- every visible apple has one parent box;
- every visible anomaly has one tight `class_defect` box regardless of reference grade;
- every child box intersects the correct apple;
- duplicate overlapping boxes do not describe the same anomaly;
- discard triggers are local to the affected apple;
- class names exactly match `data.yaml`;
- the five profile views remain grouped.

## Roboflow operating procedure

### Seed phase

1. Create an object-detection project with the three exact class names.
2. Upload profiles with `profile_id`, `reference_grade`, batch, cultivar, lot, and view type as metadata or tags.
3. Manually annotate a balanced seed set across G1, G2, G3, and DISCARD reference grades.
4. Include multiple lots, cultivars, lighting conditions, and defect presentations where available.
5. Review 100% of the seed annotations before the first training run.

A practical first seed is 200 profiles: approximately 50 per reference grade, yielding 1,000 views. This is a planning target, not proof of sufficiency.

### Label-assist phase

1. Train the first candidate only after seed QA.
2. Use Roboflow label assist to propose boxes on new profiles.
3. Human-review every proposed parent, defect, and discard box before acceptance.
4. Prioritize corrections on missed small defects, glare, calyx texture, edge defects, and overlapping anomalies.
5. Version the dataset after each approved annotation batch.
6. Never accept an automatically labeled batch without review.

### Split policy

Use a 70/20/10 profile-level split. Run `scripts/split_profiles.py` or use Roboflow grouping so all views of one `profile_id` stay in one partition. Audit the exported manifests before training.

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

Gemini 3.7 Flash may review all five views together and propose:

- a missed or questionable box;
- a suggested grade for comparison;
- a calibration hypothesis;
- a reason to request segmentation or human review.

Its output is advisory. It must be stored with `pending_human_review` status. It cannot directly edit Roboflow annotations, `grading_policy.yaml`, capture metadata, dataset versions, or model checkpoints.

## Promotion evidence

Each training dataset version must retain:

- capture manifest digest;
- Roboflow dataset/version identifier;
- exact profile split lists;
- annotation QA record;
- class counts and profiles per reference grade;
- model training configuration;
- held-out detection and profile-grade metrics;
- human promotion decision.

A high image count is not a release gate. Profile-level performance and line behavior are.
