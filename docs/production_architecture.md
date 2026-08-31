# Production Architecture

## Decision boundary

The live sorter is a bounded deterministic system. YOLO observations and optional segmentation measurements feed a versioned grading policy. A VLM may inspect evidence and create a proposal, but it is not a production control plane.

Authority is ordered as follows:

1. a qualified human defines the facility policy and supplies reference grades;
2. immutable capture and annotation records preserve observations;
3. the detector produces spatial observations;
4. deterministic code computes the operational grade;
5. a VLM proposes changes only to a human review queue;
6. a human approves datasets, policy revisions, and model promotion.

No component trains or swaps a model while the line is operating.

## Online path

```text
Arducam OV9782
  -> 1280x720 frame
  -> YOLO26 CoreML candidate at configured inference size
  -> class-name schema guard
  -> apple / class_defect / unfit_bin_discard boxes
  -> child IoA binding
  -> per-parent discard proximity
  -> clipped defect-box union coverage
  -> deterministic GradeDecision
  -> overlay + telemetry + actuator integration boundary
```

### Schema guard

The live path requires exactly:

```text
0 apple
1 unfit_bin_discard
2 class_defect
```

A COCO model or reordered schema enters benchmark mode. Benchmark frames can measure camera and inference speed, but their detections must not enter the review dataset and their grades must not be presented as valid.

### Parent-child geometry

A `class_defect` is assigned to the apple with the greatest child IoA when the score meets the policy threshold. Orphan child boxes are not graded and should enter review telemetry.

Coverage is the union of child boxes clipped to the apple parent. This prevents two overlapping predictions from counting the same pixels twice. It does not solve irregular-shape box bias; that is the selective-segmentation decision below.

`unfit_bin_discard` is evaluated per apple. A trigger near apple A must not discard apple B. If a production line can emit a discard trigger with no parent apple, the actuator contract must define explicit handling before deployment; the current grade object is parent-scoped.

### Five-view aggregation

The production target is one profile per physical apple:

- views 0-3: equatorial views at approximately 90-degree intervals;
- view 4: calyx view;
- final grade: worst visible deterministic grade across the complete profile;
- incomplete profile: review required, not silently treated as complete.

The current camera capture script implements the acquisition profile. The live loop still grades individual frames; profile tracking and actuator integration remain a later milestone and must not be described as complete.

## Deterministic policy

The policy inputs are:

- count of valid bound child boxes;
- union coverage percentage;
- local discard trigger;
- optional refined segmentation coverage;
- policy version and facility identity.

The current candidate rules are intentionally simple:

- G1 below the G2 count and coverage boundaries;
- G2 at or above the G2 count or coverage boundary;
- G3 above the configured G2 defect-count ceiling or at/above the G3 coverage boundary;
- DISCARD when the local override trigger applies.

These are hypotheses to calibrate against known-grade profiles. They are not universal apple standards.

## Selective segmentation gate

Segmentation is measurement refinement, not a new grading authority.

The box-only decision marks coverage within `refinement_margin_pct` of either threshold. Those cases may be sent to a future small segmentation refiner on an apple or defect crop. If a valid refined coverage value is returned, the same deterministic policy runs again using that measurement.

Do not add full-dataset mask annotation or a production mask model until all conditions hold:

- box-only holdout evidence identifies boundary coverage as a material error source;
- a mask pilot is independently reviewed;
- the refiner improves profile-grade performance on untouched profiles;
- edge latency remains inside the line budget;
- fallback behavior is defined for timeout or invalid masks.

## Offline review path

```text
volatile confidence / boundary flag / operator disagreement
  -> date-organized frame + typed telemetry
  -> optional local triage
  -> Gemini 3.7 Flash multi-view advisory
  -> VLMReviewProposal(status=pending_human_review)
  -> human decision
  -> approved annotation queue or rejected proposal
```

Gemini 3.7 Flash is the planned primary cloud advisor. The rationale is multimodal multi-image input, structured output, and batch economics. It is not assumed to be more accurate than a domain grader. Its deployment must be measured on a labeled review set before its suggestions influence operator priority.

A local Qwen vision model may later triage obvious cases to reduce cloud volume, but adding a second reviewer before the first review process has measured value would add complexity without evidence.

The VLM may propose:

- a missed child box;
- a questionable class assignment;
- an alternate grade for human comparison;
- a policy calibration hypothesis;
- a request for segmentation.

The VLM may not:

- overwrite the deterministic live grade;
- accept its own annotations;
- edit the policy file;
- move images into a training version;
- start training;
- promote or hot-swap a checkpoint.

## Dataset and leakage controls

The unit of independence is a physical apple profile, not an image. All five views stay in one partition. The deterministic splitter hashes `profile_id`; Roboflow exports must be audited against the same rule.

Reference grade is evaluation metadata. It does not replace image annotation and must not determine whether annotators draw a defect. The test partition stays untouched by threshold calibration, prompt tuning, class redesign, and model selection.

Track distribution by:

- reference grade;
- cultivar;
- lot and supplier;
- capture day;
- camera and lighting setup;
- facility policy version;
- grader identity;
- review and adjudication status.

## Model lifecycle

Every candidate checkpoint must be linked to:

- immutable dataset/version identifier;
- split manifest digest;
- model family and size;
- training configuration and seed;
- per-class detection metrics;
- profile-grade confusion matrix;
- discard recall and false-discard rate;
- calibration policy version;
- M4 latency and sustained line trial results;
- human approver and promotion timestamp.

YOLO26x is the accuracy-ceiling candidate, not an automatic deployment choice. Benchmark n/s/m/l/x CoreML exports and deploy the smallest candidate that meets the measured grade and throughput requirements. A larger model that misses the line cycle budget is not production-ready.

## Failure handling

- Missing or malformed grading policy: fail startup; do not silently load hidden defaults.
- Model schema mismatch: benchmark mode, no trusted grading or harvest.
- Camera disconnect: bounded reconnect attempts, then safe stop.
- Incomplete five-view profile: review required.
- Orphan defect: telemetry review.
- Boundary coverage with unavailable segmentation: retain deterministic box grade and mark for review.
- VLM timeout or malformed output: retain deterministic grade; no state change.
- Candidate regression: reject promotion and keep the current production artifact.

## Security, privacy, and licensing

Cloud review sends images outside the edge device. Customer contracts must explicitly allow that path, or cloud review remains disabled. API credentials belong in protected environment configuration and never in the repository or telemetry.

Ultralytics AGPL obligations and any commercial license must be resolved before closed commercial deployment. Model and dataset licensing are release gates, not post-launch paperwork.
