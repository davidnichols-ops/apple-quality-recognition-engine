# Evidence-Gated Roadmap

This roadmap replaces the idea of live recursive self-improvement with a governed learning loop. Stages advance on evidence, not calendar time or raw image count.

## North-star outcome

Demonstrate that a five-view, three-class detector plus deterministic policy can reproduce trusted facility grades on previously unseen apple profiles at the required line throughput, with traceable decisions and a safe human review loop.

## Global release targets

These are target gates, not achieved results:

- no physical profile crosses train, validation, and test partitions;
- per-grade profile confusion matrix is reported, not only aggregate accuracy;
- DISCARD recall target is at least 99% with false-discard rate below 1%;
- exact G1/G2/G3 profile agreement target is at least 95% on the untouched test set;
- no single grade may hide behind class imbalance; each grade target is reported separately;
- sustained M4 inference meets the measured physical line cycle with thermal and camera stability evidence;
- every production model and policy has a reversible versioned promotion record;
- no VLM output reaches production without human approval.

Targets may be revised only with a documented customer or facility requirement.

## Stage 0 — architecture and controls

### Deliverables

- three-class `data.yaml`;
- versioned count/coverage policy;
- five-view known-grade capture manifest;
- profile-level deterministic splitting;
- pure geometry and grading tests;
- typed review queue and VLM proposal contracts;
- human-gated re-ingestion;
- honest implementation/status documentation.

### Done when

- tests and lint pass in CI;
- one synthetic profile proves five views share a profile ID and split;
- geometry tests prove overlap is not double-counted;
- a discard trigger affects only the nearby parent;
- VLM proposals default to pending human review.

## Stage 1 — balanced seed and annotation contract

### Acquisition hypothesis

Capture approximately 200 profiles: about 50 reference-grade profiles each for G1, G2, G3, and DISCARD. Five views produce roughly 1,000 seed images.

### Required evidence

- reference grade source, grader, facility policy, lot, cultivar, and capture batch recorded;
- 100% seed annotation review under `docs/annotation_sop.md`;
- every visible defect boxed regardless of reference grade;
- profile split audit shows zero leakage;
- class and grade distribution report;
- duplicate and missing-view report.

### Done when

The seed is coherent enough to train a baseline and every questionable reference-grade mismatch has an adjudication status.

### Stop or repair when

- reference grades cannot be traced to a consistent facility policy;
- annotators use grade to decide whether to label defects;
- DISCARD labels are assigned without observable visual triggers;
- profiles are missing IDs or views.

## Stage 2 — baseline detector and deterministic calibration

Train YOLO26 candidates on the approved seed. Use YOLO26x as an accuracy ceiling and compare smaller variants for edge deployment.

### Required evidence

- per-class precision, recall, and mAP for `apple`, `class_defect`, and `unfit_bin_discard`;
- per-view and per-profile grade confusion matrices;
- calibration performed on validation profiles only;
- untouched test profiles remain sealed;
- error slices for glare, calyx, stem, edge defects, cultivar, lot, and lighting;
- CoreML export parity and M4 sustained throughput.

### Done when

A baseline checkpoint and policy can be reproduced from an immutable dataset version and their errors are categorized well enough to choose the next data, policy, or model experiment.

### Pivot criteria

- If generic `class_defect` recall is weak, improve data and annotation before adding taxonomy.
- If grade errors cluster at coverage boundaries, run the segmentation pilot.
- If grade errors depend on defect type despite reliable localization, test one evidence-backed class split rather than restoring eleven classes.
- If YOLO26x misses throughput, benchmark smaller variants before changing hardware.

## Stage 3 — label assist and scale toward 3,000 profiles

Roboflow label assist may propose annotations. Humans accept, correct, or reject every proposal. Each approved batch creates a new immutable dataset version.

### Sampling priority

- detector false negatives and low-confidence children;
- G1/G2 and G2/G3 boundary profiles;
- operator disagreements;
- underrepresented cultivars, lots, lighting, and defect shapes;
- DISCARD false negatives and false positives;
- profiles unlike the current training distribution.

### Required evidence per cycle

- new profile count by grade and domain slice;
- human correction rate for label assist;
- change in held-out performance by slice;
- regression comparison to the currently promoted model;
- explicit accept/reject decision.

### Done when

The system reaches the release targets or the error curve shows which architecture change is required. Reaching 3,000 profiles alone is not completion.

### Kill criteria

Stop adding random easy images when they no longer improve the weakest held-out slice. Shift acquisition to measured failure modes.

## Stage 4 — selective segmentation experiment

Run only if Stage 2 or 3 shows box-area bias is a material cause of boundary misgrading.

### Experiment

- sample human-reviewed boundary profiles;
- annotate pixel masks for visible anomalies;
- run a small crop-level segmentation candidate;
- feed refined coverage into the unchanged deterministic grade function;
- compare against box-only decisions on untouched profiles;
- measure end-to-end M4 latency and fallback behavior.

### Promotion gate

Promote segmentation only if it produces a meaningful held-out profile-grade gain, preserves line throughput, and reduces rather than redistributes boundary errors. Otherwise keep the box-only system.

## Stage 5 — Gemini 3.7 Flash advisory pilot

Gemini reviews complete five-view profiles from the human review queue, not the live stream.

### Structured proposal fields

- profile ID;
- deterministic grade and policy version;
- suggested grade;
- suspected missed annotations;
- confidence and rationale;
- optional policy hypothesis;
- `pending_human_review` status.

### Evaluation

Measure the advisor against human adjudication:

- precision of raised issues;
- missed-issue rate;
- human acceptance and correction rates;
- cost per useful accepted review;
- latency and API failure rate;
- performance by grade, cultivar, and defect presentation.

### Promotion gate

The VLM earns a review-prioritization role only if it saves human effort or catches meaningful errors at acceptable cost. It never becomes final grading authority and never self-approves its output.

## Stage 6 — scale toward 6,000 profiles only if justified

Continue collection when the weakest generalization slice remains data-limited and new approved profiles improve it. Do not double the dataset merely to hit a round number.

At this stage, freeze a geographically or temporally separated test cohort if possible. Validate across multiple lots, cultivars, graders, and operating conditions.

## Stage 7 — controlled line pilot

### Required evidence

- complete five-view profile association under line motion;
- actuator timing contract and fail-safe behavior;
- sustained camera and CoreML operation;
- throughput distribution, not only average FPS;
- false-discard and undergrade incident review;
- operator override usability;
- offline recovery after camera, model, or review service failure;
- licensing and customer data/privacy approval.

### Done when

A named facility owner signs off on measured line performance and rollback behavior for a specific model, policy, camera configuration, and dataset lineage.

## Model and policy promotion

Every promotion is offline and human-authorized:

1. freeze dataset and split manifests;
2. train candidate with recorded configuration;
3. run detection, grade, runtime, and regression evaluations;
4. compare candidate to current production artifact;
5. document failures and slice metrics;
6. approve or reject candidate;
7. deploy versioned artifact and policy together;
8. retain rollback path.

No process may save a live-trained checkpoint into production. No VLM may edit the promotion record.

## Definition of project success

The project succeeds when it produces independently reviewable evidence that the bounded system grades unseen profiles at the facility's required accuracy and speed, with safe fallback and repeatable promotion.

## Project-level pivot or stop conditions

Reconsider the architecture if any of these remain true after targeted data and calibration:

- reference grades are too inconsistent to define a learnable target;
- visible RGB surface evidence cannot separate required grades;
- generic defect geometry cannot meet the target and evidence-backed taxonomy or segmentation does not close the gap;
- the five-view acquisition cannot fit the physical line cycle;
- discard risk cannot meet the safety target;
- commercial licensing or customer privacy constraints make deployment nonviable.
