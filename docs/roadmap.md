# Evidence-Gated Roadmap

This roadmap replaces the idea of live recursive self-improvement with a governed learning loop. Stages advance on evidence, not calendar time or raw image count. The current build target is a human-editable annotation and edge-case panel; manually reviewed YOLO26-Seg labels are the training source of truth. Depth Anything V2 is a candidate supporting signal and requires a measured gain over YOLO26-Seg alone.

## North-star outcome

Demonstrate that a four-equatorial-view, four-class YOLO26-Seg detector plus deterministic policy can reproduce trusted facility grades on previously unseen apple profiles at the required line throughput, with separately captured stem/calyx views, traceable decisions, and a safe human review loop.

## Global release targets

These are target gates, not achieved results:

- no physical profile crosses train, validation, and test partitions;
- per-grade profile confusion matrix is reported, not only aggregate accuracy;
- DISCARD recall target is at least 99% with false-discard rate below 1%;
- exact G1/G2/G3/CIDER profile agreement target is at least 95% on the untouched test set;
- no single grade may hide behind class imbalance; each grade target is reported separately;
- sustained M4 inference meets the measured physical line cycle with thermal and camera stability evidence;
- every production model and policy has a reversible versioned promotion record;
- no VLM output reaches production without human approval.

Targets may be revised only with a documented customer or facility requirement.

## Stage 0 — architecture and controls

### Deliverables

- four-class `data.yaml`;
- versioned surface-ratio coverage policy;
- four-view equatorial capture manifest and separate stem/calyx capture;
- profile-level deterministic splitting;
- pure geometry and grading tests;
- typed review queue and VLM proposal contracts;
- human-gated re-ingestion;
- honest implementation/status documentation.

### Done when

- tests and lint pass in CI;
- one synthetic profile proves all linked views of an apple share its profile ID and split;
- geometry tests prove overlap is not double-counted;
- a discard trigger affects only the nearby parent;
- VLM proposals default to pending human review.

## Stage 1 — balanced seed and annotation contract

### Acquisition hypothesis

Capture approximately 200 profiles: about 40 reference-grade profiles each for G1, G2, G3, CIDER, and DISCARD. Four equatorial views produce roughly 800 seed images, with stem/calyx views captured separately. Add 5–10% pure black background frames as YOLO negative samples (see `docs/annotation_sop.md` → Background samples) so the detector does not hallucinate defects on the matte backdrop.

### Required evidence

- reference grade source, grader, facility policy, lot, cultivar, and capture batch recorded;
- 100% seed annotation review under `docs/annotation_sop.md`, including human assignment of critical versus surface and correction of SAM 2 masks;
- inspectable case records retaining original images, labels, corrections, reasons, and provenance;
- every visible defect boxed regardless of reference grade;
- background negative samples included in the train split (5–10% of total);
- profile split audit shows zero leakage;
- class and grade distribution report;
- duplicate and missing-view report.

### Done when

The seed is coherent enough to train a baseline and every questionable reference-grade mismatch has an adjudication status.

### Stop or repair when

- reference grades cannot be traced to a consistent facility policy;
- annotators use grade to decide whether to label defects;
- DISCARD labels are assigned without observable `defect_critical` triggers;
- profiles are missing IDs or views.

## Stage 2 — baseline detector and deterministic calibration

Train YOLO26-Seg candidates on the approved seed. Use YOLO26x-Seg as an accuracy ceiling and compare smaller variants for edge deployment. Compare a depth-assisted decision against this baseline only after a representative bite, surface-mark, and normal-anatomy holdout exists.

### Required evidence

- per-class precision, recall, and mAP for `apple`, `stem_calyx`, `defect_surface`, and `defect_critical`;
- per-view and per-profile grade confusion matrices;
- calibration performed on validation profiles only;
- untouched test profiles remain sealed;
- error slices for glare, calyx, stem, edge defects, cultivar, lot, and lighting;
- CoreML export parity and M4 sustained throughput.

### Done when

A baseline checkpoint and policy can be reproduced from an immutable dataset version and their errors are categorized well enough to choose the next data, policy, or model experiment.

### Pivot criteria

- If generic `defect_surface` recall is weak, improve data and annotation before adding sub-taxonomy.
- If grade errors cluster at coverage boundaries, compare mask-derived coverage with the box baseline.
- If grade errors depend on defect type despite reliable localization, test one evidence-backed class split rather than restoring eleven classes.
- If YOLO26x misses throughput, benchmark smaller variants before changing hardware.

## Stage 3 — label assist and scale toward 3,000 profiles

The trained YOLO-Seg candidate may propose annotations on new profiles, which are then refined through the frozen SAM 2 pipeline. Humans accept, correct, or reject every proposal. Each approved batch creates a new immutable dataset version.

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

## Stage 4 — segmentation coverage and optional depth experiment

Evaluate mask coverage if Stage 2 or 3 shows box-area bias is a material cause of boundary misgrading. Evaluate depth separately if YOLO26-Seg confuses bites with surface marks or normal stem/calyx anatomy.

### Experiment

- sample human-reviewed boundary profiles;
- use human-reviewed masks from the SAM 2-assisted annotation loop;
- run the trained YOLO26-Seg candidate;
- feed refined coverage into the unchanged deterministic grade function;
- compare against box-only decisions on untouched profiles;
- measure end-to-end M4 latency and fallback behavior.

### Promotion gate

Use mask coverage only if it produces a meaningful held-out profile-grade gain, preserves line throughput, and reduces rather than redistributes boundary errors. Add V2 only if it provides a consistent incremental improvement in critical/surface decisions over YOLO26-Seg alone, including normal stem/calyx controls.

## Stage 5 — Gemini 3.7 Flash advisory pilot

Gemini may review complete captured profiles from the human review queue, not the live stream.

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

- complete association of the four equatorial views and any separate stem/calyx view under line motion;
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
- the required view acquisition cannot fit the physical line cycle;
- discard risk cannot meet the safety target;
- commercial licensing or customer privacy constraints make deployment nonviable.
