# AQRE retirement decision — 2026-11-16

**Decision: retire active development of the current automated apple-grading objective. Preserve the source, partial dataset, experiments, and known limitations as a research prototype. Not suitable for operational grading.**

This is an engineering stop decision based on unmet requirements and the cost of closing them. It is not evidence that automated apple inspection is impossible, nor a claim that every possible approach was exhausted.

## Actual problem and operating context

The owner currently sorts by hand; apples are washed on a conveyor. The desired system must match or exceed human sorting speed while making useful grading decisions. No measured human throughput baseline or conveyor acceptance trial was obtained.

The turntable and four-view profiles were for dataset capture, not a production rotation requirement. Production needs persistent apple identity across the available conveyor frames, a decision deadline, and reliable clearing of state when an apple leaves. The live program does not implement those operations.

## Engineering delivered and evidence

| Work | What was implemented or exercised | Practical limit |
| --- | --- | --- |
| Capture and identity | `capture_dataset.py`, manifest metadata, four-view capture, separate anatomy cohort | Timed angles, no encoder; bench capture does not establish conveyor visibility |
| Capture contract repair | Commits `85f3f00` and `9f2bc66`: four-view split contract, calyx parsing, quarter-turn timing | Historical images retain their original acquisition limitations |
| Annotation tools | SAM-assisted masks, prompt propagation, YOLO label conversion and human review procedures | Tooling exists; complete reviewed production dataset does not |
| Grading core | Defect-parent binding, clipped union area, versioned thresholds, optional refined coverage, profile aggregation | Box geometry and candidate policy are not validated commercial standards |
| Live prototype | Camera reconnect, YOLO inference, multiple apple boxes, glove-overlap suppression, overlays | Frame-local decisions; no persistent apple tracking or physical sorter integration |
| Evidence and review contracts | Typed telemetry, persisted JPEG hashes, disagreement capture, human-approved re-ingestion and advisory VLM schema | No completed control panel; confident misses can escape automatic harvesting |
| Depth experiments | `scripts/depth_probe.py`, `scripts/apple_surface_geometry.py`, apple AOI and curved-surface residual experiments | No validated incremental grading benefit; not connected to live grading |
| Trained model smoke test | YOLO26s-Seg hosted inference on 127 images; adapter exercised grading on 120 views / 30 apples | Training overlap and profile leakage prevent generalization claims |
| Narrow mitigation | Commit `81d30f8`: candidate policy 3.1.0 requires review for zero-defect profiles | Flag only; live frame path does not consume it and it is not an actuator interlock |
| Regression checks | Hardware-independent tests and static checks, recorded in `../research/closeout/verification.txt` | Passing software tests do not establish model accuracy or line performance |

The implementation is preserved so readers can inspect the work rather than rely on a completion claim. Historical design documents describe intentions as well as code; this decision and the current source take precedence over their aspirational wording.

Closeout verification: 65 tests passed; the scoped Pyflakes check, compilation and diff checks passed. Repository-wide Ruff reported six pre-existing F541 findings in `lock_wb.py` (lines 82, 92, 102, 110, 120, 121). That file is byte-identical to HEAD and was not changed. This closeout does not claim a clean repository-wide lint result.

## What the preliminary test actually established

Model: `davids-workspace-o4qfg/test-demo-7aocj-1-yolo26s-seg-t2`, dataset version 1. The 127-image version contained 89 training, 25 validation, and 13 test images. Saved prediction payloads with remote image URLs removed, original summaries, frame tables, profile tables, and an overlay contact sheet are preserved in [the closeout evidence](../research/closeout/README.md).

- Hosted inference returned predictions for all 127 images. The adapter found apples in 120/120 apple frames and none in the seven negative frames, at the tested thresholds.
- Roboflow reported test surface recall 16.7% and critical precision/recall 50%/75%. These are small-sample, threshold-specific results, not rates established for production.
- Twenty-two of 30 physical apples had views in multiple partitions. The test is therefore not an independent apple-level holdout.
- Roboflow's critical and stem/calyx numeric IDs differed from the runtime contract. The offline adapter used class names; the live schema guard would disable grading for the reordered mapping.
- At Roboflow's per-class evaluation thresholds, the legacy policy produced 15 G1, one G2 and 14 DISCARD profile results. Fourteen of the 15 G1 profiles had no detected surface defects; the remaining one had approximately 0.69% visible mask coverage. These outputs are not verified grades.
- The smoke adapter uses segmentation-derived surface coverage. Live inference still uses bounding boxes. The adapter did not test the live camera, glove suppression, CoreML deployment or conveyor event lifecycle.

Corrections to earlier interpretation: Roboflow labeled its evaluation `object-detection-like` and returned no mIoU. Its mAP values must not be presented as demonstrated mask-IoU accuracy. Two critical predictions in one image do not prove two distinct physical wounds. Summing sightings across frames does not count unique defects. The per-class thresholds were evaluation-derived, not independently approved deployment settings.

## Why further investment stopped

1. **Required observations are missing.** The project owner reports that bruising and size imperfections are not detected in this setup. This is a limitation of the current system, not a controlled finding that RGB or other sensors can never measure them. No calibrated size measurement or validated bruise assessment exists here.
2. **Natural anatomy remains difficult.** Stem/calyx is detectable but sometimes faulty. The live code excludes that class from defect counts; it does not geometrically suppress overlapping false-critical predictions using its anatomy region.
3. **Per-apple state is theoretical.** Tracking, defect association across frames, state reset, finalization, and removal timing remain unimplemented. Repeated frame decisions are not an apple-level conveyor decision.
4. **The policy is incomplete for the real task.** The owner specified two critical defects should yield CIDER. The preserved code still maps any critical to DISCARD. One-critical treatment and the revised discard boundary were not finalized. Reference-grade capture metadata alone cannot resolve this.
5. **More data is costly and insufficient by itself.** Tens of hours of capture and annotation would be needed for expansion. More labels might improve visible-defect recognition but do not by themselves supply hidden observations, calibrated sizing, tracking or a valid owner policy.
6. **Operational value is unproven.** No human speed comparison, acceptable review workload, sustained line trial or real sorting benefit was measured. The 95% grade-agreement and 99% discard-recall roadmap targets were not achieved.

The retirement rationale is the combination of these gaps and the uncertain return on the next investment. The weak pilot alone does not establish technical impossibility. Deferring further collection is justified because the next expense would not isolate or solve all the remaining requirements.

## What is preserved and what is not claimed

The partial release includes 120 original apple JPEGs, representing 30 four-view Crimson Crisp profiles, with SHA-256 inventory and capture-grade metadata. It excludes the 33 local background frames, stale manifest paths, model weights, and Roboflow annotation exports. Seven background-frame predictions remain in the 127-image smoke evidence. The archive is a research record, not a complete training distribution or an endorsed benchmark.

No production checkpoint was promoted by this closeout. No automated policy repair, new training, actuator integration, or retroactive ground truth was fabricated. Code is left at its tested research behavior for reproducibility. The GitHub repository remains readable; retiring the objective does not require deleting or locking it.

## Conditions that could justify reopening

Reconsider only with materially different resources or scope: a funded facility pilot, representative conveyor footage with consistent expert decisions, sensing/handling that exposes required evidence, or a narrower paid inspection-assistance task. First establish observability and owner rules; then measure whether inference improves the workflow. Dataset volume alone is not a reopening criterion.
