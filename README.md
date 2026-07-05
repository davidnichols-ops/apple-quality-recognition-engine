# Apple Quality Recognition Engine

Most agricultural vision systems fail for a structural reason, not a modeling one.

They ask neural networks to perform judgment.

**This system does not.**

The Apple Quality Recognition Engine is an **edge-first** vision pipeline built for Apple Silicon (M4). It separates perception from decision-making:

- **YOLO11** detects apples, discard triggers, and surface defects.
- A **deterministic grading engine** (driven by `grading_policy.yaml`) evaluates severity, spatial relationships, and coverage.

**Neural networks identify. Algorithms decide.**

This separation keeps operational logic out of model weights, making facility rule changes fast and reliable.

---

## Core System Principles (Current Status)

| Principle              | Status     | Description |
|------------------------|------------|-------------|
| Detection ≠ Decision   | Complete   | Neural network detects; deterministic logic grades. |
| Dynamic Schema         | Complete   | Defect taxonomy loads from policy file at runtime. |
| Facility Ground Truth  | Complete   | Grading rules defined in `grading_policy.yaml`. |
| Edge-Native Execution  | Complete   | Runs on Apple Neural Engine via CoreML / Ultralytics. |
| Hardware Hardening     | Complete   | Automatic camera reconnection on Arducam drops. Camera index auto-detection via `camera_utils.py` (VID/PID + native resolution match). |
| Operator Authority     | Complete   | Telemetry capture for human overrides. |
| Active Learning Loop   | Partial    | Low-confidence frames are harvested (full loop pending). |

---

## System Architecture

### Feature Detector Pipeline

Two-stage design:

1. **Neural detection** (YOLO11)
2. **Deterministic grading engine**

```
┌─────────────────────────────────────────────────────────────────┐
│                      MACRO PARENT BOX                           │
│                 Class 0: apple (instance root)                  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              MICRO DEFECT LAYER (Classes 2-N)             │  │
│  │   Bruise     Russet     Scab      Rot     Crack          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Flow (Implemented)

1. **Neural Stage** — YOLO11 detects apples (0), discard triggers (1), defects (2-N).
2. **Spatial Binding** — Defects bound to apples via Intersection-over-Area (IoA ≥ 0.10).
3. **Grading Engine** — Deterministic scoring from `compute_grade()` using policy file.
4. **Discard Override** — Class 1 triggers immediate rejection on proximity.
5. **Edge Harvesting** — Low-confidence detections (0.40–0.65) saved with telemetry.
6. **Operator Override** — Manual disagreement logging (key `g`).

---

## Dynamic Schema & Policy Engine

The system loads defect taxonomy and rules at runtime from `grading_policy.yaml`. No code changes needed when adding defects (only dataset + retrain + redeploy).

**Key files:**
- `grading_policy.yaml` — severity mapping + thresholds
- Runtime class discovery: `num_classes = len(model.names)`

### Schema Rules

* Class 0 → apple (root instance)
* Class 1 → unfit_bin_discard (override trigger)
* Class 2-N → defect taxonomy (dynamic)

Adding a new defect requires:

1. dataset update
2. retraining
3. redeployment

No inference changes required.

---

## Hardware & Deployment

### Target Stack
- **Host**: MacBook Air (Apple M4)
- **OS**: macOS 26 (Tahoe)
- **Camera**: Arducam USB Global Shutter
- **Acceleration**: Apple Neural Engine (CoreML)

### Camera Configuration
- Camera Index: 0
- Resolution: 1280x720
- Pipeline: MJPG native stream
- Backend: OpenCV (cv2.VideoCapture)

### Inference Configuration
- Sandbox: `yolo11n.pt` (validation only)
- Production: `best.mlpackage` (CoreML)
- Sandbox Resolution: 640px
- Production Resolution: 1024px
- Execution Backend: Apple Neural Engine (ANE)

---

## Class Dictionary

### Schema Overview

| Index | Class | Type | Function |
|-------|-------|------|----------|
| 0 | apple | Parent | Instance root |
| 1 | unfit_bin_discard | Override | Immediate rejection trigger |
| 2 | z_bruise | Defect | Mild |
| 3 | z_russeting | Defect | Mild |
| 4 | z_scarf_skin | Defect | Moderate |
| 5 | z_sunburn | Defect | Moderate |
| 6 | z_stem_puncture | Defect | Moderate |
| 7 | z_split_crack | Defect | Severe |
| 8 | z_misshapen | Defect | Severe |
| 9 | z_scab | Defect | Moderate |
| 10 | z_sooty_blotch_flyspeck | Defect | Moderate |
| 11 | z_rot | Defect | Severe |
| 12 | z_insect_damage | Defect | Severe |

---

## Algorithmic Grading Matrix

Grading is computed deterministically in `compute_grade()`.

### Inputs

* defect count
* defect severity weights
* area coverage ratio

### Grade Logic

| Grade | Condition | Output |
|-------|-----------|--------|
| G1 | no defects OR minimal mild defects | green |
| G2 | moderate defects OR 5–15% coverage | orange |
| G3 | structural defects OR >15% coverage | red |
| DISCARD | Class 1 proximity event | reject |

---

## Spatial Binding Engine

Defect-to-apple association is computed via IoA:

```
IoA = Intersection Area / Defect Area
```

Binding condition:

```
IoA ≥ 0.10 → attach defect to apple instance
```

This avoids centroid instability in dense cluster environments.

---

## Data Lifecycle Architecture

```
RAW CAPTURE
  → Arducam MJPG stream
  → dataset/raw_ingest/

MANUAL ANNOTATION
  → Roboflow labeling
  → Class 0: apple
  → Class 1: discard trigger
  → Class 2-N: defects

CLOUD TRAINING
  → YOLO11 (1024px)
  → Export CoreML (.mlpackage)

EDGE DEPLOYMENT
  → M4 Neural Engine
  → local_inference.py execution
  → real-time grading
```

---

## Operational Pipeline

### 1. Environment Setup

```bash
cd apple-quality-recognition-engine
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt
```

### 2. Hardware Verification

```bash
python baseline_verify.py
```

Verifies camera + baseline inference.

### 3. Production Inference

```bash
python local_inference.py
```

Real-time grading with colors:

- Green → G1
- Orange → G2
- Red → G3 / DISCARD

### 4. Dataset Capture

```bash
python capture_dataset.py
```

Behavior:

* 1280×720 raw capture
* spacebar triggers multi-angle capture
* frames stored unprocessed

---

## Edge Learning System

Frames in the 0.40–0.65 confidence band are automatically saved to `dataset/edge_harvest/` with full telemetry. This creates the raw material for future continuous improvement.

Each entry includes:

* bounding boxes
* class IDs
* confidence scores
* timestamp metadata

---

## Operator Controls

| Key | Action |
|-----|--------|
| g | Log grading disagreement |
| q | Quit |
| space | Capture dataset frame (in capture script) |

---

## Operational Roadmap (Honest Status — June 2026)

| Milestone | Status |
|-----------|--------|
| Hardware integration + reconnection | Complete |
| Sandbox validation | Complete |
| Dataset capture | Active |
| Dynamic grading policy engine | Complete |
| Deterministic grading + spatial binding | Complete |
| Annotation pipeline | Pending |
| Cloud training / model export | Pending |
| Full CoreML production deployment | Pending (model loading in place) |
| Closed-loop continuous learning | Pending |

---

## Engineering Logbook

### Day 1 — Hardware Validation

Verified Arducam ingestion pipeline and MPS acceleration on Apple Silicon. Established baseline inference stability at 40 FPS.

### Day 2 — Requirement Collapse

Facility verification invalidated USDA-style grading assumptions. System restructured around 3-tier operational reality.

### Day 3 — Feature Detector Migration

Removed variety-grade coupling. Transitioned to 13-class feature detection model with deterministic grading engine.

### Day 4 — Dynamic Schema Deployment

Implemented runtime class discovery. Eliminated inference-time coupling to dataset taxonomy. Introduced IoA spatial binding and operator override telemetry.

### Day 5+ — Hardware Hardening & Policy Engine (June 2026)

Deployed robust camera reconnection logic. Built configurable grading policy system. Foundation ready for model training and continuous learning loop.

### Day 6 — Household Sandbox Demo (June 16, 2026)

Created household-sandbox-demo branch as a quick test from main. Built a functional proof-of-concept adapting the engine to household plant canopy monitoring using a COCO-pretrained yolo11x.mlpackage. Added demo_inference.py and demo_grading_policy.yaml with interactive mouse-driven defect injection for testing spatial binding and grading logic. Successfully validated core modularity and domain portability with zero changes to the fundamental architecture.

### Day 7 — Camera Index Bug Fix (July 2026)

Discovered all three production scripts (`capture_dataset.py`, `baseline_verify.py`, `local_inference.py`) hardcoded `CAM_INDEX=0`, which on this MacBook Air M4 maps to the built-in FaceTime camera — not the Arducam OV9782 (which enumerates at index 1). The "Hardware Hardening" milestone was previously validated against the wrong camera. Fixed by adding `camera_utils.py` with auto-detection via `system_profiler` name match + native resolution (1920x1080) probe, with graceful fallback. Re-validated against the actual Arducam hardware.

---

## Closing Statement

This system optimizes for **operational truth on real hardware**, not model complexity.

A neural network can be wrong about a fruit.
A system misaligned with facility rules is useless.

Reality remains the final authority.

**Next steps:** Focus on building a small annotated dataset and getting a working CoreML model deployed. Update this README again once training and annotation are active.

⸻
