# Apple Quality Recognition Engine

Most agricultural vision systems fail for a structural reason, not a modeling one.

They ask neural networks to perform judgment.

This system does not.

The Apple Quality Recognition Engine is a production-grade edge vision architecture deployed on Apple Silicon for cold-storage sorting operations. It separates perception from decision-making:

* The neural network detects reality.
* The software enforces grading logic.

YOLO11 performs object detection for apples, discard triggers, and surface defects. A deterministic grading engine evaluates severity, spatial relationships, and defect coverage to produce final class assignments.

Neural networks identify. Algorithms decide.

That separation is not aesthetic. It is operational necessity.

Every grading rule embedded in model weights becomes retraining debt. Every facility rule change becomes a training cycle. Every training cycle becomes downtime.

This system eliminates that dependency by design.

⸻

## Core System Principles

| Principle | Description |
|-----------|-------------|
| Detection ≠ Decision | Neural network detects objects. Deterministic logic assigns grades. |
| Dynamic Schema | Defect taxonomy scales without inference code modification. |
| Facility Ground Truth | Real-world sorting rules override theoretical taxonomies. |
| Edge-Native Execution | Runs fully on Apple Silicon via CoreML / ANE. |
| Active Learning Loop | Uncertain predictions become training data automatically. |
| Operator Authority | Human disagreement is captured as structured telemetry. |

⸻

## System Architecture

### Feature Detector Pipeline

The architecture replaces fused classification with a two-stage system:

1. Neural detection (YOLO11)
2. Deterministic grading engine

This reduces model complexity and isolates operational logic from training artifacts.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MACRO PARENT BOX                           │
│                 Class 0: apple (instance root)                  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              MICRO DEFECT LAYER (Classes 2-N)             │  │
│  │                                                           │  │
│  │   Bruise     Russet     Scab      Rot     Crack          │  │
│  │   ○          ○          ○         ○        ○             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Flow

1. **Neural Stage**
   - YOLO11 detects:
     * Apple instances (Class 0)
     * Discard triggers (Class 1)
     * Defect classes (Class 2-N)

2. **Spatial Binding**
   - Defect boxes are bound to parent apple instances using Intersection-over-Area (IoA ≥ 0.10)

3. **Grading Engine**
   - Deterministic scoring based on:
     * defect count
     * severity class
     * area coverage ratio

4. **Discard Override**
   - Class 1 triggers immediate rejection if within 50px of any apple cluster

5. **Edge Harvesting**
   - Low-confidence frames (0.40–0.65) are persisted for retraining

6. **Operator Override**
   - Press g to log disagreement with model output

⸻

## Dynamic Schema Model

The system does not hardcode defect taxonomy.

Instead, it derives structure at runtime:

```python
num_classes = len(model.names)
```

This allows defect classes to scale without modifying inference logic.

### Schema Rules

* Class 0 → apple (root instance)
* Class 1 → unfit_bin_discard (override trigger)
* Class 2-N → defect taxonomy (dynamic)

Adding a new defect requires:

1. dataset update
2. retraining
3. redeployment

No inference changes required.

⸻

## Hardware & Deployment Configuration

### Target Stack

* Host: MacBook Air (Apple M4 Silicon)
* OS: macOS 26 (Tahoe)
* Camera: Arducam USB Global Shutter
* Acceleration: Apple Neural Engine (CoreML / ANE)

⸻

### Camera Configuration

* Camera Index: 0
* Resolution: 1280x720
* Pipeline: MJPG native stream
* Backend: OpenCV (cv2.VideoCapture)

⸻

### Inference Configuration

* Sandbox Model: yolo11n.pt (validation only)
* Production Model: best.mlpackage (CoreML)
* Sandbox Resolution: 640px
* Production Resolution: 1024px
* Execution Backend: Apple Neural Engine (ANE)

⸻

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

⸻

## Algorithmic Grading Matrix

Grading is computed deterministically in `compute_grade()`.

### Inputs

* defect count
* defect severity weights
* area coverage ratio

⸻

### Grade Logic

| Grade | Condition | Output |
|-------|-----------|--------|
| G1 | no defects OR minimal mild defects | green |
| G2 | moderate defects OR 5–15% coverage | orange |
| G3 | structural defects OR >15% coverage | red |
| DISCARD | Class 1 proximity event | reject |

⸻

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

⸻

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

⸻

## Operational Pipeline

### 1. Environment Setup

```bash
cd ~/Desktop/apple-quality-recognition-engine
python3 -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt
```

⸻

### 2. Hardware Verification

```bash
python baseline_verify.py
```

Expected result:

* camera stream opens
* MPS acceleration active
* frame loop stable

⸻

### 3. Production Inference

```bash
python local_inference.py
```

Outputs:

* Green → G1
* Orange → G2
* Red → G3 / DISCARD
* Magenta → discard trigger

⸻

### 4. Dataset Capture

```bash
python capture_dataset.py
```

Behavior:

* 1280×720 raw capture
* spacebar triggers multi-angle capture
* frames stored unprocessed

⸻

## Edge Learning System

Frames in confidence band:

```
0.40 ≤ confidence ≤ 0.65
```

are automatically saved to:

```
dataset/edge_harvest/
```

Each entry includes:

* bounding boxes
* class IDs
* confidence scores
* timestamp metadata

This forms the active learning feedback loop.

⸻

## Operator Controls

| Key | Action |
|-----|--------|
| g | Log grading disagreement |
| q | Quit inference loop |
| space | Capture dataset frame |

⸻

## Operational Roadmap

| Milestone | Status |
|-----------|--------|
| Hardware integration | Complete |
| Sandbox validation | Complete |
| Dataset capture | Active |
| Annotation pipeline | Pending |
| Cloud training | Pending |
| CoreML deployment | Pending |
| Continuous learning loop | Pending |

⸻

## Engineering Logbook

### Day 1 — Hardware Validation

Verified Arducam ingestion pipeline and MPS acceleration on Apple Silicon. Established baseline inference stability at 40 FPS.

⸻

### Day 2 — Requirement Collapse

Facility verification invalidated USDA-style grading assumptions. System restructured around 3-tier operational reality.

⸻

### Day 3 — Feature Detector Migration

Removed variety-grade coupling. Transitioned to 13-class feature detection model with deterministic grading engine.

⸻

### Day 4 — Dynamic Schema Deployment

Implemented runtime class discovery. Eliminated inference-time coupling to dataset taxonomy. Introduced IoA spatial binding and operator override telemetry.

⸻

## Closing Statement

This system does not optimize for model complexity.

It optimizes for operational truth.

A neural network that misreads a fruit is incorrect.

A system that misaligns with the facility is useless.

This architecture enforces that distinction.

Reality remains the final authority.

⸻
