# Apple Quality Recognition Engine

A production-grade edge computer vision architecture deployed directly to the cold storage floor. This system implements a **Feature Detector Pipeline**—a stripped-down algorithmic approach that separates variety identification from quality grading. The neural network detects raw features (apple instances, discard triggers, surface defects), while post-processing arrays handle the grading logic programmatically. Optimized specifically for the Apple Silicon M4 Neural Engine running natively on macOS 26 (Tahoe).

## System Architecture

### Feature Detector Pipeline

The architecture shifts from fused category-grade classification to a two-stage separation: neural detection followed by algorithmic grading. This reduces model complexity and enables dynamic grade threshold adjustment without retraining.

```
┌─────────────────────────────────────────────────────────────────┐
│              MACRO PARENT BOX (Class 0: apple)                  │
│              Universal Fruit Instance Container                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           MICRO DEFECT BOXES (Classes 2-N)               │  │
│  │           Dynamic defect array (scales with dataset)     │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                  │  │
│  │  │Bruise│  │Russet│  │Scab  │  │Rot   │                  │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

```

**Execution Flow:**

1. **Neural Stage:** YOLO11 detects dynamic classes (1 parent, 1 discard trigger, N defect types)
2. **Spatial Binding:** Defect boxes bound to parent apple boxes via Intersection-over-Area (IoA) with 0.10 threshold
3. **Algorithmic Grading:** Grade computed from defect count, type severity, and area coverage
4. **Discard Override:** Class 1 (`unfit_bin_discard`) near any cluster triggers immediate DISCARD sequence
5. **Edge Harvesting:** Frames with volatile confidence (0.40-0.65) auto-save for active learning
6. **Operator Override:** Press 'g' to manually log line disagreement with model grading

**The Dynamic Schema Paradigm:** Defect class count scales based on dataset diversity. Adding or removing defect classes in `data.yaml` requires zero code changes in `local_inference.py`. The runtime auto-detects class count via `len(model.names)` and dynamically adjusts parsing thresholds.

---

## Hardware & Deployment Configuration

### Target Stack

* **Host Compute:** MacBook Air (Apple M4 Silicon)
* **Operating System:** macOS 26 (Tahoe) / Darwin 25.5.0
* **Sensor Pipeline:** Arducam USB Global Shutter Lens
* **Hardware Acceleration:** Apple Neural Engine (ANE) via CoreML compilation

### Camera Ingestion Profile

```yaml
Camera Index: 0
Resolution: 1280x720
Video Pipeline: Native MJPG Stream (Zero-compress hardware framework)
FourCC Frame Hook: cv2.VideoWriter_fourcc(*"MJPG")
```

### Inference Parameters

* **Sandbox Testing Target:** `yolo11n.pt` *(Temporary local validation tool for Metal Performance Shaders and camera bus sandboxing)*
* **Production Deployment Core:** `best.mlpackage` (Compiled CoreML format)
* **Sandbox Resolution:** 640px
* **Production Resolution:** 1024x1024px *(Native square tensor optimization for high-detail skin blemishes)*
* **Execution Backend:** Apple Neural Engine (ANE)

---

## Class Dictionary Schema

### Dynamic Feature Detector Array

**Schema Architecture:** Class 0 = apple, Class 1 = unfit_bin_discard, Classes 2-N = dynamic defects

**Current Operational Footprint:** 13 Classes (Indices 0-12)

| Index | Class Token | Type | Function |
| --- | --- | --- | --- |
| **0** | apple | Macro Parent | Universal fruit instance container |
| **1** | unfit_bin_discard | Discard Trigger | Immediate discard sequence when detected near cluster |
| **2** | z_bruise | Micro Defect | Mild surface defect |
| **3** | z_russeting | Micro Defect | Mild surface defect |
| **4** | z_scarf_skin | Micro Defect | Moderate surface defect |
| **5** | z_sunburn | Micro Defect | Moderate surface defect |
| **6** | z_stem_puncture | Micro Defect | Moderate surface defect |
| **7** | z_split_crack | Micro Defect | Structural defect |
| **8** | z_misshapen | Micro Defect | Structural defect |
| **9** | z_scab | Micro Defect | Moderate surface defect |
| **10** | z_sooty_blotch_flyspeck | Micro Defect | Moderate surface defect |
| **11** | z_rot | Micro Defect | Structural/severe defect |
| **12** | z_insect_damage | Micro Defect | Structural/severe defect |

**Dynamic Scaling:** Add or remove defect classes in `data.yaml` as needed. `local_inference.py` auto-detects class count via `len(model.names)` and adjusts parsing thresholds dynamically. No code changes required when modifying defect taxonomy.

### Algorithmic Grading Matrix

Grading is computed programmatically via `compute_grade()` in `local_inference.py`:

**Grade Determinants:**

* **Defect Count:** Number of child boxes bound to parent
* **Defect Type Severity:**
  * Mild: `z_bruise`, `z_russeting`
  * Moderate: `z_scarf_skin`, `z_sunburn`, `z_stem_puncture`, `z_scab`, `z_sooty_blotch_flyspeck`
  * Structural/Severe: `z_split_crack`, `z_misshapen`, `z_rot`, `z_insect_damage`
* **Area Coverage:** Total defect pixel area / parent box area ratio

**Grade Thresholds:**

| Grade | Condition | Color Code |
| --- | --- | --- |
| **G1** | 0 defects OR (≤2 mild defects AND ≤1 moderate AND area ratio < 5%) | Green (0, 255, 0) |
| **G2** | (>2 mild OR >1 moderate OR area ratio 5-15%) AND no structural defects | Orange (0, 165, 255) |
| **G3** | Any structural defect OR area ratio > 15% | Red (0, 0, 255) |
| **DISCARD** | Class 1 detected within 50px of parent box cluster | Red (0, 0, 255) |

---

## Local Environment & Operational Setup

### Phase 1: Sandbox Toolbox Instantiation

Isolate the development dependencies away from the global operating system. Execute activation protocols before invoking scripts.

```bash
# Move to workspace
cd ~/Desktop/apple-quality-recognition-engine

# Initialize pristine virtual environment folder onto Desktop
python3 -m venv ../venv

# Activate local environment
source ../venv/bin/activate

# Force-upgrade installer and populate system dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Phase 2: macOS Sandboxing & Hardware Clearance

macOS Tahoe applies strict sandboxing rules to camera access via the Unix shell.

1. Boot the baseline test execution block: `python baseline_verify.py`
2. Wait for the native Apple security overlay: *"Terminal" wants to access the camera*
3. Click **Allow** to white-list the hardware bus
4. *Override fallback:* Toggle manual access on via **System Settings > Privacy & Security > Camera > Terminal**

### Phase 3: Hardware Optical Verification

Verify sensor capture capabilities and frame loop throughput using the stock test model.

```bash
python baseline_verify.py
```

**Pass Condition:** Live OpenCV frame window opens at 1280x720, automatically downloads `yolo11n.pt`, and registers real-time bounding boxes via Metal Performance Shaders (MPS) acceleration. Hit `q` to terminate.

### Phase 4: Active Production Sorting Pipeline

Execute edge inference against the customized, fully trained Neural Engine network asset.

```bash
python local_inference.py
```

**Pipeline Behavior:** Code reads `best.mlpackage`, maps tensor execution directly to the Apple Neural Engine, runs at full 1024×1024 scale, and outputs color-coded bounding boxes:
- Green: G1 (Premium)
- Orange: G2 (Processing)
- Red: G3/DISCARD (Utility/Bin)
- Magenta: Discard triggers
- Red sub-boxes: Defects

**Edge Harvesting:** Frames with volatile confidence (0.40-0.65) auto-save to `dataset/edge_harvest/` with telemetry JSON for active learning loops.

### Phase 5: High-Speed Raw Data Capture

Run localized automated image acquisition using the Arducam global shutter lens for pure feature harvesting.

```bash
python capture_dataset.py
```

**Operational Optimization:** Script captures raw, uncompressed 1280x720 MJPG streams directly to `dataset/raw_ingest/`. No classification or variety bucketing—pure feature harvesting for downstream annotation. Smash **SPACEBAR** 4 times per fruit (rotating across axes). Camera window instantly prompts for next fruit—swap on dark backdrop and smash **ENTER** directly inside camera viewer. No terminal clicking required.

---

## Data Lifecycle Architecture

The production pipeline follows a strict data flow from raw capture to edge deployment:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LIFECYCLE PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

1. RAW CAPTURE (Arducam Global Shutter)
   └─> capture_dataset.py
   └─> dataset/raw_ingest/
   └─> Unlabelled 1280x720 MJPG frames with unique timestamps

2. MANUAL ANNOTATION (Roboflow Cloud Platform)
   └─> Batch upload raw frames to Roboflow
   └─> Manual bounding box annotation:
       • Class 0: apple (parent boxes)
       • Class 1: unfit_bin_discard (discard triggers)
       • Classes 2-N: dynamic defect array
   └─> Export annotated dataset in YOLO format

3. CLOUD TRAINING (Google Colab)
   └─> Import annotated dataset
   └─> Train YOLO11 model at 1024×1024 resolution
   └─> Export weights to CoreML format (.mlpackage)
   └─> Download best.mlpackage to local environment

4. LOCAL DEPLOYMENT (M4 Neural Engine)
   └─> local_inference.py loads best.mlpackage
   └─> Dynamic schema auto-detection via len(model.names)
   └─> IoA spatial binding (0.10 threshold)
   └─> Algorithmic grading via compute_grade()
   └─> Edge harvesting for volatile confidence (0.40-0.65)
   └─> Operator override via 'g' key for manual logging

```

---

## Operational Roadmap Matrix

| Target Milestone | Structural Focus | Status | Core Deliverable Artifact |
| --- | --- | --- | --- |
| **Milestone 1** | Env Configuration & Camera Ingestion | ✅ Complete | Verified `venv`, Mac hardware permissions, raw MJPG framework |
| **Milestone 2** | Sandbox Core Verification | ✅ Complete | Local `baseline_verify.py` pass; MPS hardware acceleration test |
| **Milestone 3** | Physical Rig Collection | ⏳ Active | Automated 4-shot exposure execution via `capture_dataset.py` |
| **Milestone 4** | Cloud Dataset Annotation | pending | Roboflow 13-class bounding box alignment and validation |
| **Milestone 5** | Cloud Compute Training Loop | pending | YOLO11 1024px custom network weight export via Google Colab |
| **Milestone 6** | CoreML Production Edge Push | pending | ANE optimized `best.mlpackage` execution inside `local_inference.py` |
| **Milestone 7** | Edge Harvesting Pipeline | pending | Active learning loop integration with cloud retraining |

---

## Spatial Binding & Algorithmic Grading Engine

The real-time inference processor bypasses slow multi-model structures by executing hierarchical downstream geometry filtering directly inside `local_inference.py`.

### Stage 1: Instance Parsing (13-Class Paradigm)

On every incoming frame tensor, detections route based on class index:

```python
if cls_id == 0:
    # Route to Parent Box Array (apple)
elif cls_id == 1:
    # Route to Discard Trigger Array (unfit_bin_discard)
elif cls_id >= 2:
    # Route to Defect Array (z_bruise through z_insect_damage)
```

### Stage 2: Discard Sequence Check

Class 1 (`unfit_bin_discard`) triggers immediate DISCARD mode when detected within 50px proximity of any parent box cluster. This overrides algorithmic grading and forces red-box rendering.

### Stage 3: Spatial Binding Layer

Defect bounding boxes transform into spatial coordinates to calculate center-mass points:

$$C_x = x_{min} + \frac{x_{max} - x_{min}}{2}$$

$$C_y = y_{min} + \frac{y_{max} - y_{min}}{2}$$

Logic runs containment loop to determine if micro point $(C_x, C_y)$ resides geometrically inside parent apple macro box boundaries.

### Stage 4: Deterministic Grading

Grade computed via `compute_grade()` function based on three determinants:

1. **Defect Count:** Number of child boxes bound to parent
2. **Defect Type Severity:** Categorized as mild, moderate, or structural/severe
3. **Area Coverage:** Total defect pixel area divided by parent box area ratio

**Grade Logic Flow:**
- Structural defects OR area ratio > 15% → G3
- (>2 mild OR >1 moderate OR area ratio 5-15%) AND no structural → G2
- 0 defects OR (≤2 mild AND ≤1 moderate AND area ratio < 5%) → G1

### Stage 5: Edge Harvesting (Active Learning)

Frames with any detection confidence in volatile threshold (0.40-0.65) trigger auto-save to `dataset/edge_harvest/`. Each saved frame includes telemetry JSON with:
- Timestamp
- Volatile detection class IDs, names, confidences, and bounding boxes
- Frame path reference

This enables targeted cloud retraining loops on edge cases where model confidence is unstable.

---

## 📈 Engineering Progress Journal

### 🍏 Day 1: Hardware & Sandbox Validation (June 11, 2026)

**Milestones:** Successfully bypassed macOS Tahoe sandboxing hooks; verified Arducam global shutter uncompressed pipeline streams using OpenCV.

**Inference Benchmarks:** Tested `yolo11n.pt` locally. Confirmed native Apple Silicon GPU acceleration via Metal Performance Shaders (MPS), stabilizing frames at **40 FPS** (640px).

**Architecture Design:** Built comprehensive multi-task database framework mapping overlapping bounding boxes.

### 🍏 Day 2: Cloud Ingestion Optimization & Data Collection (June 12, 2026)

**System Bottleneck Discovered:** Observed that Roboflow automatically alphabetizes mass-uploaded class arrays, scrambling sequential 0-indexed matrix paths.

**The Fix:** Implemented alphabetical padding bypass patch using character anchors to force structural layer synchronization.

**Data Collection:** Planning 4-axis fruit rotations with phone HDR and Scene Optimization filters **disabled** to maintain pixel domain parity with Arducam.

### 🍏 Day 2 Continued: The Floor-Rule Pivot & Massive Refactor (June 12, 2026)

**The Floor-Rule Drop:** Talked to the manager at the cold storage unit to double-check grading rules—he completely broke my entire architecture. I had spent hours mapping out an elaborate USDA 4-tier grading system. Turns out on the actual floor, they use a dead-simple 1-2-3 system: Grade 1 (premium retail), Grade 2 (processing/slicers), Grade 3 (low-value utility). Anything worse goes straight into the trash bin because it's not worth the labor to sort. Building an AI that doesn't match how the actual facility operates is a death sentence, so I scrapped the old schema and pivoted immediately.

**The Code & Math Rebuild:** Dropping from 4 grades to 3 changed the math on everything:
- *Old Setup:* 18 varieties × 4 grades + 11 defects = 83 classes
- *New Setup:* 18 varieties × 3 grades + 11 defects = **65 total classes**

This shifted local inference logic cutoff boundary from `cls_id < 72` down to `cls_id < 54`. Spatial binding script gets minor speed boost from smaller index array.

**Zestar Integration:** Added Zestar apples to roster. Because it starts with 'Z' but needs to be index 0, it became `a_zestar_g1`. Forced re-map of every alphabetical prefix so Roboflow's automatic sorting engine wouldn't scramble indexes.

**Model & Hardware Realignment:**
- *Cleaning up the README:* Fixed major documentation blunder. README listed `yolo11n` as production model. That Nano model was strictly a sandbox test to verify macOS Tahoe sandboxing didn't block camera bus and that MPS acceleration worked on M4 chip. Actual production model is heavy-duty multi-task YOLO11 trained at 1024×1024 and exported to CoreML for Apple Neural Engine.
- *Scrapping the Phone:* Ditched Samsung A16 for data collection. Taking training photos on phone camera and deploying on Arducam global shutter lens causes massive domain bias (model learns phone ISP artifacts instead of raw production pixels). Wrote `capture_dataset.py` to pull raw MJPG streams directly from Arducam at 1280x720 using manual spacebar trigger for 4-shot rotation sequence.

**Current Status:** Roboflow locked in at 65 classes, sorting perfectly from `a_zestar_g1` (0) down to `z_insect_damage` (64). Refactor blueprint complete, `.gitignore` set up to block raw images from bloating GitHub, hardware rig ready. Time to step into the fridge and start rolling apples.

### 🍏 Day 3: Architectural Refocus — Feature Detector Conversion (June 14, 2026)

**The Paradigm Shift:** Stripped variety/grade logic out of the weights entirely. Moved to algorithmic Feature Detector pipeline. Neural network now detects raw features only (apple instances, discard triggers, surface defects). Post-processing arrays handle grading programmatically.

**The 13-Class Refactor:**
- *Old Setup:* 18 varieties × 3 grades + 11 defects = 65 classes
- *New Setup:* 1 parent (apple) + 1 discard trigger + 11 defects = **13 total classes**

**Why This Works:**
- Reduces model complexity by 80%
- Enables dynamic grade threshold adjustment without retraining
- Separates detection from classification—neural network does what it's good at (finding things), code does what it's good at (making decisions)
- Discard trigger (Class 1) provides immediate override capability for facility-specific rejection criteria

**Implementation:**
- Updated `data.yaml` to 13-class schema
- Refactored `local_inference.py` with new spatial binding:
  - Class 0 (`apple`) = universal Macro Parent Box
  - Class 1 (`unfit_bin_discard`) = immediate DISCARD sequence when near cluster
  - Classes 2-12 = defect types bound to parent boxes
- Implemented `compute_grade()` deterministic scoring function:
  - G1: 0 defects or minor threshold
  - G2: Mild defects (bruise/russeting) or moderate coverage
  - G3: Structural/severe defects or high coverage
- Added `save_edge_harvest_frame()` active learning component:
  - Auto-saves frames with volatile confidence (0.40-0.65)
  - Stores in `dataset/edge_harvest/` with telemetry JSON
  - Enables targeted cloud retraining loops

**Current Status:** Architecture refactored to 13-class paradigm. Code updated, documentation rewritten in Systems-Operational tone. Ready for dataset re-annotation and model retraining.

### 🍏 Day 4: Core Alignment & Production Hardening (June 15, 2026)

**The Dynamic Schema Pivot:** Realized that hardcoding defect class limits creates technical debt. The exact count of skin anomaly classes is volatile and depends on real-world dataset diversity discovered during sorting. Moved to dynamic schema architecture where the runtime auto-detects class count via `len(model.names)` and adjusts parsing thresholds accordingly.

**Infrastructure Hardening:**
- **Dynamic Schema Abstracting:** Replaced hardcoded `cls_id >= 2` with `cls_id >= 2 and cls_id < num_classes` in `local_inference.py`. Adding or removing defect classes in `data.yaml` now requires zero code changes.
- **IoA Spatial Binding:** Replaced centroid containment with Intersection-over-Area (IoA) buffer evaluation. Defect boxes binding to parent apples if intersection ratio >= 0.10. Edge case resolution: defects overlapping multiple apples bind to parent with highest intersection density.
- **Operator Override:** Added 'g' key handler for manual line disagreement logging. Pressing 'g' triggers immediate edge harvest event with `operator_override: true` flag in telemetry JSON.

**Production Capture Purge:**
- Completely refactored `capture_dataset.py` to pure feature harvesting pipeline.
- Removed all obsolete grading and multi-class variety bucketing logic.
- Script now captures raw, uncompressed 1280x720 MJPG streams directly to `dataset/raw_ingest/` with unique timestamps.
- No classification logic—pure raw frame collection for downstream Roboflow annotation.

**Documentation Traceability:**
- Updated README.md to reflect dynamic anomaly schema (Class 0 = apple, Class 1 = discard trigger, Classes 2-N = dynamic defects).
- Added comprehensive Data Lifecycle Architecture section documenting the full pipeline: Raw Capture → Manual Annotation → Cloud Training → Local Deployment.
- Updated all references from "13-Class" to "Dynamic Schema" to reflect the new architecture.

**Current Status:** All infrastructure hardening complete. Dynamic schema architecture locked in. Production capture pipeline purified. Documentation updated. Ready for dataset re-annotation with new dynamic schema.