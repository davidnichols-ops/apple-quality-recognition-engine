# Apple Quality Recognition Engine

A production-grade edge computer vision architecture deployed directly to the cold storage floor. This system implements a unified multi-task network engineered for real-time apple variety identification, three-tier quality grading, and concurrent surface defect localization. Optimized specifically for the Apple Silicon M4 Neural Engine running natively on macOS 26 (Tahoe).

## System Architecture

### Multi-Task Object Detection Network

To prevent the computational overhead of running multiple parallel models or complex multi-head networks on the Apple Neural Engine (ANE), this system utilizes a **single-stage, multi-task YOLO11 architecture** paired with a flattened class mapping strategy.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MACRO PARENT BOUNDING BOX                    │
│           (Full Apple Instance - Variety + Quality Grade)        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               MICRO CHILD BOUNDING BOXES                 │  │
│  │           (Surface Anomaly / Defect Localization)         │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                  │  │
│  │  │Bruise│  │Russet│  │Scab  │  │Rot   │                  │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

```

* **Macro Inference (Indices 0–53):** Predicts bounded full fruit instances, executing categorical variety and structural quality grade extraction simultaneously.
* **Micro Overlay (Indices 54–64):** Runs concurrent localized bounding boxes directly on top of the parent fruit geometry to isolate surface defects.
* **The Flat Index Paradigm:** A single 65-class array eliminates late-stage multi-model layer synchronization latencies. A fast, single-pass conditional loop split isolates the macro/micro properties instantly.

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

* **Sandbox Testing Target:** `yolo11n.pt` *(Strictly used as a temporary local validation tool to verify Metal Performance Shaders and camera bus sandboxing)*
* **Production Deployment Core:** `best.mlpackage` (Compiled CoreML format)
* **Sandbox Resolution:** 640px
* **Production Resolution:** 1024x1024px *(Native square tensor optimization targeting high-detail skin blemishes)*
* **Execution Backend:** Apple Neural Engine (ANE)

---

## Class Dictionary Schema

### Floor-Rule Quality Matrix

This system completely bypasses theoretical USDA criteria in favor of the facility's localized 1-2-3 floor-sorting rules.

* **Grade 1 (g1):** Premium / High-Quality Retail Box Target.
* **Grade 2 (g2):** Average / Processing & Slicing Grade.
* **Grade 3 (g3):** Utility / Low-Quality with remaining market value.
* **Below Grade 3:** Bin Discard. Filtered out before sorting; zero manual data labeling footprint.

### Alphabetical Matrix Layer (Indices 0–64)

To prevent cloud-ingestion platforms (Roboflow) from automatically alphabetizing the class array and scrambling the hardcoded 0-indexed neural architecture, a strict string-padding prefix scheme (`a_` through `r_` for variety, `z_` for defects) locks the array index structure.

#### 1. Macro Class Range (Indices 0–53) | Cutoff Gate: `cls_id < 54`

| Variety Label | Grade 1 Index | Grade 2 Index | Grade 3 Index |
| --- | --- | --- | --- |
| **a_zestar** | 0 | 1 | 2 |
| **b_redfree** | 3 | 4 | 5 |
| **c_grand_gala** | 6 | 7 | 8 |
| **d_priscilla** | 9 | 10 | 11 |
| **e_freedom** | 12 | 13 | 14 |
| **f_sweet_16** | 15 | 16 | 17 |
| **g_crimson_crisp** | 18 | 19 | 20 |
| **h_spartan** | 21 | 22 | 23 |
| **i_macoun** | 24 | 25 | 26 |
| **j_snowsweet** | 27 | 28 | 29 |
| **k_liberty** | 30 | 31 | 32 |
| **l_pink_lady** | 33 | 34 | 35 |
| **m_chieftain** | 36 | 37 | 38 |
| **n_winecrisp** | 39 | 40 | 41 |
| **o_ludacrisp** | 42 | 43 | 44 |
| **p_enterprise** | 45 | 46 | 47 |
| **q_rosalee** | 48 | 49 | 50 |
| **r_evercrisp** | 51 | 52 | 53 |

#### 2. Micro Defect Range (Indices 54–64) | Cutoff Gate: `cls_id >= 54`

| Index | Anomaly Token | Index | Anomaly Token |
| --- | --- | --- | --- |
| **54** | z_bruise | **60** | z_misshapen |
| **55** | z_russeting | **61** | z_scab |
| **56** | z_scarf_skin | **62** | z_sooty_blotch_flyspeck |
| **57** | z_sunburn | **63** | z_rot |
| **58** | z_stem_puncture | **64** | z_insect_damage |
| **59** | z_split_crack |  |  |

**Total Operational Array Footprint:** Exactly 65 Classes.

---

## Local Environment & Operational Setup

### Phase 1: Sandbox Toolbox Instantiation

Isolate the development dependencies away from the global operating system. Always execute activation protocols before invoking scripts.

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

1. Boot the baseline test execution block: `python baseline_verify.py`.
2. Wait for the native Apple security overlay: *"Terminal" wants to access the camera*.
3. Click **Allow** to white-list the hardware bus.
4. *Override fallback:* Toggle manual access on via **System Settings > Privacy & Security > Camera > Terminal**.

### Phase 3: Hardware Optical Verification

Verify sensor capture capabilities and frame loop throughput using the stock test model.

```bash
python baseline_verify.py

```

* **Pass Condition:** Live OpenCV frame window opens at 1280x720, automatically downloads `yolo11n.pt`, and registers real-time bounding boxes via Metal Performance Shaders (MPS) acceleration. Hit `q` to terminate.

### Phase 4: Active Production Sorting Pipeline

Execute edge inference against the customized, fully trained Neural Engine network asset.

```bash
python local_inference.py

```

* **Pipeline Behavior:** Code reads `best.mlpackage`, maps tensor execution directly to the Apple Neural Engine, runs at a full $1024 \times 1024$ scale, and outputs green bounding boxes for fruits and red boxes for defects.

### Phase 5: High-Speed No-Click Data Capture

Run localized automated image acquisition using the Arducam global shutter lens.

```bash
python capture_dataset.py

```

* **Operational Optimization:** Type target variety and total count. The script locks focus onto the camera preview window. Smash **SPACEBAR** 4 times per apple (rotating the fruit across its axes). The camera window will instantly prompt for the next apple—swap the fruit on the dark backdrop and smash **ENTER** directly inside the camera viewer. No terminal clicking required.

---

## Operational Roadmap Matrix

| Target Milestone | Structural Focus | Status | Core Deliverable Artifact |
| --- | --- | --- | --- |
| **Milestone 1** | Env Configuration & Camera Ingestion | ✅ Complete | Verified `venv`, Mac hardware permissions, raw MJPG framework |
| **Milestone 2** | Sandbox Core Verification | ✅ Complete | Local `baseline_verify.py` pass; MPS hardware acceleration test |
| **Milestone 3** | Physical Rig Collection | ⏳ Active | Automated 4-shot exposure execution via `capture_dataset.py` |
| **Milestone 4** | Cloud Dataset Annotation | pending | Roboflow multi-task bounding box alignment and validation |
| **Milestone 5** | Cloud Compute Training Loop | pending | YOLO11 1024px custom network weight export via Google Colab |
| **Milestone 6** | CoreML Production Edge Push | pending | ANE optimized `best.mlpackage` execution inside `local_inference.py` |

---

## 📈 Engineering Progress Journal

### 🍏 Day 1: Hardware & Sandbox Validation (June 11, 2026)

* **Milestones:** Successfully bypassed macOS Tahoe sandboxing hooks; verified Arducam global shutter uncompressed pipeline streams using OpenCV.
* **Inference Benchmarks:** Tested `yolo11n.pt` locally. Confirmed native Apple Silicon GPU acceleration via Metal Performance Shaders (MPS), stabilizing frames at **40 FPS** (640px).
* **Architecture Design:** Built a comprehensive 79-class database framework mapping multi-task overlapping bounding boxes.

### 🍏 Day 2: Cloud Ingestion Optimization & Data Collection (June 12, 2026)

* **System Bottleneck Discovered:** Observed that Roboflow automatically alphabetizes mass-uploaded class arrays, which scrambled our sequential 0-indexed matrix paths (`if cls_id < 68`).
* **The Fix:** Implemented an alphabetical padding bypass patch using character anchors (`a_` through `q_` for varieties, `z_` for defects) to force structural layer synchronization.
* **Data Collection:** Planning to gather 4-axis fruit rotations with phone HDR and Scene Optimization filters **disabled** to maintain pixel domain parity with the Arducam THIS WEEK.

### 🍏 Day 2 Continued: The Floor-Rule Pivot & Massive Refactor (June 12, 2026)

* **The Floor-Rule Drop:** Talked to the manager at the cold storage unit today to double-check grading rules, and he completely broke my entire architecture. I had spent hours mapping out this elaborate USDA 4-tier grading system. Turns out on the actual floor, they use a dead-simple 1-2-3 system: Grade 1 (premium retail), Grade 2 (processing/slicers), and Grade 3 (low-value utility). Anything worse goes straight into the trash bin because it's not worth the labor to sort. Building an AI that doesn't match how the actual facility operates is a death sentence, so I had to scrap the old schema and pivot immediately.
* **The Code & Math Rebuild:** Dropping from 4 grades to 3 changed the math on everything:
* *Old Setup:* 18 varieties × 4 grades + 11 defects = 83 classes
* *New Setup:* 18 varieties × 3 grades (`g1`, `g2`, `g3`) + 11 defects = **65 total classes**
* This shifted my local inference logic cutoff boundary from `cls_id < 72` down to `cls_id < 54`. The spatial binding script actually gets a minor speed boost here because it has a smaller index array to loop through.


* **Zestar Integration:** Added Zestar apples to the roster. Because it starts with a 'Z' but needs to be index 0, it became `a_zestar_g1`. That forced me to re-map every single alphabetical prefix (`a_` through `r_` for varieties, `z_` for defects) so Roboflow's automatic sorting engine wouldn't scramble my indexes.
* **Model & Hardware Realignment:**
* *Cleaning up the README:* Fixed a major documentation blunder. The README listed `yolo11n` as the production model. That Nano model was strictly a sandbox test to make sure macOS Tahoe's sandboxing didn't block the camera bus and that MPS (Metal Performance Shaders) acceleration worked on the M4 chip. The actual production model is a heavy-duty, multi-task YOLO11 model trained at $1024 \times 1024$ and exported to CoreML for the Apple Neural Engine.
* *Scrapping the Phone:* Ditched the Samsung A16 for data collection. Taking training photos on a phone camera and deploying on an Arducam global shutter lens causes massive domain bias (the model learns phone ISP artifacts instead of raw production pixels). I wrote `capture_dataset.py` to pull raw MJPG streams directly from the Arducam at 1280x720 using a manual spacebar trigger for the 4-shot rotation sequence.


* **Current Status:** Roboflow is locked in at 65 classes, sorting perfectly from `a_zestar_g1` (0) down to `z_insect_damage` (64). Devin has the refactor blueprint, the `.gitignore` is set up to block raw images from bloating GitHub, and the hardware rig is ready. Time to step into the fridge and start rolling apples.

---

## Stage 3 Spatial Binding Engine

The real-time inference processor bypasses slow multi-model structures by executing a hierarchical downstream geometry filter directly inside `local_inference.py`.

### 1. Spatial Parsing Loop

On every incoming frame tensor, the array splits all incoming raw detections based on the class index cutoff:

```python
if cls_id < 54:
    # Route to Macro Parent Array (Variety + Grade Box)
else:
    # Route to Micro Child Array (Surface Defect Bounding Box)

```

### 2. Centroid Containment Calculus

Defect bounding boxes are transformed into spatial coordinates to calculate their center-mass points:

$$C_x = x_{min} + \frac{x_{max} - x_{min}}{2}$$

$$C_y = y_{min} + \frac{y_{max} - y_{min}}{2}$$

The logic runs a containment loop to determine if the micro point $(C_x, C_y)$ resides geometrically inside the spatial boundaries of a parent apple macro box.

### 3. String Parsing & Render Strategy

The custom classification strings are cleaned on the fly. To ensure varieties containing multiple underscores do not have their labels fractured, strings are split from the right side using a right-hand delimiter split:

```python
# Strip sorting prefix and split right-hand grade token
_, clean_name = class_name.split("_", 1)  # "crimson_crisp_g2"
variety, grade = clean_name.rsplit("_", 1)  # "crimson_crisp", "g2"

```

* **Parent Render:** Green box, formatted as uppercase `VARIETY (GRADE) [conf]`.
* **Child Render:** Red box, formatted as uppercase `DEFECT [conf]`, structurally mapped to its nearest containing parent fruit entity.