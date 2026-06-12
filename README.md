# Apple Quality Recognition Engine

A production-grade computer vision deployment system for automated apple variety identification, quality grading, and surface anomaly detection. Optimized for Apple Silicon M4 Neural Engine acceleration on macOS 26 (Tahoe).

## System Architecture

### Multi-Task Object Detection Model

The system employs a **single-model, multi-task YOLO11 architecture** with a flattened class mapping strategy:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MACRO PARENT BOUNDING BOX                    │
│                  (Full Apple - Variety + Grade)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              MICRO CHILD BOUNDING BOXES                    │  │
│  │         (Surface Anomaly/Defect Localization)              │  │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                  │  │
│  │  │Bruise│  │Russet│  │Scab  │  │Rot   │                  │  │
│  │  └──────┘  └──────┘  └──────┘  └──────┘                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Detection Strategy:**
- **Macro Boxes**: Drawn around full apples for fused Variety + Quality Grade classification
- **Micro Boxes**: Nested inside parent boundaries for surface anomaly/defect localization
- **Flattened Mapping**: 65-class array (54 variety-grade combos + 11 anomaly classes)
- **Quality Tiers**: 3-tier grading system (G1=Premium, G2=Average, G3=Utility)

## Hardware Configuration

### Target Platform
- **Device**: MacBook Air (M4 chip)
- **OS**: macOS 26 (Tahoe) / Darwin 25.5.0
- **Camera**: Arducam USB Global Shutter
- **Backend**: Apple Neural Engine (ANE) via CoreML

### Camera Ingestion Profile
```yaml
Camera Index: 0
Resolution: 1280x720
Encoding: MJPG (raw uncompressed)
FourCC: MJPG
```

### Inference Parameters
```yaml
Baseline Model: yolo11n.pt
Production Model: best.mlpackage (CoreML)
Baseline Image Size: 640
Production Image Size: 1024
Task: detect
```

## Class Dictionary Schema

### Quality Grade System
- **G1**: Premium (Very Fancy)
- **G2**: Average (Fancy)
- **G3**: Utility (Low Quality)
- **Below G3**: Discarded (not labeled)

### Variety × Quality Grade Matrix (Indices 0-53)

| Variety | G1 | G2 | G3 |
|---------|----|----|----|
| zestar | 0 | 1 | 2 |
| redfree | 3 | 4 | 5 |
| grand_gala | 6 | 7 | 8 |
| priscilla | 9 | 10 | 11 |
| freedom | 12 | 13 | 14 |
| sweet_16 | 15 | 16 | 17 |
| crimson_crisp | 18 | 19 | 20 |
| spartan | 21 | 22 | 23 |
| macoun | 24 | 25 | 26 |
| snowsweet | 27 | 28 | 29 |
| liberty | 30 | 31 | 32 |
| pink_lady | 33 | 34 | 35 |
| chieftain | 36 | 37 | 38 |
| winecrisp | 39 | 40 | 41 |
| ludacrisp | 42 | 43 | 44 |
| enterprise | 45 | 46 | 47 |
| rosalee | 48 | 49 | 50 |
| evercrisp | 51 | 52 | 53 |

### Surface Anomaly Classes (Indices 54-64)

| Index | Anomaly |
|-------|---------|
| 54 | bruise |
| 55 | russeting |
| 56 | scarf_skin |
| 57 | sunburn |
| 58 | stem_puncture |
| 59 | split_crack |
| 60 | misshapen |
| 61 | scab |
| 62 | sooty_blotch_flyspeck |
| 63 | rot |
| 64 | insect_damage |

**Total Classes: 65** (18 varieties × 3 grades + 11 defects)

## Local Setup Installation

### Phase 1: Environment Initialization

```bash
# Navigate to project directory
cd /Users/david/Project

# Create virtual environment
python3 -m venv apple_env

# Activate virtual environment
source apple_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install baseline dependencies
pip install -r requirements.txt
```

### Phase 2: macOS Camera Permissions

**Critical Step**: macOS requires explicit camera permission for terminal applications.

1. Run any camera script (e.g., `baseline_verify.py`)
2. macOS will display a system prompt: *"Terminal" wants to access the camera*
3. Click **Allow** to grant permission
4. If prompt doesn't appear, manually enable:
   - Go to **System Settings > Privacy & Security > Camera**
   - Ensure **Terminal** (or your IDE) is toggled **ON**

### Phase 3: Camera Verification

```bash
# Run baseline camera verification
python baseline_verify.py
```

**Expected Output:**
- Camera initializes at 1280x720 with MJPG encoding
- YOLO11n model downloads automatically
- Live inference window displays with FPS counter
- Press 'q' to exit cleanly

**Troubleshooting:**
- If camera fails, try changing index from 0 to 1 or 2
- Verify macOS camera permissions in System Settings
- Check Arducam USB connection

### Phase 4: Production Deployment

```bash
# Place your CoreML model in project directory
# Ensure file is named: best.mlpackage

# Run production inference
python local_inference.py
```

**Expected Output:**
- CoreML model loads on M4 Neural Engine
- Inference runs at 1024x1024 resolution
- Display format: `CRIMSON CRISP (G2) [0.89]`
- Window title: "M4 Edge Sorting Pipeline Engine"
- Real-time FPS benchmarks displayed
- Spatial binding: Green boxes for apples, red boxes for nested defects

### Phase 5: Data Capture

```bash
# Run data capture script for Arducam
python capture_dataset.py
```

**Expected Output:**
- Interactive prompt to select variety
- 4-shot exposure synchronization per apple
- Images saved to dataset/ directory with alphabetical prefix naming
- MJPG uncompressed streaming at 1280x720

## Project Timeline Matrix

| Phase | Description | Status | Deliverable |
|-------|-------------|--------|-------------|
| **Phase 1** | Local Environment & Camera Verification | ✅ Complete | Virtual environment, camera permissions, baseline_verify.py |
| **Phase 2** | Baseline Inference Test | ⏳ Pending | YOLO11n dry run, MPS acceleration verification |
| **Phase 3** | Cloud Pipeline Setup | ⏳ Pending | Roboflow workspace, dataset upload, annotation |
| **Phase 4** | Model Training & CoreML Conversion | ⏳ Pending | Trained YOLO11 model, best.mlpackage, local_inference.py |
| **Phase 5** | Data Capture | ⏳ Pending | Arducam dataset collection with capture_dataset.py |

## 📈 Engineering Progress Journal

### 🍏 Day 1: Hardware & Sandbox Validation (June 11, 2026)
* **Milestones:** Bypassed macOS Tahoe sandboxing hooks; verified Arducam global shutter uncompressed pipeline streams using OpenCV.
* **Inference Benchmarks:** Tested `yolo11n.pt` locally. Confirmed native Apple Silicon GPU acceleration via Metal Performance Shaders (MPS), stabilizing frames at **40 FPS** (640px).
* **Architecture Design:** Built a comprehensive 79-class database framework mapping multi-task overlapping bounding boxes.

### 🍏 Day 2: Cloud Ingestion Optimization & Data Collection (June 12, 2026)
* **System Bottleneck Discovered:** Observed that Roboflow automatically alphabetizes mass-uploaded class arrays, which scrambled our sequential 0-indexed matrix paths (`if cls_id < 68`).
* **The Fix:** Implemented an alphabetical padding bypass patch using character anchors (`a_` through `q_` for varieties, `z_` for defects) to force structural layer synchronization.
* **Data Collection:** Planning to gather 4-axis fruit rotations with phone HDR and Scene Optimization filters **disabled** to maintain pixel domain parity with the Arducam THIS WEEK.

### 🍏 Day 2 Continued: The Floor-Rule Pivot & Massive Refactor (June 12, 2026)
* **Architecture Breakthrough:** Talked to the manager at the cold storage unit today to double-check grading rules, and he completely broke my entire architecture. I had spent hours mapping out this elaborate USDA 4-tier grading system. Turns out on the actual floor, they use a dead-simple 1-2-3 system: Grade 1 (premium retail), Grade 2 (processing/slicers), and Grade 3 (low-value utility). Anything worse goes straight into the trash bin because it's not worth the labor to sort. Building an AI that doesn't match how the actual facility operates is a death sentence, so I had to scrap the old schema and pivot immediately.
* **The Code & Math Rebuild:** Dropping from 4 grades to 3 changed the math on everything:
  - Old Setup: 18 varieties × 4 grades + 11 defects = 83 classes
  - New Setup: 18 varieties × 3 grades (g1, g2, g3) + 11 defects = 65 total classes
  - This shifted my local inference logic cutoff boundary from cls_id < 72 down to cls_id < 54. The spatial binding script actually gets a minor speed boost here because it has a smaller index array to loop through.
* **Zestar Integration:** Added Zestar apples to the roster. Because it starts with a 'Z' but needs to be index 0, it became a_zestar_g1. That forced me to re-map every single alphabetical prefix (a_ through r_ for varieties, z_ for defects) so Roboflow's automatic sorting engine wouldn't scramble my indexes.
* **Model & Hardware Realignment:**
  - Fixed a major documentation blunder. The README listed yolo11n as the production model. That Nano model was strictly a sandbox test to make sure macOS Tahoe's sandboxing didn't block the camera bus and that MPS (Metal Performance Shaders) acceleration worked on the M4 chip. The actual production model is a heavy-duty, multi-task YOLO11 model trained at 1024×1024 and exported to CoreML for the Apple Neural Engine.
  - Scrapped the Phone: Ditched the Samsung A16 for data collection. Taking training photos on a phone camera and deploying on an Arducam global shutter lens causes massive domain bias (the model learns phone ISP artifacts instead of raw production pixels). I wrote capture_dataset.py to pull raw MJPG streams directly from the Arducam at 1280x720 using a manual spacebar trigger for the 4-shot rotation sequence.

## File Structure

```
.
├── .gitignore              # Python/ML exclusions + dataset/
├── README.md               # This file
├── requirements.txt        # Pinned baseline dependencies
├── data.yaml               # 65-class dataset configuration
├── baseline_verify.py      # Camera verification script
├── local_inference.py      # Production CoreML deployment with spatial binding
└── capture_dataset.py      # Arducam data acquisition script
```

## Display Format Examples

### Variety + Grade Predictions
```
ZESTAR (G1) [0.89]
CRIMSON CRISP (G2) [0.76]
ENTERPRISE (G3) [0.92]
```

### Surface Anomaly Predictions
```
BRUISE [0.85]
RUSSETING [0.71]
SOOTY BLOTCH FLYSPECK [0.63]
```

## Performance Benchmarks

### Expected M4 Performance (Estimated)
- **Baseline (YOLO11n @ 640)**: ~30-45 FPS
- **Production (CoreML @ 1024)**: ~15-25 FPS
- **Backend**: Apple Neural Engine (ANE) acceleration

## Stage 3 Spatial Binding Layer

The production inference engine implements a multi-stage spatial binding architecture:

### Stage 1 & 2: Instance Parsing
- **Macro Apples** (indices 0-53): Full apple bounding boxes with variety + grade classification
- **Micro Anomalies** (indices 54-64): Surface defect bounding boxes

### Stage 3: Spatial Binding
- Centroid containment check: Defect centroids are tested against apple bounding boxes
- Distance-based parent selection: Defects bind to the nearest containing apple
- Hierarchical relationship: Each apple maintains a list of child defects

### Stage 4: Output Rendering
- Green boxes: Apple variety + grade predictions
- Red boxes: Nested surface defects
- Display format: `VARIETY (GRADE) [confidence]` for apples, `DEFECT [confidence]` for anomalies

## Dependencies

```
ultralytics        # YOLO11 model framework
coremltools        # CoreML model conversion
opencv-python      # Camera ingestion and display
roboflow           # Dataset management and deployment
```

## License

Proprietary - Apple Quality Recognition Engine

## Contact

For deployment issues or architecture questions, refer to the project documentation or contact the deployment engineering team.
