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
- **Flattened Mapping**: 79-class array (68 variety-grade combos + 11 anomaly classes)

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

### Variety × Quality Grade Matrix (Indices 0-67)

| Variety | very_fancy | fancy | n1 | utility |
|---------|------------|-------|----|---------|
| redfree | 0 | 1 | 2 | 3 |
| grand_gala | 4 | 5 | 6 | 7 |
| priscilla | 8 | 9 | 10 | 11 |
| freedom | 12 | 13 | 14 | 15 |
| sweet_16 | 16 | 17 | 18 | 19 |
| crimson_crisp | 20 | 21 | 22 | 23 |
| spartan | 24 | 25 | 26 | 27 |
| macoun | 28 | 29 | 30 | 31 |
| snowsweet | 32 | 33 | 34 | 35 |
| liberty | 36 | 37 | 38 | 39 |
| pink_lady | 40 | 41 | 42 | 43 |
| chieftain | 44 | 45 | 46 | 47 |
| winecrisp | 48 | 49 | 50 | 51 |
| ludacrisp | 52 | 53 | 54 | 55 |
| enterprise | 56 | 57 | 58 | 59 |
| rosalee | 60 | 61 | 62 | 63 |
| evercrisp | 64 | 65 | 66 | 67 |

### Surface Anomaly Classes (Indices 68-78)

| Index | Anomaly |
|-------|---------|
| 68 | bruise |
| 69 | russeting |
| 70 | scarf_skin |
| 71 | sunburn |
| 72 | stem_puncture |
| 73 | split_crack |
| 74 | misshapen |
| 75 | scab |
| 76 | sooty_blotch_flyspeck |
| 77 | rot |
| 78 | insect_damage |

**Total Classes: 79**

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
- Display format: `ENTERPRISE - VERY FANCY [0.89]`
- Window title: "M4 Edge Sorting Pipeline Engine"
- Real-time FPS benchmarks displayed

## Project Timeline Matrix

| Phase | Description | Status | Deliverable |
|-------|-------------|--------|-------------|
| **Phase 1** | Local Environment & Camera Verification | ✅ Complete | Virtual environment, camera permissions, baseline_verify.py |
| **Phase 2** | Baseline Inference Test | ⏳ Pending | YOLO11n dry run, MPS acceleration verification |
| **Phase 3** | Cloud Pipeline Setup | ⏳ Pending | Roboflow workspace, dataset upload, annotation |
| **Phase 4** | Model Training & CoreML Conversion | ⏳ Pending | Trained YOLO11 model, best.mlpackage, local_inference.py |

## 📈 Engineering Progress Journal

### 🍏 Day 1: Hardware & Sandbox Validation (June 11, 2026)
* **Milestones:** Bypassed macOS Tahoe sandboxing hooks; verified Arducam global shutter uncompressed pipeline streams using OpenCV.
* **Inference Benchmarks:** Tested `yolo11n.pt` locally. Confirmed native Apple Silicon GPU acceleration via Metal Performance Shaders (MPS), stabilizing frames at **40 FPS** (640px).
* **Architecture Design:** Built a comprehensive 79-class database framework mapping multi-task overlapping bounding boxes.

### 🍏 Day 2: Cloud Ingestion Optimization & Data Collection (June 12, 2026)
* **System Bottleneck Discovered:** Observed that Roboflow automatically alphabetizes mass-uploaded class arrays, which scrambled our sequential 0-indexed matrix paths (`if cls_id < 68`).
* **The Fix:** Implemented an alphabetical padding bypass patch using character anchors (`a_` through `q_` for varieties, `z_` for defects) to force structural layer synchronization.
* **Data Collection:** Planning to gather 4-axis fruit rotations with phone HDR and Scene Optimization filters **disabled** to maintain pixel domain parity with the Arducam THIS WEEK.

## File Structure

```
.
├── .gitignore              # Python/ML exclusions
├── README.md               # This file
├── requirements.txt        # Pinned baseline dependencies
├── data.yaml               # 79-class dataset configuration
├── baseline_verify.py      # Camera verification script
└── local_inference.py      # Production CoreML deployment script
```

## Display Format Examples

### Variety + Grade Predictions
```
ENTERPRISE - VERY FANCY [0.89]
PINK LADY - FANCY [0.76]
CRIMSON CRISP - N1 [0.92]
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
