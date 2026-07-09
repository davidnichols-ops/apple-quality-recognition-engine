# Apple Quality Recognition Engine

An edge-first vision pipeline that grades apples on Apple Silicon (M4) using a two-stage architecture: **YOLO26 detects, a deterministic policy engine decides.**

Most agricultural vision systems ask a neural network to perform judgment. This one doesn't. The model identifies apples and surface defects. A configurable grading engine — driven by a YAML policy file — evaluates severity, spatial relationships, and coverage area to assign grades. Changing facility rules is a config edit, not a retraining cycle.

**Neural networks identify. Algorithms decide.**

---

## How It Works

```
Camera → YOLO26 (CoreML/ANE) → Detection Boxes → Spatial Binding → Grading Engine → G1/G2/G3/DISCARD
```

1. **Detection** — YOLO26x (FP16 CoreML, 640x640) detects apples (class 0), discard triggers (class 1), and surface defects (classes 2-12) on the Apple Neural Engine.

2. **Spatial Binding** — Each defect box is bound to the apple box with the highest Intersection-over-Area ratio (IoA >= 0.10). This associates defects with specific fruit instances without centroid tracking.

3. **Grading** — A deterministic function counts bound defects by severity, computes coverage area as a percentage of the apple box, and applies thresholds from `grading_policy.yaml`:

   | Grade | Condition | Color |
   |-------|-----------|-------|
   | G1 | <= 2 mild defects, < 5% coverage | Green |
   | G2 | > 2 mild OR > 1 moderate OR 5-15% coverage | Orange |
   | G3 | Any severe defect OR > 15% coverage | Red |
   | DISCARD | Class 1 proximity trigger | Red |

4. **Edge Harvesting** — Low-confidence detections (0.40-0.65) are saved with telemetry for active learning. Disabled in benchmark mode (COCO placeholder model).

5. **Operator Override** — Press `g` during inference to log a grading disagreement. Overrides are persisted separately from general harvest data for review.

---

## Class Schema

13 classes defined in `data.yaml`. Severity buckets defined in `grading_policy.yaml`.

| Index | Class | Role | Severity |
|------:|-------|------|----------|
| 0 | `apple` | Macro parent box (one per fruit) | — |
| 1 | `unfit_bin_discard` | Discard override trigger | — |
| 2 | `z_bruise` | Surface defect | Mild |
| 3 | `z_russeting` | Surface defect | Mild |
| 4 | `z_scarf_skin` | Surface defect | Moderate |
| 5 | `z_sunburn` | Surface defect | Moderate |
| 6 | `z_stem_puncture` | Surface defect | Moderate |
| 7 | `z_split_crack` | Surface defect | Severe |
| 8 | `z_misshapen` | Surface defect | Severe |
| 9 | `z_scab` | Surface defect | Moderate |
| 10 | `z_sooty_blotch_flyspeck` | Surface defect | Moderate |
| 11 | `z_rot` | Surface defect | Severe |
| 12 | `z_insect_damage` | Surface defect | Severe |

Adding a new defect class requires updating `data.yaml`, adding it to a severity bucket in `grading_policy.yaml`, and retraining. No inference code changes needed — the schema is discovered at runtime via `len(model.names)`.

---

## Hardware

| Component | Spec |
|-----------|------|
| Host | MacBook Air (Apple M4) |
| OS | macOS 26 (Tahoe) |
| Camera | Arducam OV9782 USB Global Shutter (auto-detected via VID/PID) |
| Acceleration | Apple Neural Engine via CoreML FP16 |
| Inference Resolution | 640x640 |
| Camera Capture | 1280x720 MJPG |

Camera auto-detection lives in `camera_utils.py`. If the Arducam isn't found, the system falls back to the built-in camera with a warning. The inference loop includes warmup reads and bounded retry logic for USB disconnects.

---

## Quick Start

```bash
# Create venv and install dependencies
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.lock.txt

# Run inference (requires camera + model file)
python local_inference.py

# Run with a specific facility policy
python local_inference.py --policy grading_policy.yaml

# Headless benchmarking (no display, faster FPS)
python local_inference.py --no-display

# Verify hardware + baseline inference
python baseline_verify.py

# Compare PyTorch vs CoreML performance
python baseline_verify.py --compare
```

### Operator Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `g` | Log grading disagreement (operator override) |
| `space` | Capture dataset frame (in `capture_dataset.py`) |

---

## Project Files

| File | Purpose |
|------|---------|
| `local_inference.py` | Production inference loop with grading, display, and edge harvesting |
| `baseline_verify.py` | Hardware verification + PyTorch/CoreML benchmark comparison |
| `capture_dataset.py` | Raw frame capture from Arducam (1280x720, multi-angle) |
| `camera_utils.py` | Arducam auto-detection via system_profiler (VID/PID + resolution) |
| `kernel_dispatch.py` | Hardware-agnostic kernel dispatch layer |
| `kernel_apple_coreml.py` | CoreML FP16/INT8 backend for ANE (4-11x speedup over PyTorch MPS) |
| `edge_harvest_schema.py` | Typed telemetry schema for harvested frames (validation + persistence) |
| `override_persistence.py` | Operator override logging (separate from general harvest) |
| `grading_policy.yaml` | Facility grading rules (severity buckets, thresholds, IoA binding) |
| `data.yaml` | 13-class dataset schema for training |
| `environment.yaml` | Conda environment spec (Python 3.13, ultralytics 8.4.90) |
| `requirements.lock.txt` | Pinned dependency lockfile |
| `scripts/reingest_harvest.py` | Promote harvested frames back into raw_ingest for re-annotation |
| `scripts/override_report.py` | Daily operator override summary report |
| `docs/annotation_sop.md` | Annotation standard operating procedure for labelers |
| `.github/workflows/ci.yml` | CI: pyflakes + compile check on push and PR |

---

## Policy Engine

The grading engine reads severity mappings and thresholds from `grading_policy.yaml` at runtime. Different facilities can have different policies — swap them with the `--policy` flag:

```yaml
# grading_policy.yaml
facility_id: "DSM_COLD_STORAGE_01"

severity_mapping:
  mild_defects: [z_bruise, z_russeting]
  moderate_defects: [z_scarf_skin, z_sunburn, z_stem_puncture, z_scab, z_sooty_blotch_flyspeck]
  severe_defects: [z_split_crack, z_misshapen, z_rot, z_insect_damage]

rules:
  max_mild_for_g1: 2
  max_moderate_for_g2: 1
  area_threshold_g2_pct: 5.0
  area_threshold_g3_pct: 15.0
  ioa_binding_threshold: 0.10
```

If the requested policy file doesn't exist, the system lists available `*_grading_policy.yaml` files and falls back to default rules.

---

## Status

| Milestone | Status |
|-----------|--------|
| Hardware integration + camera auto-detection | Complete |
| YOLO26x + CoreML FP16 ANE backend | Complete |
| Kernel dispatch architecture | Complete |
| Deterministic grading + spatial binding | Complete |
| Dynamic policy engine + facility profiles | Complete |
| Edge harvest telemetry schema | Complete |
| Operator override persistence | Complete |
| CI (lint + compile check) | Complete |
| Model training on apple dataset | In progress |
| CoreML export of trained model | Pending |
| Closed-loop continuous learning | Pending |

Currently running a COCO-pretrained placeholder model (80 classes) in benchmark mode. FPS numbers are valid; detections are not. Once a model trained on the 13-class apple dataset is loaded, grading and edge harvesting activate automatically.

---

## License

The system uses Ultralytics YOLO26 (AGPL-3.0). Be aware of copyleft obligations if commercializing.
