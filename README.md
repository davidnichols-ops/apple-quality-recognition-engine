# Plant Health Recognition Engine

An edge-first vision system for grading household plant health on Apple Silicon.

**Models observe. Deterministic policy decides. Humans authorize learning.**

This system separates perception from decision-making:

- **YOLO26n** detects plants, individual leaves, discard triggers, and defect regions.
- A **deterministic 3-tier grading engine** (driven by `grading_policy.yaml`) binds defects to leaves, grades each leaf, then aggregates leaf grades into a plant health grade.

This separation keeps operational logic out of model weights, making policy changes fast and reliable without retraining.

---

## 3-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PLANT (parent box)                         │
│                 Class 0: plant (instance root)                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LEAF (child instance box)                   │   │
│  │              Class 1: leaf                               │   │
│  │                                                          │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │         DEFECT (grandchild box)                  │    │   │
│  │  │         Class 3: class_defect                    │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │         DEFECT (grandchild box)                  │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              LEAF (child instance box)                   │   │
│  │              (no defects → HEALTHY)                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Grading flow (bottom-up)

1. **Bind defects to leaves** — each `class_defect` is bound to the leaf with the highest Intersection-over-Area (IoA) above threshold.
2. **Grade each leaf** — HEALTHY / MODERATE / POOR / DISCARD based on defect count and clipped union coverage area.
3. **Bind leaves to plants** — each leaf is bound to the plant with the highest IoA above threshold.
4. **Grade each plant** — from the aggregate of leaf grades (worst leaf + percentage of unhealthy leaves).
5. **Discard override** — `unfit_discard` boxes near a plant trigger immediate DISCARD.

---

## 4-Class Detector Schema

| ID | Class | Role |
|---:|---|---|
| 0 | `plant` | Parent bounding box for overall plant health |
| 1 | `leaf` | Individual leaf instance (intermediate child) |
| 2 | `unfit_discard` | Discard trigger — dead/dying plant signal |
| 3 | `class_defect` | Generic defect on a leaf (disease, damage, pest) |

Why one generic defect class:

- concentrates available examples instead of starving rare defect types;
- makes annotation and Roboflow label assist easier to review;
- lets leaf-level geometry, count, and coverage carry the first production decision;
- defers taxonomic expansion until confusion data proves a specific split adds grade value.

---

## Health Grades

| Grade | Meaning | Criteria (candidate) |
|---|---|---|
| HEALTHY | No significant issues | ≤2 defects per leaf, <5% coverage |
| MODERATE | Minor issues | 3-5 defects per leaf or 5-15% coverage |
| POOR | Significant damage | >5 defects per leaf or >15% coverage |
| DISCARD | Dead/dying or discard trigger | `unfit_discard` detected or all leaves poor |

---

## Production Flow

```text
Known-grade plant
  → five-view capture (4 equatorial + 1 top-down)
  → profile manifest
  → Roboflow annotation / reviewed label assist
  → profile-level train/val/test split
  → YOLO26n candidate training
  → held-out evaluation + M4 benchmark
  → human checkpoint promotion

Live plant
  → YOLO26n CoreML candidate
  → plant + leaf + defect boxes
  → defect-to-leaf IoA binding
  → leaf-to-plant IoA binding
  → deterministic per-leaf grade
  → aggregate plant grade (worst leaf + unhealthy %)
  → HEALTHY / MODERATE / POOR / DISCARD
  → review queue when confidence is volatile or coverage is near a boundary
```

---

## Dataset Pipeline

### 1. Download and merge from Roboflow Universe

```bash
export ROBOFLOW_API_KEY="your-key"
python scripts/download_universe_datasets.py \
    --output-dir plant_dataset \
    --api-key "$ROBOFLOW_API_KEY"
```

Downloads and merges multiple public plant disease datasets from Roboflow Universe into a unified 4-class YOLO dataset:

- **all_eggplant_diseases** (34K images) — has `Healthy Plant`, `Healthy Leaf`, and disease classes
- **Strawberry Diseases Detection** (2.7K images) — `Healthy Leaf` + disease classes
- **Plant Disease Detection** (5K images) — diverse leaf types + diseases
- **plant disease detection** (10K images) — multi-crop healthy + disease classes

Source class names are mapped to our 4-class schema. Missing `plant` parent boxes are computed from the union of leaf boxes.

### 2. Train YOLO26n

```bash
python scripts/train_yolo26n.py \
    --data plant_dataset/data.yaml \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --device mps
```

Trains from COCO-pretrained `yolo11n.pt` base weights, evaluates on the test split, and exports to CoreML for M4 Neural Engine deployment.

### 3. Capture known-grade profiles

```bash
python capture_dataset.py \
    --grade HEALTHY \
    --count 10 \
    --batch-id monstera-batch-20260831 \
    --grader-id david \
    --species monstera \
    --location living-room
```

### 4. Split profiles (no data leakage)

```bash
python scripts/split_profiles.py \
    --manifest dataset/raw_ingest/manifest.jsonl \
    --output-dir splits \
    --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 \
    --seed 42
```

All five views of one plant stay in the same partition.

---

## Local Inference

```bash
# CoreML (production — M4 Neural Engine)
python local_inference.py --model best.mlpackage

# PyTorch (development)
python local_inference.py --model runs/detect/plant_yolo26n/weights/best.pt

# Benchmark mode (FPS only, no grading)
python local_inference.py --benchmark
```

Keyboard controls:
- `q` — quit
- `g` — log operator override (saves frame + telemetry to edge harvest)

---

## Hardware & Deployment

| Component | Specification |
|---|---|
| Host | MacBook Air (Apple M4) |
| OS | macOS 26 (Tahoe) |
| Camera | Arducam USB Global Shutter / built-in webcam |
| Acceleration | Apple Neural Engine (CoreML) |
| Resolution | 1280×720 MJPG |
| Model | YOLO26n (4-class OD) |
| Inference size | 640px (sandbox) / 1024px (production) |

---

## Key Files

| File | Purpose |
|---|---|
| `plant_grading_engine.py` | Deterministic 3-tier grading engine (pure, no I/O) |
| `grading_policy.yaml` | Candidate thresholds and severity mapping |
| `data.yaml` | 4-class YOLO dataset schema |
| `local_inference.py` | Live inference loop with CoreML/PyTorch |
| `capture_dataset.py` | Five-view known-grade capture with manifest |
| `baseline_verify.py` | Camera + model FPS benchmark |
| `scripts/download_universe_datasets.py` | Roboflow Universe multi-dataset download/merge |
| `scripts/train_yolo26n.py` | YOLO26n training pipeline |
| `scripts/split_profiles.py` | Deterministic profile-level dataset splitting |

---

## Status

This is an engineering prototype. The current model artifact is a COCO placeholder until the training pipeline is run against the merged Universe dataset. Production claims begin only after:

1. The merged dataset is downloaded and verified
2. YOLO26n is trained and evaluated on held-out profiles
3. A real household trial passes the gates

---

## License

See `LICENSE` for details.
