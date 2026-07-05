# Annotation Standard Operating Procedure — YOLO26 13-Class Dataset

> **Scope:** This SOP governs manual labeling of raw frames captured by
> `capture_dataset.py` (1280×720 MJPG, 4 views per fruit) into the YOLO26
> object-detection dataset defined by `data.yaml`.

---

## 1. Purpose & Scope

This SOP exists to enforce the project's core architectural principle:
**detection ≠ decision**. The annotator's job is to produce a faithful
*spatial ground truth* of apples, discard triggers, and surface defects —
nothing more. Severity buckets, grade assignment (G1/G2/G3/DISCARD), and
discard-override logic are computed deterministically by the grading engine
from `grading_policy.yaml` at inference time. By keeping judgment out of the
labels, facility rule changes can be made in policy without re-annotating the
dataset. This document is the single source of truth for box conventions,
class usage, spatial binding, edge cases, and Roboflow project settings for
every labeler and reviewer on the project.

---

## 2. Class Manifest

All 13 classes are defined in `data.yaml`. Severity buckets are defined in
`grading_policy.yaml`. Class names **must match `data.yaml` exactly** —
including the `z_` prefix on defect classes.

| Index | Class Name              | Type     | Severity Bucket | Labeling Rule |
|------:|-------------------------|----------|-----------------|---------------|
| 0     | `apple`                 | Parent   | —               | One macro box per fruit, tight around the whole visible fruit. |
| 1     | `unfit_bin_discard`     | Override | —               | Box the discard signal (fruit on floor, badly crushed, gross contamination), not the whole fruit. |
| 2     | `z_bruise`              | Defect   | Mild            | Micro box tight around the bruised region only. |
| 3     | `z_russeting`           | Defect   | Mild            | Micro box around the russeted (corky/brown) skin patch. |
| 4     | `z_scarf_skin`          | Defect   | Moderate        | Micro box around the waxy, smeared-skin overlay region. |
| 5     | `z_sunburn`             | Defect   | Moderate        | Micro box around the bleached/sunken sun-scald patch. |
| 6     | `z_stem_puncture`       | Defect   | Moderate        | Micro box around the stem-puncture wound. |
| 7     | `z_split_crack`         | Defect   | Severe          | Micro box around the split/crack opening. |
| 8     | `z_misshapen`           | Defect   | Severe          | Micro box around the deformed region that deviates from round. |
| 9     | `z_scab`                | Defect   | Moderate        | Micro box around the olive-green/black scab lesion cluster. |
| 10    | `z_sooty_blotch_flyspeck` | Defect | Moderate        | Micro box around the sooty blotch / flyspeck fungal complex. |
| 11    | `z_rot`                 | Defect   | Severe          | Micro box around the decayed/rotten tissue. |
| 12    | `z_insect_damage`       | Defect   | Severe          | Micro box around the insect-feeding / entry-wound area. |

> **Note on severity:** The "Severity Bucket" column is for reference only.
> Annotators do **not** assign severity. Severity is resolved by the grading
> engine from `grading_policy.yaml` at inference time.

---

## 3. Bounding Box Conventions

The dataset uses three distinct box roles. Getting these right is the most
important part of the job.

### 3.1 Class 0 — `apple` (Macro Parent Box)

- One box per fruit — the **instance root**.
- Tight around the **entire visible fruit**, not just one face.
- Axis-aligned rectangle (no oriented/rotated boxes).
- This box is what every defect binds to via IoA.

### 3.2 Class 1 — `unfit_bin_discard` (Override Trigger Box)

- Box the **discard signal**, not the whole fruit.
- Examples: fruit on the floor, fruit crushed flat, gross contamination/mold
  covering the fruit, fruit in the reject bin.
- This class triggers an immediate rejection at inference time via proximity
  override — it does not require a parent apple box.

### 3.3 Classes 2-N — Defects (Micro Boxes)

- Box the **defect region only**, NOT the whole fruit.
- Tight to the visible boundary of the defect.
- One box per contiguous defect region. If a defect has two separate patches
  on the same apple, draw two boxes.
- Axis-aligned rectangles only.

```
┌─────────────────────────────────┐
│  Class 0: apple (macro parent)  │
│                                 │
│   ┌──────┐         ┌────────┐   │
│   │bruise│         │  scab  │   │   <- micro defect boxes (2-N)
│   └──────┘         └────────┘   │
└─────────────────────────────────┘
```

---

## 4. Spatial Binding Rule (IoA)

Every defect box (classes 2-N) **must** overlap a parent `apple` box with an
Intersection-over-Area ratio of at least **0.10**:

```
IoA = Intersection Area / Defect Area
Binding condition: IoA ≥ 0.10
```

This threshold is defined in `grading_policy.yaml` (`ioa_binding_threshold`).

**Annotator responsibility:**
- Ensure each defect box actually overlaps an apple box.
- An **orphan defect** (no overlapping apple box) is **invalid** and must be
  removed or the missing apple box added.
- Aim for the defect box to sit fully inside the apple box when possible; the
  0.10 threshold is a floor, not a target.

---

## 5. Edge Cases

| Case | Rule |
|------|------|
| **Occluded apple** | Box the **visible portion** only. If Roboflow supports a comment/note field, note "occluded" on the `apple` box. |
| **Clustered apples** | **One macro box per fruit.** Never group multiple apples into a single box. Overlapping apple boxes are acceptable. |
| **Defect spanning multiple apples** | Draw **one defect box** around the defect region. At inference time it binds to whichever apple has the highest IoA. Do not split it. |
| **Ambiguous mild vs moderate** (e.g. small bruise vs large bruise) | Label the **defect class** (`z_bruise`). Severity is computed by the grading engine from `grading_policy.yaml` — the annotator does not decide mild/moderate/severe. |
| **Defect on the fruit edge / partially out of frame** | Box the visible defect region. Ensure it still overlaps the visible apple box with IoA ≥ 0.10. |
| **Multiple defects of the same class on one apple** | Draw a separate micro box for each contiguous region. Do not merge non-adjacent patches into one box. |
| **Fruit with no visible defects** | Draw only the `apple` macro box. No defect boxes. |
| **Discard signal present** | Draw the `unfit_bin_discard` box around the signal. An `apple` macro box is optional here — add one only if the fruit body is clearly visible and distinct. |

---

## 6. Roboflow Project Settings

| Setting | Value |
|---------|-------|
| Project type | Object Detection |
| Annotation group | Per-class |
| Job size | 50 images per job |
| Reviewer assignment | 10% random sample review |
| Train / val / test split | 70 / 20 / 10 |
| Export format | YOLO26 |
| Oriented bounding boxes (OBB) | **OFF** — axis-aligned boxes only |

> **Capture context:** Raw frames arrive from `capture_dataset.py` at
> 1280×720 MJPG, 4 views per fruit, stored under `dataset/raw_ingest/`.
> Upload these directly to Roboflow without resizing — the export pipeline
> handles resolution.

---

## 7. Labeler QA Checklist

Run through this checklist before marking a job complete:

- [ ] **Every apple has a macro `apple` (class 0) box** — no fruit left unboxed.
- [ ] **Every defect box overlaps an `apple` box with IoA ≥ 0.10** — no orphan defects.
- [ ] **No duplicate boxes** — each region is boxed exactly once; no overlapping same-class duplicates.
- [ ] **Class names match `data.yaml` exactly** — including the `z_` prefix on all defect classes.
- [ ] **Defect boxes are micro (defect region only)** — no defect box encloses the whole fruit.
- [ ] **Severity is NOT assigned by the labeler** — no severity tags or notes; the policy engine handles it.
- [ ] **One apple box per fruit** — clustered fruits are split into individual boxes, never grouped.

---

## 8. Cross-Reference Note — Schema ↔ Policy Sync

The severity buckets in `grading_policy.yaml` **must** cover every defect class
in `data.yaml`. The two files are coupled:

- `data.yaml` defines the class manifest (indices 2-N are dynamic defects).
- `grading_policy.yaml` maps each defect class into one of three buckets:
  `mild_defects`, `moderate_defects`, `severe_defects`.

**Rule:** If a new defect is added to `data.yaml`, it **MUST** be added to a
severity bucket in `grading_policy.yaml` in the **same PR**. A defect class
that exists in `data.yaml` but is missing from `grading_policy.yaml` will be
silently ignored by the grading engine and produce no grade contribution.

Current mapping (verified against both files):

| Bucket | Classes |
|--------|---------|
| `mild_defects` | `z_bruise`, `z_russeting` |
| `moderate_defects` | `z_scarf_skin`, `z_sunburn`, `z_stem_puncture`, `z_scab`, `z_sooty_blotch_flyspeck` |
| `severe_defects` | `z_split_crack`, `z_misshapen`, `z_rot`, `z_insect_damage` |

---

## 9. File References

| File | Role |
|------|------|
| `data.yaml` | 13-class schema; class indices and names. |
| `grading_policy.yaml` | Severity buckets, grade thresholds, IoA binding threshold. |
| `capture_dataset.py` | Raw frame capture (1280×720, 4 views/fruit → `dataset/raw_ingest/`). |
| `README.md` | System architecture, class dictionary, spatial binding engine spec. |
