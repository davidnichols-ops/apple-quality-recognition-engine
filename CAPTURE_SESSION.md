# Apple Quality Capture Session — Full Run

Target: ~110 profiles, ~440 equatorial frames + ~110 stem frames + 30 black negatives
Time: 4-5 hours

## Setup

```bash
cd ~/Projects/apple-quality-recognition-engine
```

All commands use `ARDUCAM_CAMERA_INDEX=0` and the project venv. Run each from Terminal.

## 0. Calibrate White Balance (1 min — do this ONCE at the start)

Turn on your 5600K lighting. Let it warm up for a minute. Hold a white or gray
card in front of the camera, fill the center green box, press SPACE.

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python wb_lock.py --calibrate
```

This saves `wb_calibration.json`. All subsequent captures (negatives, equatorial,
stem) automatically load and apply these gains. You only need to recalibrate if
you change lighting or replug the camera.

## 1. Black Background Negatives (5 min)

Capture 30 frames of the empty matte backdrop. No fruit, no gloves, no turntable
label. Same 5600K lighting.

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_simple.py
```

Press SPACE 30 times. ESC to quit. Files save to `dataset/raw_ingest/` with
WB calibration applied.

## 2. Sort Your Fruit First (20 min)

Before turning on the camera, sort your apple pile by grade:
- DISCARD — active rot, soft decay, punctures (capture first, rotting doesn't wait)
- CIDER — heavy russet/hail/scab, sound flesh, no rot
- G3 — moderate cosmetic defects
- G2 — minor blemishes
- G1 — clean

## 3. Equatorial Passes (turntable, 4 shots per apple)

Each run prints `[CAMERA] Software WB lock loaded: R=... G=... B=... (target 5600K)`
at startup confirming the calibration is active.

### 3a. DISCARD — 15 apples

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --grade DISCARD --count 15 --batch-id batch-20260903-discard --grader-id david --lot-id lot-discard-01 --cultivar mixed
```

### 3b. CIDER — 50 apples

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --grade CIDER --count 50 --batch-id batch-20260903-cider --grader-id david --lot-id lot-cider-01 --cultivar mixed
```

### 3c. G3 — 20 apples

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --grade G3 --count 20 --batch-id batch-20260903-g3 --grader-id david --lot-id lot-g3-01 --cultivar mixed
```

### 3d. G2 — 20 apples

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --grade G2 --count 20 --batch-id batch-20260903-g2 --grader-id david --lot-id lot-g2-01 --cultivar mixed
```

### 3e. G1 — 20 apples

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --grade G1 --count 20 --batch-id batch-20260903-g1 --grader-id david --lot-id lot-g1-01 --cultivar mixed
```

## 4. Stem/Calyx Passes (no turntable, flip apples stem-down)

After all equatorial passes are done. Same apples, same order, same grade — just
flip them stem-down and press SPACE. No rush, no turntable. WB calibration
applied automatically.

### 4a. DISCARD stems — 15 photos

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --mode stem --grade DISCARD --count 15 --batch-id batch-20260903-discard-stem --grader-id david
```

### 4b. CIDER stems — 50 photos

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --mode stem --grade CIDER --count 50 --batch-id batch-20260903-cider-stem --grader-id david
```

### 4c. G3 stems — 20 photos

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --mode stem --grade G3 --count 20 --batch-id batch-20260903-g3-stem --grader-id david
```

### 4d. G2 stems — 20 photos

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --mode stem --grade G2 --count 20 --batch-id batch-20260903-g2-stem --grader-id david
```

### 4e. G1 stems — 20 photos

```bash
ARDUCAM_CAMERA_INDEX=0 .venv/bin/python capture_dataset.py --mode stem --grade G1 --count 20 --batch-id batch-20260903-g1-stem --grader-id david
```

## 5. Verify

```bash
# Count captured frames
ls dataset/raw_ingest/*.jpg | wc -l

# Check manifest
wc -l dataset/raw_ingest/capture_manifest.jsonl

# Quick grade breakdown
python3 -c "
import json
from collections import Counter
grades = Counter()
with open('dataset/raw_ingest/capture_manifest.jsonl') as f:
    for line in f:
        r = json.loads(line)
        grades[r['reference_grade']] += 1
for g in ['G1','G2','G3','CIDER','DISCARD']:
    print(f'{g}: {grades.get(g, 0)} frames')
"

# Check WB calibration is loaded
.venv/bin/python wb_lock.py --show
```

## Expected Output

| Grade | Equatorial (4x) | Stem (1x) | Total |
|-------|-----------------|-----------|-------|
| DISCARD | 60 | 15 | 75 |
| CIDER | 200 | 50 | 250 |
| G3 | 80 | 20 | 100 |
| G2 | 80 | 20 | 100 |
| G1 | 80 | 20 | 100 |
| Negatives | — | — | 30 |
| **Total** | **500** | **125** | **655** |

## Controls

- SPACE = capture / advance
- ESC = quit current batch (safe, saves what's been captured)

## Tips

- **Calibrate WB first** — turn on 5600K lights, let warm up, run `wb_lock.py --calibrate`
- Capture DISCARD first — rotting fruit degrades fast
- Sort by grade before touching the camera
- CIDER: grab variety (russet, hail, scab, limb rub) not just the first 50
- Stem passes: no rush, turntable is off, take your time framing
- If the camera stalls: unplug Arducam USB, replug, wait 5 seconds, re-run
- If the window doesn't appear: run from Terminal (not from Devin), macOS camera permissions require GUI context
- All scripts print the WB gains at startup — verify you see `Software WB lock loaded` before capturing
