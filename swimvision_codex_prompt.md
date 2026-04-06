# SwimVision — Codex Handoff Prompt

## Instructions for Codex

- **Do not hallucinate APIs, functions, or library methods.** If you are unsure whether a method exists, check the library docs or use a simpler alternative you are certain of.
- **Do not leave placeholder comments** like `# TODO`, `# implement this`, or `# add logic here`. Every function must be fully implemented.
- **Do not stub out functions.** If a function is defined, it must work end to end.
- **Do not skip error handling.** Every file I/O operation, model forward pass, and external library call must have a try/except with a descriptive error message.
- **Follow the repo structure exactly** as specified below. Do not add, rename, or merge files.
- **Implement files in the order listed** under "Build order." Do not jump ahead.
- **After each file, write a one-line comment at the top** describing what the file does.
- **All functions must have docstrings** with args, returns, and a one-sentence description.
- **Use only these libraries:** Python stdlib, PyTorch, MediaPipe, OpenCV, NumPy, SciPy, scikit-learn, Weights & Biases, ReportLab. Do not introduce any other dependencies without noting it explicitly.

---

## Project: SwimVision — Competitive Swim Start Biomechanical Analyzer

An end-to-end computer vision pipeline that analyzes competitive swimming track starts from video and outputs a structured joint angle deviation report, comparing the swimmer against published biomechanical optimal ranges and a learned pro distribution.

---

## Repo Structure

```
swim-vision/
├── data/
│   ├── raw/                  # uploaded video clips (any angle, any resolution)
│   ├── processed/            # extracted keypoints as .npy
│   └── labels.csv            # clip_id, swimmer_id, level, camera_angle, notes
├── src/
│   ├── ingest.py             # video ingestion, frame extraction, phase detection
│   ├── extract.py            # MediaPipe pose extraction + confidence scoring
│   ├── phases/
│   │   ├── __init__.py
│   │   ├── block.py          # block phase analysis: take your marks → explosion off block
│   │   ├── flight.py         # flight phase analysis: feet leave block → water entry
│   │   └── entry.py          # entry phase analysis: water contact → full streamline
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── joint_angles.py   # compute all joint angles from keypoints per frame
│   │   ├── deviation.py      # compare computed angles vs optimal reference ranges
│   │   └── reaction_time.py  # detect beep frame → feet-off-block frame delta
│   ├── reference/
│   │   ├── __init__.py
│   │   └── optimal_ranges.py # published biomechanical optimal angle ranges per phase
│   ├── overlay.py            # skeleton overlay renderer
│   ├── report.py             # JSON + annotated video + PDF summary output
│   └── train/
│       ├── __init__.py
│       ├── dataset.py        # PyTorch Dataset over keypoint sequences
│       ├── classifier.py     # CNN+LSTM phase boundary detector
│       └── train.py          # training loop with W&B logging
├── notebooks/
│   └── analysis.ipynb        # angle distributions, deviation heatmaps, pro vs novice
├── results/
├── requirements.txt
└── README.md
```

---

## Build Order

Implement files strictly in this order. Do not proceed to the next file until the current one is complete and functional.

1. `requirements.txt`
2. `src/reference/optimal_ranges.py`
3. `src/ingest.py`
4. `src/extract.py`
5. `src/metrics/joint_angles.py`
6. `src/metrics/deviation.py`
7. `src/metrics/reaction_time.py`
8. `src/phases/block.py`
9. `src/phases/flight.py`
10. `src/phases/entry.py`
11. `src/overlay.py`
12. `src/train/dataset.py`
13. `src/train/classifier.py`
14. `src/train/train.py`
15. `src/report.py`
16. `README.md`

---

## Full Implementation Spec

### `src/reference/optimal_ranges.py`

Hardcode published biomechanical optimal ranges for competitive track starts derived from peer-reviewed sports science literature on elite swimmer starts.

```python
OPTIMAL_RANGES = {
    "block_phase": {
        "front_knee_angle":  (90, 110),   # degrees, at set position
        "rear_knee_angle":   (110, 130),
        "hip_angle":         (55, 75),
        "torso_lean":        (15, 30),    # forward lean from vertical
    },
    "flight_phase": {
        "body_linearity":    (0, 8),      # max hip deviation in degrees
        "entry_angle":       (30, 45),    # relative to water surface
        "elbow_extension":   (165, 180),  # near full extension at entry
    },
    "entry_phase": {
        "streamline_angle":  (0, 10),     # hands-to-hips vs horizontal
        "elbow_lock_angle":  (170, 180),
    }
}

DEVIATION_THRESHOLDS = {
    "OPTIMAL":     0,
    "MINOR":       10,
    "SIGNIFICANT": 20,
    # anything above 20 degrees off = CRITICAL
}
```

Also include a `get_range(phase, metric)` function that returns the tuple or raises a `KeyError` with a helpful message.

---

### `src/ingest.py`

```
def extract_frames(video_path, output_dir, fps=30) -> dict
```
- Accept any video file format supported by OpenCV
- Extract frames at the specified FPS
- Detect water surface line using Canny edge detection + Hough line transform on the lower third of the frame
- Return metadata dict: `{fps, resolution, total_frames, water_surface_y, output_dir}`
- Save frames as `frame_{n:06d}.jpg` in output_dir
- If water surface cannot be detected, set `water_surface_y` to 75% of frame height and log a warning

```
def detect_phase_boundaries(keypoints_path, confidence_path, metadata) -> dict
```
- Load keypoints array (shape: `[T, 33, 4]`) and confidence array (shape: `[T, 33]`)
- Detect block phase start: first frame where hip keypoint (index 23 or 24) visibility > 0.7 and center-of-mass y-velocity < threshold
- Detect flight phase start: frame where both ankle keypoints (27, 28) velocity exceeds 2.0 px/frame (explosion off block)
- Detect entry phase start: frame where wrist keypoints (15, 16) decelerate sharply (velocity sign flip + magnitude drop > 50%)
- Return: `{block_start, block_end, flight_start, flight_end, entry_start, entry_end}` as frame indices

---

### `src/extract.py`

```
def extract_keypoints(frames_dir, output_dir, clip_id) -> tuple[np.ndarray, np.ndarray]
```
- Run MediaPipe Pose (model_complexity=2) on every frame in frames_dir
- Return keypoints array shape `[T, 33, 4]` (x, y, z, visibility) and confidence array shape `[T, 33]`
- Flag any keypoint with visibility < 0.6 as low-confidence (store as 0.0 in confidence array)
- Save as `{clip_id}_keypoints.npy` and `{clip_id}_confidence.npy` in output_dir
- Normalize x, y coordinates to [0, 1] relative to frame resolution
- Log per-frame mean confidence and flag frames where mean confidence < 0.5

MediaPipe keypoint indices to use:
- Shoulders: 11 (left), 12 (right)
- Elbows: 13 (left), 14 (right)
- Wrists: 15 (left), 16 (right)
- Hips: 23 (left), 24 (right)
- Knees: 25 (left), 26 (right)
- Ankles: 27 (left), 28 (right)

---

### `src/metrics/joint_angles.py`

```
def angle_between(a, b, c) -> float
```
Compute angle at point b formed by vectors b→a and b→c. Use arctan2, return degrees.

```
def compute_all_angles(keypoints) -> pd.DataFrame
```
Input: keypoints array shape `[T, 33, 4]`
Output: DataFrame indexed by frame with these columns:
- `front_knee_angle`: hip(23) → knee(25) → ankle(27)
- `rear_knee_angle`: hip(24) → knee(26) → ankle(28)
- `hip_angle`: shoulder_midpoint → hip_midpoint → knee_midpoint
- `torso_lean`: angle of shoulder_midpoint → hip_midpoint vector vs vertical axis
- `left_elbow_angle`: shoulder(11) → elbow(13) → wrist(15)
- `right_elbow_angle`: shoulder(12) → elbow(14) → wrist(16)
- `body_linearity`: deviation of hip_midpoint from the line between shoulder_midpoint and ankle_midpoint, in degrees
- `entry_angle`: wrist_midpoint → ankle_midpoint vector vs horizontal (computed only at entry frame)
- `streamline_angle`: wrist_midpoint → hip_midpoint vector vs horizontal

Use only x, y coordinates (index 0 and 1) for all 2D angle computations.

---

### `src/metrics/deviation.py`

```
def score_deviation(measured_angle, optimal_range) -> tuple[float, str]
```
- Compute deviation = 0 if within range, else distance to nearest boundary
- Return (deviation_degrees, flag) where flag is one of: "OPTIMAL", "MINOR", "SIGNIFICANT", "CRITICAL"

```
def compute_deviations(angles_df, phase, phase_boundaries) -> pd.DataFrame
```
- For each metric relevant to the given phase, compute deviation score and flag
- Average angle values over the phase frame window
- Return DataFrame with columns: `metric, measured, optimal_min, optimal_max, deviation, flag`

```
def aggregate_report(block_dev, flight_dev, entry_dev) -> dict
```
- Combine all three phase deviation DataFrames into a single structured dict
- Include overall severity: worst flag across all metrics

---

### `src/metrics/reaction_time.py`

```
def detect_reaction_time(keypoints, audio_beep_frame, fps) -> float
```
- `audio_beep_frame`: frame index when the starting beep occurs (passed in by caller)
- Detect feet-off-block as first frame where both ankle keypoints have upward velocity > 1.5 px/frame
- Return reaction time in milliseconds: `(feet_off_frame - audio_beep_frame) / fps * 1000`
- Return None if detection fails, log a warning

---

### `src/phases/block.py`, `flight.py`, `entry.py`

Each module exposes one function:

```
def analyze(keypoints, angles_df, phase_boundaries, metadata) -> dict
```

Returns a dict of key metrics for that phase:
- Block: front_knee_angle, rear_knee_angle, hip_angle, torso_lean (averaged over block window)
- Flight: body_linearity (max over flight window), entry_angle (at final flight frame), elbow_extension (at final flight frame)
- Entry: streamline_angle (averaged over entry window), elbow_lock_angle (averaged over entry window)

---

### `src/overlay.py`

```
def render_overlay(frames_dir, keypoints, angles_df, deviations, phase_boundaries, output_path)
```

Render annotated video with:
- **Joints** as filled circles (radius 5px), color-coded by deviation flag:
  - Green `(0, 200, 0)` = OPTIMAL
  - Yellow `(0, 200, 200)` = MINOR
  - Orange `(0, 140, 255)` = SIGNIFICANT
  - Red `(0, 0, 220)` = CRITICAL
  - Gray `(150, 150, 150)` = not analyzed in this phase
- **Bones** as lines (thickness 2px) connecting joint pairs using standard MediaPipe skeleton connectivity (use the 32-pair connectivity list from MediaPipe docs)
- **Angle values** rendered as white text (font scale 0.4) next to each joint
- **Phase label** in top-left corner: "BLOCK", "FLIGHT", or "ENTRY" in bold white text
- **Deviation flag** per joint in small text below the angle value
- Output: `.mp4` at same resolution and FPS as input using `cv2.VideoWriter`

---

### `src/train/dataset.py`

```
class SwimStartDataset(Dataset)
```
- Load keypoints sequences from `data/processed/`
- Read phase labels from `labels.csv`
- Return sequences of length T=90 frames (pad or truncate), normalized keypoints, and phase label as integer (0=BLOCK, 1=FLIGHT, 2=ENTRY, 3=OTHER)
- Implement `__len__` and `__getitem__`

---

### `src/train/classifier.py`

```
class PhaseClassifier(nn.Module)
```
CNN + LSTM that classifies each frame into a phase:
- Input: `[batch, T, 33, 2]` (x, y keypoints per frame)
- CNN: flatten 33×2 → Linear(66, 128) → ReLU → Linear(128, 64) per frame (applied identically across T)
- LSTM: hidden_size=128, num_layers=2, bidirectional=True, dropout=0.3
- Output head: Linear(256, 4) → softmax over 4 phase classes
- Implement `forward(x) -> logits`

---

### `src/train/train.py`

Full training loop:
- CrossEntropyLoss
- Adam optimizer, lr=1e-3, weight decay=1e-4
- 50 epochs, batch size 16
- Log train loss, val loss, val accuracy to W&B every epoch
- Save best checkpoint to `results/best_classifier.pt`
- Early stopping: patience=10 epochs
- Train/val split: 80/20 from `labels.csv`

---

### `src/report.py`

```
def generate_report(clip_id, deviations, reaction_time, annotated_video_path, output_dir)
```

Output three files:
1. **JSON** (`{clip_id}_report.json`): all angles, deviations, phase timestamps, reaction time, overall severity flag
2. **Annotated video** (already generated by `overlay.py`, just reference path in JSON)
3. **PDF** (`{clip_id}_report.pdf`) using ReportLab:
   - Page 1: title, swimmer ID, date, overall severity
   - Page 2: phase breakdown table (metric | measured | optimal range | deviation | flag), color-coded rows by flag
   - Page 3: key flagged issues (SIGNIFICANT and CRITICAL only), plain-language descriptions of what each deviation means biomechanically

---

### `README.md`

Write as a mini research paper with these sections:

**Abstract** (4–5 sentences): problem (quantifying swim start technique without expensive motion capture), approach (MediaPipe keypoint extraction + geometric angle computation + phase segmentation), contribution (fully automatic three-phase segmentation from monocular video using keypoint velocity thresholds + pro distribution learning), results placeholder.

**Introduction** — why race start optimization matters, current state of technique analysis in competitive swimming, gap this fills.

**Related Work** — cite 3 real peer-reviewed papers on competitive swimming start biomechanics (use real papers that exist, do not fabricate citations).

**Methods** — pipeline stages, phase detection approach, joint angle computation, deviation scoring.

**Experiments** — dataset description, train/val split, evaluation metrics.

**Results** — placeholder tables for: phase detection accuracy, per-joint deviation distributions, pro vs novice comparison.

**Limitations and Future Work** — monocular depth limitation, underwater entry occlusion, extension to grab start.

**Novel contribution to highlight:** fully automatic three-phase segmentation of competitive swim starts from monocular consumer video using keypoint velocity thresholds, enabling technique deviation scoring without motion capture, manual annotation, or specialized equipment.

---

## Pro vs Novice Learning

In `notebooks/analysis.ipynb`, implement:
- Load all processed keypoints grouped by `level` column in `labels.csv` (pro / competitive / novice)
- For each joint angle metric, compute mean and std per level per phase
- Plot overlapping histograms (pro = blue, competitive = orange, novice = red) for each metric
- Compute a `pro_distribution` dict: `{phase: {metric: (mean, std)}}` saved as `results/pro_distribution.json`
- When analyzing a new video in `deviation.py`, compare against both `OPTIMAL_RANGES` (hard-coded) and `pro_distribution` (data-driven), and include both comparisons in the report

---

## Data Format

`labels.csv` columns:
```
clip_id, swimmer_id, level, camera_angle, notes
```
Where `level` is one of: `pro`, `competitive`, `novice`
Where `camera_angle` is one of: `side`, `45_degree`, `overhead`, `unknown`

Pro footage: downloadable from YouTube using `yt-dlp`. Document the exact yt-dlp commands used to collect data in a `data/README.md`.

---

## Key Constraints

- Target hardware: MacBook Air M4 (Apple Silicon). Use `torch.device("mps")` if available, fall back to CPU.
- MediaPipe model complexity: use 2 (heavy) for accuracy, add a `--fast` CLI flag that drops to complexity 1.
- All scripts must be runnable from the command line with argparse. Example: `python src/extract.py --input data/raw/clip_01.mp4 --output data/processed/ --clip_id clip_01`
- Log all major steps to stdout with timestamps using Python's `logging` module at INFO level.
