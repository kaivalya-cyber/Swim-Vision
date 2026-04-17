# SwimVision: Competitive Swim Start Biomechanical Analyzer

Last validated: April 2026 on macOS (Apple Silicon M4), `mediapipe==0.10.30`, Python 3.12.

## Abstract
SwimVision is a monocular video-analysis pipeline for quantifying competitive swim-start mechanics without motion-capture hardware. The current implementation combines MediaPipe pose extraction, crop-aware single-swimmer isolation, framewise joint-angle computation, three-phase segmentation, deviation scoring against literature-derived reference ranges, annotated video rendering, and JSON/PDF reporting. The pipeline has been validated on real Caeleb Dressel 2020 Olympic footage processed from `data/raw/dressel2020olympic.mov`, with confirmed keypoint extraction in the `0.74-0.85` mean-confidence range across the tracked portion of the clip and successful generation of annotated video and deviation reports.

## Introduction
Sprint swimming races are highly sensitive to start quality. Coaches often need fast feedback on block posture, flight alignment, and entry angle, but laboratory systems are expensive and difficult to deploy in day-to-day training. SwimVision addresses that gap by turning ordinary race or training footage into phase-specific biomechanical measurements. The current codebase is designed to run locally on macOS Apple Silicon hardware and has been debugged against real race footage rather than synthetic examples.

## Related Work
SwimVision uses the same high-level start-phase framing found in prior swim-start biomechanics research and compares measurements against literature-derived target ranges.

1. Julien Vantorre, Didier Chollet, and Ludovic Seifert. “Biomechanical Analysis of the Swim-Start: A Review.” *Journal of Sports Science & Medicine* 13(2), 2014, 223-231. [PubMed](https://pubmed.ncbi.nlm.nih.gov/24790473/) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3990873/)
2. Sebastian Fischer and Armin Kibele. “The biomechanical structure of swim start performance.” *Sports Biomechanics* 15(4), 2016, 397-408. [PubMed](https://pubmed.ncbi.nlm.nih.gov/27239685/) | [DOI](https://doi.org/10.1080/14763141.2016.1171893)
3. H. Galbraith, J. Scurr, C. Hencken, L. Wood, and P. Graham-Smith. “Biomechanical comparison of the track start and the modified one-handed track start in competitive swimming: an intervention study.” *Journal of Applied Biomechanics* 24(4), 2008, 307-315. [PubMed](https://pubmed.ncbi.nlm.nih.gov/19075299/) | [DOI](https://doi.org/10.1123/jab.24.4.307)

## Installation and Setup
### Python environment
The pose extractor uses the legacy MediaPipe Solutions API, not the current Tasks API. The important consequence is that the codebase cannot rely on `mediapipe.solutions` from a modern default install.

For a clean setup, create a Python 3.12 environment and install MediaPipe:

```bash
python3.12 -m venv .mp312
./.mp312/bin/pip install mediapipe==0.10.30
```

The exact package install command that should be documented for a fresh clone is:

```bash
pip install mediapipe==0.10.30
```

In practice, the repository also contains compatibility code because MediaPipe packaging is inconsistent across environments:

- [src/mediapipe/__init__.py](/Users/kaivalyasingh/Downloads/SwimVision/src/mediapipe/__init__.py) is a local shim that redirects imports to the repo-local MediaPipe package when needed.
- [sitecustomize.py](/Users/kaivalyasingh/Downloads/SwimVision/sitecustomize.py) sets local import and cache behavior early in interpreter startup.
- [src/extract.py](/Users/kaivalyasingh/Downloads/SwimVision/src/extract.py) forces a repo-local Python environment and imports `mediapipe.python.solutions.pose` directly.

### MediaPipe import path
Document this explicitly for anyone cloning the repo:

- `mediapipe.solutions` is not available in the current working setup used by this project.
- The codebase uses `mediapipe.python.solutions.pose` instead.
- The extractor already contains this import path and should not be changed back to `mp.solutions.pose`.

### Required environment variables
These must be set before any MediaPipe import. The extractor already sets them internally, but they are part of the required documented setup because they were necessary to stabilize execution on Apple Silicon:

```python
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
```

### Model assets
Model files are stored in [models/](/Users/kaivalyasingh/Downloads/SwimVision/models).

Files currently present:

- [models/pose_landmark_lite.tflite](/Users/kaivalyasingh/Downloads/SwimVision/models/pose_landmark_lite.tflite)
- [models/pose_landmarker_heavy.task](/Users/kaivalyasingh/Downloads/SwimVision/models/pose_landmarker_heavy.task)
- [models/openpose_pose_coco.prototxt](/Users/kaivalyasingh/Downloads/SwimVision/models/openpose_pose_coco.prototxt)
- [models/pose_iter_440000.caffemodel](/Users/kaivalyasingh/Downloads/SwimVision/models/pose_iter_440000.caffemodel)

Current usage:

- The active extraction path uses the legacy MediaPipe Solutions runtime and relies on `pose_landmark_lite.tflite`.
- `pose_landmarker_heavy.task` was downloaded during earlier MediaPipe Tasks debugging and remains in the repo as a reference asset.
- The OpenPose files remain from earlier debugging but are not used by the current pipeline because the OpenCV DNN fallback was removed.

How the files were obtained:

- MediaPipe assets were downloaded into `models/` during implementation to support local execution and fallback testing.
- The current code path copies `models/pose_landmark_lite.tflite` into the repo-local MediaPipe package location when needed.

## Running the Pipeline
### One-command pipeline
The simplest way to run the full project is now:

```bash
python src/run_pipeline.py --input data/raw/YOUR_VIDEO.mov --clip_id YOUR_ID
python src/run_pipeline.py --input data/raw/YOUR_VIDEO.mov --clip_id YOUR_ID --crop 0 0 430 730
```

This command runs extraction, phase detection, joint angles, deviation scoring, overlay rendering, and report generation in sequence, stopping on the first failure.

### Browser interface
The repository now includes a dedicated React frontend with a premium landing page and analysis studio, while Flask remains the backend for uploads, background jobs, and artifact downloads.

#### Build and run
Install web dependencies for the Flask + React app:

```bash
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
python webapp.py
```

Open the local URL printed by Flask. The site will:

- render the homepage from the Vite build in `frontend/dist/`
- upload clips through the Flask `/api/jobs` endpoint
- run the existing `src/run_pipeline.py` workflow in a background thread
- show live per-step progress and a polished results view
- expose the annotated MP4, JSON, CSV, and PDF artifacts directly in the browser

For full SwimVision processing (MediaPipe, OpenCV, Torch, etc.), install the heavy pipeline dependencies too:

```bash
pip install -r requirements.pipeline.txt
```

On Vercel, `api/` serverless routes run in a lightweight mode and reject new long-running pipeline jobs by design.

#### Frontend development
For UI iteration, run the frontend dev server separately:

```bash
cd frontend
npm install
npm run dev
```

The frontend source lives in `frontend/src/`. Local/full backend behavior is served by [webapp.py](/Users/kaivalyasingh/Downloads/SwimVision/webapp.py), while Vercel uses lightweight `api/` functions for deployment-safe API responses.

### Step-by-step quickstart
If you want to run the steps manually, use these commands in order.

```bash
# Step 1: Extract keypoints (always use --crop to isolate primary swimmer)
python src/extract.py --input data/raw/YOUR_VIDEO.mov --output data/processed/ --clip_id YOUR_ID --crop 0 0 430 730

# Step 2: Preview crop before full extraction
python src/extract.py --input data/raw/YOUR_VIDEO.mov --output data/processed/ --clip_id YOUR_ID --crop 0 0 430 730 --preview

# Step 3: Detect phase boundaries
python src/ingest.py detect-phases --keypoints data/processed/YOUR_ID_keypoints.npy --confidence data/processed/YOUR_ID_confidence.npy --output results/YOUR_ID_boundaries.json

# Step 4: Compute joint angles
python src/metrics/joint_angles.py --input data/processed/YOUR_ID_keypoints.npy --output results/ --clip_id YOUR_ID

# Step 5: Render annotated overlay
python src/overlay.py --input data/raw/YOUR_VIDEO.mov --keypoints data/processed/YOUR_ID_keypoints.npy --angles results/YOUR_ID_angles.csv --output results/YOUR_ID_annotated.mp4 --crop 0 0 430 730

# Step 6: Generate report
python src/report.py --clip_id YOUR_ID --keypoints data/processed/YOUR_ID_keypoints.npy --angles results/YOUR_ID_angles.csv --video results/YOUR_ID_annotated.mp4 --output results/
```

### Crop preview workflow
Always verify the crop before committing to full extraction:

```bash
python src/extract.py --input data/raw/YOUR_VIDEO.mov --output data/processed/ --clip_id YOUR_ID --crop 0 0 430 730 --preview
```

This writes:

- [results/crop_preview.jpg](/Users/kaivalyasingh/Downloads/SwimVision/results/crop_preview.jpg)

The preview should contain the target swimmer only, with the full body visible during the key phase of interest.

## Methods
### Pipeline stages
1. Load a raw race or training video.
2. Optionally crop the frame to a single swimmer lane before inference.
3. Run MediaPipe pose extraction on each frame and save `keypoints.npy` and `confidence.npy`.
4. Detect block, flight, and entry boundaries from smoothed torso translation, ankle lift, and confidence traces.
5. Compute per-frame joint-angle and alignment metrics.
6. Aggregate phase-wise deviations against literature-derived optimal ranges.
7. Render an annotated MP4 with skeleton, joint values, phase labels, and deviation flags.
8. Generate JSON and PDF summaries.

### Phase detection
The original codebase used early velocity heuristics and then a simple confidence-only threshold. The current implementation was updated after debugging on real Dressel footage.

Current logic in [src/ingest.py](/Users/kaivalyasingh/Downloads/SwimVision/src/ingest.py):

- Smooth mean confidence with a 5-frame window.
- Smooth torso-center x position using shoulders and hips.
- Smooth ankle midpoint y position.
- Detect flight start when torso translation and ankle lift both indicate takeoff and confidence remains stable.
- Detect flight end at the peak forward extension before the pose track begins collapsing.
- Treat the next short window as entry.

This is the logic that produced working boundaries for the validated Dressel clip.

### Joint-angle computation
The current metric path in [src/metrics/joint_angles.py](/Users/kaivalyasingh/Downloads/SwimVision/src/metrics/joint_angles.py) computes:

- `front_knee_angle`
- `rear_knee_angle`
- `hip_angle`
- `torso_lean`
- `left_elbow_angle`
- `right_elbow_angle`
- `body_linearity`
- `entry_angle`
- `streamline_angle`
- `elbow_extension`
- `elbow_lock_angle`

Important implementation details discovered during debugging:

- All x-coordinates are scaled by `width / height` before distance or angle calculations.
- `entry_angle` is stored as an acute angle.
- Fully missing frames are recorded as `NaN`, not zero, so post-dropout frames do not corrupt phase means.
- `streamline_angle` and `elbow_lock_angle` use the best visible arm rather than averaging a missing arm.

### Deviation scoring
Deviation scoring in [src/metrics/deviation.py](/Users/kaivalyasingh/Downloads/SwimVision/src/metrics/deviation.py) compares each phase mean against literature-derived target intervals and assigns one of:

- `OPTIMAL`
- `MINOR`
- `SIGNIFICANT`
- `CRITICAL`

The current validated deviation JSON is written as:

- `results/{clip_id}_deviations.json`

## Experiments
### Validated clip
The current implementation was validated on:

- `data/raw/dressel2020olympic.mov`

This clip was downloaded from YouTube using `yt-dlp`, renamed locally, and processed end to end through extraction, phase detection, metric computation, overlay rendering, and report generation.

### Observed extraction quality
On the validated Dressel clip:

- mean confidence remained in the `0.74-0.85` range for the tracked portion of the clip
- confidence increased during the airborne phase when the full body was visible
- confidence dropped sharply after underwater entry when the swimmer left the fixed crop and pose detection stopped being reliable

### Current benchmark status
This repository now has one validated end-to-end pipeline run, but it is not yet a dataset-scale benchmark study. The training code and notebook remain present, but the README should be read as documentation for a working single-clip analysis pipeline rather than a completed multi-athlete evaluation paper.

## Results
### Dressel validation summary
For the validated cropped Dressel run, the pipeline produced:

- keypoint extraction arrays
- phase boundaries JSON
- angle CSV
- deviations JSON
- annotated MP4
- report JSON
- report PDF

Representative validated boundaries for the cropped run were:

- block: `0-262`
- flight: `263-267`
- entry: `268-275`

Representative sanity-checked metric means were:

| Metric | Phase | Mean |
| --- | --- | --- |
| front_knee_angle | block | 110.19 |
| rear_knee_angle | block | 101.97 |
| hip_angle | block | 36.88 |
| torso_lean | block | 51.68 |
| body_linearity | flight | 12.30 |
| entry_angle | flight | 38.75 |
| elbow_extension | flight | 164.14 |
| streamline_angle | entry | 14.57 |

Representative deviation table values from the validated run were:

| Phase | Metric | Measured | Optimal Min | Optimal Max | Deviation | Flag |
| --- | --- | --- | --- | --- | --- | --- |
| block | front_knee_angle | 110.19 | 90.0 | 110.0 | 0.19 | MINOR |
| block | rear_knee_angle | 101.97 | 110.0 | 130.0 | 8.03 | MINOR |
| block | hip_angle | 36.88 | 55.0 | 75.0 | 18.12 | SIGNIFICANT |
| block | torso_lean | 51.68 | 15.0 | 30.0 | 21.68 | CRITICAL |
| flight | body_linearity | 12.30 | 0.0 | 8.0 | 4.30 | MINOR |
| flight | entry_angle | 38.75 | 30.0 | 45.0 | 0.00 | OPTIMAL |
| flight | elbow_extension | 164.14 | 165.0 | 180.0 | 0.86 | MINOR |
| entry | streamline_angle | 14.57 | 0.0 | 10.0 | 4.57 | MINOR |
| entry | elbow_lock_angle | 171.92 | 170.0 | 180.0 | 0.00 | OPTIMAL |

## Troubleshooting
Each entry below is written as symptom, root cause, exact fix.

### Symptom
`module 'mediapipe' has no attribute 'solutions'`

### Root cause
Current MediaPipe installs in this project environment did not expose the old `mediapipe.solutions` API.

### Exact fix
Use `mediapipe.python.solutions.pose` instead of `mp.solutions.pose`, and install:

```bash
pip install mediapipe==0.10.30
```

Also keep the local compatibility shim in:

- [src/mediapipe/__init__.py](/Users/kaivalyasingh/Downloads/SwimVision/src/mediapipe/__init__.py)

### Symptom
Mean confidence around `0.037`

### Root cause
The MediaPipe Tasks backend was pushing execution through a broken OpenGL GPU path on Apple Silicon and returning unusable detections.

### Exact fix
Set these before any MediaPipe import:

```python
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
```

Then use the legacy Solutions import path:

```python
import mediapipe.python.solutions.pose as mp_pose
```

The OpenCV DNN OpenPose fallback was removed entirely because it produced unusable confidence scores.

### Symptom
Phase detection returned a 6-frame flight phase starting at frame 1

### Root cause
The original velocity-threshold detector was triggering on noise and posture adjustments early in the set position.

### Exact fix
The original velocity logic was removed. The intermediate fix was a confidence-based detector using smoothed mean confidence after frame 90. The final working code then used torso translation, ankle lift, and confidence together in [src/ingest.py](/Users/kaivalyasingh/Downloads/SwimVision/src/ingest.py), which matched the actual validated clip more reliably.

### Symptom
`body_linearity` returned `83-128°` instead of near `0°` for an elite swimmer

### Root cause
Three separate issues were found:

1. The original formula computed an absolute angle instead of line deviation.
2. In a head-first dive, hips can appear above shoulders in image coordinates, so naive top-to-bottom assumptions break.
3. MediaPipe was sometimes tracking the wrong swimmer or the wrong leg when multiple swimmers were visible.

### Exact fix
1. Replace the angle-based calculation with a perpendicular projection formula.
2. Sort the shoulder, hip, and ankle points by y-coordinate before measuring deviation from the body line.
3. Add `--crop` to isolate the primary swimmer before extraction.

### Symptom
MediaPipe tracked the wrong person or the wrong landmarks

### Root cause
MediaPipe Pose is a single-person model. When more than one swimmer is visible, it may lock onto the most prominent pose instead of the intended athlete.

### Exact fix
Always isolate the target swimmer with `--crop`, and verify the crop first with `--preview`.

Use:

```bash
python src/extract.py --input data/raw/YOUR_VIDEO.mov --output data/processed/ --clip_id YOUR_ID --crop 0 0 430 730 --preview
```

Verification artifact from the debugging session:

- [results/keypoint_debug_frame210.jpg](/Users/kaivalyasingh/Downloads/SwimVision/results/keypoint_debug_frame210.jpg)

### Symptom
`body_linearity` stayed high (`26-47°`) even after formula fixes

### Root cause
Normalized coordinates were being treated as isotropic. In a non-square frame, x-distances and y-distances are not comparable without aspect-ratio correction.

### Exact fix
Scale all x-coordinates by `width / height` before any distance or angle computation. For a `1286x730` frame:

```text
aspect ratio = 1286 / 730 = 1.7614
```

This correction is now built into [src/metrics/joint_angles.py](/Users/kaivalyasingh/Downloads/SwimVision/src/metrics/joint_angles.py).

### Symptom
`joint_angles.py: error: unrecognized arguments: --clip_id`

### Root cause
The CLI originally did not accept `--clip_id`.

### Exact fix
Add `--clip_id` to the argparse interface and write the output as:

```text
results/{clip_id}_angles.csv
```

### Symptom
Crop values worked for one assumed resolution but failed on the actual file

### Root cause
The source video `dressel2020olympic.mov` is `1286x730`, not `1920x1080`.

### Exact fix
Always probe the actual resolution before choosing crop values or aspect-ratio assumptions:

```bash
python -c "import cv2; cap=cv2.VideoCapture('data/raw/YOUR_VIDEO.mov'); print(cap.get(3), cap.get(4))"
```

### Symptom
Entry-phase metrics collapsed to zero after the dive

### Root cause
The swimmer left the fixed crop after underwater entry, so pose detection returned missing frames and the earlier metric code treated missing frames as zero values.

### Exact fix
Record fully missing frames as `NaN`, not zero, and restrict the entry phase to the short pre-dropout window. This is now handled in [src/metrics/joint_angles.py](/Users/kaivalyasingh/Downloads/SwimVision/src/metrics/joint_angles.py) and [src/ingest.py](/Users/kaivalyasingh/Downloads/SwimVision/src/ingest.py).

## Data Collection Notes
### Filming recommendations
- Film from a fixed side angle with only one swimmer in frame, or crop tightly to the target lane before processing.
- A poolside side-angle view at roughly 45 degrees worked well in the validated run.
- Confidence naturally rises during the flight phase because the full body is visible in air. This is expected and useful.

### Source video notes
The validated clip `dressel2020olympic.mov` was sourced from YouTube using `yt-dlp`, renamed locally, and processed successfully end to end.

Example download pattern:

```bash
yt-dlp -o data/raw/dressel2020olympic.%(ext)s "YOUR_SOURCE_URL"
```

## Known Limitations and Future Work
- The pose model is single-person. Always crop to isolate the target swimmer.
- Underwater entry causes keypoint dropout after the entry phase. Block and flight measurements are more reliable than late entry/underwater frames.
- `body_linearity` is the most sensitive metric to crop quality. If it reads above `15°` during a clearly extended flight phase, re-check the crop with `--preview`.
- The Apple Silicon GPU path in MediaPipe Tasks was unstable during implementation. CPU-only mode is required for the current working setup.
- The one-command pipeline now works for a similar new clip, but crop selection is still the main per-video manual decision.
- The training code remains in the repository, but the project is currently validated as a deterministic analysis pipeline rather than a trained multi-class benchmark system.

## Novel Contribution
The practical contribution of the current implementation is not just the original research framing, but a fully working local analysis loop: crop-aware extraction, robust single-clip phase segmentation, verified metric computation, deviation scoring, annotated overlay generation, and PDF/JSON report output from a single command.

## Repository Layout
```text
data/
  raw/                            source videos
  processed/                      extracted keypoints and confidences
  labels.csv                      clip metadata
models/
  pose_landmark_lite.tflite       MediaPipe legacy pose asset used by extraction
  pose_landmarker_heavy.task      retained MediaPipe Tasks asset from debugging
  openpose_pose_coco.prototxt     retained OpenPose debug asset, not used now
  pose_iter_440000.caffemodel     retained OpenPose debug asset, not used now
src/
  extract.py                      crop-aware MediaPipe pose extraction
  ingest.py                       phase-boundary detection
  overlay.py                      annotated video renderer
  report.py                       JSON and PDF report generator
  run_pipeline.py                 one-command pipeline runner
  compare.py                      side-by-side deviation comparison
  mediapipe/__init__.py           local MediaPipe shim for Apple Silicon compatibility
  metrics/
    joint_angles.py               per-frame angle and alignment metrics
    deviation.py                  phase-wise deviation scoring
    reaction_time.py              reaction time and first-stroke proxy timing
  phases/                         phase analyzers
  reference/                      literature-derived optimal ranges
  train/                          dataset, model, and training utilities
sitecustomize.py                  local startup environment setup
notebooks/
  analysis.ipynb                  exploratory analysis notebook
results/
  session_log.md                  autonomous implementation/debug log
  keypoint_debug_frame210.jpg     landmark verification output
  *.json                          boundaries, deviations, report artifacts
  *.csv                           angle and comparison outputs
  *.mp4                           annotated videos
  *.pdf                           generated reports
```

## Command-Line Usage
### Full pipeline
```bash
python src/run_pipeline.py --input data/raw/YOUR_VIDEO.mov --clip_id YOUR_ID
python src/run_pipeline.py --input data/raw/YOUR_VIDEO.mov --clip_id YOUR_ID --crop 0 0 430 730
```

### Extraction only
```bash
python src/extract.py --input data/raw/YOUR_VIDEO.mov --output data/processed/ --clip_id YOUR_ID --crop 0 0 430 730
```

### Batch extraction
```bash
python src/extract.py --batch --output data/processed/
python src/extract.py --batch --input data/raw --output data/processed/ --crop 0 0 430 730
```

### Phase detection
```bash
python src/ingest.py detect-phases --keypoints data/processed/YOUR_ID_keypoints.npy --confidence data/processed/YOUR_ID_confidence.npy --output results/YOUR_ID_boundaries.json
```

### Joint angles
```bash
python src/metrics/joint_angles.py --input data/processed/YOUR_ID_keypoints.npy --output results/ --clip_id YOUR_ID
```

### Overlay
```bash
python src/overlay.py --input data/raw/YOUR_VIDEO.mov --keypoints data/processed/YOUR_ID_keypoints.npy --angles results/YOUR_ID_angles.csv --output results/YOUR_ID_annotated.mp4 --crop 0 0 430 730
```

### Report
```bash
python src/report.py --clip_id YOUR_ID --keypoints data/processed/YOUR_ID_keypoints.npy --angles results/YOUR_ID_angles.csv --video results/YOUR_ID_annotated.mp4 --output results/
```

### Compare two processed clips
```bash
python src/compare.py --clip_a CLIP_A --clip_b CLIP_B --results_dir results
```
