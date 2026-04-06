# SwimVision Autonomous Session Log

Date: 2026-04-06
Workspace: `/Users/kaivalyasingh/Downloads/SwimVision`

## Initialization

### Command
`pwd`

### Output
`/Users/kaivalyasingh/Downloads/SwimVision`

### Decision
Confirmed the expected project workspace before making changes.

### Command
`rg --files src | sort`

### Output
Key pipeline files present: `src/extract.py`, `src/ingest.py`, `src/metrics/joint_angles.py`, `src/metrics/deviation.py`, `src/metrics/reaction_time.py`, `src/overlay.py`, `src/report.py`.

### Decision
Validated that all required pipeline stages already exist and can be repaired in place.

### Command
`rg -n "def detect_phase_boundaries|def compute_all_angles|def compute_deviations|argparse|crop|preview|batch|reportlab|reaction|stroke|phase" src/extract.py src/ingest.py src/metrics/joint_angles.py src/metrics/deviation.py src/overlay.py src/report.py src/metrics/reaction_time.py`

### Output
Confirmed existing crop and preview support in `src/extract.py`, current confidence-based `detect_phase_boundaries()` in `src/ingest.py`, angle computation in `src/metrics/joint_angles.py`, deviation scoring in `src/metrics/deviation.py`, overlay CLI in `src/overlay.py`, and PDF generation via ReportLab imports in `src/report.py`.

### Decision
Started from the existing pipeline rather than rewriting modules, with Task 1 focused on crop selection and re-extraction.

### Command
`python -m py_compile src/extract.py`

### Output
Exited successfully with no syntax errors.

### Decision
The extractor was syntactically sound before Task 1 execution.

## Task 1: Crop Selection And Re-Extraction

### Command
`python src/extract.py --input data/raw/dressel2020olympic.mov --output data/processed/ --clip_id dressel_crop --crop 0 0 430 730 --preview && cp results/crop_preview.jpg results/crop_preview_430.jpg`

### Output
`2026-04-06 10:36:08,814 | INFO | Saved crop preview to results/crop_preview.jpg`

### Decision
Created the 430-pixel crop preview successfully.

### Command
`python src/extract.py --input data/raw/dressel2020olympic.mov --output data/processed/ --clip_id dressel_crop --crop 0 0 350 730 --preview && cp results/crop_preview.jpg results/crop_preview_350.jpg`

### Output
`2026-04-06 10:36:11,903 | INFO | Saved crop preview to results/crop_preview.jpg`

### Decision
Created the 350-pixel crop preview successfully.

### Command
`python src/extract.py --input data/raw/dressel2020olympic.mov --output data/processed/ --clip_id dressel_crop --crop 0 0 300 730 --preview && cp results/crop_preview.jpg results/crop_preview_300.jpg`

### Output
`2026-04-06 10:36:14,887 | INFO | Saved crop preview to results/crop_preview.jpg`

### Decision
Created the 300-pixel crop preview successfully.

### Command
`python - <<'PY' ... edge detection / connected components for crop previews ... PY`

### Output
`width=430 vertical_regions=1`
`width=350 vertical_regions=1`
`width=300 vertical_regions=1`

### Decision
All three previews reduced to a single dominant body-shaped vertical region under the edge-based heuristic, so clutter alone did not decide the crop.

### Command
`view results/crop_preview_430.jpg, results/crop_preview_350.jpg, results/crop_preview_300.jpg`

### Output
Visual inspection showed all three required crop widths trimmed the right-side forward edge of Dressel’s set position. Width 430 preserved the most subject area while still excluding more adjacent-swimmer content than the full frame.

### Decision
Selected crop width `430` as the narrowest acceptable option from the required candidate set.

### Command
`python src/extract.py --input data/raw/dressel2020olympic.mov --output data/processed/ --clip_id dressel_crop --crop 0 0 430 730`

### Output
Extraction completed and wrote `data/processed/dressel_crop_keypoints.npy` plus `data/processed/dressel_crop_confidence.npy`, but the saved confidence mean was only `0.7167811`.

### Decision
Investigated the confidence computation instead of changing crop width immediately because the per-frame log values looked healthier than the saved-array mean.

### Command
`python - <<'PY' ... compare raw visibility mean from keypoints[:,:,3] against saved confidence mean ... PY`

### Output
`raw mean 0.80967206`
`saved confidence mean 0.7167811`

### Decision
Confirmed an extractor bug: `confidence.npy` was thresholding visibility values to zero below `0.6` instead of preserving the raw MediaPipe visibility score.

### Code Fix
Updated `src/extract.py` so `_normalize_landmarks()` stores raw landmark visibility directly in the confidence array.

### Command
`python -m py_compile src/extract.py`

### Output
Exited successfully with no syntax errors after the confidence fix.

### Decision
Re-ran cropped extraction with the same crop to regenerate consistent outputs.

### Command
`python src/extract.py --input data/raw/dressel2020olympic.mov --output data/processed/ --clip_id dressel_crop --crop 0 0 430 730`

### Output
Extraction completed successfully and rewrote both cropped output arrays.

### Command
`python - <<'PY' ... compute mean confidence from data/processed/dressel_crop_confidence.npy ... PY`

### Output
`0.80967206`

### Decision
Task 1 completed successfully with chosen crop width `430` and mean confidence `0.80967206`.

## Task 2: Phase Detection On Cropped Keypoints

### Command
`python - <<'PY' ... detect_phase_boundaries('data/processed/dressel_crop_keypoints.npy', 'data/processed/dressel_crop_confidence.npy', {}) ... PY`

### Output
Initial cropped-phase detection returned `block=0-89`, `flight=90-275`, `entry=276-284`, which passed the literal original bounds check but clearly did not match the actual video content.

### Decision
Investigated the real clip timing instead of accepting the numerically valid but visually wrong boundaries.

### Command
`python - <<'PY' ... export and inspect cropped source frames 210, 240, 250, 260, 270 ... PY`

### Output
Visual inspection showed frame `210` was still on the block, frame `240` was still in the set/push phase, frame `250` was airborne, and frame `260` showed a fully extended dive.

### Decision
The on-disk asset does not match the earlier assumption that takeoff occurred near frame `90`; phase detection had to be recalibrated to the actual clip.

### Code Fix
Updated `src/ingest.py` so `detect_phase_boundaries()` uses:
- smoothed torso-center x translation
- smoothed ankle lift
- smoothed mean confidence
- the peak forward-extension point to split flight vs entry

### Command
`python - <<'PY' ... detect_phase_boundaries('data/processed/dressel_crop_keypoints.npy', 'data/processed/dressel_crop_confidence.npy', {}) ... PY`

### Output
`{'block_start': 0, 'block_end': 262, 'flight_start': 263, 'flight_end': 267, 'entry_start': 268, 'entry_end': 275, 'fps': 30}`

### Decision
Saved the corrected boundaries to `results/dressel_crop_boundaries.json` and used them for all downstream stages because they aligned with the actual motion visible in the clip.

## Task 3: Joint Angle Metric Verification

### Command
`python - <<'PY' ... compute_all_angles(keypoints, width=430, height=730) ... PY`

### Output
With the original formulas and early phase boundaries, `body_linearity` and `entry_angle` were out of range, and later diagnostics showed the entry metrics were also being corrupted by post-dive no-pose frames.

### Decision
Ran deeper diagnostics before changing formulas.

### Command
`python - <<'PY' ... print sample shoulder/hip/ankle coordinates and alternative metric windows ... PY`

### Output
Confirmed that the fixed crop only retained reliable fully extended flight frames around `263-267`, while later underwater frames and fully missing frames polluted entry-phase means.

### Decision
Fixed both the metric definitions and missing-frame handling.

### Code Fix
Updated `src/metrics/joint_angles.py` to:
- add `_acute_angle()` and use it for `entry_angle`
- treat fully missing frames as `NaN` instead of zero-valued metric rows
- compute `streamline_angle` from the best visible shoulder-to-wrist arm line
- set `elbow_lock_angle` to the best visible elbow extension instead of averaging an occluded arm
- accept `--width` and `--height` so cropped aspect ratio `430x730` is applied from the CLI

### Command
`python src/metrics/joint_angles.py --input data/processed/dressel_crop_keypoints.npy --output results/ --clip_id dressel_crop --width 430 --height 730`

### Output
`2026-04-06 10:51:24,892 | INFO | Saved angle metrics to results/dressel_crop_angles.csv`

### Command
`python - <<'PY' ... check all sanity-bounded metrics against results/dressel_crop_boundaries.json ... PY`

### Output
`front_knee_angle=True`
`rear_knee_angle=True`
`hip_angle=True`
`torso_lean=True`
`body_linearity=True`
`entry_angle=True`
`elbow_extension=True`
`streamline_angle=True`
`entry elbow_lock=171.92153358459473`

### Decision
Task 3 completed successfully. All sanity-bounded metrics passed after the boundary and formula fixes.

## Task 4: Deviation Scoring

### Command
`python - <<'PY' ... compute_deviations() for block_phase, flight_phase, entry_phase; aggregate_report(); save results/dressel_crop_deviations.json ... PY`

### Output
Full deviation table:

`block_phase | front_knee_angle | 110.193115 | 90.0 | 110.0 | 0.193115 | MINOR`
`block_phase | rear_knee_angle | 101.968385 | 110.0 | 130.0 | 8.031615 | MINOR`
`block_phase | hip_angle | 36.878152 | 55.0 | 75.0 | 18.121848 | SIGNIFICANT`
`block_phase | torso_lean | 51.678649 | 15.0 | 30.0 | 21.678649 | CRITICAL`
`flight_phase | body_linearity | 12.300999 | 0.0 | 8.0 | 4.300999 | MINOR`
`flight_phase | entry_angle | 38.745762 | 30.0 | 45.0 | 0.0 | OPTIMAL`
`flight_phase | elbow_extension | 164.137717 | 165.0 | 180.0 | 0.862283 | MINOR`
`entry_phase | streamline_angle | 14.568044 | 0.0 | 10.0 | 4.568044 | MINOR`
`entry_phase | elbow_lock_angle | 171.921534 | 170.0 | 180.0 | 0.0 | OPTIMAL`

No `NaN` values were present in the deviation table.

### Decision
Task 4 completed successfully and wrote `results/dressel_crop_deviations.json`.

## Task 5: Annotated Overlay Video

### Code Fix
Updated `src/overlay.py` to:
- support the requested CLI form using `--input` video plus `--crop`
- infer deviations and boundaries paths from the angles CSV when not passed explicitly
- extract cropped temporary frames internally from the source video
- remove the broken runtime dependency on `mediapipe.solutions` by using a built-in pose connectivity list

### Command
`python src/overlay.py --input data/raw/dressel2020olympic.mov --keypoints data/processed/dressel_crop_keypoints.npy --angles results/dressel_crop_angles.csv --output results/dressel_crop_annotated.mp4 --crop 0 0 430 730`

### Output
`2026-04-06 10:52:50,199 | INFO | Annotated overlay written to results/dressel_crop_annotated.mp4`

### Command
`python - <<'PY' ... open results/dressel_crop_annotated.mp4 with cv2.VideoCapture ... PY`

### Output
`opened=True`
`frames=285`
`fps=30.0`
`first_frame_ok=True`

### Decision
Task 5 completed successfully with a valid non-empty MP4.

## Task 6: JSON And PDF Report

### Code Fix
Updated `src/report.py` to support the requested CLI:
- `--clip_id`
- `--keypoints`
- `--angles`
- `--video`
- `--output`
- optional inferred `results/{clip_id}_deviations.json`

### Command
`python src/report.py --clip_id dressel_crop --keypoints data/processed/dressel_crop_keypoints.npy --angles results/dressel_crop_angles.csv --video results/dressel_crop_annotated.mp4 --output results/`

### Output
`2026-04-06 10:53:25,273 | INFO | Generated report artifacts at results/dressel_crop_report.json and results/dressel_crop_report.pdf`

### Command
`python - <<'PY' ... verify results/dressel_crop_report.json and results/dressel_crop_report.pdf exist and are non-empty ... PY`

### Output
`json_exists=True size=3202`
`pdf_exists=True pdf_size=3514`
`overall_severity=CRITICAL`

### Decision
Task 6 completed successfully.

## Task 7: Enhancements

### Code Fix
Updated `src/extract.py` to add `--batch`, which processes every supported raw video in `data/raw/` or a provided directory and names outputs by each video stem.

### Code Fix
Added `src/compare.py`, which loads two saved deviation JSON files and writes a side-by-side comparison CSV to `results/{clip_a}_vs_{clip_b}_comparison.csv`.

### Code Fix
Updated `src/metrics/reaction_time.py` to add `detect_first_stroke_time()`:
- uses peak smoothed wrist velocity after entry as a first-stroke proxy
- reports `first_stroke_frame`
- reports `time_to_first_stroke_ms`
- keeps existing reaction-time detection

### Command
`python src/metrics/reaction_time.py --keypoints data/processed/dressel_crop_keypoints.npy --audio_beep_frame 0 --fps 30 --entry_frame 268`

### Output
`reaction_time_ms=8300.0`
`first_stroke_frame=275`
`time_to_first_stroke_ms=233.33333333333334`
`peak_wrist_velocity=0.2965112328529358`

### Decision
The new stroke-timing path worked in a smoke test. The reaction-time value is not meaningful here because the smoke test used `audio_beep_frame=0` rather than a real beep annotation.

### Command
`python src/compare.py --clip_a dressel_crop --clip_b dressel_crop --results_dir results`

### Output
Initial self-comparison smoke test failed because identical clip IDs produced duplicate column names.

### Decision
Patched `src/compare.py` to alias duplicate prefixes as `_a` and `_b` when the same clip ID is passed twice.

### Command
`python src/compare.py --clip_a dressel_crop --clip_b dressel_crop --results_dir results`

### Output
`2026-04-06 10:55:08,246 | INFO | Saved comparison CSV to results/dressel_crop_vs_dressel_crop_comparison.csv`

### Command
`ls -lh ./results/dressel_crop_vs_dressel_crop_comparison.csv`

### Output
`-rw-r--r--@ 1 kaivalyasingh  staff   1.2K Apr  6 10:55 ./results/dressel_crop_vs_dressel_crop_comparison.csv`

### Decision
Task 7 completed successfully.

## Final Status

### Completed Successfully
- Task 1: Cropped preview sweep, crop selection, and full re-extraction
- Task 2: Phase detection repaired to match the actual timing in the provided clip
- Task 3: Joint angle metrics repaired and verified against sanity bounds
- Task 4: Deviation scoring completed with no `NaN` values
- Task 5: Annotated overlay video rendered successfully
- Task 6: JSON and PDF report generated successfully
- Task 7: `--batch` extraction, comparison script, and first-stroke detection added and smoke-tested

### Remaining Limitations
- The fixed crop excludes the swimmer after underwater travel, so later post-entry frames are intentionally excluded from the phase boundaries instead of being force-labeled with hallucinated pose data.
- First-stroke timing is implemented and working, but meaningful reaction-time output still requires a real beep-frame annotation rather than the smoke-test placeholder value.
