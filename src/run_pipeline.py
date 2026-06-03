# This file runs the SwimVision pipeline end to end for a single clip.
"""Single-command pipeline runner for SwimVision."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

from src.pipeline.config_manager import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ProgressCallback = Callable[[dict[str, object]], None]


def _display_command(command: list[str]) -> str:
    """Render a subprocess command for human-readable logging."""

    return " ".join(command)


def _emit_progress(progress_callback: ProgressCallback | None, payload: dict[str, object]) -> None:
    """Send a progress event to an optional callback."""

    if progress_callback is not None:
        progress_callback(payload)


def _run_step(
    index: int,
    total: int,
    label: str,
    command: list[str],
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Run a pipeline step and stop immediately on failure."""

    print(f"[{index}/{total}] {label}...")
    _emit_progress(
        progress_callback,
        {
            "event": "step_started",
            "step_index": index,
            "total_steps": total,
            "label": label,
            "command": _display_command(command),
        },
    )
    start_time = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.monotonic() - start_time
    if result.returncode != 0:
        print(f"[{index}/{total}] FAILED: {label} ({elapsed:.1f}s)")
        print(f"Command: {_display_command(command)}")
        print(f"Error: step '{label}' exited with code {result.returncode}.")
        _emit_progress(
            progress_callback,
            {
                "event": "step_failed",
                "step_index": index,
                "total_steps": total,
                "label": label,
                "elapsed_seconds": elapsed,
                "command": _display_command(command),
                "return_code": result.returncode,
            },
        )
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"[{index}/{total}] Done ({elapsed:.1f}s)")
    _emit_progress(
        progress_callback,
        {
            "event": "step_completed",
            "step_index": index,
            "total_steps": total,
            "label": label,
            "elapsed_seconds": elapsed,
            "command": _display_command(command),
        },
    )


def _resolve_dimensions(input_path: Path, crop: list[int] | None) -> tuple[int, int]:
    """Resolve the frame dimensions for angle computation."""

    if crop is not None:
        return int(crop[2]), int(crop[3])

    probe_command = [
        sys.executable,
        "-c",
        (
            "import cv2, sys; "
            "cap = cv2.VideoCapture(sys.argv[1]); "
            "ok = cap.isOpened(); "
            "width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); "
            "height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); "
            "cap.release(); "
            "print(f'{width} {height}' if ok else '0 0')"
        ),
        str(input_path),
    ]
    result = subprocess.run(
        probe_command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to inspect video dimensions for '{input_path}'.")
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        raise RuntimeError(f"Unexpected video-dimension probe output: {result.stdout!r}")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video dimensions detected for '{input_path}': {width}x{height}")
    return width, height


def _run_stroke_pipeline(
    resolved_input: Path,
    clip_id: str,
    crop_values: list[str],
    crop: list[int] | None,
    keypoints_path: Path,
    confidence_path: Path,
    angles_path: Path,
    boundaries_path: Path,
    deviations_path: Path,
    annotated_path: Path,
    report_json_path: Path,
    report_pdf_path: Path,
    stroke_start_frame: int,
    progress_callback: ProgressCallback | None,
    width: int,
    height: int,
    total_steps: int,
    stroke_type: str = "auto",
    config: PipelineConfig | None = None,
) -> dict[str, Path]:
    """Run the stroke-analysis pipeline variant.

    Shares the initial extraction and angle computation steps with the dive
    pipeline, then runs stroke-cycle detection, stroke metrics, and an
    extended report generation pass.  Respects feature flags from *config*.

    Args:
        stroke_type: Stroke type override (auto, freestyle, butterfly, backstroke).
        config: Optional pipeline configuration for feature gating.
    """

    if config is None:
        config = PipelineConfig(preset="standard", analysis_mode="stroke")

    stroke_boundaries_path = RESULTS_DIR / f"{clip_id}_stroke_boundaries.json"
    stroke_metrics_path = RESULTS_DIR / f"{clip_id}_stroke_metrics.json"
    stroke_deviations_path = RESULTS_DIR / f"{clip_id}_stroke_deviations.json"

    # Step 1: Extract keypoints (shared with dive pipeline)
    extract_command = [
        sys.executable,
        "src/extract.py",
        "--input",
        str(resolved_input),
        "--output",
        str(PROCESSED_DIR),
        "--clip_id",
        clip_id,
    ]
    if crop_values:
        extract_command.extend(["--crop", *crop_values])
    _run_step(1, total_steps, "Extracting keypoints", extract_command, progress_callback=progress_callback)

    # Step 2: Compute joint angles (shared)
    joint_command = [
        sys.executable,
        "src/metrics/joint_angles.py",
        "--input",
        str(keypoints_path),
        "--output",
        str(RESULTS_DIR),
        "--clip_id",
        clip_id,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    _run_step(2, total_steps, "Computing joint angles", joint_command, progress_callback=progress_callback)

    # Step 3: Detect stroke cycles
    stroke_detect_command = [
        sys.executable,
        "src/ingest.py",
        "detect-strokes",
        "--keypoints",
        str(keypoints_path),
        "--output",
        str(stroke_boundaries_path),
        "--stroke_start_frame",
        str(stroke_start_frame),
        "--fps",
        str(30),
        "--width",
        str(width),
        "--height",
        str(height),
        "--stroke_type",
        stroke_type,
    ]
    _run_step(3, total_steps, "Detecting stroke cycles", stroke_detect_command, progress_callback=progress_callback)

    # Step 4: Compute stroke metrics
    stroke_metrics_command = [
        sys.executable,
        "-m",
        "src.metrics.stroke_metrics",
        "--keypoints",
        str(keypoints_path),
        "--boundaries",
        str(stroke_boundaries_path),
        "--output",
        str(stroke_metrics_path),
    ]
    _run_step(4, total_steps, "Computing stroke metrics", stroke_metrics_command, progress_callback=progress_callback)

    # Step 5: Compute stroke deviations with type-specific ranges
    stroke_deviation_script = (
        "import json, sys; "
        "from src.metrics.deviation import score_deviation; "
        "from src.reference.optimal_ranges import get_range; "
        "metrics = json.load(open(sys.argv[1], 'r', encoding='utf-8')); "
        "agg = metrics.get('aggregate', {}); "
        "detected_type = metrics.get('cycles', [{}])[0].get('stroke_type', 'freestyle') if metrics.get('cycles') else 'freestyle'; "
        f"stroke_type = sys.argv[3] if sys.argv[3] != 'auto' else detected_type; "
        "results = []; "
        "if stroke_type == 'butterfly': "
        "    metric_map = {'stroke_rate': 'butterfly_cycle', 'body_roll': 'butterfly_cycle', 'symmetry_index': 'butterfly_cycle', 'bilateral_elbow_flexion': 'butterfly_catch', 'bilateral_hand_speed': 'butterfly_pull'}; "
        "elif stroke_type == 'backstroke': "
        "    metric_map = {'stroke_rate': 'backstroke_cycle', 'body_roll': 'backstroke_cycle', 'symmetry_index': 'backstroke_cycle', 'supine_elbow_flexion': 'backstroke_catch', 'supine_hand_speed': 'backstroke_pull'}; "
        "else: "
        "    metric_map = {'stroke_rate': 'stroke_cycle', 'body_roll': 'stroke_cycle', 'symmetry_index': 'stroke_cycle', 'left_elbow_flexion': 'stroke_catch_left', 'right_elbow_flexion': 'stroke_catch_right', 'left_hand_speed': 'stroke_pull_left', 'right_hand_speed': 'stroke_pull_right', 'left_elbow_extension_rate': 'stroke_pull_left', 'right_elbow_extension_rate': 'stroke_pull_right'}; "
        "for metric, phase in metric_map.items(): "
        "    if metric in agg: "
        "        try: "
        "            o_min, o_max = get_range(phase.replace('butterfly_catch', 'butterfly_catch').replace('backstroke_catch', 'backstroke_catch').replace('backstroke_pull', 'backstroke_pull').replace('butterfly_cycle', 'butterfly_cycle').replace('backstroke_cycle', 'backstroke_cycle'), metric); "
        "            dev, flag = score_deviation(float(agg[metric]), (o_min, o_max)); "
        "            results.append({'metric': metric, 'phase': phase, 'measured': float(agg[metric]), 'optimal_min': o_min, 'optimal_max': o_max, 'deviation': dev, 'flag': flag}); "
        "        except Exception: pass; "
        "worst_flags = [r['flag'] for r in results]; "
        "flag_order = ['OPTIMAL', 'MINOR', 'SIGNIFICANT', 'CRITICAL']; "
        "overall = max(worst_flags, key=lambda f: flag_order.index(f)) if worst_flags else 'OPTIMAL'; "
        "json.dump({'stroke_cycle': results, 'overall_severity': overall, 'stroke_type': stroke_type}, open(sys.argv[2], 'w', encoding='utf-8'), indent=2)"
    )
    stroke_deviation_command = [
        sys.executable,
        "-c",
        stroke_deviation_script,
        str(stroke_metrics_path),
        str(stroke_deviations_path),
        stroke_type,
    ]
    _run_step(5, total_steps, "Computing stroke deviations", stroke_deviation_command, progress_callback=progress_callback)

    step_idx = 5
    stroke_symmetry_path = RESULTS_DIR / f"{clip_id}_stroke_symmetry.json"
    stroke_risk_path = RESULTS_DIR / f"{clip_id}_stroke_risk.json"

    if config.is_enabled("analyze_symmetry"):
        step_idx += 1
        stroke_symmetry_command = [
            sys.executable,
            "-m",
            "src.metrics.symmetry_analysis",
            "--stroke-metrics",
            str(stroke_metrics_path),
            "--mode",
            "stroke",
            "--output",
            str(stroke_symmetry_path),
        ]
        _run_step(step_idx, total_steps, "Analyzing stroke symmetry", stroke_symmetry_command, progress_callback=progress_callback)

    if config.is_enabled("assess_injury_risk"):
        step_idx += 1
        stroke_symm_arg = str(stroke_symmetry_path) if stroke_symmetry_path.exists() else ""
        stroke_risk_command = [
            sys.executable,
            "-m",
            "src.analytics.injury_risk",
            "--stroke-metrics",
            str(stroke_metrics_path),
            "--mode",
            "stroke",
            "--output",
            str(stroke_risk_path),
        ]
        if stroke_symm_arg:
            stroke_risk_command.extend(["--symmetry", stroke_symm_arg])
        _run_step(step_idx, total_steps, "Assessing injury risk", stroke_risk_command, progress_callback=progress_callback)

    if config.is_enabled("render_overlay"):
        step_idx += 1
        overlay_command = [
            sys.executable,
            "src/overlay.py",
            "--input",
            str(resolved_input),
            "--keypoints",
            str(keypoints_path),
            "--angles",
            str(angles_path),
            "--output",
            str(annotated_path),
            "--analysis_mode",
            "stroke",
            "--stroke_boundaries",
            str(stroke_boundaries_path),
        ]
        if crop_values:
            overlay_command.extend(["--crop", *crop_values])
        _run_step(step_idx, total_steps, "Rendering annotated overlay", overlay_command, progress_callback=progress_callback)

    # Final step: Generate report
    step_idx += 1
    report_command = [
        sys.executable,
        "src/report.py",
        "--clip_id",
        clip_id,
        "--keypoints",
        str(keypoints_path),
        "--angles",
        str(angles_path),
        "--video",
        str(annotated_path),
        "--output",
        str(RESULTS_DIR),
        "--analysis_mode",
        "stroke",
        "--stroke_metrics",
        str(stroke_metrics_path),
        "--stroke_deviations",
        str(stroke_deviations_path),
    ]
    _run_step(step_idx, total_steps, "Generating report", report_command, progress_callback=progress_callback)

    outputs = {
        "keypoints": keypoints_path,
        "confidence": confidence_path,
        "angles_csv": angles_path,
        "stroke_boundaries_json": stroke_boundaries_path,
        "stroke_metrics_json": stroke_metrics_path,
        "stroke_deviations_json": stroke_deviations_path,
        "stroke_symmetry_json": stroke_symmetry_path,
        "stroke_risk_json": stroke_risk_path,
        "annotated_video": annotated_path,
        "report_json": report_json_path,
        "report_pdf": report_pdf_path,
    }

    _emit_progress(
        progress_callback,
        {
            "event": "pipeline_completed",
            "clip_id": clip_id,
            "total_steps": total_steps,
            "analysis_mode": "stroke",
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the end-to-end pipeline."""

    parser = argparse.ArgumentParser(description="Run the SwimVision pipeline end to end.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--clip_id", required=True, help="Clip identifier.")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional crop region in pixel coordinates.",
    )
    parser.add_argument(
        "--analysis_mode",
        choices=["dive", "stroke"],
        default="dive",
        help="Analysis mode: dive (block/flight/entry) or stroke (freestyle cycles).",
    )
    parser.add_argument(
        "--stroke_start_frame",
        type=int,
        default=0,
        help="Frame index where stroke analysis begins (after dive entry).",
    )
    parser.add_argument(
        "--stroke_type",
        choices=["freestyle", "butterfly", "backstroke", "auto"],
        default="auto",
        help="Stroke type override (default: auto-detect).",
    )
    parser.add_argument(
        "--preset",
        choices=["quick", "standard", "research"],
        default="standard",
        help="Pipeline preset (default: standard).",
    )
    return parser


def run_pipeline(
    input_path: str | Path,
    clip_id: str,
    crop: list[int] | None = None,
    progress_callback: ProgressCallback | None = None,
    analysis_mode: str = "dive",
    stroke_start_frame: int = 0,
    stroke_type: str = "auto",
    preset: str = "standard",
) -> dict[str, Path]:
    """Run the full SwimVision pipeline and return the generated artifact paths.

    Args:
        input_path: Path to the input video file.
        clip_id: Identifier used to name all output artifacts.
        crop: Optional [x, y, w, h] crop region in pixels.
        progress_callback: Optional callback for progress events.
        analysis_mode: "dive" for block/flight/entry or "stroke" for freestyle cycles.
        stroke_start_frame: Frame index where stroke analysis begins.
        stroke_type: Stroke type override (auto, freestyle, butterfly, backstroke).
    """

    resolved_input = Path(input_path).expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input video not found: {resolved_input}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    crop_values = [str(value) for value in crop] if crop is not None else []

    keypoints_path = PROCESSED_DIR / f"{clip_id}_keypoints.npy"
    confidence_path = PROCESSED_DIR / f"{clip_id}_confidence.npy"
    angles_path = RESULTS_DIR / f"{clip_id}_angles.csv"
    boundaries_path = RESULTS_DIR / f"{clip_id}_boundaries.json"
    deviations_path = RESULTS_DIR / f"{clip_id}_deviations.json"
    annotated_path = RESULTS_DIR / f"{clip_id}_annotated.mp4"
    report_json_path = RESULTS_DIR / f"{clip_id}_report.json"
    report_pdf_path = RESULTS_DIR / f"{clip_id}_report.pdf"

    width, height = _resolve_dimensions(resolved_input, crop)

    config = PipelineConfig(preset=preset, analysis_mode=analysis_mode)

    # Compute total steps dynamically based on enabled features
    base_steps = 5  # extract + phases/ingest + angles + deviations + report
    if analysis_mode == "stroke":
        base_steps += 1  # stroke detection
    else:
        if config.is_enabled("compute_velocity_acceleration"):
            base_steps += 1
        if config.is_enabled("compute_dynamic_estimates"):
            base_steps += 1
    if config.is_enabled("analyze_symmetry"):
        base_steps += 1
    if config.is_enabled("assess_injury_risk"):
        base_steps += 1
    if config.is_enabled("render_overlay"):
        base_steps += 1
    if config.is_enabled("compute_joint_contributions"):
        base_steps += 1
    if config.is_enabled("analyze_entry"):
        base_steps += 1
    if config.is_enabled("compute_glide_analysis"):
        base_steps += 1
    if config.is_enabled("predict_performance"):
        base_steps += 1
    total_steps = base_steps

    LOGGER.info(
        "Using pipeline preset: %s (complexity=%s, %d steps)",
        preset,
        config.get_param("extraction_model_complexity", "default"),
        total_steps,
    )

    _emit_progress(
        progress_callback,
        {
            "event": "pipeline_started",
            "clip_id": clip_id,
            "input_path": str(resolved_input),
            "total_steps": total_steps,
            "analysis_mode": analysis_mode,
            "crop": crop or [],
        },
    )

    if analysis_mode == "stroke":
        return _run_stroke_pipeline(
            resolved_input=resolved_input,
            clip_id=clip_id,
            crop_values=crop_values,
            crop=crop,
            keypoints_path=keypoints_path,
            confidence_path=confidence_path,
            angles_path=angles_path,
            boundaries_path=boundaries_path,
            deviations_path=deviations_path,
            annotated_path=annotated_path,
            report_json_path=report_json_path,
            report_pdf_path=report_pdf_path,
            stroke_start_frame=stroke_start_frame,
            progress_callback=progress_callback,
            width=width,
            height=height,
            total_steps=total_steps,
            stroke_type=stroke_type,
            config=config,
        )

    # --- Dive analysis pipeline (default) ---
    step_idx = 0

    step_idx += 1
    extract_command = [
        sys.executable,
        "src/extract.py",
        "--input",
        str(resolved_input),
        "--output",
        str(PROCESSED_DIR),
        "--clip_id",
        clip_id,
    ]
    if crop_values:
        extract_command.extend(["--crop", *crop_values])
    _run_step(step_idx, total_steps, "Extracting keypoints", extract_command, progress_callback=progress_callback)

    step_idx += 1
    ingest_command = [
        sys.executable,
        "src/ingest.py",
        "detect-phases",
        "--keypoints",
        str(keypoints_path),
        "--confidence",
        str(confidence_path),
        "--output",
        str(boundaries_path),
    ]
    _run_step(step_idx, total_steps, "Detecting phase boundaries", ingest_command, progress_callback=progress_callback)

    step_idx += 1
    joint_command = [
        sys.executable,
        "src/metrics/joint_angles.py",
        "--input",
        str(keypoints_path),
        "--output",
        str(RESULTS_DIR),
        "--clip_id",
        clip_id,
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    _run_step(step_idx, total_steps, "Computing joint angles", joint_command, progress_callback=progress_callback)

    # Optional: Compute velocity & acceleration profiles
    vel_accel_path = RESULTS_DIR / f"{clip_id}_vel_accel.json"
    if config.is_enabled("compute_velocity_acceleration"):
        step_idx += 1
        vel_accel_command = [
            sys.executable,
            "-m",
            "src.metrics.velocity_acceleration",
            "--angles",
            str(angles_path),
            "--boundaries",
            str(boundaries_path),
            "--output",
            str(vel_accel_path),
        ]
        _run_step(step_idx, total_steps, "Computing velocity & acceleration", vel_accel_command, progress_callback=progress_callback)

    # Optional: Dynamic estimates (COM velocity, head alignment, entry trajectory)
    dynamic_path = RESULTS_DIR / f"{clip_id}_dynamic.json"
    if config.is_enabled("compute_dynamic_estimates"):
        step_idx += 1
        dynamic_command = [
            sys.executable,
            "-m",
            "src.metrics.dynamic_estimates",
            "--keypoints",
            str(keypoints_path),
            "--boundaries",
            str(boundaries_path),
            "--output",
            str(dynamic_path),
        ]
        _run_step(step_idx, total_steps, "Computing dynamic estimates", dynamic_command, progress_callback=progress_callback)

    step_idx += 1
    deviation_command = [
        sys.executable,
        "-m",
        "src.metrics.deviation",
        "--angles",
        str(angles_path),
        "--boundaries",
        str(boundaries_path),
        "--output",
        str(deviations_path),
        "--all-phases",
    ]
    _run_step(step_idx, total_steps, "Computing deviations", deviation_command, progress_callback=progress_callback)

    # Optional: Symmetry analysis
    symmetry_path = RESULTS_DIR / f"{clip_id}_symmetry.json"
    if config.is_enabled("analyze_symmetry"):
        step_idx += 1
        symmetry_command = [
            sys.executable,
            "-m",
            "src.metrics.symmetry_analysis",
            "--angles",
            str(angles_path),
            "--boundaries",
            str(boundaries_path),
            "--mode",
            "dive",
            "--output",
            str(symmetry_path),
        ]
        _run_step(step_idx, total_steps, "Analyzing bilateral symmetry", symmetry_command, progress_callback=progress_callback)

    # Optional: Injury risk assessment
    risk_path = RESULTS_DIR / f"{clip_id}_risk.json"
    if config.is_enabled("assess_injury_risk"):
        step_idx += 1
        risk_command = [
            sys.executable,
            "-m",
            "src.analytics.injury_risk",
            "--deviations",
            str(deviations_path),
            "--mode",
            "dive",
            "--output",
            str(risk_path),
        ]
        if symmetry_path.exists():
            risk_command.extend(["--symmetry", str(symmetry_path)])
        _run_step(step_idx, total_steps, "Assessing injury risk", risk_command, progress_callback=progress_callback)

    if config.is_enabled("render_overlay"):
        step_idx += 1
        overlay_command = [
            sys.executable,
            "src/overlay.py",
            "--input",
            str(resolved_input),
            "--keypoints",
            str(keypoints_path),
            "--angles",
            str(angles_path),
            "--output",
            str(annotated_path),
        ]
        if crop_values:
            overlay_command.extend(["--crop", *crop_values])
        _run_step(step_idx, total_steps, "Rendering annotated overlay", overlay_command, progress_callback=progress_callback)

    # Optional: Joint contributions analysis
    joint_contrib_path = RESULTS_DIR / f"{clip_id}_joint_contributions.json"
    if config.is_enabled("compute_joint_contributions"):
        step_idx += 1
        jc_command = [
            sys.executable,
            "-m",
            "src.metrics.joint_contributions",
            "--angles",
            str(angles_path),
            "--output",
            str(joint_contrib_path),
        ]
        if vel_accel_path.exists():
            jc_command.extend(["--vel-accel", str(vel_accel_path)])
        _run_step(step_idx, total_steps, "Analyzing joint contributions", jc_command, progress_callback=progress_callback)

    # Optional: Entry analysis
    entry_analysis_path = RESULTS_DIR / f"{clip_id}_entry_analysis.json"
    if config.is_enabled("analyze_entry"):
        step_idx += 1
        entry_command = [
            sys.executable,
            "-m",
            "src.metrics.entry_analysis",
            "--angles",
            str(angles_path),
            "--boundaries",
            str(boundaries_path),
            "--output",
            str(entry_analysis_path),
        ]
        if keypoints_path.exists():
            entry_command.extend(["--keypoints", str(keypoints_path)])
        if dynamic_path.exists():
            entry_command.extend(["--dynamic", str(dynamic_path)])
        if vel_accel_path.exists():
            entry_command.extend(["--vel-accel", str(vel_accel_path)])
        _run_step(step_idx, total_steps, "Analyzing entry mechanics", entry_command, progress_callback=progress_callback)

    # Optional: Underwater glide analysis
    glide_analysis_path = RESULTS_DIR / f"{clip_id}_glide_analysis.json"
    if config.is_enabled("compute_glide_analysis"):
        step_idx += 1
        glide_command = [
            sys.executable,
            "-m",
            "src.metrics.glide_analysis",
            "--keypoints",
            str(keypoints_path),
            "--angles",
            str(angles_path),
            "--boundaries",
            str(boundaries_path),
            "--output",
            str(glide_analysis_path),
        ]
        _run_step(step_idx, total_steps, "Analyzing underwater glide", glide_command, progress_callback=progress_callback)

    # Optional: Performance prediction
    predictions_path = RESULTS_DIR / f"{clip_id}_predictions.json"
    if config.is_enabled("predict_performance"):
        step_idx += 1
        pred_command = [
            sys.executable,
            "-m",
            "src.analytics.performance_predictor",
            "--deviations",
            str(deviations_path),
            "--output",
            str(predictions_path),
        ]
        if symmetry_path.exists():
            pred_command.extend(["--symmetry", str(symmetry_path)])
        # Reaction time from report may not exist yet — skip
        _run_step(step_idx, total_steps, "Predicting performance", pred_command, progress_callback=progress_callback)

    # Final step: Generate report
    step_idx += 1
    report_command = [
        sys.executable,
        "src/report.py",
        "--clip_id",
        clip_id,
        "--keypoints",
        str(keypoints_path),
        "--angles",
        str(angles_path),
        "--video",
        str(annotated_path),
        "--output",
        str(RESULTS_DIR),
    ]
    if vel_accel_path.exists():
        report_command.extend(["--vel_accel", str(vel_accel_path)])
    if symmetry_path.exists():
        report_command.extend(["--symmetry", str(symmetry_path)])
    if predictions_path.exists():
        report_command.extend(["--predictions", str(predictions_path)])
    if entry_analysis_path.exists():
        report_command.extend(["--entry_analysis", str(entry_analysis_path)])
    _run_step(step_idx, total_steps, "Generating report", report_command, progress_callback=progress_callback)

    outputs = {
        "keypoints": keypoints_path,
        "confidence": confidence_path,
        "angles_csv": angles_path,
        "boundaries_json": boundaries_path,
        "deviations_json": deviations_path,
        "vel_accel_json": vel_accel_path,
        "dynamic_json": dynamic_path,
        "symmetry_json": symmetry_path,
        "risk_json": risk_path,
        "joint_contributions_json": joint_contrib_path,
        "entry_analysis_json": entry_analysis_path,
        "glide_analysis_json": glide_analysis_path,
        "predictions_json": predictions_path,
        "annotated_video": annotated_path,
        "report_json": report_json_path,
        "report_pdf": report_pdf_path,
    }

    _emit_progress(
        progress_callback,
        {
            "event": "pipeline_completed",
            "clip_id": clip_id,
            "total_steps": total_steps,
            "outputs": {name: str(path) for name, path in outputs.items()},
        },
    )
    return outputs


def main() -> int:
    """Run the full SwimVision pipeline for a single clip."""

    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve() if not os.path.isabs(args.input) else Path(args.input)
    try:
        outputs = run_pipeline(
            input_path=input_path,
            clip_id=args.clip_id,
            crop=args.crop,
            analysis_mode=args.analysis_mode,
            stroke_start_frame=args.stroke_start_frame,
            stroke_type=args.stroke_type,
            preset=args.preset,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)

    print("\nPipeline complete. Output files found:")
    for output_path in outputs.values():
        exists = output_path.exists() and output_path.stat().st_size > 0
        status = "OK" if exists else "MISSING"
        print(f"- [{status}] {output_path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
