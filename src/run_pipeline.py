# This file runs the SwimVision pipeline end to end for a single clip.
"""Single-command pipeline runner for SwimVision."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable


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
    return parser


def run_pipeline(
    input_path: str | Path,
    clip_id: str,
    crop: list[int] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Path]:
    """Run the full SwimVision pipeline and return the generated artifact paths."""

    resolved_input = Path(input_path).expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input video not found: {resolved_input}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if crop is None:
        try:
            from src.extract import auto_detect_crop
            crop = auto_detect_crop(resolved_input)
        except Exception as exc:
            print(f"Auto-crop failed: {exc}")

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

    # Pre-calculate automation metrics needed for deviations
    reaction_time = None
    breakout_dist = None
    try:
        from src.metrics.reaction_time import detect_beep_frame, detect_reaction_time, detect_first_stroke_time, estimate_breakout_distance
        import numpy as np

        # We need keypoints for breakout distance, so we run extraction first?
        # Actually, let's keep the pipeline order but move the deviation computation after we have the metrics.
        # Or better: run the automation metrics as a separate step before Step 4.
    except Exception as exc:
        print(f"Automation import failed: {exc}")

    total_steps = 6

    _emit_progress(
        progress_callback,
        {
            "event": "pipeline_started",
            "clip_id": clip_id,
            "input_path": str(resolved_input),
            "total_steps": total_steps,
            "crop": crop or [],
        },
    )

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
    _run_step(2, total_steps, "Detecting phase boundaries", ingest_command, progress_callback=progress_callback)

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
        "--fps",
        "30.0", # This should ideally be probed from the video
    ]
    _run_step(3, total_steps, "Computing joint angles", joint_command, progress_callback=progress_callback)

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
    _run_step(4, total_steps, "Rendering annotated overlay", overlay_command, progress_callback=progress_callback)

    # Automatic reaction time and breakout distance calculation
    reaction_time = None
    breakout_dist = None
    try:
        from src.metrics.reaction_time import detect_beep_frame, detect_reaction_time, detect_first_stroke_time, estimate_breakout_distance
        import numpy as np
        kp = np.load(keypoints_path)
        beep = detect_beep_frame(str(resolved_input), 30.0)
        if beep is not None:
            reaction_time = detect_reaction_time(kp, beep, 30.0)

        boundaries = json.load(open(boundaries_path, "r"))
        stroke_data = detect_first_stroke_time(kp, boundaries["entry_start"], 30.0)
        if stroke_data["first_stroke_frame"]:
            breakout_dist = estimate_breakout_distance(kp, boundaries["entry_start"], stroke_data["first_stroke_frame"])
    except Exception as exc:
        print(f"Reaction time/breakout detection failed: {exc}")

    deviation_command = [
        sys.executable,
        "-c",
        (
            "import json, sys, pandas as pd; "
            "from src.metrics.deviation import compute_deviations, aggregate_report; "
            "from src.ingest import analyze_entry_splash; "
            "angles = pd.read_csv(sys.argv[1], index_col=0); "
            "boundaries = json.load(open(sys.argv[2], 'r', encoding='utf-8')); "
            "splash = analyze_entry_splash(sys.argv[4], boundaries['entry_start'], boundaries['entry_end'], sys.argv[5].split() if sys.argv[5] else None); "
            "angles['splash_score'] = splash; "
            "block = compute_deviations(angles, 'block_phase', boundaries); "
            "flight = compute_deviations(angles, 'flight_phase', boundaries); "
            "entry = compute_deviations(angles, 'entry_phase', boundaries); "
            "report = aggregate_report(block, flight, entry); "
            "report['phase_boundaries'] = boundaries; "
            "if sys.argv[6] != 'None': report['breakout_distance_m'] = float(sys.argv[6]); "
            "json.dump(report, open(sys.argv[3], 'w', encoding='utf-8'), indent=2)"
        ),
        str(angles_path),
        str(boundaries_path),
        str(deviations_path),
        str(resolved_input),
        " ".join(crop_values),
        str(breakout_dist),
    ]
    _run_step(5, total_steps, "Computing deviations", deviation_command, progress_callback=progress_callback)

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
    if reaction_time is not None:
        report_command.extend(["--reaction_time", str(reaction_time)])
    _run_step(6, total_steps, "Generating report", report_command, progress_callback=progress_callback)

    outputs = {
        "keypoints": keypoints_path,
        "confidence": confidence_path,
        "angles_csv": angles_path,
        "boundaries_json": boundaries_path,
        "deviations_json": deviations_path,
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
        outputs = run_pipeline(input_path=input_path, clip_id=args.clip_id, crop=args.crop)
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
