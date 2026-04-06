# This file runs the SwimVision pipeline end to end for a single clip.
"""Single-command pipeline runner for SwimVision."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def _display_command(command: list[str]) -> str:
    """Render a subprocess command for human-readable logging."""

    return " ".join(command)


def _run_step(index: int, total: int, label: str, command: list[str]) -> None:
    """Run a pipeline step and stop immediately on failure."""

    print(f"[{index}/{total}] {label}...")
    start_time = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.monotonic() - start_time
    if result.returncode != 0:
        print(f"[{index}/{total}] FAILED: {label} ({elapsed:.1f}s)")
        print(f"Command: {_display_command(command)}")
        print(f"Error: step '{label}' exited with code {result.returncode}.")
        raise SystemExit(result.returncode)
    print(f"[{index}/{total}] Done ({elapsed:.1f}s)")


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


def main() -> int:
    """Run the full SwimVision pipeline for a single clip."""

    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input).resolve() if not os.path.isabs(args.input) else Path(args.input)
    if not input_path.exists():
        print(f"Input video not found: {input_path}")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    clip_id = args.clip_id
    crop = [str(value) for value in args.crop] if args.crop is not None else []

    keypoints_path = PROCESSED_DIR / f"{clip_id}_keypoints.npy"
    confidence_path = PROCESSED_DIR / f"{clip_id}_confidence.npy"
    angles_path = RESULTS_DIR / f"{clip_id}_angles.csv"
    boundaries_path = RESULTS_DIR / f"{clip_id}_boundaries.json"
    deviations_path = RESULTS_DIR / f"{clip_id}_deviations.json"
    annotated_path = RESULTS_DIR / f"{clip_id}_annotated.mp4"
    report_json_path = RESULTS_DIR / f"{clip_id}_report.json"
    report_pdf_path = RESULTS_DIR / f"{clip_id}_report.pdf"

    width, height = _resolve_dimensions(input_path, args.crop)

    total_steps = 6

    extract_command = [
        sys.executable,
        "src/extract.py",
        "--input",
        str(input_path),
        "--output",
        str(PROCESSED_DIR),
        "--clip_id",
        clip_id,
    ]
    if crop:
        extract_command.extend(["--crop", *crop])
    _run_step(1, total_steps, "Extracting keypoints", extract_command)

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
    _run_step(2, total_steps, "Detecting phase boundaries", ingest_command)

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
    _run_step(3, total_steps, "Computing joint angles", joint_command)

    deviation_command = [
        sys.executable,
        "-c",
        (
            "import json, sys, pandas as pd; "
            "from src.metrics.deviation import compute_deviations, aggregate_report; "
            "angles = pd.read_csv(sys.argv[1], index_col=0); "
            "boundaries = json.load(open(sys.argv[2], 'r', encoding='utf-8')); "
            "block = compute_deviations(angles, 'block_phase', boundaries); "
            "flight = compute_deviations(angles, 'flight_phase', boundaries); "
            "entry = compute_deviations(angles, 'entry_phase', boundaries); "
            "report = aggregate_report(block, flight, entry); "
            "report['phase_boundaries'] = boundaries; "
            "json.dump(report, open(sys.argv[3], 'w', encoding='utf-8'), indent=2)"
        ),
        str(angles_path),
        str(boundaries_path),
        str(deviations_path),
    ]
    _run_step(4, total_steps, "Computing deviations", deviation_command)

    overlay_command = [
        sys.executable,
        "src/overlay.py",
        "--input",
        str(input_path),
        "--keypoints",
        str(keypoints_path),
        "--angles",
        str(angles_path),
        "--output",
        str(annotated_path),
    ]
    if crop:
        overlay_command.extend(["--crop", *crop])
    _run_step(5, total_steps, "Rendering annotated overlay", overlay_command)

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
    _run_step(6, total_steps, "Generating report", report_command)

    outputs = [
        keypoints_path,
        confidence_path,
        angles_path,
        boundaries_path,
        deviations_path,
        annotated_path,
        report_json_path,
        report_pdf_path,
    ]

    print("\nPipeline complete. Output files found:")
    for output_path in outputs:
        exists = output_path.exists() and output_path.stat().st_size > 0
        status = "OK" if exists else "MISSING"
        print(f"- [{status}] {output_path.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
