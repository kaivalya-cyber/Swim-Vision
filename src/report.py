# This file writes SwimVision JSON and PDF reports from computed analysis outputs.
"""Report generation utilities for SwimVision analysis results."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

FLAG_TO_COLOR = {
    "OPTIMAL": colors.lightgreen,
    "MINOR": colors.khaki,
    "SIGNIFICANT": colors.orange,
    "CRITICAL": colors.salmon,
}
FLAG_EXPLANATIONS = {
    "front_knee_angle": "Front-leg setup may reduce horizontal force production off the block.",
    "rear_knee_angle": "Rear-leg drive geometry may limit explosive push-off mechanics.",
    "hip_angle": "Hip positioning may reduce force transfer from the lower body into the launch.",
    "torso_lean": "Torso lean may alter the balance between forward projection and block stability.",
    "body_linearity": "Body alignment in flight may increase drag before water entry.",
    "entry_angle": "Entry angle may cause excess splash or an overly deep trajectory.",
    "elbow_extension": "Arm extension at entry may be incomplete, reducing streamline quality.",
    "streamline_angle": "Streamline alignment may increase frontal resistance after entry.",
    "elbow_lock_angle": "Elbow position may soften the streamline during underwater travel.",
    "left_elbow_flexion": "Left elbow catch angle may reduce propulsion efficiency.",
    "right_elbow_flexion": "Right elbow catch angle may reduce propulsion efficiency.",
    "left_shoulder_rotation": "Left shoulder rotation may indicate insufficient body roll.",
    "right_shoulder_rotation": "Right shoulder rotation may indicate insufficient body roll.",
    "left_hand_speed": "Left hand pull speed may be suboptimal.",
    "right_hand_speed": "Right hand pull speed may be suboptimal.",
    "stroke_rate": "Stroke rate may be too low or too high for optimal efficiency.",
    "body_roll": "Body roll may be insufficient for effective breathing and propulsion.",
    "symmetry_index": "Left/right arm asymmetry may indicate technique imbalance.",
}


def _flatten_phase_rows(deviations: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten per-phase deviation rows into a single list.

    Args:
        deviations: Aggregate deviation payload.

    Returns:
        List of metric rows with phase annotations.
    """

    rows: List[Dict[str, Any]] = []
    for phase_name in ("block_phase", "flight_phase", "entry_phase"):
        for row in deviations.get(phase_name, []):
            if isinstance(row, dict):
                enriched_row = dict(row)
                enriched_row["phase"] = phase_name
                rows.append(enriched_row)
    return rows


def _draw_page_header(pdf: canvas.Canvas, title: str) -> None:
    """Draw a consistent page header.

    Args:
        pdf: Active ReportLab canvas.
        title: Header title text.

    Returns:
        None.
    """

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, 750, title)


def generate_report(
    clip_id: str, deviations: Dict[str, Any], reaction_time: float | None, annotated_video_path: str, output_dir: str
) -> Dict[str, str]:
    """Generate JSON and PDF SwimVision reports for an analyzed clip.

    Args:
        clip_id: Clip identifier.
        deviations: Aggregate deviation payload and optional report context.
        reaction_time: Detected reaction time in milliseconds.
        annotated_video_path: Path to the annotated video.
        output_dir: Directory where report artifacts will be written.

    Returns:
        Paths to the generated JSON and PDF reports.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{clip_id}_report.json"
    pdf_path = output_path / f"{clip_id}_report.pdf"

    phase_rows = _flatten_phase_rows(deviations)
    report_payload: Dict[str, Any] = {
        "clip_id": clip_id,
        "swimmer_id": deviations.get("swimmer_id", "unknown"),
        "date": deviations.get("date", datetime.now().strftime("%Y-%m-%d")),
        "overall_severity": deviations.get("overall_severity", "OPTIMAL"),
        "reaction_time_ms": reaction_time,
        "annotated_video_path": annotated_video_path,
        "phase_timestamps": deviations.get("phase_boundaries", {}),
        "angles": deviations.get("angles", {}),
        "deviations": {
            "block_phase": deviations.get("block_phase", []),
            "flight_phase": deviations.get("flight_phase", []),
            "entry_phase": deviations.get("entry_phase", []),
        },
    }

    try:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to write JSON report '{json_path}': {exc}") from exc

    try:
        pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize PDF report '{pdf_path}': {exc}") from exc

    try:
        _draw_page_header(pdf, f"SwimVision Report: {clip_id}")
        pdf.setFont("Helvetica", 12)
        pdf.drawString(72, 715, f"Swimmer ID: {report_payload['swimmer_id']}")
        pdf.drawString(72, 695, f"Date: {report_payload['date']}")
        pdf.drawString(72, 675, f"Overall Severity: {report_payload['overall_severity']}")
        pdf.drawString(72, 655, f"Reaction Time (ms): {reaction_time if reaction_time is not None else 'N/A'}")
        pdf.drawString(72, 635, f"Annotated Video: {annotated_video_path}")
        pdf.showPage()

        _draw_page_header(pdf, "Phase Breakdown")
        pdf.setFont("Helvetica-Bold", 10)
        header_y = 725
        headers = ["Phase", "Metric", "Measured", "Optimal", "Deviation", "Flag"]
        header_x = [40, 110, 250, 325, 405, 485]
        for index, header in enumerate(headers):
            pdf.drawString(header_x[index], header_y, header)

        current_y = 705
        pdf.setFont("Helvetica", 9)
        for row in phase_rows:
            row_flag = str(row.get("flag", "OPTIMAL"))
            pdf.setFillColor(FLAG_TO_COLOR.get(row_flag, colors.white))
            pdf.rect(36, current_y - 4, 540, 16, fill=1, stroke=0)
            pdf.setFillColor(colors.black)
            if row.get("optimal_min") is not None and row.get("optimal_max") is not None:
                optimal_range = f"{float(row['optimal_min']):.1f}-{float(row['optimal_max']):.1f}"
            else:
                optimal_range = "N/A"
            values = [
                str(row.get("phase", "")),
                str(row.get("metric", "")),
                f"{float(row.get('measured', 0.0)):.1f}",
                optimal_range,
                f"{float(row.get('deviation', 0.0)):.1f}",
                row_flag,
            ]
            for index, value in enumerate(values):
                pdf.drawString(header_x[index], current_y, value)
            current_y -= 20
            if current_y < 72:
                pdf.showPage()
                _draw_page_header(pdf, "Phase Breakdown (cont.)")
                pdf.setFont("Helvetica", 9)
                current_y = 725

        pdf.showPage()
        _draw_page_header(pdf, "Key Flagged Issues")
        pdf.setFont("Helvetica", 11)
        current_y = 715
        flagged_rows = [row for row in phase_rows if row.get("flag") in {"SIGNIFICANT", "CRITICAL"}]
        if not flagged_rows:
            pdf.drawString(72, current_y, "No SIGNIFICANT or CRITICAL deviations were detected.")
        for row in flagged_rows:
            explanation = FLAG_EXPLANATIONS.get(
                str(row.get("metric", "")),
                "This metric deviated meaningfully from the expected biomechanical target.",
            )
            line = (
                f"{row.get('phase', '')} | {row.get('metric', '')} | "
                f"{row.get('flag', '')}: {explanation}"
            )
            pdf.drawString(72, current_y, line[:100])
            current_y -= 22
            if current_y < 72:
                pdf.showPage()
                _draw_page_header(pdf, "Key Flagged Issues (cont.)")
                pdf.setFont("Helvetica", 11)
                current_y = 715

        pdf.save()
    except Exception as exc:
        raise RuntimeError(f"Failed while composing PDF report '{pdf_path}': {exc}") from exc

    LOGGER.info("Generated report artifacts at %s and %s", json_path, pdf_path)
    return {"json_path": str(json_path), "pdf_path": str(pdf_path)}


def generate_stroke_report(
    clip_id: str,
    stroke_metrics: Dict[str, Any],
    stroke_deviations: Dict[str, Any],
    annotated_video_path: str,
    output_dir: str,
) -> Dict[str, str]:
    """Generate stroke-specific JSON and PDF reports with cyclical averaging.

    Args:
        clip_id: Clip identifier.
        stroke_metrics: Stroke metrics payload (per-cycle and aggregate).
        stroke_deviations: Stroke deviation scoring payload.
        annotated_video_path: Path to the annotated video.
        output_dir: Directory where report artifacts will be written.

    Returns:
        Paths to JSON and PDF report files.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{clip_id}_report.json"
    pdf_path = output_path / f"{clip_id}_report.pdf"

    aggregate = stroke_metrics.get("aggregate", {})
    cycles = stroke_metrics.get("cycles", [])
    num_cycles = int(aggregate.get("num_cycles", 0))
    overall_severity = stroke_deviations.get("overall_severity", "OPTIMAL")

    report_payload: Dict[str, Any] = {
        "clip_id": clip_id,
        "analysis_mode": "stroke",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "overall_severity": overall_severity,
        "annotated_video_path": annotated_video_path,
        "num_cycles": num_cycles,
        "aggregate_metrics": aggregate,
        "per_cycle_metrics": cycles,
        "stroke_deviations": stroke_deviations,
    }

    try:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to write stroke JSON report '{json_path}': {exc}") from exc

    try:
        pdf = canvas.Canvas(str(pdf_path), pagesize=letter)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize stroke PDF report '{pdf_path}': {exc}") from exc

    try:
        _draw_page_header(pdf, f"SwimVision Stroke Report: {clip_id}")
        pdf.setFont("Helvetica", 12)
        pdf.drawString(72, 715, f"Date: {report_payload['date']}")
        pdf.drawString(72, 695, f"Overall Severity: {overall_severity}")
        pdf.drawString(72, 675, f"Stroke Cycles Detected: {num_cycles}")
        pdf.drawString(72, 655, f"Annotated Video: {annotated_video_path}")
        pdf.showPage()

        _draw_page_header(pdf, "Aggregate Stroke Metrics")
        pdf.setFont("Helvetica", 11)
        metric_labels = {
            "stroke_rate": "Stroke Rate (spm)",
            "body_roll": "Body Roll (deg)",
            "symmetry_index": "Symmetry Index (%)",
            "left_elbow_flexion": "Left Elbow Flexion (deg)",
            "right_elbow_flexion": "Right Elbow Flexion (deg)",
            "left_shoulder_rotation": "Left Shoulder Rot (deg)",
            "right_shoulder_rotation": "Right Shoulder Rot (deg)",
            "left_hand_speed": "Left Hand Speed (norm)",
            "right_hand_speed": "Right Hand Speed (norm)",
            "cycle_duration_seconds": "Cycle Duration (s)",
        }
        current_y = 715
        for key, label in metric_labels.items():
            value = aggregate.get(key, 0.0)
            pdf.drawString(72, current_y, f"{label}: {float(value):.1f}")
            current_y -= 18
        pdf.showPage()

        if cycles:
            _draw_page_header(pdf, "Per-Cycle Breakdown")
            pdf.setFont("Helvetica-Bold", 9)
            header_y = 725
            headers = [
                "Cycle", "SR (spm)", "L Elbow", "R Elbow",
                "Body Roll", "Sym %", "Dur (s)",
            ]
            header_x = [40, 100, 170, 240, 310, 380, 440]
            for index, header in enumerate(headers):
                pdf.drawString(header_x[index], header_y, header)

            current_y = 705
            pdf.setFont("Helvetica", 8)
            for cycle in cycles:
                sr = float(cycle.get("stroke_rate", 0))
                le = float(cycle.get("left_elbow_flexion", 0))
                re = float(cycle.get("right_elbow_flexion", 0))
                br = float(cycle.get("body_roll", 0))
                si = float(cycle.get("symmetry_index", 0))
                dur = float(cycle.get("cycle_duration_seconds", 0))
                row_values = [f"{sr:.1f}", f"{le:.1f}", f"{re:.1f}", f"{br:.1f}", f"{si:.1f}", f"{dur:.2f}"]
                pdf.drawString(header_x[0], current_y, str(cycle.get("cycle_index", "")))
                for col, val in enumerate(row_values):
                    pdf.drawString(header_x[col + 1], current_y, val)
                current_y -= 16
                if current_y < 72:
                    pdf.showPage()
                    current_y = 725

        pdf.showPage()
        _draw_page_header(pdf, "Key Flagged Issues")
        pdf.setFont("Helvetica", 11)
        current_y = 715
        flagged_rows = stroke_deviations.get("stroke_cycle", [])
        sig_critical = [
            row for row in flagged_rows
            if isinstance(row, dict) and row.get("flag") in {"SIGNIFICANT", "CRITICAL"}
        ]
        if not sig_critical:
            pdf.drawString(72, current_y, "No SIGNIFICANT or CRITICAL stroke deviations were detected.")
        for row in sig_critical:
            explanation = FLAG_EXPLANATIONS.get(
                str(row.get("metric", "")),
                "This metric deviated meaningfully from the expected biomechanical target.",
            )
            line = (
                f"stroke | {row.get('metric', '')} | "
                f"{row.get('flag', '')}: {explanation}"
            )
            pdf.drawString(72, current_y, line[:100])
            current_y -= 22
            if current_y < 72:
                pdf.showPage()
                current_y = 715

        pdf.save()
    except Exception as exc:
        raise RuntimeError(f"Failed while composing stroke PDF report '{pdf_path}': {exc}") from exc

    LOGGER.info("Generated stroke report artifacts at %s and %s", json_path, pdf_path)
    return {"json_path": str(json_path), "pdf_path": str(pdf_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for report generation.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Generate SwimVision JSON and PDF reports.")
    parser.add_argument("--clip_id", required=True, help="Clip identifier.")
    parser.add_argument("--keypoints", help="Path to keypoints .npy file.")
    parser.add_argument("--angles", help="Path to angles CSV.")
    parser.add_argument("--video", required=True, help="Annotated video path.")
    parser.add_argument("--output", required=True, help="Directory for report artifacts.")
    parser.add_argument("--deviations", help="Optional aggregate deviation JSON path.")
    parser.add_argument("--reaction_time", type=float, help="Optional reaction time in milliseconds.")
    parser.add_argument("--vel_accel", help="Optional velocity/acceleration profile JSON.")
    parser.add_argument("--symmetry", help="Optional symmetry analysis JSON.")
    parser.add_argument("--joint_contributions", help="Optional joint contributions JSON.")
    parser.add_argument("--entry_analysis", help="Optional entry analysis JSON.")
    parser.add_argument(
        "--analysis_mode",
        choices=["dive", "stroke"],
        default="dive",
        help="Analysis mode: dive or stroke.",
    )
    parser.add_argument("--stroke_metrics", help="Path to stroke metrics JSON.")
    parser.add_argument("--stroke_deviations", help="Path to stroke deviations JSON.")
    return parser


def _main_stroke(args: Any) -> int:
    """Handle stroke report generation from the CLI."""

    stroke_metrics_path = Path(args.stroke_metrics) if args.stroke_metrics else Path(args.output) / f"{args.clip_id}_stroke_metrics.json"
    stroke_deviations_path = Path(args.stroke_deviations) if args.stroke_deviations else Path(args.output) / f"{args.clip_id}_stroke_deviations.json"

    try:
        with open(stroke_metrics_path, "r", encoding="utf-8") as handle:
            stroke_metrics = json.load(handle)
    except Exception as exc:
        LOGGER.error("Failed to read stroke metrics %s: %s", stroke_metrics_path, exc)
        return 1

    try:
        with open(stroke_deviations_path, "r", encoding="utf-8") as handle:
            stroke_deviations = json.load(handle)
    except Exception as exc:
        LOGGER.error("Failed to read stroke deviations %s: %s", stroke_deviations_path, exc)
        return 1

    try:
        artifact_paths = generate_stroke_report(
            args.clip_id,
            stroke_metrics,
            stroke_deviations,
            args.video,
            args.output,
        )
    except Exception as exc:
        LOGGER.error("Stroke report generation failed: %s", exc)
        return 1

    print(json.dumps(artifact_paths, indent=2))
    return 0


def main() -> int:
    """Run the command-line interface for report generation.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.analysis_mode == "stroke":
        return _main_stroke(args)

    deviations_path = Path(args.deviations) if args.deviations else Path(args.output) / f"{args.clip_id}_deviations.json"
    try:
        with open(deviations_path, "r", encoding="utf-8") as handle:
            deviations = json.load(handle)
    except Exception as exc:
        LOGGER.error("Failed to read deviation JSON %s: %s", deviations_path, exc)
        return 1

    if args.angles:
        try:
            deviations["angles"] = {"csv_path": str(Path(args.angles))}
        except Exception:
            pass
    if args.keypoints:
        try:
            deviations["keypoints_path"] = str(Path(args.keypoints))
        except Exception:
            pass

    # Load optional enhanced analysis results
    if args.vel_accel:
        try:
            with open(args.vel_accel, "r", encoding="utf-8") as f:
                deviations["velocity_acceleration"] = json.load(f)
        except Exception:
            pass
    if args.symmetry:
        try:
            with open(args.symmetry, "r", encoding="utf-8") as f:
                deviations["symmetry_analysis"] = json.load(f)
        except Exception:
            pass
    if args.joint_contributions:
        try:
            with open(args.joint_contributions, "r", encoding="utf-8") as f:
                deviations["joint_contributions"] = json.load(f)
        except Exception:
            pass
    if args.entry_analysis:
        try:
            with open(args.entry_analysis, "r", encoding="utf-8") as f:
                deviations["entry_analysis"] = json.load(f)
        except Exception:
            pass

    try:
        artifact_paths = generate_report(
            args.clip_id,
            deviations,
            args.reaction_time,
            args.video,
            args.output,
        )
    except Exception as exc:
        LOGGER.error("Report generation failed: %s", exc)
        return 1

    print(json.dumps(artifact_paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
