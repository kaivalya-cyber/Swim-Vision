# This file writes SwimVision JSON and PDF reports from computed analysis outputs.
"""Report generation utilities for SwimVision analysis results."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
    "velocity": "Low velocity indicates a loss of momentum during this phase.",
    "splash_score": "A high splash score suggests an inefficient water entry that creates excess drag.",
    "angle_of_attack": "The angle at which your arms enter the water. An optimal angle minimizes resistance.",
    "stability_score": "Measures body control during flight. High scores indicate a steady, streamlined jump.",
}
DRILL_SUGGESTIONS = {
    "front_knee_angle": "Practice 'Weight Shift' drills: ensure your center of gravity is forward so your front knee is loaded and ready to explode.",
    "rear_knee_angle": "Focus on 'Rear Leg Drive' drills: Use the back foot to kick the block away, ensuring it provides initial horizontal momentum.",
    "hip_angle": "Try 'Block Squats': improve hip flexibility and strength to maintain a powerful, coiled position.",
    "torso_lean": "Practice 'Head and Torso Alignment': keep your chest close to your knees but eyes looking slightly forward, not at your feet.",
    "body_linearity": "Perform 'Pencil Dives' or 'Tight Streamline Jumps' from the block to focus on a straight line from fingers to toes.",
    "entry_angle": "Use 'Target Entry' drills: place a hula hoop or marker on the water to practice a consistent 35-40 degree entry.",
    "elbow_extension": "Practice 'Wall Streamline Holds': focus on locking elbows behind the head to minimize frontal drag.",
    "streamline_angle": "Focus on 'Finger-First Entry': ensure hands lead the body into the water at a sharp angle to create a 'hole' for the rest of the body.",
    "elbow_lock_angle": "Try 'Push-and-Glide' drills: maintain a rock-solid streamline for at least 5 meters after every push-off.",
    "velocity": "Focus on explosive power drills like 'Box Jumps' or 'Resistance Starts' to increase takeoff and breakout speed.",
    "splash_score": "Practice 'Clean Entry' drills: focus on entering through a single 'hole' in the water with minimal surface disturbance.",
    "angle_of_attack": "Try 'Angle Entry' drills: use a visual marker to guide your arms into the water at a steep, piercing angle.",
    "stability_score": "Practice 'Tight Streamline' jumps and core-strengthening exercises to maintain a rock-solid position in mid-air.",
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
    # Enrich deviations with drills
    enriched_deviations = {}
    for phase in ["block_phase", "flight_phase", "entry_phase"]:
        phase_rows = deviations.get(phase, [])
        enriched_rows = []
        for row in phase_rows:
            metric = row.get("metric")
            enriched_row = dict(row)
            enriched_row["drill"] = DRILL_SUGGESTIONS.get(metric, "Focus on general biomechanical alignment drills.")
            enriched_rows.append(enriched_row)
        enriched_deviations[phase] = enriched_rows

    report_payload: Dict[str, Any] = {
        "clip_id": clip_id,
        "swimmer_id": deviations.get("swimmer_id", "unknown"),
        "date": deviations.get("date", datetime.now().strftime("%Y-%m-%d")),
        "overall_severity": deviations.get("overall_severity", "OPTIMAL"),
        "reaction_time_ms": reaction_time,
        "annotated_video_path": annotated_video_path,
        "phase_timestamps": deviations.get("phase_boundaries", {}),
        "angles": deviations.get("angles", {}),
        "deviations": enriched_deviations,
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

        breakout_dist = deviations.get("breakout_distance_m")
        if breakout_dist is not None:
            pdf.drawString(72, 635, f"Breakout Distance (m): {breakout_dist:.2f}")
            current_video_y = 615
        else:
            current_video_y = 635


        pdf.drawString(72, current_video_y, f"Annotated Video: {annotated_video_path}")
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
            metric = str(row.get("metric", ""))
            explanation = FLAG_EXPLANATIONS.get(
                metric,
                "This metric deviated meaningfully from the expected biomechanical target.",
            )
            drill = DRILL_SUGGESTIONS.get(metric, "Focus on general biomechanical alignment drills.")

            pdf.setFont("Helvetica-Bold", 10)
            line_header = f"{row.get('phase', '')} | {metric} | {row.get('flag', '')}"
            pdf.drawString(72, current_y, line_header)
            current_y -= 14

            pdf.setFont("Helvetica", 9)
            pdf.drawString(82, current_y, f"Issue: {explanation}")
            current_y -= 12
            pdf.drawString(82, current_y, f"Drill: {drill}")
            current_y -= 20

            if current_y < 80:
                pdf.showPage()
                _draw_page_header(pdf, "Key Flagged Issues (cont.)")
                current_y = 715

        pdf.save()
    except Exception as exc:
        raise RuntimeError(f"Failed while composing PDF report '{pdf_path}': {exc}") from exc

    LOGGER.info("Generated report artifacts at %s and %s", json_path, pdf_path)
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
    return parser


def main() -> int:
    """Run the command-line interface for report generation.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()
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
