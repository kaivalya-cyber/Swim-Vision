# This file assesses biomechanical injury risk from SwimVision analysis results.
"""Injury risk assessment for SwimVision — pattern detection and risk scoring."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Risk factor definitions: (metric, condition, weight, description)
SHOULDER_RISK_FACTORS = [
    ("elbow_extension", ">175", 2.0, "Extreme elbow hyperextension (>175°) increases shoulder joint stress"),
    ("left_elbow_flexion", "<85", 1.5, "Low left elbow flexion (<85°) may indicate shoulder impingement risk"),
    ("right_elbow_flexion", "<85", 1.5, "Low right elbow flexion (<85°) may indicate shoulder impingement risk"),
]

KNEE_RISK_FACTORS = [
    ("front_knee_angle", "<85", 2.0, "Front knee angle below 85° indicates excessive loading"),
    ("rear_knee_angle", "<100", 2.0, "Rear knee angle below 100° may stress patellar tendon"),
    ("front_knee_angle", ">135", 1.5, "Front knee angle above 135° may indicate unstable block position"),
]

ASYMMETRY_RISK_FACTORS = [
    ("elbow_flexion_asymmetry", ">15", 2.5, "Elbow flexion asymmetry >15% — unilateral overloading risk"),
    ("knee_angle_asymmetry", ">12", 2.0, "Knee angle asymmetry >12% — uneven force distribution"),
    ("hand_speed_asymmetry", ">20", 2.0, "Hand speed asymmetry >20% — compensatory movement pattern"),
]

RISK_LEVELS = {
    "LOW": (0, 5),
    "MEDIUM": (5, 10),
    "HIGH": (10, 15),
    "CRITICAL": (15, float("inf")),
}


def classify_risk(total_score: float) -> str:
    """Classify risk level from total score."""
    for level, (min_val, max_val) in RISK_LEVELS.items():
        if min_val <= total_score < max_val:
            return level
    return "CRITICAL"


def evaluate_condition(value: float, condition: str) -> bool:
    """Evaluate a numeric condition string against a value."""
    if condition.startswith(">"):
        return value > float(condition[1:])
    if condition.startswith("<"):
        return value < float(condition[1:])
    if condition.startswith(">="):
        return value >= float(condition[2:])
    if condition.startswith("<="):
        return value <= float(condition[2:])
    return False


def assess_dive_risk(
    deviations: Dict[str, Any],
    symmetry_data: Dict[str, Any] | None = None,
    angles_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Assess injury risk from dive analysis results.

    Args:
        deviations: Deviation scoring payload.
        symmetry_data: Optional symmetry analysis results.
        angles_df: Optional per-frame angle DataFrame.

    Returns:
        Risk assessment dictionary with overall level and flagged factors.
    """
    total_score = 0.0
    flagged_factors: List[Dict[str, Any]] = []

    # Collect all measured values from deviations
    measured_values: Dict[str, float] = {}
    for phase_name in ("block_phase", "flight_phase", "entry_phase"):
        for row in deviations.get(phase_name, []):
            if isinstance(row, dict):
                measured_values[str(row.get("metric", ""))] = float(row.get("measured", 0))

    # Check shoulder risk factors
    for metric, condition, weight, description in SHOULDER_RISK_FACTORS:
        if metric in measured_values:
            if evaluate_condition(measured_values[metric], condition):
                total_score += weight
                flagged_factors.append({
                    "category": "shoulder",
                    "metric": metric,
                    "measured": measured_values[metric],
                    "condition": condition,
                    "weight": weight,
                    "description": description,
                })

    # Check knee risk factors
    for metric, condition, weight, description in KNEE_RISK_FACTORS:
        if metric in measured_values:
            if evaluate_condition(measured_values[metric], condition):
                total_score += weight
                flagged_factors.append({
                    "category": "knee",
                    "metric": metric,
                    "measured": measured_values[metric],
                    "condition": condition,
                    "weight": weight,
                    "description": description,
                })

    # Check asymmetry risk factors
    if symmetry_data:
        phases = symmetry_data.get("phases", {})
        all_pairs: Dict[str, Dict[str, Any]] = {}
        for phase_data in phases.values():
            all_pairs.update(phase_data)

        # Elbow asymmetry
        elbow_data = all_pairs.get("elbow_flexion", {})
        elbow_si = elbow_data.get("symmetry_index_pct")
        if elbow_si is not None and elbow_si > 15:
            weight = 2.5
            total_score += weight
            flagged_factors.append({
                "category": "asymmetry",
                "metric": "elbow_flexion_asymmetry",
                "measured": elbow_si,
                "condition": ">15",
                "weight": weight,
                "description": "Elbow flexion asymmetry >15% — unilateral overloading risk",
            })

        # Knee asymmetry
        knee_data = all_pairs.get("knee_angle", {})
        knee_si = knee_data.get("symmetry_index_pct")
        if knee_si is not None and knee_si > 12:
            weight = 2.0
            total_score += weight
            flagged_factors.append({
                "category": "asymmetry",
                "metric": "knee_angle_asymmetry",
                "measured": knee_si,
                "condition": ">12",
                "weight": weight,
                "description": "Knee angle asymmetry >12% — uneven force distribution",
            })

    risk_level = classify_risk(total_score)

    return {
        "overall_risk_level": risk_level,
        "total_risk_score": round(total_score, 1),
        "num_flagged_factors": len(flagged_factors),
        "flagged_factors": flagged_factors,
        "preventive_recommendations": _generate_recommendations(risk_level, flagged_factors),
    }


def assess_stroke_risk(
    stroke_metrics: Dict[str, Any],
    symmetry_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assess injury risk from stroke analysis results.

    Args:
        stroke_metrics: Stroke metrics payload.
        symmetry_data: Optional symmetry analysis results.

    Returns:
        Risk assessment dictionary.
    """
    total_score = 0.0
    flagged_factors: List[Dict[str, Any]] = []

    aggregate = stroke_metrics.get("aggregate", {})

    # Elbow flexion checks
    for side, key in [("left", "left_elbow_flexion"), ("right", "right_elbow_flexion")]:
        val = aggregate.get(key)
        if val is not None:
            val = float(val)
            if val < 85:
                total_score += 1.5
                flagged_factors.append({
                    "category": "shoulder",
                    "metric": key,
                    "measured": val,
                    "condition": "<85",
                    "weight": 1.5,
                    "description": f"Low {side} elbow flexion (<85°) may indicate shoulder impingement risk",
                })

    # Asymmetry checks
    if symmetry_data:
        pairs = symmetry_data.get("pairs", {})
        for label, data in pairs.items():
            si = data.get("symmetry_index_pct")
            if si is not None and si > 15:
                total_score += 2.0
                flagged_factors.append({
                    "category": "asymmetry",
                    "metric": f"{label}_asymmetry",
                    "measured": si,
                    "condition": ">15",
                    "weight": 2.0,
                    "description": f"{label.replace('_', ' ').title()} asymmetry >15% — compensatory pattern",
                })

    risk_level = classify_risk(total_score)

    return {
        "overall_risk_level": risk_level,
        "total_risk_score": round(total_score, 1),
        "num_flagged_factors": len(flagged_factors),
        "flagged_factors": flagged_factors,
        "preventive_recommendations": _generate_recommendations(risk_level, flagged_factors),
    }


def _generate_recommendations(
    risk_level: str,
    flagged_factors: List[Dict[str, Any]],
) -> List[str]:
    """Generate preventive recommendations based on risk level and factors."""
    recommendations: List[str] = []

    if risk_level == "LOW":
        recommendations.append("Continue current technique with regular monitoring.")
        return recommendations

    if risk_level == "CRITICAL":
        recommendations.append("URGENT: Consult a sports medicine professional for evaluation.")
        recommendations.append("Discontinue high-intensity training until assessed.")

    shoulder_issues = [f for f in flagged_factors if f["category"] == "shoulder"]
    if shoulder_issues:
        recommendations.append("Incorporate rotator cuff strengthening exercises into warm-up routine.")
        recommendations.append("Review arm entry mechanics — focus on relaxed hand entry.")

    knee_issues = [f for f in flagged_factors if f["category"] == "knee"]
    if knee_issues:
        recommendations.append("Focus on knee tracking alignment during block setup.")
        recommendations.append("Include eccentric quadriceps strengthening exercises.")

    asymmetry_issues = [f for f in flagged_factors if f["category"] == "asymmetry"]
    if asymmetry_issues:
        recommendations.append("Incorporate unilateral strength training to address asymmetry.")
        recommendations.append("Video review with coach — focus on bilateral symmetry drills.")

    return recommendations


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for injury risk assessment."""
    parser = argparse.ArgumentParser(
        description="Assess biomechanical injury risk from SwimVision results."
    )
    parser.add_argument("--deviations", help="Deviations JSON path (dive mode).")
    parser.add_argument("--stroke-metrics", help="Stroke metrics JSON path (stroke mode).")
    parser.add_argument("--symmetry", help="Symmetry analysis JSON path.")
    parser.add_argument(
        "--mode",
        choices=["dive", "stroke"],
        default="dive",
        help="Analysis mode.",
    )
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run injury risk CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    symmetry_data = None
    if args.symmetry:
        try:
            with open(args.symmetry, "r", encoding="utf-8") as f:
                symmetry_data = json.load(f)
        except Exception as exc:
            LOGGER.warning("Failed to load symmetry data: %s", exc)

    try:
        if args.mode == "stroke":
            if not args.stroke_metrics:
                LOGGER.error("--stroke-metrics required for stroke mode")
                return 1
            with open(args.stroke_metrics, "r", encoding="utf-8") as f:
                stroke_metrics = json.load(f)
            result = assess_stroke_risk(stroke_metrics, symmetry_data)
        else:
            if not args.deviations:
                LOGGER.error("--deviations required for dive mode")
                return 1
            with open(args.deviations, "r", encoding="utf-8") as f:
                deviations = json.load(f)
            result = assess_dive_risk(deviations, symmetry_data)
    except Exception as exc:
        LOGGER.error("Risk assessment failed: %s", exc)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        LOGGER.info("Saved risk assessment to %s", args.output)
    else:
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
