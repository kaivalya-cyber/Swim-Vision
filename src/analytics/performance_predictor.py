# This file predicts swim performance outcomes from biomechanical analysis using heuristic models.
"""Performance prediction for SwimVision — race time, skill level, and improvement potential."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Base race times (seconds) for reference levels — heuristic estimates
BASE_TIMES: Dict[str, Dict[str, float]] = {
    "50m_freestyle": {"elite": 21.0, "advanced": 23.5, "intermediate": 26.0, "beginner": 30.0},
    "100m_freestyle": {"elite": 47.0, "advanced": 52.0, "intermediate": 58.0, "beginner": 68.0},
    "50m_butterfly": {"elite": 23.0, "advanced": 25.5, "intermediate": 29.0, "beginner": 34.0},
    "50m_backstroke": {"elite": 24.5, "advanced": 27.0, "intermediate": 31.0, "beginner": 36.0},
}

# Metric impact factors: how much each metric deviation affects race time (seconds per severity level)
METRIC_TIME_IMPACT: Dict[str, Dict[str, float]] = {
    "reaction_time_ms": {"MINOR": 0.05, "SIGNIFICANT": 0.15, "CRITICAL": 0.30},
    "front_knee_angle": {"MINOR": 0.08, "SIGNIFICANT": 0.20, "CRITICAL": 0.35},
    "rear_knee_angle": {"MINOR": 0.10, "SIGNIFICANT": 0.25, "CRITICAL": 0.40},
    "hip_angle": {"MINOR": 0.06, "SIGNIFICANT": 0.15, "CRITICAL": 0.30},
    "torso_lean": {"MINOR": 0.04, "SIGNIFICANT": 0.10, "CRITICAL": 0.20},
    "body_linearity": {"MINOR": 0.05, "SIGNIFICANT": 0.12, "CRITICAL": 0.25},
    "entry_angle": {"MINOR": 0.06, "SIGNIFICANT": 0.15, "CRITICAL": 0.30},
    "elbow_extension": {"MINOR": 0.03, "SIGNIFICANT": 0.08, "CRITICAL": 0.15},
    "streamline_angle": {"MINOR": 0.05, "SIGNIFICANT": 0.12, "CRITICAL": 0.22},
    "stroke_rate": {"MINOR": 0.07, "SIGNIFICANT": 0.18, "CRITICAL": 0.35},
    "symmetry_index": {"MINOR": 0.04, "SIGNIFICANT": 0.10, "CRITICAL": 0.20},
}

# Skill classification scoring weights
SKILL_FEATURE_WEIGHTS: Dict[str, float] = {
    "reaction_time_score": 0.15,
    "body_position_score": 0.20,
    "entry_quality_score": 0.20,
    "consistency_score": 0.15,
    "deviation_severity_score": 0.15,
    "symmetry_score": 0.15,
}


@dataclass
class PerformancePrediction:
    """Predicted performance metrics for a swimmer."""

    estimated_50m_time_sec: Optional[float] = None
    estimated_100m_time_sec: Optional[float] = None
    skill_level: str = "intermediate"
    skill_confidence: float = 0.5
    improvement_potential_sec: Optional[float] = None
    top_improvement_areas: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def _score_reaction_time(rt_ms: Optional[float]) -> float:
    """Score reaction time: elite <600ms, good <700ms, fair <800ms."""
    if rt_ms is None:
        return 0.5
    if rt_ms < 580:
        return 1.0
    if rt_ms < 650:
        return 0.8
    if rt_ms < 750:
        return 0.6
    if rt_ms < 850:
        return 0.4
    return 0.2


def _score_body_position(deviations: Dict[str, Any]) -> float:
    """Score body position from block/flight deviations."""
    scores = []
    for phase in ("block_phase", "flight_phase"):
        for row in deviations.get(phase, []):
            if not isinstance(row, dict):
                continue
            flag = row.get("flag", "OPTIMAL")
            if flag == "OPTIMAL":
                scores.append(1.0)
            elif flag == "MINOR":
                scores.append(0.7)
            elif flag == "SIGNIFICANT":
                scores.append(0.4)
            else:
                scores.append(0.1)
    return float(np.mean(scores)) if scores else 0.5


def _score_entry_quality(deviations: Dict[str, Any]) -> float:
    """Score entry quality from entry phase deviations."""
    scores = []
    for row in deviations.get("entry_phase", []):
        if not isinstance(row, dict):
            continue
        flag = row.get("flag", "OPTIMAL")
        if flag == "OPTIMAL":
            scores.append(1.0)
        elif flag == "MINOR":
            scores.append(0.7)
        elif flag == "SIGNIFICANT":
            scores.append(0.4)
        else:
            scores.append(0.1)
    return float(np.mean(scores)) if scores else 0.5


def _score_consistency(consistency_data: Optional[Dict[str, Any]]) -> float:
    """Score consistency from consistency analyzer output."""
    if consistency_data is None:
        return 0.5
    overall = consistency_data.get("overall_consistency_score", 50)
    return min(1.0, max(0.0, overall / 100.0))


def _score_deviation_severity(deviations: Dict[str, Any]) -> float:
    """Score overall deviation severity (lower is better)."""
    overall = deviations.get("overall_severity", "OPTIMAL")
    if overall == "OPTIMAL":
        return 1.0
    if overall == "MINOR":
        return 0.7
    if overall == "SIGNIFICANT":
        return 0.4
    return 0.1


def _score_symmetry(symmetry_data: Optional[Dict[str, Any]]) -> float:
    """Score bilateral symmetry from symmetry analysis output."""
    if symmetry_data is None:
        return 0.5
    phases = symmetry_data.get("phases", {})
    si_values = []
    for phase_data in phases.values():
        for pair_data in phase_data.values():
            si = pair_data.get("symmetry_index_pct")
            if si is not None:
                si_values.append(si)
    if not si_values:
        return 0.5
    avg_si = float(np.mean(si_values))
    # Lower SI is better: <5% excellent, >20% poor
    if avg_si < 5:
        return 1.0
    if avg_si < 10:
        return 0.8
    if avg_si < 15:
        return 0.5
    if avg_si < 20:
        return 0.3
    return 0.1


def classify_skill_level(feature_scores: Dict[str, float]) -> tuple[str, float]:
    """Classify swimmer skill level from feature scores.

    Args:
        feature_scores: Dictionary of feature name to 0-1 score.

    Returns:
        (skill_level, confidence) tuple.
    """
    total = 0.0
    total_weight = 0.0
    for feature, weight in SKILL_FEATURE_WEIGHTS.items():
        if feature in feature_scores:
            total += feature_scores[feature] * weight
            total_weight += weight

    if total_weight < 0.3:
        return "intermediate", 0.3

    weighted_score = total / total_weight

    if weighted_score >= 0.85:
        return "elite", weighted_score
    if weighted_score >= 0.65:
        return "advanced", weighted_score
    if weighted_score >= 0.45:
        return "intermediate", weighted_score
    return "beginner", weighted_score


def predict_race_time(
    skill_level: str,
    deviations: Dict[str, Any],
    event: str = "50m_freestyle",
) -> float:
    """Predict race time based on skill level and technique deviations.

    Args:
        skill_level: Classified skill level.
        deviations: Deviation scoring payload.
        event: Event name (e.g., "50m_freestyle").

    Returns:
        Predicted time in seconds.
    """
    base = BASE_TIMES.get(event, BASE_TIMES["50m_freestyle"]).get(skill_level, 26.0)

    # Add time penalties for each deviation
    penalty = 0.0
    for phase_name in ("block_phase", "flight_phase", "entry_phase", "stroke_cycle"):
        for row in deviations.get(phase_name, []):
            if not isinstance(row, dict):
                continue
            metric = str(row.get("metric", ""))
            flag = str(row.get("flag", "OPTIMAL"))
            if flag == "OPTIMAL":
                continue
            impact = METRIC_TIME_IMPACT.get(metric, {}).get(flag, 0.0)
            penalty += impact

    return round(base + penalty, 2)


def compute_improvement_potential(
    deviations: Dict[str, Any],
    event: str = "50m_freestyle",
) -> Dict[str, Any]:
    """Compute improvement potential by identifying fixable deviations.

    Args:
        deviations: Deviation scoring payload.
        event: Target event.

    Returns:
        Dictionary with total potential time savings and ranked improvement areas.
    """
    improvement_areas: List[Dict[str, Any]] = []
    total_potential = 0.0

    for phase_name in ("block_phase", "flight_phase", "entry_phase", "stroke_cycle"):
        for row in deviations.get(phase_name, []):
            if not isinstance(row, dict):
                continue
            flag = str(row.get("flag", "OPTIMAL"))
            if flag == "OPTIMAL":
                continue
            metric = str(row.get("metric", ""))
            impact = METRIC_TIME_IMPACT.get(metric, {}).get(flag, 0.0)
            total_potential += impact
            improvement_areas.append({
                "metric": metric,
                "phase": phase_name,
                "current_flag": flag,
                "potential_time_savings_sec": impact,
                "measured": row.get("measured"),
                "optimal_min": row.get("optimal_min"),
                "optimal_max": row.get("optimal_max"),
            })

    # Sort by biggest impact
    improvement_areas.sort(key=lambda x: x["potential_time_savings_sec"], reverse=True)

    return {
        "total_potential_improvement_sec": round(total_potential, 2),
        "num_fixable_deviations": len(improvement_areas),
        "top_improvements": improvement_areas[:5],
        "all_improvements": improvement_areas,
    }


def generate_recommendations(
    skill_level: str,
    improvement_data: Dict[str, Any],
) -> List[str]:
    """Generate personalized coaching recommendations.

    Args:
        skill_level: Classified skill level.
        improvement_data: Improvement potential data.

    Returns:
        List of recommendation strings.
    """
    recs: List[str] = []

    top = improvement_data.get("top_improvements", [])
    potential = improvement_data.get("total_potential_improvement_sec", 0)

    if potential < 0.1:
        recs.append("Technique is near-optimal — focus on explosive power training.")
        return recs

    if skill_level == "beginner":
        recs.append("Prioritize fundamental body position before power development.")
        for imp in top[:2]:
            recs.append(f"Drill: improve {imp['metric']} — currently {imp['current_flag']}")
    elif skill_level == "intermediate":
        recs.append("Focus on consistency and reducing minor technique deviations.")
        for imp in top[:3]:
            phase_label = imp["phase"].replace("_phase", "").replace("_cycle", "")
            recs.append(f"{phase_label.title()}: correct {imp['metric']} (saves ~{imp['potential_time_savings_sec']:.2f}s)")
    elif skill_level == "advanced":
        recs.append("Fine-tune technique details for marginal gains.")
        for imp in top[:3]:
            recs.append(f"Refine {imp['metric']} ({imp['phase']}) — potential {imp['potential_time_savings_sec']:.2f}s gain")
    else:  # elite
        recs.append("Elite-level technique — focus on race strategy and reaction optimization.")
        if top:
            recs.append(f"Marginal gain: address {top[0]['metric']} ({top[0]['potential_time_savings_sec']:.2f}s)")

    recs.append(f"Total estimated improvement potential: {potential:.2f}s")

    return recs


def predict_performance(
    deviations: Dict[str, Any],
    symmetry_data: Optional[Dict[str, Any]] = None,
    consistency_data: Optional[Dict[str, Any]] = None,
    reaction_time_ms: Optional[float] = None,
    event: str = "50m_freestyle",
) -> Dict[str, Any]:
    """Full performance prediction from biomechanical analysis.

    Args:
        deviations: Deviation scoring payload.
        symmetry_data: Optional symmetry analysis results.
        consistency_data: Optional consistency analysis results.
        reaction_time_ms: Detected reaction time in ms.
        event: Target event name.

    Returns:
        Comprehensive performance prediction dictionary.
    """
    # 1. Compute feature scores
    feature_scores = {
        "reaction_time_score": _score_reaction_time(reaction_time_ms),
        "body_position_score": _score_body_position(deviations),
        "entry_quality_score": _score_entry_quality(deviations),
        "consistency_score": _score_consistency(consistency_data),
        "deviation_severity_score": _score_deviation_severity(deviations),
        "symmetry_score": _score_symmetry(symmetry_data),
    }

    # 2. Classify skill level
    skill_level, confidence = classify_skill_level(feature_scores)

    # 3. Predict race times
    time_50m = predict_race_time(skill_level, deviations, event)
    time_100m = predict_race_time(
        skill_level, deviations,
        event.replace("50m", "100m") if "50m" in event else "100m_freestyle",
    )

    # 4. Compute improvement potential
    improvement = compute_improvement_potential(deviations, event)

    # 5. Generate recommendations
    recommendations = generate_recommendations(skill_level, improvement)

    return {
        "skill_level": skill_level,
        "skill_confidence": round(confidence, 2),
        "feature_scores": {k: round(v, 2) for k, v in feature_scores.items()},
        "estimated_50m_time_sec": time_50m,
        "estimated_100m_time_sec": time_100m,
        "event": event,
        "improvement_potential": improvement,
        "recommendations": recommendations,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for performance prediction."""
    parser = argparse.ArgumentParser(
        description="Predict swim performance from biomechanical analysis."
    )
    parser.add_argument("--deviations", required=True, help="Deviations JSON path.")
    parser.add_argument("--symmetry", help="Symmetry analysis JSON path.")
    parser.add_argument("--consistency", help="Consistency analysis JSON path.")
    parser.add_argument("--reaction-time", type=float, help="Reaction time in ms.")
    parser.add_argument("--event", default="50m_freestyle", help="Target event name.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run performance prediction CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        with open(args.deviations, "r", encoding="utf-8") as f:
            deviations = json.load(f)
    except Exception as exc:
        LOGGER.error("Failed to load deviations: %s", exc)
        return 1

    symmetry_data = None
    if args.symmetry:
        try:
            with open(args.symmetry, "r", encoding="utf-8") as f:
                symmetry_data = json.load(f)
        except Exception:
            pass

    consistency_data = None
    if args.consistency:
        try:
            with open(args.consistency, "r", encoding="utf-8") as f:
                consistency_data = json.load(f)
        except Exception:
            pass

    result = predict_performance(
        deviations,
        symmetry_data=symmetry_data,
        consistency_data=consistency_data,
        reaction_time_ms=args.reaction_time,
        event=args.event,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved performance prediction to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
