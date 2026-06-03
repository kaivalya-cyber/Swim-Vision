# This file computes stroke efficiency indices from SwimVision stroke metrics.
"""Stroke efficiency analysis — distance per stroke, energy cost estimation, efficiency ratios."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def compute_distance_per_stroke(
    stroke_rate_spm: float,
    estimated_speed_ms: Optional[float] = None,
    cycle_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate distance covered per stroke cycle.

    DPS = speed / stroke_rate (converted to per-cycle basis).

    Args:
        stroke_rate_spm: Stroke rate in strokes per minute.
        estimated_speed_ms: Estimated swimming speed in m/s.
        cycle_duration_sec: Duration of one full cycle.

    Returns:
        Distance per stroke metrics.
    """
    if estimated_speed_ms is None:
        estimated_speed_ms = 1.8  # typical swim speed m/s

    if stroke_rate_spm < 1:
        return {"error": "Invalid stroke rate"}

    # Strokes per second
    strokes_per_sec = stroke_rate_spm / 60.0
    dps = estimated_speed_ms / strokes_per_sec if strokes_per_sec > 0 else 0

    # Efficiency: higher DPS = more efficient
    if dps > 2.5:
        dps_label = "EXCELLENT"
    elif dps > 2.0:
        dps_label = "GOOD"
    elif dps > 1.5:
        dps_label = "MODERATE"
    else:
        dps_label = "POOR"

    return {
        "distance_per_stroke_m": round(dps, 2),
        "stroke_rate_spm": round(stroke_rate_spm, 1),
        "estimated_speed_ms": round(estimated_speed_ms, 2),
        "dps_label": dps_label,
    }


def compute_stroke_index(
    speed_ms: float,
    stroke_length_m: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the Stroke Index (SI = speed × stroke_length).

    SI is a widely-used swimming efficiency metric. Higher = more efficient.

    Args:
        speed_ms: Swimming speed in m/s.
        stroke_length_m: Distance per stroke cycle in meters.

    Returns:
        Stroke index metrics.
    """
    if stroke_length_m is None:
        stroke_length_m = 2.0  # default for freestyle

    si = speed_ms * stroke_length_m

    if si > 4.5:
        label = "ELITE"
    elif si > 3.5:
        label = "ADVANCED"
    elif si > 2.5:
        label = "INTERMEDIATE"
    else:
        label = "DEVELOPING"

    return {
        "stroke_index": round(si, 2),
        "speed_ms": round(speed_ms, 1),
        "stroke_length_m": round(stroke_length_m, 2),
        "stroke_index_label": label,
    }


def estimate_energy_cost(
    stroke_rate_spm: float,
    body_roll_deg: Optional[float] = None,
    symmetry_index_pct: Optional[float] = None,
    num_cycles: int = 1,
) -> Dict[str, Any]:
    """Estimate relative energy cost of the stroke technique.

    Higher stroke rate + asymmetry + low body roll → higher energy cost.

    Args:
        stroke_rate_spm: Stroke rate.
        body_roll_deg: Body roll angle.
        symmetry_index_pct: Symmetry index percentage.
        num_cycles: Number of cycles analyzed.

    Returns:
        Energy cost estimation metrics.
    """
    # Base cost from stroke rate: higher rate = more energy
    base_cost = stroke_rate_spm * 0.15

    # Body roll penalty: less roll (<30°) = higher drag = higher cost
    roll_penalty = 0.0
    if body_roll_deg is not None:
        if body_roll_deg < 20:
            roll_penalty = 3.0
        elif body_roll_deg < 30:
            roll_penalty = 1.5
        elif body_roll_deg < 40:
            roll_penalty = 0.5

    # Asymmetry penalty: higher asymmetry = wasted energy
    asym_penalty = 0.0
    if symmetry_index_pct is not None:
        asym_penalty = (symmetry_index_pct / 100.0) * 5.0

    total_cost = base_cost + roll_penalty + asym_penalty

    if total_cost < 8:
        efficiency_label = "HIGHLY-EFFICIENT"
    elif total_cost < 12:
        efficiency_label = "EFFICIENT"
    elif total_cost < 16:
        efficiency_label = "MODERATE"
    else:
        efficiency_label = "INEFFICIENT"

    return {
        "estimated_energy_cost": round(total_cost, 1),
        "efficiency_label": efficiency_label,
        "components": {
            "stroke_rate_cost": round(base_cost, 1),
            "body_roll_penalty": round(roll_penalty, 1),
            "asymmetry_penalty": round(asym_penalty, 1),
        },
        "num_cycles_analyzed": num_cycles,
    }


def compute_propulsive_efficiency(
    left_elbow_flexion: Optional[float] = None,
    right_elbow_flexion: Optional[float] = None,
    left_hand_speed: Optional[float] = None,
    right_hand_speed: Optional[float] = None,
) -> Dict[str, Any]:
    """Estimate propulsive efficiency from catch and pull mechanics.

    Optimal elbow angle (~90-110°) with high hand speed = efficient propulsion.

    Args:
        left_elbow_flexion: Left elbow flexion during catch (degrees).
        right_elbow_flexion: Right elbow flexion during catch.
        left_hand_speed: Left hand speed during pull.
        right_hand_speed: Right hand speed during pull.

    Returns:
        Propulsive efficiency metrics.
    """
    def _elbow_score(angle: Optional[float]) -> float:
        if angle is None:
            return 0.5
        # Optimal range: 90-110°
        if 90 <= angle <= 110:
            return 1.0
        if 80 <= angle <= 120:
            return 0.7
        if 70 <= angle <= 130:
            return 0.4
        return 0.2

    left_score = _elbow_score(left_elbow_flexion)
    right_score = _elbow_score(right_elbow_flexion)
    elbow_avg = (left_score + right_score) / 2.0

    # Hand speed contribution: higher = better (normalized 0-1)
    max_speed = max(
        left_hand_speed or 0,
        right_hand_speed or 0,
    )
    speed_score = min(1.0, max_speed / 10.0) if max_speed > 0 else 0.5

    # Combined: 60% elbow position, 40% hand speed
    prop_eff = elbow_avg * 0.6 + speed_score * 0.4

    if prop_eff >= 0.85:
        label = "HIGHLY-EFFICIENT"
    elif prop_eff >= 0.65:
        label = "EFFICIENT"
    elif prop_eff >= 0.45:
        label = "MODERATE"
    else:
        label = "INEFFICIENT"

    return {
        "propulsive_efficiency_score": round(prop_eff, 2),
        "propulsive_efficiency_label": label,
        "components": {
            "elbow_position_score": round(elbow_avg, 2),
            "hand_speed_score": round(speed_score, 2),
        },
    }


def analyze_stroke_efficiency(
    stroke_metrics: Dict[str, Any],
    estimated_speed_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Full stroke efficiency analysis from aggregated stroke metrics.

    Args:
        stroke_metrics: Aggregated stroke metrics dictionary.
        estimated_speed_ms: Optional estimated swim speed in m/s.

    Returns:
        Comprehensive stroke efficiency analysis.
    """
    result: Dict[str, Any] = {}

    sr = stroke_metrics.get("stroke_rate", 0)
    body_roll = stroke_metrics.get("body_roll")
    sym_index = stroke_metrics.get("symmetry_index")
    num_cycles = int(stroke_metrics.get("num_cycles", 1))
    le = stroke_metrics.get("left_elbow_flexion")
    re = stroke_metrics.get("right_elbow_flexion")
    lh = stroke_metrics.get("left_hand_speed")
    rh = stroke_metrics.get("right_hand_speed")
    cycle_dur = stroke_metrics.get("cycle_duration_seconds")

    # 1. Distance per stroke
    result["distance_per_stroke"] = compute_distance_per_stroke(
        sr, estimated_speed_ms, cycle_dur
    )

    # 2. Stroke index
    speed = estimated_speed_ms or 1.8
    dps = result["distance_per_stroke"].get("distance_per_stroke_m", 2.0)
    result["stroke_index"] = compute_stroke_index(speed, dps)

    # 3. Energy cost
    result["energy_cost"] = estimate_energy_cost(sr, body_roll, sym_index, num_cycles)

    # 4. Propulsive efficiency
    result["propulsive_efficiency"] = compute_propulsive_efficiency(le, re, lh, rh)

    # Overall efficiency score (0-100)
    scores = []
    si = result["stroke_index"].get("stroke_index", 2.5)
    scores.append(min(100.0, si * 20.0))  # SI 5.0 → 100

    pe = result["propulsive_efficiency"].get("propulsive_efficiency_score", 0.5)
    scores.append(pe * 100.0)

    ec = result["energy_cost"].get("estimated_energy_cost", 12)
    scores.append(max(0.0, 100.0 - ec * 6.0))  # cost 0 → 100, cost 16 → 4

    if scores:
        overall = round(float(np.mean(scores)), 1)
        result["overall_efficiency_score"] = overall
        if overall >= 80:
            result["overall_efficiency_label"] = "ELITE-EFFICIENCY"
        elif overall >= 60:
            result["overall_efficiency_label"] = "GOOD-EFFICIENCY"
        elif overall >= 40:
            result["overall_efficiency_label"] = "MODERATE-EFFICIENCY"
        else:
            result["overall_efficiency_label"] = "NEEDS-IMPROVEMENT"

    result["generated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for stroke efficiency analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze stroke efficiency from SwimVision stroke metrics."
    )
    parser.add_argument("--stroke-metrics", required=True, help="Stroke metrics JSON path.")
    parser.add_argument("--speed-ms", type=float, help="Estimated swim speed in m/s.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run stroke efficiency CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        with open(args.stroke_metrics, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as exc:
        LOGGER.error("Failed to load stroke metrics: %s", exc)
        return 1

    aggregate = metrics.get("aggregate", metrics)
    result = analyze_stroke_efficiency(aggregate, estimated_speed_ms=args.speed_ms)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved stroke efficiency analysis to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
