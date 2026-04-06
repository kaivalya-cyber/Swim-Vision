# This file scores measured swim-start angles against literature ranges and learned pro distributions.
"""Deviation scoring utilities for SwimVision phase metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.reference.optimal_ranges import DEVIATION_THRESHOLDS, OPTIMAL_RANGES, get_range


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

PHASE_TO_METRICS = {
    "block_phase": ["front_knee_angle", "rear_knee_angle", "hip_angle", "torso_lean"],
    "flight_phase": ["body_linearity", "entry_angle", "elbow_extension"],
    "entry_phase": ["streamline_angle", "elbow_lock_angle"],
}
FLAG_ORDER = ["OPTIMAL", "MINOR", "SIGNIFICANT", "CRITICAL"]


def _load_pro_distribution(pro_distribution_path: str | None = None) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Load the learned pro distribution if it exists.

    Args:
        pro_distribution_path: Optional JSON path override.

    Returns:
        Parsed pro-distribution dictionary or an empty dictionary.
    """

    candidate_path = Path(pro_distribution_path) if pro_distribution_path else PROJECT_ROOT / "results" / "pro_distribution.json"
    if not candidate_path.exists():
        LOGGER.warning("Pro distribution file not found at %s; skipping learned comparison.", candidate_path)
        return {}
    try:
        with open(candidate_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to load pro distribution from '{candidate_path}': {exc}") from exc

    parsed: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for phase, metrics in payload.items():
        parsed[phase] = {}
        for metric, stats in metrics.items():
            parsed[phase][metric] = (float(stats[0]), float(stats[1]))
    return parsed


def score_deviation(measured_angle: float, optimal_range: Tuple[float, float]) -> tuple[float, str]:
    """Score the angular deviation from an optimal range.

    Args:
        measured_angle: Measured metric value in degrees.
        optimal_range: Inclusive ``(min_value, max_value)`` optimal range.

    Returns:
        A tuple of deviation degrees and categorical severity flag.
    """

    minimum, maximum = optimal_range
    if minimum <= measured_angle <= maximum:
        deviation = 0.0
    elif measured_angle < minimum:
        deviation = float(minimum - measured_angle)
    else:
        deviation = float(measured_angle - maximum)

    if deviation <= DEVIATION_THRESHOLDS["OPTIMAL"]:
        flag = "OPTIMAL"
    elif deviation <= DEVIATION_THRESHOLDS["MINOR"]:
        flag = "MINOR"
    elif deviation <= DEVIATION_THRESHOLDS["SIGNIFICANT"]:
        flag = "SIGNIFICANT"
    else:
        flag = "CRITICAL"
    return deviation, flag


def compute_deviations(
    angles_df: pd.DataFrame, phase: str, phase_boundaries: Dict[str, int]
) -> pd.DataFrame:
    """Compute deviation scores for the metrics relevant to a phase.

    Args:
        angles_df: Per-frame joint-angle DataFrame.
        phase: Phase identifier such as ``block_phase``.
        phase_boundaries: Dictionary of detected phase boundaries.

    Returns:
        A DataFrame with metric values, optimal ranges, and deviation scores.
    """

    if phase not in PHASE_TO_METRICS:
        raise KeyError(f"Unsupported phase '{phase}'. Expected one of {sorted(PHASE_TO_METRICS)}.")

    start_key = phase.replace("_phase", "_start")
    end_key = phase.replace("_phase", "_end")
    if start_key not in phase_boundaries or end_key not in phase_boundaries:
        raise KeyError(f"Missing boundary keys '{start_key}'/'{end_key}' in phase boundaries.")

    start_idx = int(phase_boundaries[start_key])
    end_idx = int(phase_boundaries[end_key])
    if end_idx < start_idx:
        raise ValueError(f"Invalid phase window for {phase}: start={start_idx}, end={end_idx}.")

    try:
        phase_window = angles_df.loc[start_idx:end_idx]
    except Exception as exc:
        raise RuntimeError(f"Failed to select frame window {start_idx}:{end_idx} for {phase}: {exc}") from exc

    pro_distribution = _load_pro_distribution()
    rows: List[Dict[str, Any]] = []

    for metric in PHASE_TO_METRICS[phase]:
        if metric not in phase_window.columns:
            raise KeyError(f"Metric '{metric}' not found in angles DataFrame.")
        measured_value = float(phase_window[metric].mean())
        optimal_min, optimal_max = get_range(phase, metric)
        deviation, flag = score_deviation(measured_value, (optimal_min, optimal_max))

        learned_mean = None
        learned_std = None
        learned_z_score = None
        if phase in pro_distribution and metric in pro_distribution[phase]:
            learned_mean, learned_std = pro_distribution[phase][metric]
            if learned_std and learned_std > 0.0:
                learned_z_score = float((measured_value - learned_mean) / learned_std)
            else:
                learned_z_score = 0.0

        rows.append(
            {
                "metric": metric,
                "measured": measured_value,
                "optimal_min": float(optimal_min),
                "optimal_max": float(optimal_max),
                "deviation": deviation,
                "flag": flag,
                "pro_mean": learned_mean,
                "pro_std": learned_std,
                "pro_z_score": learned_z_score,
            }
        )

    return pd.DataFrame(rows)


def aggregate_report(
    block_dev: pd.DataFrame, flight_dev: pd.DataFrame, entry_dev: pd.DataFrame
) -> Dict[str, Any]:
    """Combine per-phase deviation DataFrames into a structured report.

    Args:
        block_dev: Deviation DataFrame for the block phase.
        flight_dev: Deviation DataFrame for the flight phase.
        entry_dev: Deviation DataFrame for the entry phase.

    Returns:
        Structured deviation report with overall severity.
    """

    all_flags = list(block_dev["flag"]) + list(flight_dev["flag"]) + list(entry_dev["flag"])
    worst_flag = max(all_flags, key=lambda item: FLAG_ORDER.index(item)) if all_flags else "OPTIMAL"
    return {
        "block_phase": block_dev.to_dict(orient="records"),
        "flight_phase": flight_dev.to_dict(orient="records"),
        "entry_phase": entry_dev.to_dict(orient="records"),
        "overall_severity": worst_flag,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for deviation scoring.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Compute SwimVision phase deviations.")
    parser.add_argument("--angles", required=True, help="Path to angle CSV output.")
    parser.add_argument("--phase", required=True, choices=sorted(PHASE_TO_METRICS), help="Phase to score.")
    parser.add_argument("--boundaries", required=True, help="Path to phase boundary JSON.")
    parser.add_argument("--output", help="Optional CSV output path.")
    return parser


def main() -> int:
    """Run the command-line interface for deviation scoring.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        angles_df = pd.read_csv(args.angles, index_col=0)
    except Exception as exc:
        LOGGER.error("Failed to read angle CSV %s: %s", args.angles, exc)
        return 1

    try:
        with open(args.boundaries, "r", encoding="utf-8") as handle:
            phase_boundaries = json.load(handle)
    except Exception as exc:
        LOGGER.error("Failed to read boundary JSON %s: %s", args.boundaries, exc)
        return 1

    try:
        deviations_df = compute_deviations(angles_df, args.phase, phase_boundaries)
    except Exception as exc:
        LOGGER.error("Failed to compute deviations: %s", exc)
        return 1

    if args.output:
        try:
            deviations_df.to_csv(args.output, index=False)
        except Exception as exc:
            LOGGER.error("Failed to write deviation CSV %s: %s", args.output, exc)
            return 1
        LOGGER.info("Saved deviations to %s", args.output)
    else:
        print(deviations_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
