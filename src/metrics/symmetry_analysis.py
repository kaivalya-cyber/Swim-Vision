# This file computes bilateral symmetry indices from paired left/right joint metrics.
"""Symmetry analysis for SwimVision — bilateral comparison and asymmetry scoring."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Paired metrics: (left_col, right_col, label)
PAIRED_JOINTS = [
    ("left_elbow_angle", "right_elbow_angle", "elbow_flexion"),
    ("front_knee_angle", "rear_knee_angle", "knee_angle"),
]

# Stroke-mode paired metrics (derived from aggregate key names)
STROKE_PAIRS = [
    ("left_elbow_flexion", "right_elbow_flexion", "elbow_flexion"),
    ("left_hand_speed", "right_hand_speed", "hand_speed"),
    ("left_shoulder_rotation", "right_shoulder_rotation", "shoulder_rotation"),
]

ASYMMETRY_THRESHOLDS = {
    "symmetry_index": {
        "excellent": 5.0,   # <5% asymmetry
        "good": 10.0,       # 5-10%
        "moderate": 15.0,   # 10-15%
        "poor": float("inf"),  # >15%
    },
}


def symmetry_index(left_val: float | None, right_val: float | None) -> Optional[float]:
    """Compute the symmetry index (%) between two paired values.

    Uses the formula: SI = |(L - R)| / max(|L|, |R|) * 100

    Args:
        left_val: Left-side metric value.
        right_val: Right-side metric value.

    Returns:
        Symmetry index percentage (0 = perfect symmetry), or None if invalid.
    """
    if left_val is None or right_val is None:
        return None
    if np.isnan(left_val) or np.isnan(right_val):
        return None
    denominator = max(abs(left_val), abs(right_val))
    if denominator < 1e-9:
        return 0.0
    return float(abs(left_val - right_val) / denominator * 100.0)


def classify_asymmetry(si: float) -> str:
    """Classify asymmetry severity from symmetry index.

    Args:
        si: Symmetry index percentage.

    Returns:
        String classification: excellent, good, moderate, or poor.
    """
    if si <= ASYMMETRY_THRESHOLDS["symmetry_index"]["excellent"]:
        return "excellent"
    if si <= ASYMMETRY_THRESHOLDS["symmetry_index"]["good"]:
        return "good"
    if si <= ASYMMETRY_THRESHOLDS["symmetry_index"]["moderate"]:
        return "moderate"
    return "poor"


def analyze_dive_symmetry(
    angles_df: pd.DataFrame,
    phase_boundaries: Dict[str, int],
) -> Dict[str, Any]:
    """Compute bilateral symmetry indices across dive phases.

    Args:
        angles_df: Per-frame angle DataFrame.
        phase_boundaries: Phase boundary dictionary.

    Returns:
        Dictionary with per-phase symmetry analysis results.
    """
    result: Dict[str, Any] = {"overall": {}, "phases": {}}

    for phase_name, start_key, end_key in [
        ("block_phase", "block_start", "block_end"),
        ("flight_phase", "flight_start", "flight_end"),
        ("entry_phase", "entry_start", "entry_end"),
    ]:
        if start_key not in phase_boundaries or end_key not in phase_boundaries:
            continue

        start_idx = int(phase_boundaries[start_key])
        end_idx = int(phase_boundaries[end_key])
        if end_idx <= start_idx:
            continue

        try:
            phase_window = angles_df.loc[start_idx:end_idx]
        except Exception:
            continue

        phase_pairs: Dict[str, Dict[str, Any]] = {}

        for left_col, right_col, label in PAIRED_JOINTS:
            if left_col not in phase_window.columns or right_col not in phase_window.columns:
                continue

            left_vals = phase_window[left_col].dropna()
            right_vals = phase_window[right_col].dropna()
            if len(left_vals) == 0 or len(right_vals) == 0:
                continue

            left_mean = float(left_vals.mean())
            right_mean = float(right_vals.mean())
            si = symmetry_index(left_mean, right_mean)

            if si is not None:
                phase_pairs[label] = {
                    "left_mean": round(left_mean, 2),
                    "right_mean": round(right_mean, 2),
                    "symmetry_index_pct": round(si, 2),
                    "classification": classify_asymmetry(si),
                    "delta": round(abs(left_mean - right_mean), 2),
                    "dominant_side": "left" if left_mean > right_mean else "right" if right_mean > left_mean else "equal",
                }

        if phase_pairs:
            result["phases"][phase_name] = phase_pairs

    all_sis = []
    for phase_data in result["phases"].values():
        for pair_data in phase_data.values():
            si = pair_data.get("symmetry_index_pct")
            if si is not None:
                all_sis.append(si)

    if all_sis:
        overall_si = float(np.mean(all_sis))
        result["overall"] = {
            "mean_symmetry_index_pct": round(overall_si, 2),
            "classification": classify_asymmetry(overall_si),
            "num_pairs_analyzed": len(all_sis),
        }

    return result


def analyze_stroke_symmetry(
    aggregate_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Compute symmetry indices from stroke aggregate metrics.

    Args:
        aggregate_metrics: Dictionary of aggregate stroke metrics.

    Returns:
        Symmetry analysis result for stroke mode.
    """
    pairs_result: Dict[str, Dict[str, Any]] = {}

    for left_key, right_key, label in STROKE_PAIRS:
        left_val = aggregate_metrics.get(left_key)
        right_val = aggregate_metrics.get(right_key)
        if left_val is None or right_val is None:
            continue

        si = symmetry_index(left_val, right_val)
        if si is not None and not np.isnan(si):
            pairs_result[label] = {
                "left_value": round(float(left_val), 2),
                "right_value": round(float(right_val), 2),
                "symmetry_index_pct": round(si, 2),
                "classification": classify_asymmetry(si),
                "delta": round(abs(float(left_val) - float(right_val)), 2),
                "dominant_side": "left" if float(left_val) > float(right_val) else "right" if float(right_val) > float(left_val) else "equal",
            }

    all_sis = [d["symmetry_index_pct"] for d in pairs_result.values()]
    overall_si = float(np.mean(all_sis)) if all_sis else None

    return {
        "overall": {
            "mean_symmetry_index_pct": round(overall_si, 2) if overall_si is not None else None,
            "classification": classify_asymmetry(overall_si) if overall_si is not None else None,
            "num_pairs_analyzed": len(all_sis),
        },
        "pairs": pairs_result,
    }


def compute_frame_by_frame_symmetry(
    angles_df: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> pd.Series:
    """Compute per-frame symmetry index for a paired joint.

    Args:
        angles_df: Per-frame angle DataFrame.
        left_col: Left-side column name.
        right_col: Right-side column name.

    Returns:
        Series of symmetry index percentages per frame.
    """
    if left_col not in angles_df.columns or right_col not in angles_df.columns:
        raise KeyError(f"Columns '{left_col}' and '{right_col}' must exist.")

    si_values = []
    for _, row in angles_df.iterrows():
        lv = row[left_col]
        rv = row[right_col]
        if pd.isna(lv) or pd.isna(rv):
            si_values.append(np.nan)
        else:
            si_values.append(symmetry_index(float(lv), float(rv)))

    return pd.Series(si_values, index=angles_df.index, name=f"{left_col}_vs_{right_col}_si")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for symmetry analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze bilateral symmetry from SwimVision angle data."
    )
    parser.add_argument("--angles", required=True, help="Path to angle CSV file.")
    parser.add_argument("--boundaries", help="Phase boundary JSON for dive mode.")
    parser.add_argument(
        "--mode",
        choices=["dive", "stroke"],
        default="dive",
        help="Analysis mode.",
    )
    parser.add_argument("--stroke-metrics", help="Stroke metrics JSON for stroke mode.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument(
        "--output-csv",
        help="Optional CSV path for frame-by-frame symmetry indices.",
    )
    return parser


def main() -> int:
    """Run the CLI for symmetry analysis."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.mode == "stroke":
            if not args.stroke_metrics:
                LOGGER.error("--stroke-metrics is required for stroke mode.")
                return 1
            with open(args.stroke_metrics, "r", encoding="utf-8") as f:
                stroke_data = json.load(f)
            aggregate = stroke_data.get("aggregate", {})
            result = analyze_stroke_symmetry(aggregate)
        else:
            angles_df = pd.read_csv(args.angles, index_col=0)
            boundaries = {}
            if args.boundaries:
                with open(args.boundaries, "r", encoding="utf-8") as f:
                    boundaries = json.load(f)
            result = analyze_dive_symmetry(angles_df, boundaries)
    except Exception as exc:
        LOGGER.error("Symmetry analysis failed: %s", exc)
        return 1

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            LOGGER.info("Saved symmetry analysis to %s", args.output)
        except Exception as exc:
            LOGGER.error("Failed to write: %s", exc)
            return 1
    else:
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
