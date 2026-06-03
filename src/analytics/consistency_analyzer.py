# This file analyzes temporal consistency across repeated trials for technique stability assessment.
"""Temporal consistency analysis for SwimVision — CV, outliers, fatigue, and trend stability."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Metrics that should decrease with improvement (lower is better)
LOWER_IS_BETTER = {
    "symmetry_index",
    "cycle_duration_seconds",
    "reaction_time_ms",
    "head_deviation",
    "entry_splash_angle",
    "asymmetry_pct",
}


def _is_improvement(metric_name: str, delta: float) -> bool:
    """Determine if a delta represents improvement based on the metric's direction."""
    if any(lower in metric_name.lower() for lower in LOWER_IS_BETTER):
        return delta < 0
    return delta > 0


def compute_cv(values: List[float]) -> Optional[float]:
    """Compute coefficient of variation (CV%) for a list of values.

    Args:
        values: List of float metric values.

    Returns:
        CV% or None if insufficient data.
    """
    if len(values) < 2:
        return None
    arr = np.array(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return None
    mean_val = float(np.mean(arr))
    if abs(mean_val) < 1e-9:
        return None
    std_val = float(np.std(arr, ddof=1))
    return round(std_val / abs(mean_val) * 100.0, 2)


def detect_outliers(
    values: List[float],
    z_threshold: float = 2.0,
) -> List[int]:
    """Detect outlier indices using modified Z-score (MAD-based).

    Args:
        values: List of metric values.
        z_threshold: Z-score threshold for outlier detection.

    Returns:
        List of outlier indices.
    """
    if len(values) < 4:
        return []
    arr = np.array(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return []

    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad < 1e-9:
        return []

    outliers = []
    for i, v in enumerate(values):
        if v is None or not np.isfinite(v):
            outliers.append(i)
            continue
        z = 0.6745 * (v - median) / mad
        if abs(z) > z_threshold:
            outliers.append(i)
    return outliers


def compute_trend_stability(values: List[float]) -> Dict[str, Any]:
    """Compute trend stability metrics for a time series.

    Args:
        values: Time-ordered metric values.

    Returns:
        Dictionary with slope, R², direction, and volatility.
    """
    result: Dict[str, Any] = {
        "slope": None,
        "r_squared": None,
        "direction": "stable",
        "volatility": None,
        "num_points": len(values),
    }

    clean = [v for v in values if v is not None and np.isfinite(v)]
    if len(clean) < 2:
        return result

    arr = np.array(clean, dtype=np.float32)
    x = np.arange(len(arr), dtype=np.float32)

    # Linear regression
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(arr))
    numerator = float(np.sum((x - x_mean) * (arr - y_mean)))
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator > 0:
        slope = numerator / denominator
        y_pred = slope * x + (y_mean - slope * x_mean)
        ss_res = float(np.sum((arr - y_pred) ** 2))
        ss_tot = float(np.sum((arr - y_mean) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result["slope"] = round(slope, 6)
        result["r_squared"] = round(max(0.0, min(1.0, r_squared)), 4)

    # Direction
    if result["slope"] is not None:
        if abs(result["slope"]) < 1e-6:
            result["direction"] = "stable"
        elif result["slope"] > 0:
            result["direction"] = "increasing"
        else:
            result["direction"] = "decreasing"

    # Volatility = std of pairwise differences
    if len(arr) >= 3:
        diffs = np.diff(arr)
        result["volatility"] = round(float(np.std(diffs, ddof=1)), 4)

    return result


def detect_fatigue_pattern(
    values: List[float],
    within_session: bool = True,
) -> Dict[str, Any]:
    """Detect fatigue patterns in metric degradation.

    For within-session analysis, checks progressive decline across repetitions.
    For cross-session, checks if recent sessions show degraded metrics vs baseline.

    Args:
        values: Ordered metric values (within-session or across sessions).
        within_session: True if values are within one session, False if cross-session.

    Returns:
        Dictionary with fatigue score, flagged status, and degradation slope.
    """
    result: Dict[str, Any] = {
        "fatigue_detected": False,
        "fatigue_score": 0.0,
        "degradation_slope": None,
        "first_to_last_change_pct": None,
    }

    clean = [v for v in values if v is not None and np.isfinite(v)]
    if len(clean) < 3:
        return result

    arr = np.array(clean, dtype=np.float32)
    n = len(arr)

    # Compute slope over the series
    x = np.arange(n, dtype=np.float32)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(arr))
    numerator = float(np.sum((x - x_mean) * (arr - y_mean)))
    denominator = float(np.sum((x - x_mean) ** 2))
    if denominator > 0:
        slope = numerator / denominator
        result["degradation_slope"] = round(slope, 6)

    # Compare first third to last third
    third = max(1, n // 3)
    first_segment = arr[:third]
    last_segment = arr[-third:]
    first_mean = float(np.mean(first_segment))
    last_mean = float(np.mean(last_segment))

    if abs(first_mean) > 1e-9:
        change_pct = round((last_mean - first_mean) / abs(first_mean) * 100.0, 2)
        result["first_to_last_change_pct"] = change_pct

        # Fatigue if degradation exceeds threshold
        threshold = 5.0 if within_session else 8.0  # % change threshold
        if abs(change_pct) > threshold:
            fatigue_score = min(100.0, abs(change_pct) * 2.0)
            result["fatigue_score"] = round(fatigue_score, 1)
            if fatigue_score > 15.0:
                result["fatigue_detected"] = True

    return result


def analyze_consistency(
    report_paths: List[str],
    metric_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze temporal consistency across multiple session reports.

    Args:
        report_paths: Paths to *_report.json files in chronological order.
        metric_filter: Optional list of metric names to analyze.

    Returns:
        Dictionary with per-metric CV, outliers, trend stability, fatigue.
    """
    # Load all sessions
    sessions: List[Dict[str, Any]] = []
    for path_str in report_paths:
        try:
            with open(Path(path_str), "r", encoding="utf-8") as f:
                sessions.append(json.load(f))
        except Exception as exc:
            LOGGER.warning("Failed to load %s: %s", path_str, exc)

    if len(sessions) < 2:
        return {"error": "Need at least 2 sessions for consistency analysis.", "sessions_loaded": len(sessions)}

    # Extract metrics from all sessions
    metric_values: Dict[str, List[Optional[float]]] = defaultdict(list)
    session_ids: List[str] = []

    for session in sessions:
        sid = session.get("clip_id", "unknown")
        session_ids.append(sid)

        # Extract from deviations (dive) or aggregate_metrics (stroke)
        if session.get("analysis_mode") == "stroke":
            agg = session.get("aggregate_metrics", {})
            for key, val in agg.items():
                if metric_filter and key not in metric_filter:
                    continue
                try:
                    metric_values[key].append(float(val))
                except (ValueError, TypeError):
                    metric_values[key].append(None)
        else:
            for phase_name in ("block_phase", "flight_phase", "entry_phase"):
                phase_rows = session.get("deviations", {}).get(phase_name, [])
                for row in (phase_rows or []):
                    if not isinstance(row, dict):
                        continue
                    metric_name = str(row.get("metric", ""))
                    full_name = f"{phase_name}_{metric_name}"
                    if metric_filter and metric_name not in metric_filter and full_name not in metric_filter:
                        continue
                    try:
                        metric_values[full_name].append(float(row.get("measured", 0)))
                    except (ValueError, TypeError):
                        metric_values[full_name].append(None)

    if not metric_values:
        return {"error": "No metrics extracted from sessions.", "sessions_loaded": len(sessions)}

    # Compute per-metric consistency stats
    consistency_results: Dict[str, Dict[str, Any]] = {}
    overall_cv_values: List[float] = []

    for metric_name, values in sorted(metric_values.items()):
        clean = [v for v in values if v is not None and np.isfinite(v)]
        if len(clean) < 2:
            continue

        cv = compute_cv(clean)
        outliers = detect_outliers(clean)
        trend = compute_trend_stability(clean)
        fatigue = detect_fatigue_pattern(clean, within_session=False)

        result = {
            "cv_pct": cv,
            "mean": round(float(np.mean(clean)), 3),
            "std": round(float(np.std(clean, ddof=1)), 3),
            "min": round(float(np.min(clean)), 3),
            "max": round(float(np.max(clean)), 3),
            "num_sessions": len(clean),
            "outlier_indices": outliers,
            "outlier_count": len(outliers),
            "trend": trend,
            "fatigue": fatigue,
        }

        consistency_results[metric_name] = result
        if cv is not None:
            overall_cv_values.append(cv)

    # Overall consistency score (100 - avg CV, clamped)
    avg_cv = float(np.mean(overall_cv_values)) if overall_cv_values else 50.0
    overall_score = max(0.0, min(100.0, 100.0 - avg_cv))

    # Severity classification
    if overall_score >= 85:
        level = "EXCELLENT"
    elif overall_score >= 70:
        level = "GOOD"
    elif overall_score >= 50:
        level = "MODERATE"
    else:
        level = "POOR"

    # Per-metric severity classification
    for metric_name, result in consistency_results.items():
        cv_val = result.get("cv_pct")
        if cv_val is None:
            result["severity"] = "N/A"
        elif cv_val < 5:
            result["severity"] = "EXCELLENT"
        elif cv_val < 10:
            result["severity"] = "GOOD"
        elif cv_val < 20:
            result["severity"] = "MODERATE"
        else:
            result["severity"] = "POOR"

    # Build recommendations
    recommendations: List[str] = []
    poor_metrics = [
        name for name, r in consistency_results.items()
        if r.get("severity") in ("MODERATE", "POOR")
    ]
    if poor_metrics:
        recommendations.append(
            f"Focus on consistency for: {', '.join(poor_metrics[:5])}"
            f"{' +{} more'.format(len(poor_metrics) - 5) if len(poor_metrics) > 5 else ''}"
        )

    # Check for fatigue patterns
    fatigued = [
        name for name, r in consistency_results.items()
        if r.get("fatigue", {}).get("fatigue_detected")
    ]
    if fatigued:
        recommendations.append(
            f"Fatigue patterns detected in: {', '.join(fatigued[:3])}"
            f"{' +{} more'.format(len(fatigued) - 3) if len(fatigued) > 3 else ''}"
            " — consider conditioning work"
        )

    # Check for outliers
    outlier_metrics = [
        name for name, r in consistency_results.items()
        if r.get("outlier_count", 0) > 0
    ]
    if outlier_metrics:
        recommendations.append(
            f"Outlier sessions detected in: {', '.join(outlier_metrics[:3])}"
            f"{' +{} more'.format(len(outlier_metrics) - 3) if len(outlier_metrics) > 3 else ''}"
            " — review these sessions for anomalies"
        )

    return {
        "overall_consistency_score": round(overall_score, 1),
        "overall_consistency_level": level,
        "overall_avg_cv_pct": round(avg_cv, 2),
        "num_sessions_analyzed": len(sessions),
        "num_metrics_analyzed": len(consistency_results),
        "session_ids": session_ids,
        "metric_consistency": consistency_results,
        "recommendations": recommendations,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def analyze_within_session_consistency(
    angles_csv: str,
    phase: str = "entry_phase",
) -> Dict[str, Any]:
    """Analyze within-session consistency from angle data for a specific phase.

    Detects if technique degrades across repeated movements within a single session.

    Args:
        angles_csv: Path to joint angles CSV.
        phase: Phase to analyze for fatigue patterns.

    Returns:
        Dictionary with within-session consistency metrics.
    """
    import pandas as pd

    try:
        df = pd.read_csv(angles_csv)
    except Exception as exc:
        return {"error": f"Failed to load angles CSV: {exc}"}

    if df.empty:
        return {"error": "Empty angle data"}

    # Split session into thirds to compare early vs late performance
    n = len(df)
    if n < 10:
        return {"error": "Too few frames for within-session analysis (need ≥10)"}

    third = n // 3
    first_third = df.iloc[:third]
    last_third = df.iloc[-third:]

    metric_stability: Dict[str, Dict[str, Any]] = {}
    angle_columns = [c for c in df.columns if c not in ("frame", "timestamp", "confidence")]

    for col in angle_columns:
        try:
            early_vals = pd.to_numeric(first_third[col], errors="coerce").dropna().values
            late_vals = pd.to_numeric(last_third[col], errors="coerce").dropna().values
        except Exception:
            continue

        if len(early_vals) < 2 or len(late_vals) < 2:
            continue

        early_mean = float(np.mean(early_vals))
        late_mean = float(np.mean(late_vals))
        early_std = float(np.std(early_vals, ddof=1))
        late_std = float(np.std(late_vals, ddof=1))

        mean_shift = late_mean - early_mean
        mean_shift_pct = round(mean_shift / abs(early_mean) * 100, 2) if abs(early_mean) > 1e-6 else 0.0

        # Full series fatigue detection
        full_vals = pd.to_numeric(df[col], errors="coerce").dropna().tolist()
        fatigue = detect_fatigue_pattern(full_vals, within_session=True)

        metric_stability[col] = {
            "early_mean": round(early_mean, 3),
            "late_mean": round(late_mean, 3),
            "mean_shift": round(mean_shift, 3),
            "mean_shift_pct": mean_shift_pct,
            "early_std": round(early_std, 3),
            "late_std": round(late_std, 3),
            "std_change_pct": round((late_std - early_std) / early_std * 100, 1) if early_std > 1e-6 else 0.0,
            "fatigue": fatigue,
        }

    # Overall within-session score
    shifts = [
        abs(r["mean_shift_pct"])
        for r in metric_stability.values()
        if r.get("mean_shift_pct") is not None
    ]
    avg_shift = float(np.mean(shifts)) if shifts else 0.0

    return {
        "phase": phase,
        "num_frames": n,
        "num_metrics_analyzed": len(metric_stability),
        "average_mean_shift_pct": round(avg_shift, 2),
        "technique_stability": "STABLE" if avg_shift < 5 else "MODERATE_DRIFT" if avg_shift < 12 else "SIGNIFICANT_DRIFT",
        "per_metric_stability": metric_stability,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for consistency analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze temporal consistency across SwimVision sessions."
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Paths to *_report.json files in chronological order.",
    )
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument(
        "--metrics",
        nargs="*",
        help="Optional metric names to filter analysis.",
    )
    parser.add_argument(
        "--angles-csv",
        help="Optional angles CSV for within-session analysis.",
    )
    parser.add_argument(
        "--phase",
        default="entry_phase",
        help="Phase for within-session analysis (default: entry_phase).",
    )
    return parser


def main() -> int:
    """Run consistency analysis CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = analyze_consistency(
            args.reports,
            metric_filter=args.metrics if args.metrics else None,
        )
    except Exception as exc:
        LOGGER.error("Consistency analysis failed: %s", exc)
        return 1

    # Optional within-session analysis
    if args.angles_csv:
        try:
            within = analyze_within_session_consistency(args.angles_csv, args.phase)
            result["within_session_analysis"] = within
        except Exception as exc:
            LOGGER.warning("Within-session analysis failed: %s", exc)
            result["within_session_analysis"] = {"error": str(exc)}

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved consistency analysis to %s", args.output)
    except Exception as exc:
        LOGGER.error("Failed to write output: %s", exc)
        return 1

    print(json.dumps(
        {k: v for k, v in result.items() if k != "metric_consistency"},
        indent=2,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
