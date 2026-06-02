# This file analyzes trends across multiple SwimVision session reports for longitudinal tracking.
"""Longitudinal trend analysis across SwimVision session reports."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from collections import defaultdict
from datetime import datetime as dt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """Aggregated metrics for a single analysis session.

    Attributes:
        session_id: Identifier for the session (clip_id or custom label).
        date: ISO format date string.
        analysis_mode: dive or stroke.
        overall_severity: Overall deviation severity (OPTIMAL, MINOR, etc.).
        num_cycles: Number of stroke cycles detected (stroke mode only).
        metrics: Flat dictionary of key metric values.
    """

    session_id: str
    date: str = ""
    analysis_mode: str = "stroke"
    overall_severity: str = "OPTIMAL"
    num_cycles: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


def _load_report(report_path: Path) -> Optional[SessionRecord]:
    """Parse a single SwimVision report JSON into a SessionRecord.

    Args:
        report_path: Path to a *_report.json file.

    Returns:
        SessionRecord or None if parsing fails.
    """

    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        LOGGER.warning("Failed to load report %s: %s", report_path, exc)
        return None

    analysis_mode = data.get("analysis_mode", "dive")
    metrics: Dict[str, float] = {}

    if analysis_mode == "stroke":
        aggregate = data.get("aggregate_metrics", {})
        if isinstance(aggregate, dict):
            for key in (
                "stroke_rate",
                "body_roll",
                "symmetry_index",
                "left_elbow_flexion",
                "right_elbow_flexion",
                "left_hand_speed",
                "right_hand_speed",
                "cycle_duration_seconds",
                "num_cycles",
            ):
                val = aggregate.get(key)
                if val is not None:
                    try:
                        metrics[key] = float(val)
                    except (ValueError, TypeError):
                        pass
    else:
        # Dive mode: extract key metrics from deviations
        deviations = data.get("deviations", {})
        for phase_name in ("block_phase", "flight_phase", "entry_phase"):
            phase_rows = deviations.get(phase_name, [])
            if isinstance(phase_rows, list):
                for row in phase_rows:
                    if isinstance(row, dict):
                        metric_name = str(row.get("metric", ""))
                        measured = row.get("measured")
                        if measured is not None:
                            try:
                                metrics[f"{phase_name}_{metric_name}"] = float(measured)
                            except (ValueError, TypeError):
                                pass

    return SessionRecord(
        session_id=data.get("clip_id", report_path.stem.replace("_report", "")),
        date=data.get("date", ""),
        analysis_mode=analysis_mode,
        overall_severity=str(data.get("overall_severity", "OPTIMAL")),
        num_cycles=int(data.get("num_cycles", 0)),
        metrics=metrics,
    )


def _is_improvement_direction(metric_name: str) -> bool:
    """Determine if an increase in this metric is an improvement.

    Most swim metrics (stroke_rate, hand_speed, elbow_extension_rate, body_roll)
    improve with higher values. Some (symmetry_index, cycle_duration_seconds)
    improve with lower values.
    """

    lower_is_better = {
        "symmetry_index",
        "cycle_duration_seconds",
    }
    if metric_name in lower_is_better:
        return False
    return True


def _aggregate_sessions(
    sessions: List[SessionRecord],
    aggregation: str,
) -> List[SessionRecord]:
    """Group sessions by week or month and average metrics within each period.

    Args:
        sessions: List of session records sorted by date.
        aggregation: "week" or "month".

    Returns:
        New list of SessionRecord with aggregated metrics per period.
    """

    if aggregation not in ("week", "month"):
        return sessions

    groups: Dict[str, List[SessionRecord]] = defaultdict(list)
    for session in sessions:
        if not session.date:
            LOGGER.warning("Skipping session '%s' in aggregation: no date", session.session_id)
            continue
        try:
            d = dt.fromisoformat(session.date[:10])
        except (ValueError, TypeError):
            continue
        if aggregation == "week":
            iso_week = d.isocalendar()
            key = f"{iso_week[0]}-W{iso_week[1]:02d}"
        else:
            key = d.strftime("%Y-%m")
        groups[key].append(session)

    aggregated: List[SessionRecord] = []
    for key in sorted(groups.keys()):
        group = groups[key]
        if not group:
            continue
        combined_metrics: Dict[str, List[float]] = defaultdict(list)
        for s in group:
            for m_name, m_val in s.metrics.items():
                if m_val is not None and not (isinstance(m_val, float) and np.isnan(m_val)):
                    combined_metrics[m_name].append(m_val)
        avg_metrics: Dict[str, float] = {}
        for m_name, vals in combined_metrics.items():
            if vals:
                avg_metrics[m_name] = float(np.mean(vals))
        severities = [s.overall_severity for s in group]
        flag_order = ["OPTIMAL", "MINOR", "SIGNIFICANT", "CRITICAL"]
        worst = max(severities, key=lambda f: flag_order.index(f)) if severities else "OPTIMAL"
        modes = {s.analysis_mode for s in group}
        aggregated.append(SessionRecord(
            session_id=key,
            date=group[0].date if group else "",
            analysis_mode=",".join(sorted(modes)) if modes else group[0].analysis_mode,
            overall_severity=worst,
            num_cycles=sum(s.num_cycles for s in group),
            metrics=avg_metrics,
        ))

    return aggregated


def _compute_trend(values: List[float], metric_name: str = "") -> Dict[str, float]:
    """Compute trend statistics for a series of metric values.

    Args:
        values: Time-ordered list of metric values.

    Returns:
        Dictionary with mean, std, min, max, slope (per-session change), and trend direction.
    """

    if not values:
        return {}

    arr = np.array(values, dtype=np.float32)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    change = float(arr[-1] - arr[0]) if len(arr) > 1 else 0.0

    slope = 0.0
    if len(arr) > 1:
        x_vals = np.arange(len(arr), dtype=np.float32)
        x_mean = float(np.mean(x_vals))
        y_mean = float(np.mean(arr))
        numerator = float(np.sum((x_vals - x_mean) * (arr - y_mean)))
        denominator = float(np.sum((x_vals - x_mean) ** 2))
        if denominator > 0:
            slope = numerator / denominator

    direction = "stable"
    if slope > 0.01:
        direction = "improving" if _is_improvement_direction(metric_name) else "declining"
    elif slope < -0.01:
        direction = "improving" if not _is_improvement_direction(metric_name) else "declining"

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "change": change,
        "slope": slope,
        "direction": direction,
        "num_sessions": float(len(values)),
    }


def analyze_trends(
    report_paths: List[str],
    primary_metric: str = "stroke_rate",
    analysis_mode: str = "",
    start_date: str = "",
    end_date: str = "",
    aggregation: str = "",
) -> Dict[str, Any]:
    """Analyze longitudinal trends across multiple SwimVision session reports.

    Args:
        report_paths: List of paths to *_report.json files, in chronological order.
        primary_metric: Key metric to focus trend analysis on.
        analysis_mode: Filter sessions by mode (dive, stroke, or empty for all).
        start_date: ISO date string for earliest session to include.
        end_date: ISO date string for latest session to include.

    Returns:
        aggregation: Group sessions by "week" or "month" (empty for no aggregation).

    Returns:
        Dictionary with sessions, trend_summary, and per-metric trend data.
    """

    sessions: List[SessionRecord] = []
    for path_str in report_paths:
        record = _load_report(Path(path_str))
        if record is None:
            continue
        if analysis_mode and record.analysis_mode != analysis_mode:
            continue
        if start_date and record.date and record.date < start_date:
            continue
        if end_date and record.date and record.date > end_date:
            continue
        sessions.append(record)

    if not sessions:
        LOGGER.warning("No valid reports loaded for trend analysis.")
        return {"sessions": [], "trend_summary": {}, "metric_trends": {}}

    # Apply time aggregation if requested
    if aggregation in ("week", "month"):
        sessions = _aggregate_sessions(sessions, aggregation)
        if not sessions:
            LOGGER.warning("No sessions after aggregation.")
            return {"sessions": [], "trend_summary": {}, "metric_trends": {}}

    # Collect all metric names across sessions
    all_metrics: set[str] = set()
    for session in sessions:
        all_metrics.update(session.metrics.keys())

    metric_trends: Dict[str, Dict[str, float]] = {}
    for metric_name in sorted(all_metrics):
        values = [s.metrics.get(metric_name) for s in sessions]
        values_float = [v for v in values if v is not None]
        if len(values_float) >= 2:
            metric_trends[metric_name] = _compute_trend(values_float, metric_name=metric_name)

    # Overall trend summary
    primary_trend = metric_trends.get(primary_metric, {})
    severities = [s.overall_severity for s in sessions]
    flag_order = ["OPTIMAL", "MINOR", "SIGNIFICANT", "CRITICAL"]
    worst_severity = max(severities, key=lambda f: flag_order.index(f)) if severities else "OPTIMAL"

    trend_summary: Dict[str, Any] = {
        "num_sessions": len(sessions),
        "date_range": {
            "first": sessions[0].date if sessions else "",
            "last": sessions[-1].date if sessions else "",
        },
        "primary_metric": primary_metric,
        "primary_trend": primary_trend,
        "overall_worst_severity": worst_severity,
        "metrics_with_trends": len(metric_trends),
    }

    # Check for sustained improvement
    if primary_trend.get("direction") == "improving" and primary_trend.get("slope", 0) > 0.01:
        trend_summary["summary_verdict"] = (
            f"{primary_metric} shows consistent improvement "
            f"(+{primary_trend.get('change', 0):.1f} over {len(sessions)} sessions)"
        )
    elif primary_trend.get("direction") == "declining":
        trend_summary["summary_verdict"] = (
            f"{primary_metric} shows a declining trend "
            f"({primary_trend.get('change', 0):.1f} over {len(sessions)} sessions) — review technique"
        )
    else:
        trend_summary["summary_verdict"] = (
            f"{primary_metric} is stable across {len(sessions)} sessions"
        )

    return {
        "sessions": [asdict(s) for s in sessions],
        "trend_summary": trend_summary,
        "metric_trends": metric_trends,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for trend analysis."""

    parser = argparse.ArgumentParser(
        description="Analyze longitudinal trends across SwimVision session reports."
    )
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Paths to *_report.json files in chronological order.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for trend analysis JSON.",
    )
    parser.add_argument(
        "--primary-metric",
        default="stroke_rate",
        help="Primary metric for trend focus (default: stroke_rate).",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional CSV path for session metrics table.",
    )
    return parser


def main() -> int:
    """Run the CLI for trend analysis."""

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = analyze_trends(args.reports, primary_metric=args.primary_metric)
    except Exception as exc:
        LOGGER.error("Trend analysis failed: %s", exc)
        return 1

    try:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        LOGGER.info("Saved trend analysis to %s", args.output)
    except Exception as exc:
        LOGGER.error("Failed to write trend output: %s", exc)
        return 1

    # Optional CSV export
    if args.output_csv and result.get("sessions"):
        try:
            import pandas as pd
            rows = []
            for session in result["sessions"]:
                row = {
                    "session_id": session["session_id"],
                    "date": session["date"],
                    "analysis_mode": session["analysis_mode"],
                    "overall_severity": session["overall_severity"],
                }
                row.update(session.get("metrics", {}))
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(args.output_csv, index=False)
            LOGGER.info("Saved session metrics CSV to %s", args.output_csv)
        except Exception as exc:
            LOGGER.warning("Failed to write CSV: %s", exc)

    print(json.dumps(result["trend_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
