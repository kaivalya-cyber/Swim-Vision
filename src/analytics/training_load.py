# This file tracks training load and recovery status across SwimVision sessions for a swimmer.
"""Training load and recovery tracking — acute/chronic workload, overreaching, and recovery recommendations."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.storage.session_manager import init_db, get_swimmer_history


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Session severity → load score mapping
SEVERITY_LOAD: Dict[str, float] = {
    "OPTIMAL": 3.0,
    "MINOR": 5.0,
    "SIGNIFICANT": 8.0,
    "CRITICAL": 12.0,
}

# Risk level → load multiplier
RISK_LOAD_MULTIPLIER: Dict[str, float] = {
    "LOW": 1.0,
    "MEDIUM": 1.3,
    "HIGH": 1.6,
    "CRITICAL": 2.0,
}

# Reaction time bonus load (faster reaction = less load, slower = more)
def _reaction_load(rt_ms: Optional[float]) -> float:
    if rt_ms is None:
        return 0.0
    if rt_ms < 600:
        return 1.0
    if rt_ms < 700:
        return 2.0
    if rt_ms < 800:
        return 3.0
    return 4.0


def compute_session_load(
    overall_severity: str,
    reaction_time_ms: Optional[float] = None,
    num_cycles: int = 0,
    risk_level: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the training load score for a single session.

    Args:
        overall_severity: Overall deviation severity.
        reaction_time_ms: Reaction time in milliseconds.
        num_cycles: Number of stroke cycles completed.
        risk_level: Injury risk level.

    Returns:
        Session load metrics.
    """
    base_load = SEVERITY_LOAD.get(overall_severity, 5.0)
    rt_load = _reaction_load(reaction_time_ms)
    risk_mult = RISK_LOAD_MULTIPLIER.get(risk_level or "LOW", 1.0)
    cycle_bonus = min(num_cycles * 0.5, 5.0)  # up to 5 bonus for high volume

    total_load = (base_load + rt_load + cycle_bonus) * risk_mult

    if total_load < 5:
        intensity = "LOW"
    elif total_load < 10:
        intensity = "MODERATE"
    elif total_load < 15:
        intensity = "HIGH"
    else:
        intensity = "VERY-HIGH"

    return {
        "session_load_score": round(total_load, 1),
        "intensity": intensity,
        "components": {
            "base_severity_load": base_load,
            "reaction_time_load": rt_load,
            "cycle_volume_bonus": cycle_bonus,
            "risk_multiplier": risk_mult,
        },
    }


def compute_acute_chronic_ratio(
    session_loads: List[Dict[str, Any]],
    acute_window_days: int = 7,
    chronic_window_days: int = 28,
) -> Dict[str, Any]:
    """Compute the acute:chronic workload ratio for overreaching detection.

    ACWR > 1.5 = high injury risk, 0.8-1.3 = optimal training zone.

    Args:
        session_loads: List of session dicts with 'date' and 'load_score'.
        acute_window_days: Days for acute (short-term) window.
        chronic_window_days: Days for chronic (long-term) window.

    Returns:
        ACWR metrics and training status.
    """
    if not session_loads:
        return {"acwr": None, "status": "INSUFFICIENT-DATA"}

    now = datetime.now()
    acute_loads: List[float] = []
    chronic_loads: List[float] = []

    for s in session_loads:
        try:
            s_date = datetime.fromisoformat(s.get("date", "")[:10])
        except (ValueError, TypeError):
            continue
        load = s.get("load_score", 0)
        if load is None:
            continue
        days_ago = (now - s_date).days
        if days_ago <= acute_window_days:
            acute_loads.append(float(load))
        if days_ago <= chronic_window_days:
            chronic_loads.append(float(load))

    acute_total = sum(acute_loads)
    chronic_total = sum(chronic_loads)

    # Average daily load
    acute_avg = acute_total / acute_window_days if acute_window_days > 0 else 0
    chronic_avg = chronic_total / chronic_window_days if chronic_window_days > 0 else 0

    if chronic_avg < 0.01:
        return {"acwr": None, "status": "BASELINE-BUILDING", "acute_avg_load": round(acute_avg, 2)}

    acwr = round(acute_avg / chronic_avg, 2)

    if acwr < 0.8:
        status = "DETRAINING"
    elif acwr <= 1.3:
        status = "OPTIMAL"
    elif acwr <= 1.5:
        status = "OVERREACHING"
    else:
        status = "HIGH-RISK"

    return {
        "acwr": acwr,
        "status": status,
        "acute_avg_load": round(acute_avg, 2),
        "chronic_avg_load": round(chronic_avg, 2),
        "acute_sessions": len(acute_loads),
        "chronic_sessions": len(chronic_loads),
    }


def detect_overreaching(
    session_loads: List[Dict[str, Any]],
    lookback_days: int = 14,
) -> Dict[str, Any]:
    """Detect overreaching patterns from recent session loads.

    Flags when: 3+ high-intensity sessions in a week, or monotonic load increase.

    Args:
        session_loads: List of session load dicts sorted by date.
        lookback_days: Days to look back for pattern detection.

    Returns:
        Overreaching detection results.
    """
    if len(session_loads) < 3:
        return {"overreaching_detected": False, "confidence": 0.0}

    now = datetime.now()
    recent = [
        s for s in session_loads
        if s.get("date") and (now - datetime.fromisoformat(s["date"][:10])).days <= lookback_days
    ]

    if len(recent) < 3:
        return {"overreaching_detected": False, "confidence": 0.0, "reason": "insufficient recent sessions"}

    signals = 0
    reasons: List[str] = []

    # Check for high-intensity cluster
    high_intensity = [
        s for s in recent
        if s.get("intensity") in ("HIGH", "VERY-HIGH")
    ]
    if len(high_intensity) >= 3:
        signals += 1
        reasons.append(f"{len(high_intensity)} high-intensity sessions in {lookback_days} days")

    # Check for monotonic load increase
    loads = [s.get("load_score", 0) for s in recent]
    loads_clean = [l for l in loads if l is not None and l > 0]
    if len(loads_clean) >= 3:
        increasing = all(loads_clean[i] <= loads_clean[i + 1] for i in range(len(loads_clean) - 1))
        if increasing:
            signals += 1
            reasons.append("Monotonic load increase across recent sessions")

    # Check for performance degradation
    severities = [s.get("overall_severity", "OPTIMAL") for s in recent]
    flag_order = ["OPTIMAL", "MINOR", "SIGNIFICANT", "CRITICAL"]
    if len(severities) >= 3:
        recent_half = severities[-len(severities) // 2 :]
        early_half = severities[: len(severities) // 2]
        recent_worst = max(recent_half, key=lambda f: flag_order.index(f)) if recent_half else "OPTIMAL"
        early_worst = max(early_half, key=lambda f: flag_order.index(f)) if early_half else "OPTIMAL"
        if flag_order.index(recent_worst) > flag_order.index(early_worst):
            signals += 1
            reasons.append(f"Technique degradation: {early_worst} → {recent_worst}")

    confidence = min(1.0, signals / 3.0)

    return {
        "overreaching_detected": signals >= 2,
        "confidence": round(confidence, 2),
        "signals": signals,
        "reasons": reasons,
    }


def compute_recovery_recommendations(
    acwr_data: Dict[str, Any],
    overreaching_data: Dict[str, Any],
    recent_load: float,
) -> List[str]:
    """Generate recovery recommendations based on load analysis.

    Args:
        acwr_data: ACWR results.
        overreaching_data: Overreaching detection results.
        recent_load: Most recent session load score.

    Returns:
        List of recommendation strings.
    """
    recs: List[str] = []

    acwr_status = acwr_data.get("status", "")
    if acwr_status == "HIGH-RISK":
        recs.append("URGENT: Reduce training volume by 40-50% for 7-10 days.")
        recs.append("Incorporate active recovery sessions (light technique work only).")
    elif acwr_status == "OVERREACHING":
        recs.append("CAUTION: Deload week recommended — reduce volume by 20-30%.")
        recs.append("Monitor sleep quality and resting heart rate daily.")
    elif acwr_status == "DETRAINING":
        recs.append("Training load is below maintenance — consider progressive overload.")
        recs.append("Increase session frequency or intensity gradually (10% per week).")
    elif acwr_status == "OPTIMAL":
        recs.append("Training load is in the optimal zone — maintain current programming.")

    if overreaching_data.get("overreaching_detected"):
        recs.append("Overreaching pattern detected — prioritize recovery modalities (massage, nutrition, sleep).")
        recs.append("Consider a 48-72 hour complete rest period before next high-intensity session.")

    if recent_load > 15:
        recs.append("Last session was very high load — ensure 48+ hours recovery before next session.")
        recs.append("Perform a dynamic warm-up and mobility assessment before next training block.")

    return recs


def analyze_training_load(
    swimmer_id: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Full training load analysis for a swimmer.

    Queries session storage DB and computes load metrics, ACWR, and recommendations.

    Args:
        swimmer_id: Swimmer identifier.
        db_path: Optional database path.

    Returns:
        Comprehensive training load analysis.
    """
    conn = init_db(db_path)
    sessions = get_swimmer_history(conn, swimmer_id, limit=50)
    conn.close()

    if not sessions:
        return {"error": f"No sessions found for swimmer {swimmer_id}", "swimmer_id": swimmer_id}

    # Compute per-session load
    session_loads: List[Dict[str, Any]] = []
    for s in sessions:
        load = compute_session_load(
            overall_severity=s.get("overall_severity", "OPTIMAL"),
            reaction_time_ms=s.get("reaction_time_ms"),
            num_cycles=int(s.get("num_cycles", 0) or 0),
        )
        session_loads.append({
            "session_id": s.get("id", ""),
            "clip_id": s.get("clip_id", ""),
            "date": s.get("created_at", ""),
            "analysis_mode": s.get("analysis_mode", "dive"),
            "load_score": load["session_load_score"],
            "intensity": load["intensity"],
            "overall_severity": s.get("overall_severity"),
        })

    if not session_loads:
        return {"error": "No valid session loads computed", "swimmer_id": swimmer_id}

    # Sort by date
    session_loads.sort(key=lambda s: s.get("date", ""))

    # Compute metrics
    acwr = compute_acute_chronic_ratio(session_loads)
    overreaching = detect_overreaching(session_loads)
    recent_load = session_loads[-1]["load_score"] if session_loads else 0
    total_load_28d = sum(
        s["load_score"] for s in session_loads
        if s.get("date") and (datetime.now() - datetime.fromisoformat(s["date"][:10])).days <= 28
    )
    recommendations = compute_recovery_recommendations(acwr, overreaching, recent_load)

    # Load trend
    loads = [s["load_score"] for s in session_loads[-10:]]
    load_trend = "stable"
    if len(loads) >= 3:
        slope = np.polyfit(range(len(loads)), loads, 1)[0]
        if slope > 0.5:
            load_trend = "increasing"
        elif slope < -0.5:
            load_trend = "decreasing"

    return {
        "swimmer_id": swimmer_id,
        "num_sessions_analyzed": len(sessions),
        "current_load": round(recent_load, 1),
        "total_load_28d": round(total_load_28d, 1),
        "load_trend": load_trend,
        "acwr": acwr,
        "overreaching": overreaching,
        "recovery_recommendations": recommendations,
        "recent_sessions": session_loads[-7:],
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for training load analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze training load and recovery status from SwimVision session history."
    )
    parser.add_argument("--swimmer-id", required=True, help="Swimmer identifier.")
    parser.add_argument("--db-path", help="Optional SQLite database path.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run training load CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    result = analyze_training_load(
        args.swimmer_id,
        db_path=Path(args.db_path) if args.db_path else None,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved training load analysis to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
