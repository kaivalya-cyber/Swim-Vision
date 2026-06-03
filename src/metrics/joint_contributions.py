# This file computes relative joint contributions from SwimVision angle time series.
"""Relative joint contribution analysis — percentage impact of each joint on total movement."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Anatomically-informed significance weights for swim start phases
# Higher weight = joint contributes more to effective propulsion
JOINT_SIGNIFICANCE_WEIGHTS: Dict[str, float] = {
    "front_knee_angle": 1.2,
    "rear_knee_angle": 1.3,
    "hip_angle": 1.5,
    "torso_lean": 1.0,
    "elbow_extension": 0.8,
    "streamline_angle": 1.2,
    "elbow_lock_angle": 0.7,
    "body_linearity": 1.0,
    "entry_angle": 1.3,
    "shoulder_angle": 0.9,
    "hip_rotation": 0.8,
    "shoulder_hip_separation": 0.9,
}

PHASE_TO_JOINTS: Dict[str, List[str]] = {
    "block_phase": [
        "front_knee_angle", "rear_knee_angle", "hip_angle",
        "torso_lean", "shoulder_angle",
    ],
    "flight_phase": [
        "body_linearity", "hip_angle", "torso_lean",
        "shoulder_angle", "hip_rotation",
    ],
    "entry_phase": [
        "entry_angle", "elbow_extension", "streamline_angle",
        "elbow_lock_angle", "body_linearity",
    ],
}


def compute_rom_contribution(
    angles_df: pd.DataFrame,
    phase_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute Range-of-Motion (ROM) contribution for each joint angle.

    Contribution = (joint ROM / total ROM) × 100, weighted by significance.

    Args:
        angles_df: DataFrame with per-frame angle columns.
        phase_columns: Optional subset of columns to analyze.

    Returns:
        Dictionary with per-joint ROM contributions.
    """
    columns = phase_columns or [c for c in angles_df.columns if c not in ("frame", "timestamp", "confidence")]
    if not columns:
        return {"error": "No angle columns found"}

    contributions: Dict[str, Dict[str, Any]] = {}
    total_weighted_rom = 0.0

    # First pass: compute ROM for each joint
    joint_roms: Dict[str, float] = {}
    for col in columns:
        try:
            vals = pd.to_numeric(angles_df[col], errors="coerce").dropna()
            if len(vals) < 2:
                continue
            rom = float(vals.max() - vals.min())
            if rom < 1e-6:
                continue
            joint_roms[col] = rom
        except Exception:
            continue

    if not joint_roms:
        return {"error": "No valid ROM computed for any joint"}

    # Weight ROMs by anatomical significance
    weighted_roms: Dict[str, float] = {}
    for col, rom in joint_roms.items():
        weight = JOINT_SIGNIFICANCE_WEIGHTS.get(col, 0.8)
        weighted_roms[col] = rom * weight
        total_weighted_rom += weighted_roms[col]

    if total_weighted_rom < 1e-6:
        return {"error": "Total weighted ROM too small"}

    # Compute percentage contributions
    for col, w_rom in sorted(weighted_roms.items(), key=lambda x: x[1], reverse=True):
        pct = round(w_rom / total_weighted_rom * 100.0, 2)
        raw_rom = joint_roms.get(col, 0.0)
        contributions[col] = {
            "rom_degrees": round(raw_rom, 2),
            "weight": JOINT_SIGNIFICANCE_WEIGHTS.get(col, 0.8),
            "weighted_rom": round(w_rom, 2),
            "contribution_pct": pct,
        }

    # Rank joints by contribution
    ranked = sorted(contributions.items(), key=lambda x: x[1]["contribution_pct"], reverse=True)

    return {
        "total_weighted_rom": round(total_weighted_rom, 2),
        "num_joints_analyzed": len(contributions),
        "top_contributors": [
            {"joint": name, "contribution_pct": data["contribution_pct"]}
            for name, data in ranked[:5]
        ],
        "joint_contributions": contributions,
    }


def compute_phase_contributions(
    angles_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute joint contributions broken down by dive phase.

    Assumes phase columns exist as named groups in the DataFrame.

    Args:
        angles_df: DataFrame with angle columns.

    Returns:
        Dictionary with per-phase contribution analysis.
    """
    result: Dict[str, Any] = {}

    for phase_name, phase_cols in PHASE_TO_JOINTS.items():
        available = [c for c in phase_cols if c in angles_df.columns]
        if not available:
            continue
        try:
            rom_result = compute_rom_contribution(angles_df, phase_columns=available)
            if "error" not in rom_result:
                result[phase_name] = rom_result
        except Exception as exc:
            LOGGER.warning("Phase contribution for %s failed: %s", phase_name, exc)

    return result


def compute_velocity_contribution(
    angles_df: pd.DataFrame,
    vel_accel_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Estimate joint velocity contributions as percentage of total angular velocity.

    Uses max velocity from velocity_acceleration module if available,
    otherwise estimates from angle derivatives.

    Args:
        angles_df: DataFrame with per-frame angles.
        vel_accel_data: Optional velocity/acceleration profile data.

    Returns:
        Dictionary with velocity-based contribution percentages.
    """
    contributions: Dict[str, Dict[str, Any]] = {}
    total_peak_vel = 0.0

    # Use pre-computed velocity data if available
    if vel_accel_data:
        phase_profiles = vel_accel_data.get("phase_profiles", {})
        for phase_name, profile in phase_profiles.items():
            if isinstance(profile, dict):
                for col, col_data in profile.items():
                    if isinstance(col_data, dict):
                        max_vel = abs(col_data.get("max_velocity", 0.0) or 0.0)
                        if max_vel > 0:
                            total_peak_vel += max_vel
                            contributions[col] = contributions.get(col, {"peak_velocity": 0.0})
                            contributions[col]["peak_velocity"] = max(
                                contributions[col].get("peak_velocity", 0.0), max_vel
                            )
    else:
        # Estimate velocity from angle derivatives
        columns = [c for c in angles_df.columns if c not in ("frame", "timestamp", "confidence")]
        for col in columns:
            try:
                vals = pd.to_numeric(angles_df[col], errors="coerce").dropna().values
                if len(vals) < 3:
                    continue
                velocities = np.abs(np.diff(vals))
                peak_vel = float(np.max(velocities))
                if peak_vel > 0:
                    total_peak_vel += peak_vel
                    contributions[col] = {"peak_velocity": round(peak_vel, 4)}
            except Exception:
                continue

    if total_peak_vel < 1e-6:
        return {"error": "No valid velocity data"}

    # Compute velocity contribution percentages
    for col, data in contributions.items():
        pv = data.get("peak_velocity", 0.0)
        data["velocity_contribution_pct"] = round(pv / total_peak_vel * 100.0, 2)

    # Sort by velocity contribution
    ranked = sorted(contributions.items(), key=lambda x: x[1].get("velocity_contribution_pct", 0), reverse=True)

    return {
        "total_peak_velocity": round(total_peak_vel, 4),
        "num_joints_analyzed": len(contributions),
        "top_velocity_contributors": [
            {"joint": name, "velocity_contribution_pct": data.get("velocity_contribution_pct", 0)}
            for name, data in ranked[:5]
        ],
        "velocity_contributions": contributions,
    }


def analyze_joint_contributions(
    angles_csv: str,
    vel_accel_json: Optional[str] = None,
    boundaries_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Full joint contribution analysis from angle data.

    Args:
        angles_csv: Path to joint angles CSV.
        vel_accel_json: Optional velocity/acceleration profile JSON.
        boundaries_json: Optional phase boundaries JSON.

    Returns:
        Comprehensive joint contribution analysis dictionary.
    """
    try:
        df = pd.read_csv(angles_csv)
    except Exception as exc:
        return {"error": f"Failed to load angles CSV: {exc}"}

    if df.empty:
        return {"error": "Empty angle data"}

    result: Dict[str, Any] = {}

    # ROM-based contributions (all phases combined)
    try:
        result["rom_contributions"] = compute_rom_contribution(df)
    except Exception as exc:
        LOGGER.warning("ROM contribution failed: %s", exc)
        result["rom_contributions"] = {"error": str(exc)}

    # Per-phase ROM contributions
    try:
        result["phase_contributions"] = compute_phase_contributions(df)
    except Exception as exc:
        LOGGER.warning("Phase contributions failed: %s", exc)

    # Velocity-based contributions
    vel_data = None
    if vel_accel_json:
        try:
            with open(vel_accel_json, "r", encoding="utf-8") as f:
                vel_data = json.load(f)
        except Exception as exc:
            LOGGER.warning("Failed to load vel/accel data: %s", exc)

    try:
        result["velocity_contributions"] = compute_velocity_contribution(df, vel_data)
    except Exception as exc:
        LOGGER.warning("Velocity contribution failed: %s", exc)
        result["velocity_contributions"] = {"error": str(exc)}

    # Dominant joints summary
    dominant = set()
    rom_top = result.get("rom_contributions", {}).get("top_contributors", [])
    vel_top = result.get("velocity_contributions", {}).get("top_velocity_contributors", [])
    for item in rom_top[:3]:
        dominant.add(item.get("joint", ""))
    for item in vel_top[:3]:
        dominant.add(item.get("joint", ""))

    result["dominant_joints"] = sorted(dominant)
    result["summary"] = (
        f"Primary movement drivers: {', '.join(result['dominant_joints']) if result['dominant_joints'] else 'N/A'}. "
        f"Analyzed {result.get('rom_contributions', {}).get('num_joints_analyzed', 0)} joints across "
        f"{len(result.get('phase_contributions', {}))} phases."
    )

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for joint contribution analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze relative joint contributions from SwimVision angle data."
    )
    parser.add_argument("--angles", required=True, help="Path to angles CSV.")
    parser.add_argument("--vel-accel", help="Optional velocity/acceleration profile JSON.")
    parser.add_argument("--boundaries", help="Optional phase boundaries JSON.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run joint contributions CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = analyze_joint_contributions(
            args.angles,
            vel_accel_json=args.vel_accel,
            boundaries_json=args.boundaries,
        )
    except Exception as exc:
        LOGGER.error("Joint contribution analysis failed: %s", exc)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved joint contributions to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
