# This file computes angular velocity and acceleration profiles from joint angle time series.
"""Angular velocity and acceleration computation for SwimVision phase metrics."""

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

# Joint angle columns to compute derivatives for
ANGLE_COLUMNS = [
    "front_knee_angle",
    "rear_knee_angle",
    "hip_angle",
    "torso_lean",
    "left_elbow_angle",
    "right_elbow_angle",
    "body_linearity",
    "entry_angle",
    "streamline_angle",
    "elbow_extension",
    "elbow_lock_angle",
]

# Phase-specific columns of interest
PHASE_COLUMNS = {
    "block_phase": ["front_knee_angle", "rear_knee_angle", "hip_angle", "torso_lean"],
    "flight_phase": ["body_linearity", "entry_angle", "elbow_extension"],
    "entry_phase": ["streamline_angle", "elbow_lock_angle"],
}


def compute_velocity(
    angles_df: pd.DataFrame,
    fps: float = 30.0,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    """Compute angular velocity (deg/s) from joint angle time series.

    Uses central differences with forward/backward differences at endpoints.

    Args:
        angles_df: Per-frame angle DataFrame indexed by frame.
        fps: Frame rate of the source video.
        columns: Optional subset of angle columns. Defaults to all ANGLE_COLUMNS.

    Returns:
        DataFrame with velocity columns suffixed with ``_vel``.
    """
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    target_columns = columns or [c for c in ANGLE_COLUMNS if c in angles_df.columns]
    vel_data: Dict[str, List[float]] = {}

    dt = 1.0 / fps

    for col in target_columns:
        if col not in angles_df.columns:
            continue
        values = angles_df[col].to_numpy(dtype=np.float32)
        vel = np.full_like(values, np.nan, dtype=np.float32)

        # Interior points: central difference
        for i in range(1, len(values) - 1):
            if np.isnan(values[i - 1]) or np.isnan(values[i + 1]):
                continue
            vel[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)

        # Endpoints: forward/backward difference
        if len(values) >= 2:
            if not np.isnan(values[0]) and not np.isnan(values[1]):
                vel[0] = (values[1] - values[0]) / dt
            if not np.isnan(values[-2]) and not np.isnan(values[-1]):
                vel[-1] = (values[-1] - values[-2]) / dt

        vel_data[f"{col}_vel"] = vel.tolist()

    vel_df = pd.DataFrame(vel_data, index=angles_df.index)
    return vel_df


def compute_acceleration(
    velocity_df: pd.DataFrame,
    fps: float = 30.0,
) -> pd.DataFrame:
    """Compute angular acceleration (deg/s²) from angular velocity time series.

    Uses central differences with forward/backward differences at endpoints.

    Args:
        velocity_df: Per-frame velocity DataFrame (output of compute_velocity).
        fps: Frame rate of the source video.

    Returns:
        DataFrame with acceleration columns suffixed with ``_acc``.
    """
    if fps <= 0:
        raise ValueError("FPS must be positive.")

    vel_columns = [c for c in velocity_df.columns if c.endswith("_vel")]
    acc_data: Dict[str, List[float]] = {}
    dt = 1.0 / fps

    for col in vel_columns:
        base_name = col.replace("_vel", "")
        values = velocity_df[col].to_numpy(dtype=np.float32)
        acc = np.full_like(values, np.nan, dtype=np.float32)

        for i in range(1, len(values) - 1):
            if np.isnan(values[i - 1]) or np.isnan(values[i + 1]):
                continue
            acc[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)

        if len(values) >= 2:
            if not np.isnan(values[0]) and not np.isnan(values[1]):
                acc[0] = (values[1] - values[0]) / dt
            if not np.isnan(values[-2]) and not np.isnan(values[-1]):
                acc[-1] = (values[-1] - values[-2]) / dt

        acc_data[f"{base_name}_acc"] = acc.tolist()

    acc_df = pd.DataFrame(acc_data, index=velocity_df.index)
    return acc_df


def compute_phase_profile(
    angles_df: pd.DataFrame,
    phase_boundaries: Dict[str, int],
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Compute velocity and acceleration profiles for each detected phase.

    Args:
        angles_df: Per-frame angle DataFrame indexed by frame.
        phase_boundaries: Phase boundary dictionary with start/end frame indices.
        fps: Frame rate of the source video.

    Returns:
        Nested dict with per-phase velocity/acceleration summary statistics.
    """
    result: Dict[str, Any] = {}

    for phase_name, columns in PHASE_COLUMNS.items():
        start_key = phase_name.replace("_phase", "_start")
        end_key = phase_name.replace("_phase", "_end")

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

        phase_result: Dict[str, Any] = {}

        for col in columns:
            if col not in phase_window.columns:
                continue

            values = phase_window[col].dropna().to_numpy(dtype=np.float32)
            if len(values) < 2:
                continue

            # Compute velocity and acceleration for this column segment
            vel_df = compute_velocity(phase_window[[col]], fps=fps, columns=[col])
            acc_df = compute_acceleration(vel_df, fps=fps)

            vel_col = f"{col}_vel"
            acc_col = f"{col}_acc"

            vel_series = vel_df[vel_col].dropna()
            acc_series = acc_df[acc_col].dropna()

            phase_result[col] = {
                "mean_velocity_deg_s": float(vel_series.mean()) if len(vel_series) > 0 else None,
                "max_velocity_deg_s": float(vel_series.max()) if len(vel_series) > 0 else None,
                "mean_acceleration_deg_s2": float(acc_series.mean()) if len(acc_series) > 0 else None,
                "max_acceleration_deg_s2": float(acc_series.max()) if len(acc_series) > 0 else None,
                "peak_velocity_frame": int(vel_series.idxmax()) if len(vel_series) > 0 else None,
                "peak_acceleration_frame": int(acc_series.idxmax()) if len(acc_series) > 0 else None,
            }

        if phase_result:
            result[phase_name] = phase_result

    return result


def compute_full_profiles(
    angles_df: pd.DataFrame,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Compute velocity and acceleration across the full sequence.

    Args:
        angles_df: Per-frame angle DataFrame indexed by frame.
        fps: Frame rate of the source video.

    Returns:
        Dictionary with velocity and acceleration DataFrames and summary stats.
    """
    vel_df = compute_velocity(angles_df, fps=fps)
    acc_df = compute_acceleration(vel_df, fps=fps)

    summary: Dict[str, Any] = {}
    for col in [c for c in ANGLE_COLUMNS if c in angles_df.columns]:
        vel_col = f"{col}_vel"
        acc_col = f"{col}_acc"

        vel_series = vel_df[vel_col].dropna()
        acc_series = acc_df[acc_col].dropna()

        summary[col] = {
            "mean_velocity_deg_s": float(vel_series.mean()) if len(vel_series) > 0 else None,
            "max_velocity_deg_s": float(vel_series.max()) if len(vel_series) > 0 else None,
            "mean_acceleration_deg_s2": float(acc_series.mean()) if len(acc_series) > 0 else None,
            "max_acceleration_deg_s2": float(acc_series.max()) if len(acc_series) > 0 else None,
        }

    return {
        "summary": summary,
        "velocity_columns": list(vel_df.columns),
        "acceleration_columns": list(acc_df.columns),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for velocity/acceleration computation."""
    parser = argparse.ArgumentParser(
        description="Compute angular velocity and acceleration from SwimVision joint angles."
    )
    parser.add_argument("--angles", required=True, help="Path to angle CSV file.")
    parser.add_argument("--boundaries", help="Path to phase boundary JSON file.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
    parser.add_argument("--output", help="Optional JSON output path for profiles.")
    parser.add_argument(
        "--output-csv",
        help="Optional CSV path for full velocity+acceleration time series.",
    )
    return parser


def main() -> int:
    """Run the CLI for velocity/acceleration computation."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        angles_df = pd.read_csv(args.angles, index_col=0)
    except Exception as exc:
        LOGGER.error("Failed to load angles CSV: %s", exc)
        return 1

    try:
        profiles = compute_full_profiles(angles_df, fps=args.fps)
    except Exception as exc:
        LOGGER.error("Failed to compute profiles: %s", exc)
        return 1

    if args.boundaries:
        try:
            with open(args.boundaries, "r", encoding="utf-8") as f:
                boundaries = json.load(f)
            phase_profiles = compute_phase_profile(angles_df, boundaries, fps=args.fps)
            profiles["phase_profiles"] = phase_profiles
        except Exception as exc:
            LOGGER.warning("Phase profile computation failed: %s", exc)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(profiles, f, indent=2, default=str)
            LOGGER.info("Saved profiles to %s", args.output)
        except Exception as exc:
            LOGGER.error("Failed to write profile JSON: %s", exc)
            return 1
    else:
        print(json.dumps(profiles, indent=2, default=str))

    if args.output_csv:
        vel_df = compute_velocity(angles_df, fps=args.fps)
        acc_df = compute_acceleration(vel_df, fps=args.fps)
        combined = pd.concat([angles_df, vel_df, acc_df], axis=1)
        try:
            combined.to_csv(args.output_csv)
            LOGGER.info("Saved time series to %s", args.output_csv)
        except Exception as exc:
            LOGGER.error("Failed to write CSV: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
