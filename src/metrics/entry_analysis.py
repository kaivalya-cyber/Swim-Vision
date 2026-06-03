# This file performs dedicated entry-phase analysis for swim start biomechanics.
"""Entry analysis module — splash score, depth trajectory, streamline quality, and velocity retention."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.metrics.dynamic_estimates import _midpoint


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the midpoint of two 2D points — imported from dynamic_estimates."""
    from src.metrics.dynamic_estimates import _midpoint as _mid
    return _mid(a, b)


def compute_splash_score(
    entry_angle: float,
    entry_velocity: Optional[float] = None,
    body_linearity: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute a splash/efficiency score for the entry phase.

    Lower entry angle (more horizontal) and higher linearity yield better scores.
    Ideal: entry_angle 30-45°, good linearity.

    Args:
        entry_angle: Measured entry angle in degrees.
        entry_velocity: Optional entry velocity (normalized).
        body_linearity: Optional body linearity score (0-1).

    Returns:
        Splash score dictionary with 0-100 rating and qualitative label.
    """
    # Convert to 0-100 score where 100 = minimal splash
    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    # Entry angle: optimal range is 30-45 degrees from horizontal
    if entry_angle is not None:
        if 30 <= entry_angle <= 45:
            angle_score = 100.0
        elif entry_angle < 30:
            angle_score = max(40.0, 100.0 - (30.0 - entry_angle) * 3.0)
        elif entry_angle <= 60:
            angle_score = max(40.0, 100.0 - (entry_angle - 45.0) * 2.5)
        else:
            angle_score = max(20.0, 100.0 - (entry_angle - 45.0) * 2.0)
        components["angle"] = round(angle_score, 1)
        weights["angle"] = 0.40

    # Velocity: lower entry velocity reduces splash
    if entry_velocity is not None:
        # Normalized velocity: <0.5 excellent, 0.5-1.0 good, >1.0 high
        vel_score = max(20.0, 100.0 - abs(entry_velocity) * 50.0)
        components["velocity"] = round(vel_score, 1)
        weights["velocity"] = 0.30

    # Body linearity: higher is better
    if body_linearity is not None:
        lin_score = min(100.0, body_linearity * 100.0) if 0 <= body_linearity <= 1 else 50.0
        components["linearity"] = round(lin_score, 1)
        weights["linearity"] = 0.30

    if not components:
        return {"splash_score": None, "label": "N/A"}

    # Weighted average
    total_weight = sum(weights.values())
    if total_weight > 0:
        score = sum(components[k] * weights[k] for k in components) / total_weight
    else:
        score = float(np.mean(list(components.values())))

    score = round(score, 1)

    if score >= 85:
        label = "EXCELLENT"
    elif score >= 70:
        label = "GOOD"
    elif score >= 50:
        label = "FAIR"
    else:
        label = "POOR"

    return {
        "splash_score": score,
        "label": label,
        "components": components,
        "weights": weights,
    }


def compute_depth_trajectory(
    keypoints: np.ndarray,
    phase_boundaries: Dict[str, int],
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Estimate depth trajectory during the entry phase.

    Tracks the vertical (y) position of the hip midpoint across entry frames
    to estimate descent angle and rate.

    Args:
        keypoints: Array with shape [T, 33, 4].
        phase_boundaries: Phase boundary dictionary with entry_start, entry_end.
        fps: Video frame rate.

    Returns:
        Depth trajectory metrics.
    """
    if "entry_start" not in phase_boundaries or "entry_end" not in phase_boundaries:
        return {"error": "Missing entry phase boundaries"}

    entry_start = int(phase_boundaries["entry_start"])
    entry_end = int(phase_boundaries["entry_end"])
    if entry_end <= entry_start or entry_start >= keypoints.shape[0]:
        return {"error": f"Invalid entry boundaries: {entry_start}-{entry_end}"}

    end_idx = min(entry_end, keypoints.shape[0] - 1)
    frames = range(entry_start, end_idx + 1)

    hip_y_positions: List[float] = []
    shoulder_y_positions: List[float] = []

    for t in frames:
        hip = keypoints[t, 23, 1]  # Left hip y
        hip_r = keypoints[t, 24, 1]  # Right hip y
        shoulder = keypoints[t, 11, 1]
        shoulder_r = keypoints[t, 12, 1]

        hip_y = float((hip + hip_r) / 2.0) if hip > 0 and hip_r > 0 else float(max(hip, hip_r))
        sh_y = float((shoulder + shoulder_r) / 2.0) if shoulder > 0 and shoulder_r > 0 else float(max(shoulder, shoulder_r))

        hip_y_positions.append(hip_y)
        shoulder_y_positions.append(sh_y)

    if len(hip_y_positions) < 2:
        return {"error": "Too few entry frames"}

    hip_arr = np.array(hip_y_positions, dtype=np.float32)
    shoulder_arr = np.array(shoulder_y_positions, dtype=np.float32)

    # Descent rate (px per frame, lower = deeper)
    descent_total = float(hip_arr[-1] - hip_arr[0])
    descent_per_frame = descent_total / len(hip_arr) if len(hip_arr) > 1 else 0.0

    # Descent angle approximation (assuming camera perpendicular to water)
    # tan(theta) ≈ vertical_displacement / horizontal_displacement
    horizontal_span = len(hip_arr)  # frames as proxy for horizontal distance
    if horizontal_span > 0:
        descent_angle_rad = np.arctan2(abs(descent_total), float(horizontal_span))
        descent_angle_deg = float(np.degrees(descent_angle_rad))
    else:
        descent_angle_deg = 0.0

    # Body angle relative to horizontal during entry
    body_angles: List[float] = []
    for t_idx in range(min(len(hip_arr), len(shoulder_arr))):
        dy = float(hip_arr[t_idx] - shoulder_arr[t_idx])
        dx = 1.0  # Assume normalized horizontal
        angle = float(np.degrees(np.arctan2(dy, dx)))
        body_angles.append(angle)

    mean_body_angle = float(np.mean(body_angles)) if body_angles else None

    return {
        "num_entry_frames": len(hip_arr),
        "descent_total_px": round(descent_total, 3),
        "descent_per_frame_px": round(descent_per_frame, 4),
        "descent_angle_deg": round(descent_angle_deg, 2),
        "mean_body_angle_deg": round(mean_body_angle, 2) if mean_body_angle is not None else None,
        "hip_y_start": round(float(hip_arr[0]), 3),
        "hip_y_end": round(float(hip_arr[-1]), 3),
    }


def compute_streamline_quality(
    angles_df: pd.DataFrame,
    entry_start_frame: Optional[int] = None,
    entry_end_frame: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute streamline quality metrics from entry phase angle data.

    Streamline quality = combination of body linearity, elbow lock, and streamline angle.

    Args:
        angles_df: DataFrame with angle columns.
        entry_start_frame: Optional entry start frame index.
        entry_end_frame: Optional entry end frame index.

    Returns:
        Streamline quality score and per-component breakdown.
    """
    # Determine entry range
    if entry_start_frame is not None and entry_end_frame is not None:
        df_slice = angles_df.iloc[entry_start_frame:entry_end_frame + 1]
    else:
        df_slice = angles_df

    if df_slice.empty:
        return {"error": "No entry data available"}

    components: Dict[str, float] = {}
    weights: Dict[str, float] = {}

    # Body linearity (higher = better, 0-1 range assumed)
    if "body_linearity" in angles_df.columns:
        vals = pd.to_numeric(df_slice["body_linearity"], errors="coerce").dropna()
        if len(vals) > 0:
            lin_mean = float(vals.mean())
            components["body_linearity"] = round(lin_mean * 100.0, 1)
            weights["body_linearity"] = 0.35

    # Elbow lock angle (closer to 180° = better)
    for col in ("elbow_lock_angle", "elbow_extension"):
        if col in angles_df.columns:
            vals = pd.to_numeric(df_slice[col], errors="coerce").dropna()
            if len(vals) > 0:
                mean_angle = float(vals.mean())
                # 180° is perfect lock, score decreases with deviation
                score = max(30.0, 100.0 - abs(180.0 - mean_angle) * 1.5)
                components["elbow_lock"] = round(score, 1)
                weights["elbow_lock"] = 0.30
                break

    # Streamline angle (lower = better, <15° excellent)
    if "streamline_angle" in angles_df.columns:
        vals = pd.to_numeric(df_slice["streamline_angle"], errors="coerce").dropna()
        if len(vals) > 0:
            mean_angle = float(vals.mean())
            score = max(20.0, 100.0 - mean_angle * 3.0)
            components["streamline_angle"] = round(score, 1)
            weights["streamline_angle"] = 0.35

    if not components:
        return {"error": "No streamline metrics available"}

    # Weighted score
    total_weight = sum(weights.values())
    if total_weight > 0:
        overall = sum(components[k] * weights[k] for k in components) / total_weight
    else:
        overall = float(np.mean(list(components.values())))

    overall = round(overall, 1)

    if overall >= 85:
        label = "EXCELLENT-STREAMLINE"
    elif overall >= 70:
        label = "GOOD-STREAMLINE"
    elif overall >= 50:
        label = "FAIR-STREAMLINE"
    else:
        label = "POOR-STREAMLINE"

    return {
        "streamline_quality_score": overall,
        "streamline_label": label,
        "components": components,
        "weights": weights,
    }


def compute_velocity_retention(
    entry_start_frame: int,
    entry_end_frame: int,
    com_velocities: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Estimate velocity retention through the entry phase.

    Compares velocity at entry start to velocity at entry end.

    Args:
        entry_start_frame: Frame index where entry begins.
        entry_end_frame: Frame index where entry ends.
        com_velocities: Optional list of per-frame COM velocities.

    Returns:
        Velocity retention metrics.
    """
    if com_velocities is None or len(com_velocities) == 0:
        return {"error": "No velocity data available"}

    start_idx = min(entry_start_frame, len(com_velocities) - 1)
    end_idx = min(entry_end_frame, len(com_velocities) - 1)

    if end_idx <= start_idx:
        return {"error": "Invalid entry frame range for velocity retention"}

    vel_start = com_velocities[start_idx]
    vel_end = com_velocities[end_idx]

    if vel_start is None or abs(vel_start) < 1e-9:
        return {"error": "Invalid entry start velocity"}

    retention_pct = round((vel_end / vel_start) * 100.0, 1) if vel_start else None

    if retention_pct is not None:
        if retention_pct >= 90:
            label = "EXCELLENT-RETENTION"
        elif retention_pct >= 75:
            label = "GOOD-RETENTION"
        elif retention_pct >= 60:
            label = "MODERATE-LOSS"
        else:
            label = "SIGNIFICANT-LOSS"
    else:
        label = "N/A"

    return {
        "entry_start_velocity": round(vel_start, 4) if vel_start else None,
        "entry_end_velocity": round(vel_end, 4) if vel_end else None,
        "velocity_retention_pct": retention_pct,
        "retention_label": label,
    }


def analyze_entry(
    angles_csv: str,
    keypoints_path: Optional[str] = None,
    boundaries_json: Optional[str] = None,
    dynamic_json: Optional[str] = None,
    vel_accel_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Full entry-phase analysis.

    Args:
        angles_csv: Path to joint angles CSV.
        keypoints_path: Optional path to keypoints .npy file for depth trajectory.
        boundaries_json: Optional phase boundaries JSON.
        dynamic_json: Optional dynamic estimates JSON.
        vel_accel_json: Optional velocity/acceleration JSON.

    Returns:
        Comprehensive entry analysis dictionary.
    """
    result: Dict[str, Any] = {}

    # Load data
    try:
        df = pd.read_csv(angles_csv)
    except Exception as exc:
        return {"error": f"Failed to load angles CSV: {exc}"}

    boundaries: Dict[str, int] = {}
    if boundaries_json:
        try:
            with open(boundaries_json, "r", encoding="utf-8") as f:
                boundaries = json.load(f)
        except Exception as exc:
            LOGGER.warning("Failed to load boundaries: %s", exc)

    entry_start = boundaries.get("entry_start", 0)
    entry_end = boundaries.get("entry_end", len(df) - 1)

    # Extract entry phase metrics
    entry_angle = None
    body_linearity_mean = None
    if "entry_angle" in df.columns:
        try:
            vals = pd.to_numeric(df.iloc[entry_start:entry_end + 1]["entry_angle"], errors="coerce").dropna()
            if len(vals) > 0:
                entry_angle = float(vals.mean())
        except Exception:
            pass

    if "body_linearity" in df.columns:
        try:
            vals = pd.to_numeric(df.iloc[entry_start:entry_end + 1]["body_linearity"], errors="coerce").dropna()
            if len(vals) > 0:
                body_linearity_mean = float(vals.mean())
        except Exception:
            pass

    # 1. Splash Score
    try:
        result["splash_score"] = compute_splash_score(
            entry_angle=entry_angle,
            body_linearity=body_linearity_mean,
        )
    except Exception as exc:
        result["splash_score"] = {"error": str(exc)}

    # 2. Depth Trajectory (requires keypoints)
    if keypoints_path:
        try:
            kp = np.load(keypoints_path)
            if boundaries:
                result["depth_trajectory"] = compute_depth_trajectory(kp, boundaries)
        except Exception as exc:
            LOGGER.warning("Depth trajectory failed: %s", exc)

    # 3. Streamline Quality
    try:
        result["streamline_quality"] = compute_streamline_quality(
            df,
            entry_start_frame=entry_start if entry_end > entry_start else None,
            entry_end_frame=entry_end if entry_end > entry_start else None,
        )
    except Exception as exc:
        result["streamline_quality"] = {"error": str(exc)}

    # 4. Velocity Retention
    com_velocities = None
    if dynamic_json:
        try:
            with open(dynamic_json, "r", encoding="utf-8") as f:
                dynamic = json.load(f)
            com_data = dynamic.get("center_of_mass_velocity", {})
            com_velocities = com_data.get("velocity_time_series")
        except Exception:
            pass

    if not com_velocities and vel_accel_json:
        try:
            with open(vel_accel_json, "r", encoding="utf-8") as f:
                va_data = json.load(f)
            # Try to get per-phase velocity profiles
            phase_profiles = va_data.get("phase_profiles", {})
            if phase_profiles:
                # Extract velocity time series from first available phase
                for phase_name in ("entry_phase", "flight_phase", "block_phase"):
                    if phase_name in phase_profiles:
                        phase_data = phase_profiles[phase_name]
                        # Collect all per-column velocity time series
                        velocities = []
                        for col, col_data in phase_data.items():
                            if isinstance(col_data, dict):
                                ts = col_data.get("velocity_time_series", [])
                                if ts:
                                    velocities.extend(ts)
                                    break
                        if velocities:
                            com_velocities = velocities
                            break
        except Exception:
            pass

    if com_velocities and entry_end > entry_start:
        try:
            result["velocity_retention"] = compute_velocity_retention(
                entry_start, entry_end, com_velocities
            )
        except Exception as exc:
            result["velocity_retention"] = {"error": str(exc)}

    # Overall entry quality summary
    scores = []
    splash = result.get("splash_score", {}).get("splash_score")
    streamline = result.get("streamline_quality", {}).get("streamline_quality_score")
    retention = result.get("velocity_retention", {}).get("velocity_retention_pct")

    if splash is not None:
        scores.append(("splash", splash))
    if streamline is not None:
        scores.append(("streamline", streamline))
    if retention is not None:
        scores.append(("retention", retention))

    if scores:
        avg_score = float(np.mean([s for _, s in scores]))
        result["overall_entry_score"] = round(avg_score, 1)
        if avg_score >= 80:
            result["overall_entry_label"] = "ELITE-ENTRY"
        elif avg_score >= 65:
            result["overall_entry_label"] = "EFFICIENT-ENTRY"
        elif avg_score >= 50:
            result["overall_entry_label"] = "ADEQUATE-ENTRY"
        else:
            result["overall_entry_label"] = "NEEDS-IMPROVEMENT"

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for entry analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze entry-phase mechanics from SwimVision data."
    )
    parser.add_argument("--angles", required=True, help="Path to angles CSV.")
    parser.add_argument("--keypoints", help="Optional keypoints .npy path for depth trajectory.")
    parser.add_argument("--boundaries", help="Phase boundaries JSON path.")
    parser.add_argument("--dynamic", help="Dynamic estimates JSON path.")
    parser.add_argument("--vel-accel", help="Velocity/acceleration JSON path.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run entry analysis CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = analyze_entry(
            args.angles,
            keypoints_path=args.keypoints,
            boundaries_json=args.boundaries,
            dynamic_json=args.dynamic,
            vel_accel_json=args.vel_accel,
        )
    except Exception as exc:
        LOGGER.error("Entry analysis failed: %s", exc)
        return 1

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved entry analysis to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
