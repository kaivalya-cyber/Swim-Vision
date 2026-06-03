# This file analyzes underwater glide mechanics from SwimVision keypoint data post-entry.
"""Underwater glide analysis for SwimVision — depth, lateral deviation, and streamline effectiveness."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Keypoint indices
HIP_LEFT, HIP_RIGHT = 23, 24
SHOULDER_LEFT, SHOULDER_RIGHT = 11, 12
ANKLE_LEFT, ANKLE_RIGHT = 27, 28
WRIST_LEFT, WRIST_RIGHT = 15, 16
NOSE = 0


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the midpoint of two 2D points — imported from dynamic_estimates."""
    from src.metrics.dynamic_estimates import _midpoint as _mid
    return _mid(a, b)


def compute_glide_depth(
    keypoints: np.ndarray,
    entry_start: int,
    num_glide_frames: int = 30,
) -> Dict[str, Any]:
    """Track depth maintenance during the underwater glide phase.

    Tracks hip midpoint vertical position to estimate depth stability.

    Args:
        keypoints: Array [T, 33, 4].
        entry_start: Frame index where entry begins.
        num_glide_frames: Number of frames to analyze post-entry.

    Returns:
        Depth metrics for the glide phase.
    """
    T = keypoints.shape[0]
    end_frame = min(entry_start + num_glide_frames, T)
    if end_frame <= entry_start or entry_start >= T:
        return {"error": "Insufficient glide frames for depth analysis"}

    hip_y_positions: List[float] = []

    for t in range(entry_start, end_frame):
        hip_l = keypoints[t, HIP_LEFT, 1]
        hip_r = keypoints[t, HIP_RIGHT, 1]
        if hip_l > 0 and hip_r > 0:
            hip_y = float((hip_l + hip_r) / 2.0)
        else:
            hip_y = float(max(hip_l, hip_r))
        hip_y_positions.append(hip_y)

    if len(hip_y_positions) < 3:
        return {"error": "Too few valid hip positions"}

    arr = np.array(hip_y_positions, dtype=np.float32)

    # Depth metrics (y increases downward in image coordinates)
    depth_start = float(arr[0])
    depth_end = float(arr[-1])
    depth_change = depth_end - depth_start  # positive = deeper
    depth_range = float(np.max(arr) - np.min(arr))
    depth_std = float(np.std(arr, ddof=1))

    # Rate of descent (px/frame)
    if len(arr) >= 2:
        x = np.arange(len(arr), dtype=np.float32)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(arr))
        numerator = float(np.sum((x - x_mean) * (arr - y_mean)))
        denominator = float(np.sum((x - x_mean) ** 2))
        descent_rate = numerator / denominator if denominator > 0 else 0.0
    else:
        descent_rate = 0.0

    # Stability: low std = stable depth maintenance
    if depth_std < 2.0:
        stability_label = "EXCELLENT"
    elif depth_std < 5.0:
        stability_label = "GOOD"
    elif depth_std < 10.0:
        stability_label = "MODERATE"
    else:
        stability_label = "POOR"

    return {
        "num_glide_frames": len(arr),
        "depth_start_px": round(depth_start, 3),
        "depth_end_px": round(depth_end, 3),
        "depth_change_px": round(depth_change, 3),
        "depth_range_px": round(depth_range, 3),
        "depth_std_px": round(depth_std, 3),
        "descent_rate_px_per_frame": round(descent_rate, 4),
        "depth_stability": stability_label,
    }


def compute_lateral_deviation(
    keypoints: np.ndarray,
    entry_start: int,
    num_glide_frames: int = 30,
) -> Dict[str, Any]:
    """Track lateral (x-axis) deviation during underwater glide.

    Measures how much the swimmer drifts sideways from the entry line.

    Args:
        keypoints: Array [T, 33, 4].
        entry_start: Frame index where entry begins.
        num_glide_frames: Number of frames to analyze.

    Returns:
        Lateral deviation metrics.
    """
    T = keypoints.shape[0]
    end_frame = min(entry_start + num_glide_frames, T)
    if end_frame <= entry_start or entry_start >= T:
        return {"error": "Insufficient glide frames for lateral analysis"}

    nose_x: List[float] = []
    hip_x: List[float] = []

    for t in range(entry_start, end_frame):
        n_x = float(keypoints[t, NOSE, 0])
        hip_l = keypoints[t, HIP_LEFT, 0]
        hip_r = keypoints[t, HIP_RIGHT, 0]
        h_x = float((hip_l + hip_r) / 2.0) if hip_l > 0 and hip_r > 0 else float(max(hip_l, hip_r))

        nose_x.append(n_x) if n_x > 0 else None
        hip_x.append(h_x) if h_x > 0 else None

    if len(hip_x) < 3:
        return {"error": "Too few valid lateral positions"}

    hip_arr = np.array(hip_x, dtype=np.float32)

    # Reference line = entry start position
    ref_x = float(hip_arr[0])
    deviations = np.abs(hip_arr - ref_x)

    max_deviation = float(np.max(deviations))
    mean_deviation = float(np.mean(deviations))
    deviation_std = float(np.std(deviations, ddof=1))

    if max_deviation < 3.0:
        drift_label = "STRAIGHT"
    elif max_deviation < 8.0:
        drift_label = "MINOR-DRIFT"
    elif max_deviation < 15.0:
        drift_label = "MODERATE-DRIFT"
    else:
        drift_label = "SIGNIFICANT-DRIFT"

    return {
        "num_frames": len(hip_arr),
        "max_lateral_deviation_px": round(max_deviation, 3),
        "mean_lateral_deviation_px": round(mean_deviation, 3),
        "lateral_deviation_std_px": round(deviation_std, 3),
        "drift_classification": drift_label,
    }


def compute_streamline_effectiveness(
    angles_df: pd.DataFrame,
    entry_start: Optional[int] = None,
    num_glide_frames: int = 30,
) -> Dict[str, Any]:
    """Compute underwater streamline effectiveness from angle data.

    Combines body linearity, streamline angle, and elbow lock maintenance
    during the glide phase.

    Args:
        angles_df: DataFrame with angle columns.
        entry_start: Optional entry start frame index.
        num_glide_frames: Number of frames to analyze.

    Returns:
        Streamline effectiveness metrics.
    """
    if entry_start is not None and entry_start < len(angles_df):
        df_slice = angles_df.iloc[entry_start:entry_start + num_glide_frames]
    else:
        df_slice = angles_df.iloc[-num_glide_frames:] if len(angles_df) > num_glide_frames else angles_df

    if df_slice.empty:
        return {"error": "No glide data available"}

    components: Dict[str, Dict[str, Any]] = {}

    # Body linearity
    if "body_linearity" in angles_df.columns:
        vals = pd.to_numeric(df_slice["body_linearity"], errors="coerce").dropna()
        if len(vals) > 0:
            components["body_linearity"] = {
                "mean": round(float(vals.mean()), 3),
                "std": round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else 0.0,
                "min": round(float(vals.min()), 3),
            }

    # Streamline angle
    if "streamline_angle" in angles_df.columns:
        vals = pd.to_numeric(df_slice["streamline_angle"], errors="coerce").dropna()
        if len(vals) > 0:
            components["streamline_angle"] = {
                "mean": round(float(vals.mean()), 3),
                "std": round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else 0.0,
                "max": round(float(vals.max()), 3),
            }

    # Elbow lock
    for col in ("elbow_lock_angle", "elbow_extension"):
        if col in angles_df.columns:
            vals = pd.to_numeric(df_slice[col], errors="coerce").dropna()
            if len(vals) > 0:
                components["elbow_lock"] = {
                    "mean": round(float(vals.mean()), 3),
                    "std": round(float(vals.std(ddof=1)), 3) if len(vals) > 1 else 0.0,
                    "min": round(float(vals.min()), 3),
                }
                break

    if not components:
        return {"error": "No streamline metrics available in angle data"}

    # Compute overall effectiveness score
    scores = []
    weights = {"body_linearity": 0.4, "streamline_angle": 0.35, "elbow_lock": 0.25}

    for comp_name, comp_data in components.items():
        if comp_name == "body_linearity":
            # Higher = better, 0-1 range
            score = min(100.0, comp_data.get("mean", 0.5) * 100.0)
        elif comp_name == "streamline_angle":
            # Lower = better, <10° excellent
            mean_angle = comp_data.get("mean", 15)
            score = max(20.0, 100.0 - mean_angle * 5.0)
        elif comp_name == "elbow_lock":
            # Closer to 180° = better
            mean_angle = comp_data.get("mean", 160)
            score = max(30.0, 100.0 - abs(180.0 - mean_angle) * 1.5)
        else:
            score = 50.0
        scores.append((comp_name, score, weights.get(comp_name, 0.33)))

    total_weight = sum(w for _, _, w in scores)
    if total_weight > 0:
        overall = sum(s * w for _, s, w in scores) / total_weight
    else:
        overall = 50.0

    overall = round(overall, 1)

    if overall >= 85:
        label = "EXCELLENT-GLIDE"
    elif overall >= 70:
        label = "GOOD-GLIDE"
    elif overall >= 50:
        label = "FAIR-GLIDE"
    else:
        label = "POOR-GLIDE"

    return {
        "glide_streamline_score": overall,
        "glide_streamline_label": label,
        "components": {k: v for k, v in components.items()},
    }


def estimate_glide_distance(
    descent_rate: float,
    num_frames: int,
    body_length_px: Optional[float] = None,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Estimate approximate glide distance from descent rate and frame count.

    Uses a simplified model: glide distance ≈ horizontal displacement estimated
    from typical swim start kinematics.

    Args:
        descent_rate: Descent rate in px/frame.
        num_frames: Number of glide frames.
        body_length_px: Optional body length in pixels for scaling.
        fps: Frame rate.

    Returns:
        Estimated glide distance metrics.
    """
    glide_duration_sec = num_frames / fps

    # Rough estimate: horizontal speed ≈ 2-3 m/s during glide
    # Using descent rate as a proxy for overall speed (steeper = faster entry = longer glide)
    estimated_speed_ms = 1.5 + abs(descent_rate) * 0.01
    estimated_distance_m = estimated_speed_ms * glide_duration_sec

    return {
        "glide_duration_sec": round(glide_duration_sec, 2),
        "estimated_speed_ms": round(estimated_speed_ms, 2),
        "estimated_glide_distance_m": round(estimated_distance_m, 2),
        "num_frames": num_frames,
    }


def analyze_glide(
    keypoints_path: str,
    angles_csv: str,
    boundaries_json: Optional[str] = None,
    num_glide_frames: int = 30,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Full underwater glide analysis from keypoints and angle data.

    Args:
        keypoints_path: Path to keypoints .npy file.
        angles_csv: Path to angles CSV.
        boundaries_json: Optional phase boundaries JSON.
        num_glide_frames: Number of post-entry frames to analyze.
        fps: Video frame rate.

    Returns:
        Comprehensive glide analysis dictionary.
    """
    result: Dict[str, Any] = {}

    # Load data
    try:
        keypoints = np.load(keypoints_path)
    except Exception as exc:
        return {"error": f"Failed to load keypoints: {exc}"}

    try:
        df = pd.read_csv(angles_csv)
    except Exception as exc:
        return {"error": f"Failed to load angles CSV: {exc}"}

    # Determine entry start frame
    entry_start = 0
    if boundaries_json:
        try:
            with open(boundaries_json, "r", encoding="utf-8") as f:
                boundaries = json.load(f)
            entry_start = int(boundaries.get("entry_start", 0))
        except Exception:
            pass

    # If no boundaries, estimate entry at 60% through the clip (typical swim start)
    if entry_start == 0:
        entry_start = int(keypoints.shape[0] * 0.6)

    # 1. Depth analysis
    try:
        result["depth"] = compute_glide_depth(keypoints, entry_start, num_glide_frames)
    except Exception as exc:
        result["depth"] = {"error": str(exc)}

    # 2. Lateral deviation
    try:
        result["lateral_deviation"] = compute_lateral_deviation(keypoints, entry_start, num_glide_frames)
    except Exception as exc:
        result["lateral_deviation"] = {"error": str(exc)}

    # 3. Streamline effectiveness
    try:
        result["streamline_effectiveness"] = compute_streamline_effectiveness(
            df, entry_start=entry_start, num_glide_frames=num_glide_frames
        )
    except Exception as exc:
        result["streamline_effectiveness"] = {"error": str(exc)}

    # 4. Glide distance estimate
    depth_data = result.get("depth", {})
    descent_rate = depth_data.get("descent_rate_px_per_frame", 0)
    depth_frames = depth_data.get("num_glide_frames", num_glide_frames)
    try:
        result["glide_distance"] = estimate_glide_distance(
            descent_rate, depth_frames, fps=fps
        )
    except Exception as exc:
        result["glide_distance"] = {"error": str(exc)}

    # Overall glide quality
    scores = []
    depth_stab = result.get("depth", {}).get("depth_stability")
    drift = result.get("lateral_deviation", {}).get("drift_classification")
    streamline = result.get("streamline_effectiveness", {}).get("glide_streamline_label")

    score_map = {
        "EXCELLENT": 100, "EXCELLENT-GLIDE": 100, "GOOD": 80, "GOOD-GLIDE": 80,
        "STRAIGHT": 100, "MINOR-DRIFT": 75, "MODERATE-DRIFT": 50,
        "FAIR": 60, "FAIR-GLIDE": 60, "POOR": 30, "POOR-GLIDE": 30,
        "MODERATE": 50, "SIGNIFICANT-DRIFT": 20,
    }

    for label in [depth_stab, drift, streamline]:
        if label:
            scores.append(score_map.get(label, 50))

    if scores:
        avg_score = float(np.mean(scores))
        result["overall_glide_score"] = round(avg_score, 1)
        if avg_score >= 85:
            result["overall_glide_label"] = "ELITE-GLIDE"
        elif avg_score >= 70:
            result["overall_glide_label"] = "EFFICIENT-GLIDE"
        elif avg_score >= 50:
            result["overall_glide_label"] = "ADEQUATE-GLIDE"
        else:
            result["overall_glide_label"] = "NEEDS-IMPROVEMENT"

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for glide analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze underwater glide mechanics from SwimVision data."
    )
    parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    parser.add_argument("--angles", required=True, help="Path to angles CSV.")
    parser.add_argument("--boundaries", help="Phase boundaries JSON path.")
    parser.add_argument("--glide-frames", type=int, default=30, help="Frames to analyze post-entry.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run glide analysis CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    result = analyze_glide(
        args.keypoints,
        args.angles,
        boundaries_json=args.boundaries,
        num_glide_frames=args.glide_frames,
        fps=args.fps,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved glide analysis to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
