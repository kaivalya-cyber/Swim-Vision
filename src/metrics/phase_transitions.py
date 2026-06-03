# This file analyzes phase transition quality between block→flight and flight→entry.
"""Phase transition quality analysis for SwimVision — smoothness and energy transfer between phases."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Keypoint indices for transition analysis
SHOULDER_L, SHOULDER_R = 11, 12
HIP_L, HIP_R = 23, 24
KNEE_L, KNEE_R = 25, 26
ANKLE_L, ANKLE_R = 27, 28
ELBOW_L, ELBOW_R = 13, 14
WRIST_L, WRIST_R = 15, 16


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from src.metrics.dynamic_estimates import _midpoint as _mid
    return _mid(a, b)


def compute_transition_smoothness(
    keypoints: np.ndarray,
    transition_frame: int,
    window_before: int = 5,
    window_after: int = 5,
) -> Dict[str, Any]:
    """Compute smoothness of movement across a phase transition boundary.

    Measures velocity continuity and jerk (derivative of acceleration) across
    the transition point. Low jerk = smooth transition.

    Args:
        keypoints: Array [T, 33, 4].
        transition_frame: Frame index of the transition.
        window_before: Frames to analyze before transition.
        window_after: Frames to analyze after transition.

    Returns:
        Transition smoothness metrics.
    """
    T = keypoints.shape[0]
    start = max(0, transition_frame - window_before)
    end = min(T - 1, transition_frame + window_after)

    if end - start < 4:
        return {"error": "Insufficient frames around transition"}

    # Track hip and shoulder positions across the transition
    hip_y: List[float] = []
    shoulder_y: List[float] = []

    for t in range(start, end + 1):
        h_mid = _midpoint(keypoints[t, HIP_L, :2], keypoints[t, HIP_R, :2])
        s_mid = _midpoint(keypoints[t, SHOULDER_L, :2], keypoints[t, SHOULDER_R, :2])
        hip_y.append(float(h_mid[1]))
        shoulder_y.append(float(s_mid[1]))

    hip_arr = np.array(hip_y, dtype=np.float32)
    shoulder_arr = np.array(shoulder_y, dtype=np.float32)

    # Velocity (1st derivative)
    hip_vel = np.diff(hip_arr)
    shoulder_vel = np.diff(shoulder_arr)

    # Acceleration (2nd derivative)
    hip_acc = np.diff(hip_vel) if len(hip_vel) > 1 else np.array([0.0])
    shoulder_acc = np.diff(shoulder_vel) if len(shoulder_vel) > 1 else np.array([0.0])

    # Jerk (3rd derivative) — measure of smoothness
    hip_jerk = np.diff(hip_acc) if len(hip_acc) > 1 else np.array([0.0])
    shoulder_jerk = np.diff(shoulder_acc) if len(shoulder_acc) > 1 else np.array([0.0])

    # Mean absolute jerk (lower = smoother)
    hip_smoothness = float(np.mean(np.abs(hip_jerk))) if len(hip_jerk) > 0 else 0.0
    shoulder_smoothness = float(np.mean(np.abs(shoulder_jerk))) if len(shoulder_jerk) > 0 else 0.0

    # Combined smoothness score
    combined_jerk = (hip_smoothness + shoulder_smoothness) / 2.0

    if combined_jerk < 0.5:
        label = "EXCELLENT"
    elif combined_jerk < 1.5:
        label = "GOOD"
    elif combined_jerk < 3.0:
        label = "MODERATE"
    else:
        label = "JERKY"

    return {
        "transition_frame": transition_frame,
        "hip_smoothness_jerk": round(hip_smoothness, 4),
        "shoulder_smoothness_jerk": round(shoulder_smoothness, 4),
        "combined_jerk": round(combined_jerk, 4),
        "smoothness_label": label,
    }


def compute_velocity_continuity(
    keypoints: np.ndarray,
    transition_frame: int,
    window: int = 5,
) -> Dict[str, Any]:
    """Measure velocity preservation across a phase transition.

    High velocity retention = efficient energy transfer between phases.

    Args:
        keypoints: Array [T, 33, 4].
        transition_frame: Frame index of the transition.
        window: Frames on each side to compute velocity.

    Returns:
        Velocity continuity metrics.
    """
    T = keypoints.shape[0]
    pre_start = max(0, transition_frame - window)
    pre_end = transition_frame
    post_start = transition_frame
    post_end = min(T - 1, transition_frame + window)

    # COM velocity before transition
    pre_vels: List[float] = []
    for t in range(pre_start, pre_end):
        com = _midpoint(
            _midpoint(keypoints[t, SHOULDER_L, :2], keypoints[t, SHOULDER_R, :2]),
            _midpoint(keypoints[t, HIP_L, :2], keypoints[t, HIP_R, :2]),
        )
        pre_vels.append(float(com[0]))

    # COM velocity after transition
    post_vels: List[float] = []
    for t in range(post_start, post_end):
        com = _midpoint(
            _midpoint(keypoints[t, SHOULDER_L, :2], keypoints[t, SHOULDER_R, :2]),
            _midpoint(keypoints[t, HIP_L, :2], keypoints[t, HIP_R, :2]),
        )
        post_vels.append(float(com[0]))

    if len(pre_vels) < 2 or len(post_vels) < 2:
        return {"error": "Insufficient frames for velocity continuity"}

    pre_speed = abs(pre_vels[-1] - pre_vels[0]) / max(1, len(pre_vels) - 1)
    post_speed = abs(post_vels[-1] - post_vels[0]) / max(1, len(post_vels) - 1)

    if pre_speed > 1e-6:
        retention_pct = round(min(100.0, (post_speed / pre_speed) * 100.0), 1)
    else:
        retention_pct = None

    if retention_pct is not None:
        if retention_pct >= 95:
            label = "EXCELLENT-TRANSFER"
        elif retention_pct >= 85:
            label = "GOOD-TRANSFER"
        elif retention_pct >= 70:
            label = "MODERATE-LOSS"
        else:
            label = "ENERGY-LEAK"
    else:
        label = "N/A"

    return {
        "pre_transition_speed": round(pre_speed, 4),
        "post_transition_speed": round(post_speed, 4),
        "velocity_retention_pct": retention_pct,
        "transfer_label": label,
    }


def analyze_transitions(
    keypoints_path: str,
    boundaries_json: str,
) -> Dict[str, Any]:
    """Full phase transition quality analysis.

    Analyzes block→flight and flight→entry transition smoothness and velocity continuity.

    Args:
        keypoints_path: Path to keypoints .npy file.
        boundaries_json: Path to phase boundaries JSON.

    Returns:
        Transition analysis dictionary.
    """
    try:
        keypoints = np.load(keypoints_path)
    except Exception as exc:
        return {"error": f"Failed to load keypoints: {exc}"}

    try:
        with open(boundaries_json, "r", encoding="utf-8") as f:
            boundaries = json.load(f)
    except Exception as exc:
        return {"error": f"Failed to load boundaries: {exc}"}

    result: Dict[str, Any] = {}

    # Block → Flight transition (at flight_start)
    flight_start = boundaries.get("flight_start")
    if flight_start is not None:
        try:
            smooth = compute_transition_smoothness(keypoints, int(flight_start))
            continuity = compute_velocity_continuity(keypoints, int(flight_start))
            result["block_to_flight"] = {"smoothness": smooth, "velocity_continuity": continuity}
        except Exception as exc:
            result["block_to_flight"] = {"error": str(exc)}

    # Flight → Entry transition (at entry_start)
    entry_start = boundaries.get("entry_start")
    if entry_start is not None:
        try:
            smooth = compute_transition_smoothness(keypoints, int(entry_start))
            continuity = compute_velocity_continuity(keypoints, int(entry_start))
            result["flight_to_entry"] = {"smoothness": smooth, "velocity_continuity": continuity}
        except Exception as exc:
            result["flight_to_entry"] = {"error": str(exc)}

    # Overall transition quality
    quality_scores: List[float] = []
    for trans in ("block_to_flight", "flight_to_entry"):
        trans_data = result.get(trans, {})
        vel = trans_data.get("velocity_continuity", {}).get("velocity_retention_pct")
        if vel is not None:
            quality_scores.append(vel)

    if quality_scores:
        avg_retention = float(np.mean(quality_scores))
        result["overall_velocity_retention_pct"] = round(avg_retention, 1)
        if avg_retention >= 90:
            result["overall_transition_quality"] = "ELITE-TRANSITIONS"
        elif avg_retention >= 80:
            result["overall_transition_quality"] = "EFFICIENT-TRANSITIONS"
        else:
            result["overall_transition_quality"] = "NEEDS-IMPROVEMENT"

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for transition analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze phase transition quality from SwimVision keypoint data."
    )
    parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    parser.add_argument("--boundaries", required=True, help="Phase boundaries JSON path.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run transition analysis CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    result = analyze_transitions(args.keypoints, args.boundaries)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved transition analysis to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
