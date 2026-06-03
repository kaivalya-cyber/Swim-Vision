# This file computes dynamic biomechanical estimates from SwimVision keypoint sequences.
"""Dynamic estimates for SwimVision: COM velocity, alignment, and entry trajectory."""

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

# MediaPipe keypoint indices
HEAD_INDICES = {"nose": 0, "left_ear": 7, "right_ear": 8}
SHOULDER_INDICES = {"left": 11, "right": 12}
HIP_INDICES = {"left": 23, "right": 24}
ANKLE_INDICES = {"left": 27, "right": 28}

# Approximate body segment mass fractions (from anthropometric tables)
SEGMENT_MASS_FRACTIONS = {
    "head": 0.069,
    "torso": 0.430,
    "upper_arm": 0.027 * 2,
    "forearm": 0.016 * 2,
    "thigh": 0.100 * 2,
    "shank": 0.045 * 2,
}


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the midpoint of two 2D points."""
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) / 2.0


def compute_com_velocity(
    keypoints: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Estimate center-of-mass horizontal velocity from key body landmarks.

    Uses a simplified weighted-average of shoulder, hip, and head midpoints.
    Velocity is computed as px/frame displacement converted to approximate m/s.

    Args:
        keypoints: Array with shape [T, 33, 4].
        fps: Video frame rate.

    Returns:
        Dictionary with time-series velocity and summary statistics.
    """
    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}")

    T = keypoints.shape[0]
    com_x = np.zeros(T, dtype=np.float32)

    for t in range(T):
        head_mid = _midpoint(keypoints[t, 0, :2], keypoints[t, 0, :2])  # nose
        shoulder_mid = _midpoint(keypoints[t, 11, :2], keypoints[t, 12, :2])
        hip_mid = _midpoint(keypoints[t, 23, :2], keypoints[t, 24, :2])
        # Weighted COM approximation (upper body focus for swim starts)
        com_x[t] = float(
            head_mid[0] * 0.15 + shoulder_mid[0] * 0.35 + hip_mid[0] * 0.50
        )

    # Velocity as displacement between consecutive frames
    # Scale factor: assume 2m body height ≈ frame height, so px velocity ≈ m/s * frames
    # For normalized coordinates (0-1 range), velocity is in body-lengths per second
    com_vel = np.zeros_like(com_x)
    dt = 1.0 / fps
    for t in range(1, T):
        com_vel[t] = (com_x[t] - com_x[t - 1]) / dt

    # Smoothed velocity
    kernel = np.ones(min(5, T), dtype=np.float32) / min(5, T)
    smoothed_vel = np.convolve(com_vel, kernel, mode="same") if T >= 5 else com_vel

    valid_vel = smoothed_vel[np.isfinite(smoothed_vel)]

    return {
        "mean_velocity_norm_per_s": float(np.mean(valid_vel)) if len(valid_vel) > 0 else None,
        "max_velocity_norm_per_s": float(np.max(valid_vel)) if len(valid_vel) > 0 else None,
        "peak_velocity_frame": int(np.argmax(np.abs(smoothed_vel))) if len(valid_vel) > 0 else None,
        "velocity_time_series": [float(v) if np.isfinite(v) else None for v in smoothed_vel],
    }


def compute_head_alignment(
    keypoints: np.ndarray,
    phase_boundaries: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Compute head alignment deviation from body line during flight and entry.

    Args:
        keypoints: Array with shape [T, 33, 4].
        phase_boundaries: Optional phase boundaries for per-phase analysis.

    Returns:
        Dictionary with head alignment metrics.
    """
    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}")

    T = keypoints.shape[0]
    deviations = []

    for t in range(T):
        nose = keypoints[t, 0, :2]
        shoulder_mid = _midpoint(keypoints[t, 11, :2], keypoints[t, 12, :2])
        hip_mid = _midpoint(keypoints[t, 23, :2], keypoints[t, 24, :2])

        # Body line vector (shoulder → hip)
        body_line = hip_mid - shoulder_mid
        # Head vector (shoulder → nose)
        head_vec = nose - shoulder_mid

        body_len = float(np.linalg.norm(body_line))
        if body_len < 1e-6:
            deviations.append(None)
            continue

        # Cross product to measure lateral deviation
        cross = float(np.cross(body_line, head_vec))
        deviations.append(abs(cross) / body_len)

    valid_devs = [d for d in deviations if d is not None and np.isfinite(d)]

    result: Dict[str, Any] = {
        "mean_head_deviation": float(np.mean(valid_devs)) if valid_devs else None,
        "max_head_deviation": float(np.max(valid_devs)) if valid_devs else None,
    }

    # Per-phase analysis if boundaries provided
    if phase_boundaries:
        for phase_name, start_key, end_key in [
            ("flight_phase", "flight_start", "flight_end"),
            ("entry_phase", "entry_start", "entry_end"),
        ]:
            if start_key in phase_boundaries and end_key in phase_boundaries:
                start_idx = int(phase_boundaries[start_key])
                end_idx = int(phase_boundaries[end_key])
                phase_devs = [
                    d for i, d in enumerate(deviations)
                    if start_idx <= i <= end_idx and d is not None and np.isfinite(d)
                ]
                if phase_devs:
                    result[f"{phase_name}_mean"] = float(np.mean(phase_devs))
                    result[f"{phase_name}_max"] = float(np.max(phase_devs))

    return result


def compute_entry_trajectory(
    keypoints: np.ndarray,
    phase_boundaries: Dict[str, int],
) -> Dict[str, Any]:
    """Compute entry trajectory angle from the final flight frames through entry.

    Args:
        keypoints: Array with shape [T, 33, 4].
        phase_boundaries: Phase boundary dictionary.

    Returns:
        Dictionary with entry trajectory metrics.
    """
    if "entry_start" not in phase_boundaries or "entry_end" not in phase_boundaries:
        return {}

    entry_start = int(phase_boundaries["entry_start"])
    entry_end = int(phase_boundaries["entry_end"])
    if entry_end <= entry_start:
        return {}

    # First 5 frames of entry
    first_n = min(5, entry_end - entry_start)
    trajectory_angles = []

    for t in range(entry_start, entry_start + first_n):
        if t >= keypoints.shape[0]:
            break
        wrist_mid = _midpoint(keypoints[t, 15, :2], keypoints[t, 16, :2])
        shoulder_mid = _midpoint(keypoints[t, 11, :2], keypoints[t, 12, :2])

        entry_vec = wrist_mid - shoulder_mid
        horizontal = np.array([1.0, 0.0], dtype=np.float32)

        vec_len = float(np.linalg.norm(entry_vec))
        if vec_len < 1e-6:
            continue

        cos_angle = np.clip(np.dot(entry_vec, horizontal) / vec_len, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(abs(cos_angle))))
        trajectory_angles.append(angle)

    if not trajectory_angles:
        return {}

    return {
        "mean_trajectory_angle_deg": round(float(np.mean(trajectory_angles)), 2),
        "min_trajectory_angle_deg": round(float(np.min(trajectory_angles)), 2),
        "max_trajectory_angle_deg": round(float(np.max(trajectory_angles)), 2),
        "num_frames_analyzed": len(trajectory_angles),
    }


def compute_all_dynamic_estimates(
    keypoints: np.ndarray,
    phase_boundaries: Dict[str, int] | None = None,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Compute all dynamic estimates for a swim start clip.

    Args:
        keypoints: Keypoint array [T, 33, 4].
        phase_boundaries: Optional phase boundary dictionary.
        fps: Video frame rate.

    Returns:
        Combined dynamic estimates dictionary.
    """
    result: Dict[str, Any] = {}

    try:
        result["center_of_mass_velocity"] = compute_com_velocity(keypoints, fps=fps)
    except Exception as exc:
        LOGGER.warning("COM velocity computation failed: %s", exc)
        result["center_of_mass_velocity"] = {"error": str(exc)}

    try:
        result["head_alignment"] = compute_head_alignment(keypoints, phase_boundaries)
    except Exception as exc:
        LOGGER.warning("Head alignment computation failed: %s", exc)
        result["head_alignment"] = {"error": str(exc)}

    if phase_boundaries:
        try:
            result["entry_trajectory"] = compute_entry_trajectory(keypoints, phase_boundaries)
        except Exception as exc:
            LOGGER.warning("Entry trajectory computation failed: %s", exc)
            result["entry_trajectory"] = {"error": str(exc)}

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for dynamic estimates."""
    parser = argparse.ArgumentParser(
        description="Compute dynamic biomechanical estimates from SwimVision keypoints."
    )
    parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    parser.add_argument("--boundaries", help="Phase boundary JSON path.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
    parser.add_argument("--output", help="JSON output path.")
    return parser


def main() -> int:
    """Run dynamic estimates CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        keypoints = np.load(args.keypoints)
    except Exception as exc:
        LOGGER.error("Failed to load keypoints: %s", exc)
        return 1

    boundaries = {}
    if args.boundaries:
        try:
            with open(args.boundaries, "r", encoding="utf-8") as f:
                boundaries = json.load(f)
        except Exception as exc:
            LOGGER.warning("Failed to load boundaries: %s", exc)

    result = compute_all_dynamic_estimates(keypoints, boundaries, fps=args.fps)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        LOGGER.info("Saved dynamic estimates to %s", args.output)
    else:
        print(json.dumps(result, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
