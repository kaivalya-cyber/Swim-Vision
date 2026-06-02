# This file computes per-cycle stroke biomechanical metrics from SwimVision keypoint sequences.
"""Stroke-specific metric computation for freestyle swimming analysis."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.stroke.cycle_detection import StrokeCycle, angle_between


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass
class StrokeMetrics:
    """Per-cycle stroke biomechanical metrics.

    Attributes:
        cycle_index: Zero-based index of the stroke cycle.
        stroke_type: Stroke classification (freestyle, butterfly, backstroke).
        left_elbow_flexion: Mean left elbow flexion during catch (degrees).
        right_elbow_flexion: Mean right elbow flexion during catch (degrees).
        left_shoulder_rotation: Left shoulder rotation relative to horizontal (degrees).
        right_shoulder_rotation: Right shoulder rotation relative to horizontal (degrees).
        body_roll: Peak body roll angle during the cycle (degrees).
        left_hand_speed: Mean left wrist speed during pull (normalized units/frame).
        right_hand_speed: Mean right wrist speed during pull (normalized units/frame).
        left_elbow_extension_rate: Rate of left elbow extension during push (deg/frame).
        right_elbow_extension_rate: Rate of right elbow extension during push (deg/frame).
        stroke_rate: Strokes per minute for this cycle.
        cycle_duration_seconds: Duration of the full cycle.
        symmetry_index: Percentage difference between left/right catch angles.
        bilateral_elbow_flexion: Mean bilateral elbow flexion for butterfly (degrees).
        bilateral_hand_speed: Mean bilateral hand speed for butterfly (norm/frame).
        supine_elbow_flexion: Mean elbow flexion for backstroke catch (degrees).
        supine_hand_speed: Mean hand speed for backstroke pull (norm/frame).
    """

    cycle_index: int
    stroke_type: str = "freestyle"
    left_elbow_flexion: float = 0.0
    right_elbow_flexion: float = 0.0
    left_shoulder_rotation: float = 0.0
    right_shoulder_rotation: float = 0.0
    body_roll: float = 0.0
    left_hand_speed: float = 0.0
    right_hand_speed: float = 0.0
    left_elbow_extension_rate: float = 0.0
    right_elbow_extension_rate: float = 0.0
    stroke_rate: float = 0.0
    cycle_duration_seconds: float = 0.0
    symmetry_index: float = 0.0
    bilateral_elbow_flexion: float = 0.0
    bilateral_hand_speed: float = 0.0
    supine_elbow_flexion: float = 0.0
    supine_hand_speed: float = 0.0


def compute_stroke_metrics(
    keypoints: np.ndarray, cycles: List[StrokeCycle], fps: float, width: int, height: int
) -> List[StrokeMetrics]:
    """Compute per-cycle biomechanical metrics for detected stroke cycles.

    Args:
        keypoints: Keypoint array [T, 33, 4].
        cycles: List of detected StrokeCycle instances.
        fps: Video frame rate.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of StrokeMetrics for each cycle.
    """

    if not cycles:
        LOGGER.warning("No stroke cycles provided; returning empty metrics.")
        return []

    num_frames = keypoints.shape[0]
    aspect_ratio = float(width) / float(height)
    metrics_list: List[StrokeMetrics] = []

    for cycle in cycles:
        left_start = max(0, cycle.left_catch_frame)
        left_end = min(cycle.left_pull_end_frame, num_frames - 1)
        right_start = max(0, cycle.right_catch_frame)
        right_end = min(cycle.right_pull_end_frame, num_frames - 1)

        # Left elbow flexion during catch
        left_elbow_flexion = _compute_mean_elbow_angle(keypoints, left_start, left_end, side="left") if left_end > left_start else 0.0

        # Right elbow flexion during catch
        right_elbow_flexion = _compute_mean_elbow_angle(keypoints, right_start, right_end, side="right") if right_end > right_start else 0.0

        # Body roll
        body_roll = cycle.body_roll_peak

        # Hand speed during pull (wrist displacement between frames)
        left_hand_speed = _compute_hand_speed(keypoints, left_start, left_end, side="left") if left_end > left_start + 1 else 0.0
        right_hand_speed = _compute_hand_speed(keypoints, right_start, right_end, side="right") if right_end > right_start + 1 else 0.0

        # Elbow extension rate during push
        left_elbow_extension_rate = _compute_extension_rate(keypoints, left_start, left_end, side="left") if left_end > left_start + 1 else 0.0
        right_elbow_extension_rate = _compute_extension_rate(keypoints, right_start, right_end, side="right") if right_end > right_start + 1 else 0.0

        # Shoulder rotation
        left_shoulder_rot = _compute_shoulder_rotation(keypoints, left_start, left_end, side="left", aspect_ratio=aspect_ratio)
        right_shoulder_rot = _compute_shoulder_rotation(keypoints, right_start, right_end, side="right", aspect_ratio=aspect_ratio)

        # Cycle timing
        cycle_start = min(cycle.left_entry_frame, cycle.right_entry_frame)
        cycle_end = max(cycle.left_recovery_end_frame, cycle.right_recovery_end_frame)
        cycle_frames = max(cycle_end - cycle_start, 1)
        cycle_duration = float(cycle_frames) / float(fps)

        # Stroke rate (strokes per minute) - one full cycle = 2 strokes (left + right)
        stroke_rate = 120.0 / cycle_duration if cycle_duration > 0 else 0.0

        # Symmetry index: percentage difference between left and right catch angles
        avg_catch = (left_elbow_flexion + right_elbow_flexion) / 2.0
        if avg_catch > 0:
            symmetry_index = abs(left_elbow_flexion - right_elbow_flexion) / avg_catch * 100.0
        else:
            symmetry_index = 0.0

        metrics_list.append(
            StrokeMetrics(
                cycle_index=cycle.cycle_index,
                stroke_type=cycle.stroke_type,
                left_elbow_flexion=left_elbow_flexion,
                right_elbow_flexion=right_elbow_flexion,
                left_shoulder_rotation=left_shoulder_rot,
                right_shoulder_rotation=right_shoulder_rot,
                body_roll=body_roll,
                left_hand_speed=left_hand_speed,
                right_hand_speed=right_hand_speed,
                left_elbow_extension_rate=left_elbow_extension_rate,
                right_elbow_extension_rate=right_elbow_extension_rate,
                stroke_rate=stroke_rate,
                cycle_duration_seconds=cycle_duration,
                symmetry_index=symmetry_index,
                bilateral_elbow_flexion=_compute_bilateral_elbow(keypoints, left_start, right_start, left_end, right_end) if cycle.stroke_type == "butterfly" else 0.0,
                bilateral_hand_speed=_compute_bilateral_speed(keypoints, left_start, right_start, left_end, right_end) if cycle.stroke_type == "butterfly" else 0.0,
                supine_elbow_flexion=_compute_supine_elbow(keypoints, left_start, right_start, left_end, right_end) if cycle.stroke_type == "backstroke" else 0.0,
                supine_hand_speed=_compute_supine_speed(keypoints, left_start, right_start, left_end, right_end) if cycle.stroke_type == "backstroke" else 0.0,
            )
        )

    LOGGER.info("Computed metrics for %d stroke cycles.", len(metrics_list))
    return metrics_list


def _compute_mean_elbow_angle(
    keypoints: np.ndarray, start_frame: int, end_frame: int, side: str = "left"
) -> float:
    """Compute the mean elbow-flexion angle over a frame window.

    Args:
        keypoints: Full keypoint array.
        start_frame: Start frame index.
        end_frame: End frame index.
        side: Arm side identifier.

    Returns:
        Mean elbow angle in degrees.
    """

    if side == "left":
        shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15
    else:
        shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16

    angles: List[float] = []
    for frame_idx in range(start_frame, end_frame + 1):
        shoulder = keypoints[frame_idx, shoulder_idx, :2]
        elbow = keypoints[frame_idx, elbow_idx, :2]
        wrist = keypoints[frame_idx, wrist_idx, :2]
        angles.append(angle_between(shoulder, elbow, wrist))

    return float(np.mean(angles)) if angles else 0.0


def _compute_hand_speed(
    keypoints: np.ndarray, start_frame: int, end_frame: int, side: str = "left"
) -> float:
    """Compute mean wrist displacement per frame over a window.

    Args:
        keypoints: Full keypoint array.
        start_frame: Start frame index.
        end_frame: End frame index.
        side: Arm side identifier.

    Returns:
        Mean normalized hand speed per frame.
    """

    wrist_idx = 15 if side == "left" else 16
    speeds: List[float] = []
    for frame_idx in range(start_frame, end_frame):
        current = keypoints[frame_idx, wrist_idx, :2]
        next_pos = keypoints[frame_idx + 1, wrist_idx, :2]
        speeds.append(float(np.linalg.norm(next_pos - current)))

    return float(np.mean(speeds)) if speeds else 0.0


def _compute_extension_rate(
    keypoints: np.ndarray, start_frame: int, end_frame: int, side: str = "left"
) -> float:
    """Compute mean elbow extension rate in degrees per frame over a window.

    Args:
        keypoints: Full keypoint array.
        start_frame: Start frame index.
        end_frame: End frame index.
        side: Arm side identifier.

    Returns:
        Mean extension rate in deg/frame (positive = extending).
    """

    if side == "left":
        shoulder_idx, elbow_idx, wrist_idx = 11, 13, 15
    else:
        shoulder_idx, elbow_idx, wrist_idx = 12, 14, 16

    rates: List[float] = []
    for frame_idx in range(start_frame, end_frame):
        curr_angle = angle_between(
            keypoints[frame_idx, shoulder_idx, :2],
            keypoints[frame_idx, elbow_idx, :2],
            keypoints[frame_idx, wrist_idx, :2],
        )
        next_angle = angle_between(
            keypoints[frame_idx + 1, shoulder_idx, :2],
            keypoints[frame_idx + 1, elbow_idx, :2],
            keypoints[frame_idx + 1, wrist_idx, :2],
        )
        rates.append(next_angle - curr_angle)

    return float(np.mean(rates)) if rates else 0.0


def _compute_shoulder_rotation(
    keypoints: np.ndarray, start_frame: int, end_frame: int, side: str = "left", aspect_ratio: float = 1.0
) -> float:
    """Compute mean shoulder rotation relative to the horizontal axis.

    Args:
        keypoints: Full keypoint array.
        start_frame: Start frame index.
        end_frame: End frame index.
        side: Arm side identifier.
        aspect_ratio: Width/height for coordinate correction.

    Returns:
        Mean shoulder rotation in degrees.
    """

    shoulder_idx = 11 if side == "left" else 12
    hip_idx = 23 if side == "left" else 24
    angles: List[float] = []
    for frame_idx in range(start_frame, end_frame + 1):
        shoulder = keypoints[frame_idx, shoulder_idx, :2].copy()
        hip = keypoints[frame_idx, hip_idx, :2].copy()
        shoulder[0] *= aspect_ratio
        hip[0] *= aspect_ratio
        vector = shoulder - hip
        horizontal = np.array([1.0, 0.0], dtype=np.float32)
        norm_v = float(np.linalg.norm(vector))
        if norm_v < 1e-6:
            continue
        cosine = float(np.clip(np.dot(vector, horizontal) / norm_v, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))

    return float(np.mean(angles)) if angles else 0.0


def aggregate_stroke_metrics(
    metrics_list: List[StrokeMetrics],
) -> Dict[str, float]:
    """Aggregate per-cycle metrics into summary statistics.

    Args:
        metrics_list: List of per-cycle StrokeMetrics.

    Returns:
        Dictionary of averaged metrics across all cycles.
    """

    if not metrics_list:
        return {}

    summary: Dict[str, float] = {}
    metric_names = [
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_shoulder_rotation",
        "right_shoulder_rotation",
        "body_roll",
        "left_hand_speed",
        "right_hand_speed",
        "left_elbow_extension_rate",
        "right_elbow_extension_rate",
        "stroke_rate",
        "cycle_duration_seconds",
        "symmetry_index",
        "bilateral_elbow_flexion",
        "bilateral_hand_speed",
        "supine_elbow_flexion",
        "supine_hand_speed",
    ]

    for name in metric_names:
        values = [getattr(m, name) for m in metrics_list]
        nonzero_values = [v for v in values if v != 0.0]
        summary[name] = float(np.mean(nonzero_values)) if nonzero_values else 0.0

    summary["num_cycles"] = float(len(metrics_list))
    # Include detected stroke type for downstream use
    if metrics_list:
        summary["stroke_type"] = metrics_list[0].stroke_type
    return summary


def _compute_bilateral_elbow(
    keypoints: np.ndarray,
    left_start: int, right_start: int,
    left_end: int, right_end: int,
) -> float:
    """Compute bilateral (both arms) mean elbow flexion for butterfly."""

    left_angle = _compute_mean_elbow_angle(keypoints, left_start, left_end, side="left") if left_end > left_start else 0.0
    right_angle = _compute_mean_elbow_angle(keypoints, right_start, right_end, side="right") if right_end > right_start else 0.0
    if left_angle > 0 or right_angle > 0:
        return (left_angle + right_angle) / 2.0
    return 0.0


def _compute_bilateral_speed(
    keypoints: np.ndarray,
    left_start: int, right_start: int,
    left_end: int, right_end: int,
) -> float:
    """Compute bilateral mean hand speed for butterfly."""

    left_speed = _compute_hand_speed(keypoints, left_start, left_end, side="left") if left_end > left_start + 1 else 0.0
    right_speed = _compute_hand_speed(keypoints, right_start, right_end, side="right") if right_end > right_start + 1 else 0.0
    if left_speed > 0 or right_speed > 0:
        return (left_speed + right_speed) / 2.0
    return 0.0


def _compute_supine_elbow(
    keypoints: np.ndarray,
    left_start: int, right_start: int,
    left_end: int, right_end: int,
) -> float:
    """Compute supine (backstroke) elbow flexion — best arm value."""

    left_angle = _compute_mean_elbow_angle(keypoints, left_start, left_end, side="left") if left_end > left_start else 0.0
    right_angle = _compute_mean_elbow_angle(keypoints, right_start, right_end, side="right") if right_end > right_start else 0.0
    return max(left_angle, right_angle)


def _compute_supine_speed(
    keypoints: np.ndarray,
    left_start: int, right_start: int,
    left_end: int, right_end: int,
) -> float:
    """Compute supine (backstroke) hand speed — best arm value."""

    left_speed = _compute_hand_speed(keypoints, left_start, left_end, side="left") if left_end > left_start + 1 else 0.0
    right_speed = _compute_hand_speed(keypoints, right_start, right_end, side="right") if right_end > right_start + 1 else 0.0
    return max(left_speed, right_speed)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for stroke metrics computation."""

    parser = argparse.ArgumentParser(description="Compute stroke biomechanical metrics.")
    parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    parser.add_argument("--boundaries", required=True, help="Path to stroke cycle boundaries JSON.")
    parser.add_argument("--output", required=True, help="Path to write stroke metrics JSON.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
    parser.add_argument("--width", type=int, default=1920, help="Frame width.")
    parser.add_argument("--height", type=int, default=1080, help="Frame height.")
    return parser


def main() -> int:
    """Run the stroke metrics CLI."""

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        keypoints = np.load(args.keypoints)
    except Exception as exc:
        LOGGER.error("Failed to load keypoints from %s: %s", args.keypoints, exc)
        return 1

    try:
        with open(args.boundaries, "r", encoding="utf-8") as handle:
            boundaries = json.load(handle)
    except Exception as exc:
        LOGGER.error("Failed to load stroke boundaries from %s: %s", args.boundaries, exc)
        return 1

    # Reconstruct StrokeCycle objects from JSON
    cycles_data = boundaries.get("cycles", [])
    cycles = [
        StrokeCycle(
            cycle_index=c.get("cycle_index", 0),
            stroke_type=c.get("stroke_type", "freestyle"),
            left_entry_frame=c.get("left_entry_frame", 0),
            left_catch_frame=c.get("left_catch_frame", 0),
            left_pull_end_frame=c.get("left_pull_end_frame", 0),
            left_recovery_end_frame=c.get("left_recovery_end_frame", 0),
            right_entry_frame=c.get("right_entry_frame", 0),
            right_catch_frame=c.get("right_catch_frame", 0),
            right_pull_end_frame=c.get("right_pull_end_frame", 0),
            right_recovery_end_frame=c.get("right_recovery_end_frame", 0),
            body_roll_peak=c.get("body_roll_peak", 0.0),
        )
        for c in cycles_data
    ]

    try:
        metrics_list = compute_stroke_metrics(keypoints, cycles, args.fps, args.width, args.height)
        aggregate = aggregate_stroke_metrics(metrics_list)
        # Convert StrokeMetrics to dicts for JSON serialization
        cycles_json = [asdict(m) for m in metrics_list]
        output_payload = {"cycles": cycles_json, "aggregate": aggregate, "clip_id": ""}
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output_payload, handle, indent=2)
        LOGGER.info("Saved stroke metrics to %s", args.output)
    except Exception as exc:
        LOGGER.error("Failed to compute stroke metrics: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
