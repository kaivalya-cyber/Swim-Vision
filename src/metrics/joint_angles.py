# This file computes per-frame joint-angle metrics from SwimVision keypoint sequences.
"""Joint-angle computation utilities for swim start analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle at point ``b`` formed by points ``a`` and ``c``.

    Args:
        a: First point in ``[x, y]`` format.
        b: Vertex point in ``[x, y]`` format.
        c: Third point in ``[x, y]`` format.

    Returns:
        Angle in degrees in the inclusive range ``[0, 180]``.
    """

    vector_ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    vector_bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)

    angle_one = np.arctan2(vector_ba[1], vector_ba[0])
    angle_two = np.arctan2(vector_bc[1], vector_bc[0])
    degrees = np.degrees(np.abs(angle_one - angle_two))
    if degrees > 180.0:
        degrees = 360.0 - degrees
    return float(degrees)


def _midpoint(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compute the midpoint between two 2D points.

    Args:
        first: First point in ``[x, y]`` format.
        second: Second point in ``[x, y]`` format.

    Returns:
        Midpoint coordinates in ``[x, y]`` format.
    """

    return (np.asarray(first, dtype=np.float32) + np.asarray(second, dtype=np.float32)) / 2.0


def _vector_angle_against_axis(vector: np.ndarray, axis: np.ndarray) -> float:
    """Compute the unsigned angle between a vector and a reference axis.

    Args:
        vector: Target vector in ``[x, y]`` format.
        axis: Reference vector in ``[x, y]`` format.

    Returns:
        Unsigned angle in degrees.
    """

    norm_vector = np.linalg.norm(vector)
    norm_axis = np.linalg.norm(axis)
    if norm_vector == 0.0 or norm_axis == 0.0:
        return 0.0
    cosine = float(np.clip(np.dot(vector, axis) / (norm_vector * norm_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _acute_angle(angle_degrees: float) -> float:
    """Convert an unsigned angle into its acute representation."""

    bounded = float(max(0.0, min(angle_degrees, 180.0)))
    return min(bounded, 180.0 - bounded)


def compute_com(frame: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Compute the Center of Mass (CoM) for a single frame using segmental weights.

    Args:
        frame: Keypoint array for a single frame [33, 2].
        aspect_ratio: Width / Height for scaling.

    Returns:
        CoM in [x, y] format.
    """

    # Segment weights and CoM positions (proximal to distal ratio)
    # Ref: Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement.
    segments = [
        # Torso: Shoulders & Hips. Weight 0.433
        {"joints": [11, 12, 23, 24], "weight": 0.433, "ratio": 0.5},
        # Head: Nose. Weight 0.081
        {"joints": [0], "weight": 0.081, "ratio": 1.0},
        # Upper Arms. Weight 0.028 each
        {"joints": [11, 13], "weight": 0.028, "ratio": 0.436},
        {"joints": [12, 14], "weight": 0.028, "ratio": 0.436},
        # Forearms. Weight 0.016 each
        {"joints": [13, 15], "weight": 0.016, "ratio": 0.430},
        {"joints": [14, 16], "weight": 0.016, "ratio": 0.430},
        # Thighs. Weight 0.100 each
        {"joints": [23, 25], "weight": 0.100, "ratio": 0.433},
        {"joints": [24, 26], "weight": 0.100, "ratio": 0.433},
        # Shanks. Weight 0.0465 each
        {"joints": [25, 27], "weight": 0.0465, "ratio": 0.433},
        {"joints": [26, 28], "weight": 0.0465, "ratio": 0.433},
        # Feet. Weight 0.0145 each
        {"joints": [27, 31], "weight": 0.0145, "ratio": 0.5},
        {"joints": [28, 32], "weight": 0.0145, "ratio": 0.5},
    ]

    total_weight = 0.0
    com_accum = np.zeros(2, dtype=np.float32)

    for seg in segments:
        joints = [_scale_point(frame[j], aspect_ratio) for j in seg["joints"]]
        # Compute segmental center
        if len(joints) == 1:
            seg_com = joints[0]
        elif len(joints) == 2:
            seg_com = joints[0] + seg["ratio"] * (joints[1] - joints[0])
        else:
            # For torso, take midpoint of the 4 points
            seg_com = np.mean(joints, axis=0)

        com_accum += seg_com * seg["weight"]
        total_weight += seg["weight"]

    return com_accum / total_weight if total_weight > 0 else np.zeros(2)


def body_linearity(shoulder_mid: np.ndarray, hip_mid: np.ndarray, ankle_mid: np.ndarray) -> float:
    """Compute deviation from straight alignment using y-sorted body landmarks.

    Args:
        shoulder_mid: Shoulder midpoint in ``[x, y]`` format.
        hip_mid: Hip midpoint in ``[x, y]`` format.
        ankle_mid: Ankle midpoint in ``[x, y]`` format.

    Returns:
        Deviation from straight alignment where ``0`` means perfectly linear.
    """

    points = sorted(
        [
            np.asarray(shoulder_mid, dtype=np.float32),
            np.asarray(hip_mid, dtype=np.float32),
            np.asarray(ankle_mid, dtype=np.float32),
        ],
        key=lambda point: float(point[1]),
    )
    top, mid, bottom = points[0], points[1], points[2]
    line_vec = bottom - top
    body_length = float(np.linalg.norm(line_vec))
    if body_length < 1e-6:
        return 0.0
    mid_vec = mid - top
    t = float(np.dot(mid_vec, line_vec) / np.dot(line_vec, line_vec))
    projection = top + t * line_vec
    perp_dist = float(np.linalg.norm(mid - projection))
    return float(np.degrees(np.arcsin(np.clip(perp_dist / body_length, 0.0, 1.0))))


def _scale_point(point: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Scale a 2D point for aspect-ratio-correct geometric computation.

    Args:
        point: Point in ``[x, y]`` format.
        aspect_ratio: Width divided by height for the source frame.

    Returns:
        Aspect-ratio-corrected point.
    """

    scaled_point = np.asarray(point, dtype=np.float32).copy()
    scaled_point[0] *= float(aspect_ratio)
    return scaled_point


def compute_all_angles(keypoints: np.ndarray, width: int | None = None, height: int | None = None, fps: float = 30.0) -> pd.DataFrame:
    """Compute all requested SwimVision joint-angle metrics per frame.

    Args:
        keypoints: Array with shape ``[T, 33, 4]``.
        width: Frame width in pixels used for aspect-ratio correction.
        height: Frame height in pixels used for aspect-ratio correction.
        fps: Video frame rate for velocity computation.

    Returns:
        A DataFrame indexed by frame containing the defined angle metrics.
    """

    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}.")
    if width is None or height is None:
        width, height = 1920, 1080
    if height == 0:
        raise ValueError("Frame height must be non-zero for aspect-ratio correction.")

    records = []
    aspect_ratio = float(width) / float(height)
    vertical_axis = np.array([0.0, -1.0], dtype=np.float32)
    horizontal_axis = np.array([1.0, 0.0], dtype=np.float32)

    for frame_index in range(keypoints.shape[0]):
        frame = keypoints[frame_index, :, :2]
        visibility = keypoints[frame_index, :, 3]
        if float(np.max(visibility)) <= 0.0:
            records.append(
                {
                    "frame": float(frame_index),
                    "front_knee_angle": np.nan,
                    "rear_knee_angle": np.nan,
                    "hip_angle": np.nan,
                    "torso_lean": np.nan,
                    "left_elbow_angle": np.nan,
                    "right_elbow_angle": np.nan,
                    "body_linearity": np.nan,
                    "entry_angle": np.nan,
                    "streamline_angle": np.nan,
                    "elbow_extension": np.nan,
                    "elbow_lock_angle": np.nan,
                    "shoulder_hip_alignment": np.nan,
                }
            )
            continue
        left_shoulder = _scale_point(frame[11], aspect_ratio)
        right_shoulder = _scale_point(frame[12], aspect_ratio)
        left_elbow = _scale_point(frame[13], aspect_ratio)
        right_elbow = _scale_point(frame[14], aspect_ratio)
        left_wrist = _scale_point(frame[15], aspect_ratio)
        right_wrist = _scale_point(frame[16], aspect_ratio)
        left_hip = _scale_point(frame[23], aspect_ratio)
        right_hip = _scale_point(frame[24], aspect_ratio)
        left_knee = _scale_point(frame[25], aspect_ratio)
        right_knee = _scale_point(frame[26], aspect_ratio)
        left_ankle = _scale_point(frame[27], aspect_ratio)
        right_ankle = _scale_point(frame[28], aspect_ratio)

        shoulder_midpoint = _midpoint(left_shoulder, right_shoulder)
        hip_midpoint = _midpoint(left_hip, right_hip)
        knee_midpoint = _midpoint(left_knee, right_knee)
        wrist_midpoint = _midpoint(left_wrist, right_wrist)
        ankle_midpoint = _midpoint(left_ankle, right_ankle)

        torso_vector = hip_midpoint - shoulder_midpoint
        entry_vector = ankle_midpoint - wrist_midpoint
        hip_to_shoulder = shoulder_midpoint - hip_midpoint
        ankle_to_hip = ankle_midpoint - hip_midpoint

        body_linearity_value = body_linearity(shoulder_midpoint, hip_midpoint, ankle_midpoint)
        left_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = angle_between(right_shoulder, right_elbow, right_wrist)
        left_streamline_angle = _acute_angle(_vector_angle_against_axis(left_shoulder - left_wrist, horizontal_axis))
        right_streamline_angle = _acute_angle(
            _vector_angle_against_axis(right_shoulder - right_wrist, horizontal_axis)
        )
        streamline_angle = min(left_streamline_angle, right_streamline_angle)
        elbow_extension = max(left_elbow_angle, right_elbow_angle)

        torso_length = float(np.linalg.norm(torso_vector))

        # Angle of Attack: Arm angle relative to horizontal at entry
        # We use the shoulder-to-wrist vector
        arm_vector = wrist_midpoint - shoulder_midpoint
        angle_of_attack = _acute_angle(_vector_angle_against_axis(arm_vector, horizontal_axis))

        record: Dict[str, float] = {
            "frame": float(frame_index),
            "front_knee_angle": angle_between(left_hip, left_knee, left_ankle),
            "rear_knee_angle": angle_between(right_hip, right_knee, right_ankle),
            "hip_angle": angle_between(shoulder_midpoint, hip_midpoint, knee_midpoint),
            "torso_lean": _vector_angle_against_axis(torso_vector, vertical_axis),
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "body_linearity": body_linearity_value,
            "entry_angle": _acute_angle(_vector_angle_against_axis(entry_vector, horizontal_axis)),
            "streamline_angle": streamline_angle,
            "elbow_extension": elbow_extension,
            "elbow_lock_angle": elbow_extension,
            "shoulder_hip_alignment": angle_between(hip_to_shoulder, np.zeros(2, dtype=np.float32), ankle_to_hip),
            "torso_x": float(hip_midpoint[0]),
            "torso_y": float(hip_midpoint[1]),
            "torso_length": torso_length,
            "com_x": float(compute_com(frame, aspect_ratio)[0]),
            "com_y": float(compute_com(frame, aspect_ratio)[1]),
            "angle_of_attack": angle_of_attack,
        }
        records.append(record)

    angles_df = pd.DataFrame.from_records(records)
    angles_df["frame"] = angles_df["frame"].astype(int)
    angles_df = angles_df.set_index("frame")

    # Compute Velocity (m/s)
    # Assume torso length is 0.6 meters for scaling
    # We use a 5-frame rolling average for smoothness
    torso_meters_ref = 0.6

    # Estimate median torso length during the middle of the clip (likely more stable)
    valid_torso = angles_df["torso_length"].dropna()
    if not valid_torso.empty:
        median_torso_scaled = float(valid_torso.median())
        meters_per_scaled_unit = torso_meters_ref / median_torso_scaled if median_torso_scaled > 0 else 0
    else:
        meters_per_scaled_unit = 0

    dx = angles_df["torso_x"].diff()
    dy = angles_df["torso_y"].diff()
    dist_scaled = np.sqrt(dx**2 + dy**2)
    velocity_mps = dist_scaled * meters_per_scaled_unit * fps
    angles_df["velocity"] = velocity_mps.rolling(window=5, center=True).mean()

    # Compute Stability Score (0-100)
    # Based on the variance of body_linearity and torso_lean during stable tracking
    linearity_var = angles_df["body_linearity"].rolling(window=10).std().fillna(0)
    lean_var = angles_df["torso_lean"].rolling(window=10).std().fillna(0)
    stability = 100.0 - (linearity_var * 2.0 + lean_var * 1.5)
    angles_df["stability_score"] = np.clip(stability, 0, 100)

    return angles_df[
        [
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
            "velocity",
            "com_x",
            "com_y",
            "torso_length",
            "angle_of_attack",
            "stability_score",
        ]
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for angle computation.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Compute SwimVision joint angles from keypoints.")
    parser.add_argument("--input", required=True, help="Path to a keypoints .npy file.")
    parser.add_argument("--output", help="Optional CSV output path.")
    parser.add_argument("--clip_id", help="Clip identifier used to name the output CSV.")
    parser.add_argument("--width", type=int, help="Optional frame width for aspect-ratio correction.")
    parser.add_argument("--height", type=int, help="Optional frame height for aspect-ratio correction.")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate for velocity computation.")
    return parser


def main() -> int:
    """Run the command-line interface for angle computation.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        keypoints = np.load(args.input)
    except Exception as exc:
        LOGGER.error("Failed to load keypoints from %s: %s", args.input, exc)
        return 1

    try:
        angles_df = compute_all_angles(keypoints, width=args.width, height=args.height, fps=args.fps)
    except Exception as exc:
        LOGGER.error("Failed to compute joint angles: %s", exc)
        return 1

    if args.output:
        if not args.clip_id:
            LOGGER.error("--clip_id is required when --output is provided.")
            return 1
        output_path = Path(args.output) / f"{args.clip_id}_angles.csv"
        try:
            angles_df.to_csv(output_path)
        except Exception as exc:
            LOGGER.error("Failed to write angles CSV to %s: %s", output_path, exc)
            return 1
        LOGGER.info("Saved angle metrics to %s", output_path)
    else:
        print(angles_df.head().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
