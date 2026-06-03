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


def hip_rotation_angle(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
) -> float:
    """Compute hip rotation angle in the transverse plane.

    Measures pelvis rotation by comparing shoulder line to hip line.
    Higher values indicate more torso rotation during the movement.

    Args:
        left_shoulder: Left shoulder point.
        right_shoulder: Right shoulder point.
        left_hip: Left hip point.
        right_hip: Right hip point.

    Returns:
        Hip rotation angle in degrees.
    """
    shoulder_vec = np.asarray(right_shoulder, dtype=np.float32) - np.asarray(left_shoulder, dtype=np.float32)
    hip_vec = np.asarray(right_hip, dtype=np.float32) - np.asarray(left_hip, dtype=np.float32)

    norm_shoulder = np.linalg.norm(shoulder_vec)
    norm_hip = np.linalg.norm(hip_vec)
    if norm_shoulder < 1e-6 or norm_hip < 1e-6:
        return 0.0

    cos_angle = np.clip(np.dot(shoulder_vec, hip_vec) / (norm_shoulder * norm_hip), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def shoulder_hip_separation(
    shoulder_mid: np.ndarray,
    hip_mid: np.ndarray,
    vertical: np.ndarray,
) -> float:
    """Compute shoulder-to-hip separation angle from vertical.

    Measures how far the upper body leans relative to the lower body.
    Important for assessing kinetic chain efficiency in swim starts.

    Args:
        shoulder_mid: Shoulder midpoint.
        hip_mid: Hip midpoint.
        vertical: Vertical reference vector (typically [0, -1]).

    Returns:
        Separation angle in degrees.
    """
    torso_vec = np.asarray(shoulder_mid, dtype=np.float32) - np.asarray(hip_mid, dtype=np.float32)
    norm_torso = np.linalg.norm(torso_vec)
    if norm_torso < 1e-6:
        return 0.0
    cos_angle = np.clip(np.dot(torso_vec, vertical) / norm_torso, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_all_angles(keypoints: np.ndarray, width: int | None = None, height: int | None = None) -> pd.DataFrame:
    """Compute all requested SwimVision joint-angle metrics per frame.

    Args:
        keypoints: Array with shape ``[T, 33, 4]``.
        width: Frame width in pixels used for aspect-ratio correction.
        height: Frame height in pixels used for aspect-ratio correction.

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
                    "hip_rotation": np.nan,
                    "shoulder_hip_separation": np.nan,
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

        # New alignment metrics
        hip_rotation = hip_rotation_angle(left_shoulder, right_shoulder, left_hip, right_hip)
        separation_angle = shoulder_hip_separation(shoulder_midpoint, hip_midpoint, vertical_axis)

        body_linearity_value = body_linearity(shoulder_midpoint, hip_midpoint, ankle_midpoint)
        left_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = angle_between(right_shoulder, right_elbow, right_wrist)
        left_streamline_angle = _acute_angle(_vector_angle_against_axis(left_shoulder - left_wrist, horizontal_axis))
        right_streamline_angle = _acute_angle(
            _vector_angle_against_axis(right_shoulder - right_wrist, horizontal_axis)
        )
        streamline_angle = min(left_streamline_angle, right_streamline_angle)
        elbow_extension = max(left_elbow_angle, right_elbow_angle)

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
            "hip_rotation": hip_rotation,
            "shoulder_hip_separation": separation_angle,
        }
        records.append(record)

    angles_df = pd.DataFrame.from_records(records)
    angles_df["frame"] = angles_df["frame"].astype(int)
    angles_df = angles_df.set_index("frame")
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
            "hip_rotation",
            "shoulder_hip_separation",
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
        angles_df = compute_all_angles(keypoints, width=args.width, height=args.height)
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
        LOGGER.info("All angle computation modules initialized.")
    else:
        print(angles_df.head().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
