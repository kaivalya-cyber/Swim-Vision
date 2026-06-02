# This file detects swim stroke cycles from pose keypoint sequences using wrist and body kinematics.
"""Stroke-cycle detection utilities for SwimVision."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


@dataclass
class StrokeCycle:
    """A detected stroke cycle with per-arm phase boundaries.

    Attributes:
        cycle_index: Zero-based cycle index within the sequence.
        stroke_type: Stroke classification (freestyle, butterfly, backstroke).
        left_entry_frame: Frame index of left hand water entry.
        left_catch_frame: Frame index of left arm catch initiation.
        left_pull_end_frame: Frame index of left arm pull completion.
        left_recovery_end_frame: Frame index of left arm recovery completion.
        right_entry_frame: Frame index of right hand water entry.
        right_catch_frame: Frame index of right arm catch initiation.
        right_pull_end_frame: Frame index of right arm pull completion.
        right_recovery_end_frame: Frame index of right arm recovery completion.
        body_roll_peak: Peak body roll angle in degrees during the cycle.
    """

    cycle_index: int
    stroke_type: str = "freestyle"
    left_entry_frame: int = 0
    left_catch_frame: int = 0
    left_pull_end_frame: int = 0
    left_recovery_end_frame: int = 0
    right_entry_frame: int = 0
    right_catch_frame: int = 0
    right_pull_end_frame: int = 0
    right_recovery_end_frame: int = 0
    body_roll_peak: float = 0.0


def _smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a moving-average filter to a 1D signal.

    Args:
        signal: 1D input array.
        window: Smoothing kernel width.

    Returns:
        Smoothed signal of the same length.
    """

    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _detect_arm_phases(
    wrist_y: np.ndarray,
    wrist_x: np.ndarray,
    elbow_angle: np.ndarray,
    fps: float,
    side: str = "left",
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Detect entry, catch, pull-end, and recovery-end frames for one arm.

    Args:
        wrist_y: Smoothed wrist y-coordinate per frame (pixel or normalized).
        wrist_x: Smoothed wrist x-coordinate per frame.
        elbow_angle: Smoothed elbow-flexion angle per frame.
        fps: Frame rate.
        side: Arm identifier for logging.

    Returns:
        Lists of entry frames, catch frames, pull-end frames, recovery-end frames.
    """

    num_frames = len(wrist_y)
    wrist_velocity_y = np.diff(wrist_y, prepend=wrist_y[0])
    entries: List[int] = []
    catches: List[int] = []
    pull_ends: List[int] = []
    recoveries: List[int] = []

    # Hand entry: wrist moving downward (y increasing in image coords) and reaching a high point
    # Catch: elbow begins flexing > 90° while wrist moves backward
    # Pull end: wrist reaches its lowest (closest to hip) point
    # Recovery end: wrist returns to high position and begins forward reach

    min_detection_gap = max(int(fps * 0.5), 10)
    min_elbow_for_catch = 90.0

    for frame_idx in range(min_detection_gap, num_frames - min_detection_gap):
        # Entry detection: wrist at local minimum y (highest in frame) with downward velocity transition
        local_window_start = max(0, frame_idx - min_detection_gap)
        local_window_end = min(num_frames, frame_idx + min_detection_gap)
        local_wrist_y = wrist_y[local_window_start:local_window_end]

        if (
            len(local_wrist_y) >= 3
            and wrist_y[frame_idx] <= np.percentile(local_wrist_y, 15)
            and wrist_velocity_y[frame_idx] > wrist_velocity_y[max(0, frame_idx - 2)]
        ):
            if not entries or frame_idx - entries[-1] >= min_detection_gap:
                entries.append(frame_idx)

    for frame_idx in range(min_detection_gap, num_frames - min_detection_gap):
        local_window_start = max(0, frame_idx - min_detection_gap // 2)
        local_window_end = min(num_frames, frame_idx + min_detection_gap // 2)
        local_wrist_y = wrist_y[local_window_start:local_window_end]

        if (
            len(local_wrist_y) >= 3
            and wrist_y[frame_idx] >= np.percentile(local_wrist_y, 85)
            and elbow_angle[frame_idx] >= min_elbow_for_catch
        ):
            if not pull_ends or frame_idx - pull_ends[-1] >= min_detection_gap:
                pull_ends.append(frame_idx)

    for entry_frame in entries:
        search_end = min(entry_frame + int(fps * 0.4), num_frames)
        best_catch = entry_frame
        for candidate in range(entry_frame + 3, search_end):
            if elbow_angle[candidate] >= min_elbow_for_catch:
                best_catch = candidate
                break
        catches.append(best_catch)

    for pull_end in pull_ends:
        search_end = min(pull_end + int(fps * 0.5), num_frames)
        best_recovery = pull_end
        min_y = wrist_y[pull_end]
        for candidate in range(pull_end + 3, search_end):
            if wrist_y[candidate] < min_y:
                min_y = wrist_y[candidate]
                best_recovery = candidate
        recoveries.append(best_recovery)

    LOGGER.info(
        "%s arm: %d entries, %d catches, %d pull_ends, %d recoveries detected.",
        side,
        len(entries),
        len(catches),
        len(pull_ends),
        len(recoveries),
    )
    return entries, catches, pull_ends, recoveries


def _compute_body_roll(
    keypoints: np.ndarray, frame_idx: int, width: int, height: int
) -> float:
    """Compute the body roll angle (shoulder line vs horizontal) for a single frame.

    Args:
        keypoints: Full keypoint array [T, 33, 4].
        frame_idx: Target frame index.
        width: Frame width for pixel conversion.
        height: Frame height for pixel conversion.

    Returns:
        Body roll angle in degrees.
    """

    left_shoulder = keypoints[frame_idx, 11, :2]
    right_shoulder = keypoints[frame_idx, 12, :2]
    dx = float(right_shoulder[0] - left_shoulder[0]) * float(width)
    dy = float(right_shoulder[1] - left_shoulder[1]) * float(height)
    if abs(dx) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan(abs(dy / dx))))


def detect_stroke_cycles(
    keypoints: np.ndarray,
    fps: float,
    width: int = 1920,
    height: int = 1080,
    min_confidence: float = 0.5,
) -> List[StrokeCycle]:
    """Detect stroke cycles from a pose keypoint sequence.

    Uses bilateral wrist kinematics, elbow angles, and body rotation to
    identify full stroke cycles for freestyle swimming.

    Args:
        keypoints: Array with shape ``[T, 33, 4]``.
        fps: Video frame rate.
        width: Frame width in pixels.
        height: Frame height in pixels.
        min_confidence: Minimum joint visibility to consider a frame valid.

    Returns:
        List of detected ``StrokeCycle`` instances.
    """

    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(
            f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}."
        )
    num_frames = keypoints.shape[0]
    if num_frames < int(fps * 2.0):
        LOGGER.warning(
            "Sequence too short (%d frames) for reliable stroke detection.", num_frames
        )
        return []

    left_wrist_x = keypoints[:, 15, 0]
    left_wrist_y = keypoints[:, 15, 1]
    right_wrist_x = keypoints[:, 16, 0]
    right_wrist_y = keypoints[:, 16, 1]
    visibility = keypoints[:, :, 3]

    left_elbow_angle_raw = np.zeros(num_frames, dtype=np.float32)
    right_elbow_angle_raw = np.zeros(num_frames, dtype=np.float32)
    for frame_idx in range(num_frames):
        ls = keypoints[frame_idx, 11, :2]
        le = keypoints[frame_idx, 13, :2]
        lw = keypoints[frame_idx, 15, :2]
        rs = keypoints[frame_idx, 12, :2]
        re = keypoints[frame_idx, 14, :2]
        rw = keypoints[frame_idx, 16, :2]
        left_elbow_angle_raw[frame_idx] = angle_between(ls, le, lw)
        right_elbow_angle_raw[frame_idx] = angle_between(rs, re, rw)

    smoothed_left_wrist_y = _smooth_signal(left_wrist_y)
    smoothed_left_wrist_x = _smooth_signal(left_wrist_x)
    smoothed_right_wrist_y = _smooth_signal(right_wrist_y)
    smoothed_right_wrist_x = _smooth_signal(right_wrist_x)
    smoothed_left_elbow = _smooth_signal(left_elbow_angle_raw)
    smoothed_right_elbow = _smooth_signal(right_elbow_angle_raw)

    left_entries, left_catches, left_pull_ends, left_recoveries = _detect_arm_phases(
        smoothed_left_wrist_y, smoothed_left_wrist_x, smoothed_left_elbow, fps, side="Left"
    )
    right_entries, right_catches, right_pull_ends, right_recoveries = _detect_arm_phases(
        smoothed_right_wrist_y, smoothed_right_wrist_x, smoothed_right_elbow, fps, side="Right"
    )

    num_cycles = max(
        len(left_entries),
        len(right_entries),
    )
    if num_cycles == 0:
        LOGGER.warning("No stroke cycles detected in the sequence.")
        return []

    cycles: List[StrokeCycle] = []
    for cycle_idx in range(num_cycles):
        left_entry = left_entries[cycle_idx] if cycle_idx < len(left_entries) else 0
        left_catch = left_catches[cycle_idx] if cycle_idx < len(left_catches) else left_entry
        left_pull_end = left_pull_ends[cycle_idx] if cycle_idx < len(left_pull_ends) else left_catch
        left_recovery_end = left_recoveries[cycle_idx] if cycle_idx < len(left_recoveries) else left_pull_end
        right_entry = right_entries[cycle_idx] if cycle_idx < len(right_entries) else 0
        right_catch = right_catches[cycle_idx] if cycle_idx < len(right_catches) else right_entry
        right_pull_end = right_pull_ends[cycle_idx] if cycle_idx < len(right_pull_ends) else right_catch
        right_recovery_end = right_recoveries[cycle_idx] if cycle_idx < len(right_recoveries) else right_pull_end

        start_frame = min(
            left_entry, right_entry,
        )
        end_frame = max(
            left_recovery_end, right_recovery_end,
        )
        if end_frame <= start_frame:
            end_frame = min(start_frame + int(fps * 2.0), num_frames - 1)

        roll_angles = []
        for frame_idx in range(max(0, start_frame), min(end_frame + 1, num_frames)):
            if float(np.mean(visibility[frame_idx])) >= min_confidence:
                roll_angles.append(_compute_body_roll(keypoints, frame_idx, width, height))
        peak_roll = float(np.max(roll_angles)) if roll_angles else 0.0

        cycles.append(
            StrokeCycle(
                cycle_index=cycle_idx,
                left_entry_frame=int(left_entry),
                left_catch_frame=int(left_catch),
                left_pull_end_frame=int(left_pull_end),
                left_recovery_end_frame=int(left_recovery_end),
                right_entry_frame=int(right_entry),
                right_catch_frame=int(right_catch),
                right_pull_end_frame=int(right_pull_end),
                right_recovery_end_frame=int(right_recovery_end),
                body_roll_peak=peak_roll,
            )
        )

    LOGGER.info("Detected %d stroke cycles.", len(cycles))
    return cycles


def detect_stroke_type(
    keypoints: np.ndarray, num_sample_frames: int = 30, start_frame: int = 0
) -> str:
    """Detect the stroke type from keypoint patterns.

    Heuristics:
    - Butterfly: bilateral arm movement (wrists move in sync), high dolphin
      kick signature (both ankles move together).
    - Backstroke: swimmer is supine (shoulders above hips in y, i.e. smaller
      y-values for shoulders than hips in image coords).
    - Freestyle: alternating arm movement, moderate body roll.

    Args:
        keypoints: Array [T, 33, 4].
        num_sample_frames: Number of frames to analyze.
        start_frame: Frame index to start sampling from.

    Returns:
        Detected stroke type string.
    """

    total_frames = keypoints.shape[0]
    sample_start = max(start_frame, total_frames // 3)
    end_frame = min(total_frames, sample_start + num_sample_frames)
    if end_frame <= sample_start + 5:
        return "freestyle"

    sample = keypoints[sample_start:end_frame]

    # Check supine position (backstroke): shoulders above hips
    shoulder_y = sample[:, 11:13, 1].mean()
    hip_y = sample[:, 23:25, 1].mean()
    if shoulder_y < hip_y - 0.05:
        return "backstroke"

    # Check bilateral arm sync (butterfly): wrists have similar y-motion
    left_wrist_y = sample[:, 15, 1]
    right_wrist_y = sample[:, 16, 1]
    lw_diff = np.diff(left_wrist_y)
    rw_diff = np.diff(right_wrist_y)
    if len(lw_diff) > 3 and len(rw_diff) > 3:
        correlation = float(np.corrcoef(lw_diff, rw_diff)[0, 1])
        if correlation > 0.6:
            return "butterfly"

    return "freestyle"


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute the angle at joint ``b`` formed by points ``a`` and ``c``.

    Args:
        a: First 2D point.
        b: Vertex 2D point.
        c: Third 2D point.

    Returns:
        Angle in degrees [0, 180].
    """

    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    angle_rad = np.arctan2(ba[1], ba[0]) - np.arctan2(bc[1], bc[0])
    degrees = float(np.abs(np.degrees(angle_rad)))
    if degrees > 180.0:
        degrees = 360.0 - degrees
    return degrees
