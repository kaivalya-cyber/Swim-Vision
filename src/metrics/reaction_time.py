# This file estimates swim-start reaction time from ankle motion after the start beep.
"""Reaction-time detection utilities for SwimVision."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def detect_reaction_time(keypoints: np.ndarray, audio_beep_frame: int, fps: float) -> float | None:
    """Estimate reaction time from the start beep to feet leaving the block.

    Args:
        keypoints: Keypoint array with shape ``[T, 33, 4]``.
        audio_beep_frame: Frame index of the starting beep.
        fps: Frame rate of the analyzed video.

    Returns:
        Reaction time in milliseconds, or ``None`` when detection fails.
    """

    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}.")
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    if audio_beep_frame < 0 or audio_beep_frame >= keypoints.shape[0]:
        raise ValueError("audio_beep_frame must lie within the keypoint sequence.")

    ankle_y = keypoints[:, [27, 28], 1]
    y_velocity = np.diff(ankle_y, axis=0, prepend=ankle_y[:1])
    threshold = 1.5 if float(np.max(np.abs(ankle_y))) > 2.0 else 0.01

    for frame_index in range(audio_beep_frame + 1, keypoints.shape[0]):
        upward_motion = np.all(y_velocity[frame_index] < -threshold)
        if upward_motion:
            reaction_time_ms = float((frame_index - audio_beep_frame) / fps * 1000.0)
            LOGGER.info(
                "Detected feet-off-block at frame %s (reaction time %.2f ms).",
                frame_index,
                reaction_time_ms,
            )
            return reaction_time_ms

    LOGGER.warning("Reaction-time detection failed after beep frame %s.", audio_beep_frame)
    return None


def detect_first_stroke_time(keypoints: np.ndarray, entry_frame: int, fps: float) -> Dict[str, Any]:
    """Detect the time to first stroke using peak wrist velocity after entry.

    Args:
        keypoints: Keypoint array with shape ``[T, 33, 4]``.
        entry_frame: Frame index marking entry into the water.
        fps: Frame rate of the analyzed video.

    Returns:
        Dictionary containing the detected peak wrist-velocity frame and time in milliseconds.
    """

    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}.")
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    if entry_frame < 0 or entry_frame >= keypoints.shape[0]:
        raise ValueError("entry_frame must lie within the keypoint sequence.")

    wrist_points = keypoints[:, [15, 16], :2]
    wrist_velocity = np.linalg.norm(np.diff(wrist_points, axis=0, prepend=wrist_points[:1]), axis=2)
    mean_wrist_velocity = wrist_velocity.mean(axis=1)
    smoothed_velocity = np.convolve(mean_wrist_velocity, np.ones(5, dtype=np.float32) / 5.0, mode="same")

    search_start = min(entry_frame + 1, len(smoothed_velocity) - 1)
    if search_start >= len(smoothed_velocity):
        return {"first_stroke_frame": None, "time_to_first_stroke_ms": None, "peak_wrist_velocity": None}

    peak_offset = int(np.argmax(smoothed_velocity[search_start:]))
    peak_frame = search_start + peak_offset
    peak_velocity = float(smoothed_velocity[peak_frame])
    time_ms = float((peak_frame - entry_frame) / fps * 1000.0)
    LOGGER.info(
        "Detected first-stroke proxy at frame %s (%.2f ms after entry, peak wrist velocity %.4f).",
        peak_frame,
        time_ms,
        peak_velocity,
    )
    return {
        "first_stroke_frame": int(peak_frame),
        "time_to_first_stroke_ms": time_ms,
        "peak_wrist_velocity": peak_velocity,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for reaction-time detection.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Estimate swim-start reaction time.")
    parser.add_argument("--keypoints", required=True, help="Path to a keypoints .npy file.")
    parser.add_argument("--audio_beep_frame", required=True, type=int, help="Beep frame index.")
    parser.add_argument("--fps", required=True, type=float, help="Video frame rate.")
    parser.add_argument("--entry_frame", type=int, help="Optional entry frame for first-stroke detection.")
    return parser


def main() -> int:
    """Run the command-line interface for reaction-time detection.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        keypoints = np.load(args.keypoints)
    except Exception as exc:
        LOGGER.error("Failed to load keypoints from %s: %s", args.keypoints, exc)
        return 1

    try:
        reaction_time = detect_reaction_time(keypoints, args.audio_beep_frame, args.fps)
    except Exception as exc:
        LOGGER.error("Reaction-time detection failed: %s", exc)
        return 1

    payload: Dict[str, Any] = {"reaction_time_ms": reaction_time}
    if args.entry_frame is not None:
        try:
            payload.update(detect_first_stroke_time(keypoints, args.entry_frame, args.fps))
        except Exception as exc:
            LOGGER.error("First-stroke detection failed: %s", exc)
            return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
