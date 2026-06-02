# This file renders annotated SwimVision videos with skeletons, angles, phases, and deviation flags.
"""Annotated overlay rendering for SwimVision analysis outputs."""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

FLAG_COLORS = {
    "OPTIMAL": (0, 200, 0),
    "MINOR": (0, 200, 200),
    "SIGNIFICANT": (0, 140, 255),
    "CRITICAL": (0, 0, 220),
    "UNUSED": (150, 150, 150),
}
FLAG_PRIORITY = {"UNUSED": -1, "OPTIMAL": 0, "MINOR": 1, "SIGNIFICANT": 2, "CRITICAL": 3}
PHASE_NAMES = {
    "block_phase": "BLOCK",
    "flight_phase": "FLIGHT",
    "entry_phase": "ENTRY",
}
PHASE_METRIC_TO_JOINTS = {
    "block_phase": {
        "front_knee_angle": [23, 25, 27],
        "rear_knee_angle": [24, 26, 28],
        "hip_angle": [11, 12, 23, 24, 25, 26],
        "torso_lean": [11, 12, 23, 24],
    },
    "flight_phase": {
        "body_linearity": [11, 12, 23, 24, 27, 28],
        "entry_angle": [15, 16, 27, 28],
        "elbow_extension": [11, 12, 13, 14, 15, 16],
    },
    "entry_phase": {
        "streamline_angle": [15, 16, 23, 24],
        "elbow_lock_angle": [11, 12, 13, 14, 15, 16],
    },
}
JOINT_ANGLE_LABELS = {
    13: ("left_elbow_angle", "L elbow"),
    14: ("right_elbow_angle", "R elbow"),
    25: ("front_knee_angle", "Front knee"),
    26: ("rear_knee_angle", "Rear knee"),
    23: ("hip_angle", "Hip"),
    24: ("hip_angle", "Hip"),
    15: ("streamline_angle", "Streamline"),
    16: ("streamline_angle", "Streamline"),
    11: ("torso_lean", "Torso"),
    12: ("torso_lean", "Torso"),
}
POSE_CONNECTIONS: Sequence[Tuple[int, int]] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


def _resolve_connectivity() -> Sequence[Tuple[int, int]]:
    """Return the MediaPipe pose connectivity list."""
    return POSE_CONNECTIONS


def _validate_crop(crop: Sequence[int] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Validate crop bounds against a frame resolution."""
    if crop is None:
        return None
    x_coord, y_coord, crop_width, crop_height = [int(value) for value in crop]
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError("Crop width and height must be positive.")
    if x_coord < 0 or y_coord < 0:
        raise ValueError("Crop x and y must be non-negative.")
    if x_coord + crop_width > width or y_coord + crop_height > height:
        raise ValueError(
            f"Crop {(x_coord, y_coord, crop_width, crop_height)} exceeds frame bounds {(width, height)}."
        )
    return x_coord, y_coord, crop_width, crop_height


def _apply_crop(frame: np.ndarray, crop: Sequence[int] | None) -> np.ndarray:
    """Crop a frame when a crop tuple is provided."""
    if crop is None:
        return frame
    height, width = frame.shape[:2]
    x_coord, y_coord, crop_width, crop_height = _validate_crop(crop, width, height)
    return frame[y_coord : y_coord + crop_height, x_coord : x_coord + crop_width].copy()


def _extract_video_to_temp_frames(
    video_path: Path, crop: Sequence[int] | None = None
) -> tuple[tempfile.TemporaryDirectory, str]:
    """Convert a video file into temporary frame images for overlay rendering."""
    temp_dir = tempfile.TemporaryDirectory(prefix="swimvision_overlay_frames_")
    output_dir = Path(temp_dir.name)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        temp_dir.cleanup()
        raise RuntimeError(f"OpenCV could not open video '{video_path}'.")
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame = _apply_crop(frame, crop)
            frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"cv2.imwrite returned False for '{frame_path}'.")
            frame_index += 1
    finally:
        capture.release()
    return temp_dir, str(output_dir)


def _infer_related_path(angles_path: str, suffix: str) -> Path:
    """Infer a sibling artifact path from an angles CSV name."""
    angles_file = Path(angles_path)
    clip_stem = angles_file.stem
    if clip_stem.endswith("_angles"):
        clip_stem = clip_stem[: -len("_angles")]
    return angles_file.parent / f"{clip_stem}_{suffix}"


def _normalize_deviation_payload(deviations: Any) -> Dict[str, Dict[str, str]]:
    """Convert deviation input into phase -> metric -> flag form."""
    normalized: Dict[str, Dict[str, str]] = {}
    if isinstance(deviations, dict):
        for phase in ("block_phase", "flight_phase", "entry_phase"):
            phase_rows = deviations.get(phase, [])
            metric_map: Dict[str, str] = {}
            if isinstance(phase_rows, list):
                for row in phase_rows:
                    if isinstance(row, dict) and "metric" in row and "flag" in row:
                        metric_map[str(row["metric"])] = str(row["flag"])
            normalized[phase] = metric_map
    return normalized


def _phase_for_frame(frame_index: int, phase_boundaries: Dict[str, int]) -> str:
    """Resolve the named phase associated with a frame index."""
    if phase_boundaries["block_start"] <= frame_index <= phase_boundaries["block_end"]:
        return "block_phase"
    if phase_boundaries["flight_start"] <= frame_index <= phase_boundaries["flight_end"]:
        return "flight_phase"
    return "entry_phase"


def _cycle_for_frame(frame_index: int, stroke_boundaries: Dict[str, Any]) -> int | None:
    """Resolve which stroke cycle a frame belongs to. Returns cycle index or None."""
    cycles = stroke_boundaries.get("cycles", [])
    if not isinstance(cycles, list):
        return None
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        start = min(
            cycle.get("left_entry_frame", 0),
            cycle.get("right_entry_frame", 0),
        )
        end = max(
            cycle.get("left_recovery_end_frame", 0),
            cycle.get("right_recovery_end_frame", 0),
        )
        if start <= frame_index <= end:
            return cycle.get("cycle_index")
    return None


def _joint_flags_for_phase(phase_name: str, deviations: Dict[str, Dict[str, str]]) -> Dict[int, str]:
    """Map deviation flags onto joints for a single phase."""
    flags = {joint_index: "UNUSED" for joint_index in range(33)}
    metric_to_flag = deviations.get(phase_name, {})
    for metric, joints in PHASE_METRIC_TO_JOINTS.get(phase_name, {}).items():
        flag = metric_to_flag.get(metric, "UNUSED")
        for joint_index in joints:
            if FLAG_PRIORITY[flag] > FLAG_PRIORITY[flags[joint_index]]:
                flags[joint_index] = flag
    return flags


def _pixel_coord(point: np.ndarray, width: int, height: int) -> Tuple[int, int]:
    """Convert normalized or pixel coordinates to integer pixel positions."""
    x_coord, y_coord = float(point[0]), float(point[1])
    if 0.0 <= x_coord <= 1.0 and 0.0 <= y_coord <= 1.0:
        return int(x_coord * width), int(y_coord * height)
    return int(x_coord), int(y_coord)


def render_overlay(
    frames_dir: str,
    keypoints: np.ndarray,
    angles_df: pd.DataFrame,
    deviations: Any,
    phase_boundaries: Dict[str, int],
    output_path: str,
) -> None:
    """Render an annotated overlay video from frames and computed metrics."""
    frame_paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No extracted frames found in '{frames_dir}'.")
    if len(frame_paths) != keypoints.shape[0]:
        raise ValueError(
            f"Frame count {len(frame_paths)} does not match keypoint count {keypoints.shape[0]}."
        )

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to load first frame '{frame_paths[0]}'.")
    height, width = first_frame.shape[:2]
    fps = int(phase_boundaries.get("fps", 30))

    try:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize video writer '{output_path}': {exc}") from exc
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for '{output_path}'.")

    connectivity = _resolve_connectivity()
    normalized_deviations = _normalize_deviation_payload(deviations)

    try:
        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to load frame '{frame_path}'.")

            phase_name = _phase_for_frame(frame_index, phase_boundaries)
            joint_flags = _joint_flags_for_phase(phase_name, normalized_deviations)
            frame_keypoints = keypoints[frame_index, :, :2]
            frame_angles = angles_df.loc[frame_index]

            for start_joint, end_joint in connectivity:
                start_point = _pixel_coord(frame_keypoints[start_joint], width, height)
                end_point = _pixel_coord(frame_keypoints[end_joint], width, height)
                try:
                    cv2.line(frame, start_point, end_point, (200, 200, 200), 2)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to draw skeleton line {(start_joint, end_joint)} on frame {frame_index}: {exc}"
                    ) from exc

            for joint_index in range(33):
                point = _pixel_coord(frame_keypoints[joint_index], width, height)
                flag = joint_flags[joint_index]
                color = FLAG_COLORS[flag]
                try:
                    cv2.circle(frame, point, 5, color, -1)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to draw joint {joint_index} on frame {frame_index}: {exc}"
                    ) from exc

                if joint_index in JOINT_ANGLE_LABELS:
                    metric_name, label = JOINT_ANGLE_LABELS[joint_index]
                    angle_value = float(frame_angles[metric_name]) if metric_name in frame_angles.index else 0.0
                    try:
                        cv2.putText(
                            frame,
                            f"{label}: {angle_value:.1f}",
                            (point[0] + 6, point[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            frame,
                            flag,
                            (point[0] + 6, point[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to draw labels for joint {joint_index} on frame {frame_index}: {exc}"
                        ) from exc

            try:
                cv2.putText(
                    frame,
                    PHASE_NAMES[phase_name],
                    (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to draw phase label on frame {frame_index}: {exc}") from exc

            try:
                writer.write(frame)
            except Exception as exc:
                raise RuntimeError(f"Failed to encode frame {frame_index}: {exc}") from exc
    finally:
        writer.release()

    LOGGER.info("Annotated overlay written to %s", output_path)


def render_stroke_overlay(
    frames_dir: str,
    keypoints: np.ndarray,
    angles_df: pd.DataFrame,
    stroke_boundaries: Dict[str, Any],
    output_path: str,
) -> None:
    """Render a stroke-annotated overlay video with cycle labels and arm phase info.

    Args:
        frames_dir: Directory of extracted frame images.
        keypoints: Keypoint array [T, 33, 4].
        angles_df: Per-frame angle DataFrame.
        stroke_boundaries: Stroke cycle boundaries with per-arm phase info.
        output_path: Output MP4 path.
    """

    frame_paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No extracted frames found in '{frames_dir}'.")
    if len(frame_paths) != keypoints.shape[0]:
        raise ValueError(
            f"Frame count {len(frame_paths)} does not match keypoint count {keypoints.shape[0]}."
        )

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to load first frame '{frame_paths[0]}'.")
    height, width = first_frame.shape[:2]
    fps = int(stroke_boundaries.get("fps", 30))

    try:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize video writer '{output_path}': {exc}") from exc
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for '{output_path}'.")

    connectivity = _resolve_connectivity()

    try:
        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to load frame '{frame_path}'.")

            frame_keypoints = keypoints[frame_index, :, :2]

            # Draw skeleton
            for start_joint, end_joint in connectivity:
                start_point = _pixel_coord(frame_keypoints[start_joint], width, height)
                end_point = _pixel_coord(frame_keypoints[end_joint], width, height)
                cv2.line(frame, start_point, end_point, (200, 200, 200), 2)

            # Draw joints with default color
            for joint_index in range(33):
                point = _pixel_coord(frame_keypoints[joint_index], width, height)
                cv2.circle(frame, point, 4, (0, 200, 0), -1)

            # Highlight wrists (stroke analysis key)
            left_wrist = _pixel_coord(frame_keypoints[15], width, height)
            right_wrist = _pixel_coord(frame_keypoints[16], width, height)
            left_elbow = _pixel_coord(frame_keypoints[13], width, height)
            right_elbow = _pixel_coord(frame_keypoints[14], width, height)
            cv2.circle(frame, left_wrist, 7, (0, 255, 255), -1)
            cv2.circle(frame, right_wrist, 7, (255, 0, 255), -1)
            cv2.circle(frame, left_elbow, 6, (0, 200, 200), -1)
            cv2.circle(frame, right_elbow, 6, (200, 0, 200), -1)

            # Draw stroke cycle label
            cycle_idx = _cycle_for_frame(frame_index, stroke_boundaries)
            cycle_label = f"STROKE CYCLE {cycle_idx}" if cycle_idx is not None else "STROKE"
            cv2.putText(
                frame,
                cycle_label,
                (20, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw arm phase indicator
            cycles = stroke_boundaries.get("cycles", [])
            if isinstance(cycles, list) and cycle_idx is not None and cycle_idx < len(cycles):
                cycle = cycles[cycle_idx]
                if isinstance(cycle, dict):
                    left_phase = _arm_phase_for_frame(frame_index, cycle, side="left")
                    right_phase = _arm_phase_for_frame(frame_index, cycle, side="right")
                    cv2.putText(
                        frame,
                        f"L: {left_phase}",
                        (left_wrist[0] + 8, left_wrist[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"R: {right_phase}",
                        (right_wrist[0] + 8, right_wrist[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

            try:
                writer.write(frame)
            except Exception as exc:
                raise RuntimeError(f"Failed to encode frame {frame_index}: {exc}") from exc
    finally:
        writer.release()

    LOGGER.info("Stroke-annotated overlay written to %s", output_path)


def _arm_phase_for_frame(frame_index: int, cycle: Dict[str, Any], side: str = "left") -> str:
    """Determine the stroke arm phase for a given frame within a cycle.

    Args:
        frame_index: Current frame index.
        cycle: Stroke cycle boundary dictionary.
        side: 'left' or 'right'.

    Returns:
        Phase label string.
    """

    entry = cycle.get(f"{side}_entry_frame", 0)
    catch = cycle.get(f"{side}_catch_frame", 0)
    pull_end = cycle.get(f"{side}_pull_end_frame", 0)
    recovery_end = cycle.get(f"{side}_recovery_end_frame", 0)

    if frame_index < entry:
        return "pre-entry"
    if frame_index < catch:
        return "entry"
    if frame_index < pull_end:
        return "pull"
    if frame_index < recovery_end:
        return "recovery"
    return "post"


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for overlay rendering."""
    parser = argparse.ArgumentParser(description="Render SwimVision annotated overlay video.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--frames_dir", help="Directory of extracted frame images.")
    source_group.add_argument("--input", help="Input video path.")
    parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    parser.add_argument("--angles", required=True, help="Path to angles CSV.")
    parser.add_argument("--deviations", help="Path to aggregate deviation JSON.")
    parser.add_argument("--boundaries", help="Path to phase boundaries JSON.")
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional crop region in pixel coordinates for video inputs.",
    )
    parser.add_argument("--output", required=True, help="Output annotated MP4 path.")
    parser.add_argument(
        "--analysis_mode",
        choices=["dive", "stroke"],
        default="dive",
        help="Analysis mode: dive or stroke.",
    )
    parser.add_argument("--stroke_boundaries", help="Path to stroke cycle boundaries JSON.")
    return parser


def main() -> int:
    """Run the command-line interface for overlay rendering."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        keypoints = np.load(args.keypoints)
    except Exception as exc:
        LOGGER.error("Failed to load keypoints from %s: %s", args.keypoints, exc)
        return 1
    try:
        angles_df = pd.read_csv(args.angles, index_col=0)
    except Exception as exc:
        LOGGER.error("Failed to load angles from %s: %s", args.angles, exc)
        return 1

    temp_dir: tempfile.TemporaryDirectory | None = None
    frames_dir = args.frames_dir
    if args.input:
        try:
            temp_dir, frames_dir = _extract_video_to_temp_frames(Path(args.input), crop=args.crop)
        except Exception as exc:
            LOGGER.error("Failed to prepare video frames from %s: %s", args.input, exc)
            return 1

    try:
        if args.analysis_mode == "stroke":
            stroke_boundaries_path = args.stroke_boundaries or _infer_related_path(args.angles, "stroke_boundaries.json")
            with open(stroke_boundaries_path, "r", encoding="utf-8") as handle:
                stroke_boundaries = json.load(handle)
            render_stroke_overlay(
                str(frames_dir),
                keypoints,
                angles_df,
                stroke_boundaries,
                args.output,
            )
        else:
            deviations_path = args.deviations or _infer_related_path(args.angles, "deviations.json")
            boundaries_path = args.boundaries or _infer_related_path(args.angles, "boundaries.json")
            with open(deviations_path, "r", encoding="utf-8") as handle:
                deviations = json.load(handle)
            with open(boundaries_path, "r", encoding="utf-8") as handle:
                phase_boundaries = json.load(handle)
            render_overlay(
                str(frames_dir),
                keypoints,
                angles_df,
                deviations,
                phase_boundaries,
                args.output,
            )
        return 0
    except Exception as exc:
        LOGGER.error("Overlay rendering failed: %s", exc)
        return 1
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
