# This file ingests raw swim videos into frames and estimates coarse phase boundaries.
"""Video ingestion, frame extraction, and heuristic phase detection for SwimVision."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _safe_float(value: float, default: float) -> float:
    """Return a finite float or the provided default.

    Args:
        value: Candidate numeric value.
        default: Fallback value.

    Returns:
        A finite float.
    """

    return float(value) if np.isfinite(value) else float(default)


def analyze_entry_splash(video_path: str, entry_start: int, entry_end: int, crop: list[int] | None = None) -> float:
    """Quantify the splash at entry using frame differencing in the water region.

    Args:
        video_path: Path to the input video.
        entry_start: Frame index when entry begins.
        entry_end: Frame index when entry ends.
        crop: Optional crop region [x, y, w, h].

    Returns:
        Normalized splash score (higher means more splash).
    """

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return 0.0

    splash_scores = []
    prev_frame = None

    # We focus on the water surface area
    # If we don't have water surface detection here, we estimate it
    # from the lower part of the frame.

    for idx in range(entry_end + 1):
        success, frame = capture.read()
        if not success:
            break
        if idx < entry_start:
            continue

        if crop:
            frame = frame[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2]]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            splash_score = np.sum(thresh) / float(thresh.size)
            splash_scores.append(splash_score)

        prev_frame = gray

    capture.release()

    if not splash_scores:
        return 0.0

    # Return peak splash normalized to a 0-100 scale (heuristic)
    return float(np.max(splash_scores) * 100.0)


def _detect_water_surface(frame: np.ndarray) -> int:
    """Estimate the water surface row using edges and line detection.

    Args:
        frame: Input BGR frame.

    Returns:
        The detected or fallback water surface y-coordinate.
    """

    height, width = frame.shape[:2]
    lower_third = frame[(2 * height) // 3 :, :]

    try:
        gray = cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        LOGGER.warning("Failed to convert frame to grayscale for water detection: %s", exc)
        return int(height * 0.75)

    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=60,
            minLineLength=max(width // 3, 50),
            maxLineGap=20,
        )
    except Exception as exc:
        LOGGER.warning("Water surface detection failed during edge processing: %s", exc)
        return int(height * 0.75)

    if lines is None or len(lines) == 0:
        LOGGER.warning("Water surface not detected; using 75%% frame-height fallback.")
        return int(height * 0.75)

    candidates = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        if x2 == x1:
            continue
        slope = (y2 - y1) / float(x2 - x1)
        if abs(slope) < 0.15:
            avg_y = (y1 + y2) / 2.0
            line_length = float(np.hypot(x2 - x1, y2 - y1))
            candidates.append((line_length, avg_y))

    if not candidates:
        LOGGER.warning("No near-horizontal water line found; using 75%% frame-height fallback.")
        return int(height * 0.75)

    _, best_y = max(candidates, key=lambda item: item[0])
    return int((2 * height) // 3 + best_y)


def extract_frames(video_path: str, output_dir: str, fps: int = 30) -> Dict[str, Any]:
    """Extract frames from a video and estimate the water surface line.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory where extracted frames should be written.
        fps: Target extraction frames per second.

    Returns:
        Metadata describing the extracted sequence.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(video_path)
    capture: Optional[cv2.VideoCapture] = None

    try:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open video '{source_path}'.")
    except Exception as exc:
        raise RuntimeError(f"Failed to open video '{source_path}': {exc}") from exc

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    except Exception as exc:
        if capture is not None:
            capture.release()
        raise RuntimeError(f"Failed to read video metadata from '{source_path}': {exc}") from exc

    effective_source_fps = _safe_float(source_fps, float(fps))
    sampling_stride = max(int(round(effective_source_fps / max(fps, 1))), 1)
    LOGGER.info(
        "Extracting frames from %s at target %s FPS (source FPS %.2f, stride %s).",
        source_path,
        fps,
        effective_source_fps,
        sampling_stride,
    )

    total_written = 0
    water_surface_y: Optional[int] = None
    current_frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if current_frame_index % sampling_stride == 0:
                if water_surface_y is None:
                    water_surface_y = _detect_water_surface(frame)
                frame_name = output_path / f"frame_{total_written:06d}.jpg"
                try:
                    if not cv2.imwrite(str(frame_name), frame):
                        raise IOError("cv2.imwrite returned False.")
                except Exception as exc:
                    raise RuntimeError(f"Failed to write frame '{frame_name}': {exc}") from exc
                total_written += 1
            current_frame_index += 1
    except Exception as exc:
        raise RuntimeError(f"Failed during frame extraction for '{source_path}': {exc}") from exc
    finally:
        if capture is not None:
            capture.release()

    metadata = {
        "fps": int(fps),
        "resolution": [width, height],
        "total_frames": int(total_written),
        "water_surface_y": int(water_surface_y if water_surface_y is not None else int(height * 0.75)),
        "output_dir": str(output_path),
        "source_video": str(source_path),
        "source_total_frames": int(frame_count),
    }
    LOGGER.info("Frame extraction complete. Wrote %s frames to %s.", total_written, output_path)
    return metadata


def _compute_joint_speed(sequence: np.ndarray) -> np.ndarray:
    """Compute per-frame point speeds from a sequence of 2D points.

    Args:
        sequence: Array of shape ``[T, J, 2]`` or ``[T, 2]``.

    Returns:
        Speed magnitudes aligned to frame indices.
    """

    if sequence.ndim == 2:
        diffs = np.diff(sequence, axis=0, prepend=sequence[0:1])
        return np.linalg.norm(diffs, axis=1)
    diffs = np.diff(sequence, axis=0, prepend=sequence[0:1])
    return np.linalg.norm(diffs, axis=2)


def detect_phase_boundaries(
    keypoints_path: str, confidence_path: str, metadata: Dict[str, Any], min_flight_start: int = 90
) -> Dict[str, int]:
    """Detect swim-start phase boundaries from torso translation, ankle lift, and confidence.

    Args:
        keypoints_path: Path to a ``[T, 33, 4]`` keypoint array.
        confidence_path: Path to a ``[T, 33]`` confidence array.
        metadata: Metadata dictionary from frame extraction.
        min_flight_start: Earliest allowed frame index for flight-phase onset.

    Returns:
        A dictionary of inclusive phase boundary frame indices.
    """

    try:
        keypoints = np.load(keypoints_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load keypoints from '{keypoints_path}': {exc}") from exc
    try:
        confidence = np.load(confidence_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load confidence from '{confidence_path}': {exc}") from exc

    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(
            f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}."
        )
    if confidence.shape != keypoints.shape[:2]:
        raise ValueError(
            "Confidence array must match keypoint frames and joints: "
            f"{tuple(confidence.shape)} vs {tuple(keypoints.shape[:2])}."
        )

    frame_confidence = confidence.mean(axis=1)
    torso_center_x = keypoints[:, [11, 12, 23, 24], 0].mean(axis=1)
    ankle_mid_y = keypoints[:, [27, 28], 1].mean(axis=1)

    kernel = np.ones(5, dtype=np.float32) / 5.0
    smoothed_confidence = np.convolve(frame_confidence, kernel, mode="same")
    smoothed_torso_x = np.convolve(torso_center_x, kernel, mode="same")
    smoothed_ankle_y = np.convolve(ankle_mid_y, kernel, mode="same")
    torso_x_velocity = np.diff(smoothed_torso_x, prepend=smoothed_torso_x[0])

    baseline_start = min(int(min_flight_start), max(len(smoothed_confidence) - 1, 0))
    baseline_end = min(len(smoothed_confidence), max(baseline_start + 20, baseline_start + 140))
    baseline_slice = slice(baseline_start, baseline_end)
    baseline_torso_x = float(np.median(smoothed_torso_x[baseline_slice]))
    baseline_ankle_y = float(np.median(smoothed_ankle_y[baseline_slice]))

    flight_start = None
    for index in range(int(min_flight_start), len(smoothed_confidence) - 3):
        if (
            np.all(smoothed_torso_x[index : index + 3] > baseline_torso_x + 0.08)
            and np.all(smoothed_ankle_y[index : index + 3] < baseline_ankle_y - 0.16)
            and np.all(smoothed_confidence[index : index + 3] > 0.8)
        ):
            flight_start = index
            break

    if flight_start is None:
        for index in range(int(min_flight_start), len(smoothed_confidence) - 5):
            if np.all(smoothed_confidence[index : index + 5] > 0.79):
                flight_start = index
                break

    if flight_start is None:
        flight_start = int(min_flight_start)

    collapse_index = None
    for index in range(flight_start + 5, len(smoothed_confidence) - 1):
        if np.all(torso_x_velocity[index : index + 2] < -0.05):
            collapse_index = index
            break
        if np.all(smoothed_confidence[index : index + 2] < 0.2):
            collapse_index = max(flight_start + 1, index - 1)
            break

    if collapse_index is None:
        collapse_index = len(smoothed_confidence) - 1

    peak_window = smoothed_torso_x[flight_start : collapse_index + 1]
    if peak_window.size > 0:
        peak_index = int(flight_start + int(np.argmax(peak_window)))
        flight_end = min(peak_index, collapse_index)
    else:
        flight_end = min(flight_start, collapse_index)

    entry_start = min(flight_end + 1, len(smoothed_confidence) - 1)
    entry_end = int(collapse_index)
    if entry_end < entry_start:
        entry_end = entry_start

    boundaries = {
        "block_start": 0,
        "block_end": int(flight_start - 1),
        "flight_start": int(flight_start),
        "flight_end": int(flight_end),
        "entry_start": int(entry_start),
        "entry_end": int(entry_end),
        "fps": int(metadata.get("fps", 30)),
    }
    LOGGER.info("Detected phase boundaries: %s", boundaries)
    LOGGER.info("Metadata context used for detection: %s", metadata)
    return boundaries


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for video ingestion utilities.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="SwimVision ingestion utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-frames", help="Extract video frames.")
    extract_parser.add_argument("--input", required=True, help="Input video path.")
    extract_parser.add_argument("--output", required=True, help="Frame output directory.")
    extract_parser.add_argument("--fps", type=int, default=30, help="Target frame rate.")
    extract_parser.add_argument(
        "--metadata-output",
        help="Optional path to write extraction metadata JSON.",
    )

    detect_parser = subparsers.add_parser(
        "detect-phases", help="Detect phase boundaries from extracted keypoints."
    )
    detect_parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    detect_parser.add_argument("--confidence", required=True, help="Path to confidence .npy file.")
    detect_parser.add_argument(
        "--metadata-json",
        help="Optional metadata JSON path from frame extraction.",
    )
    detect_parser.add_argument(
        "--output",
        help="Optional path to write detected boundary JSON.",
    )
    return parser


def _load_metadata(path: Optional[str]) -> Dict[str, Any]:
    """Load metadata JSON if provided, otherwise return an empty context.

    Args:
        path: Optional metadata JSON path.

    Returns:
        Parsed metadata dictionary or an empty dictionary.
    """

    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to load metadata JSON '{path}': {exc}") from exc


def _write_json(payload: Dict[str, Any], path: Optional[str]) -> None:
    """Write a JSON payload to disk when a path is supplied.

    Args:
        payload: Serializable dictionary to persist.
        path: Target file path or ``None``.

    Returns:
        None.
    """

    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to write JSON output '{path}': {exc}") from exc


def main() -> int:
    """Run the SwimVision ingestion command-line interface.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.command == "extract-frames":
            metadata = extract_frames(args.input, args.output, args.fps)
            _write_json(metadata, args.metadata_output)
            print(json.dumps(metadata, indent=2))
            return 0
        metadata = _load_metadata(args.metadata_json)
        boundaries = detect_phase_boundaries(args.keypoints, args.confidence, metadata)
        _write_json(boundaries, args.output)
        print(json.dumps(boundaries, indent=2))
        return 0
    except Exception as exc:
        LOGGER.error("Ingestion command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
