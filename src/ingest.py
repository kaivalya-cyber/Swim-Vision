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
    return float(value) if np.isfinite(value) else float(default)


def _detect_water_surface(frame: np.ndarray) -> int:
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
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=60, minLineLength=max(width // 3, 50), maxLineGap=20)
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
    LOGGER.info("Extracting frames from %s at target %s FPS (source FPS %.2f, stride %s).", source_path, fps, effective_source_fps, sampling_stride)
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


def detect_phase_boundaries(
    keypoints_path: str, confidence_path: str, metadata: Dict[str, Any], min_flight_start: int = 90
) -> Dict[str, int]:
    try:
        keypoints = np.load(keypoints_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load keypoints from '{keypoints_path}': {exc}") from exc
    try:
        confidence = np.load(confidence_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load confidence from '{confidence_path}': {exc}") from exc
    if keypoints.ndim != 3 or keypoints.shape[1:] != (33, 4):
        raise ValueError(f"Expected keypoints shape [T, 33, 4], got {tuple(keypoints.shape)}.")
    if confidence.shape != keypoints.shape[:2]:
        raise ValueError(f"Confidence array must match: {tuple(confidence.shape)} vs {tuple(keypoints.shape[:2])}.")
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
        if (np.all(smoothed_torso_x[index : index + 3] > baseline_torso_x + 0.08)
                and np.all(smoothed_ankle_y[index : index + 3] < baseline_ankle_y - 0.16)
                and np.all(smoothed_confidence[index : index + 3] > 0.8)):
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
    return boundaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SwimVision ingestion utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-frames", help="Extract video frames.")
    extract_parser.add_argument("--input", required=True, help="Input video path.")
    extract_parser.add_argument("--output", required=True, help="Frame output directory.")
    extract_parser.add_argument("--fps", type=int, default=30, help="Target frame rate.")
    extract_parser.add_argument("--metadata-output", help="Optional path to write extraction metadata JSON.")

    detect_parser = subparsers.add_parser("detect-phases", help="Detect phase boundaries from extracted keypoints.")
    detect_parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    detect_parser.add_argument("--confidence", required=True, help="Path to confidence .npy file.")
    detect_parser.add_argument("--metadata-json", help="Optional metadata JSON path from frame extraction.")
    detect_parser.add_argument("--output", help="Optional path to write detected boundary JSON.")

    stroke_detect_parser = subparsers.add_parser("detect-strokes", help="Detect stroke cycles from extracted keypoints.")
    stroke_detect_parser.add_argument("--keypoints", required=True, help="Path to keypoints .npy file.")
    stroke_detect_parser.add_argument("--output", help="Path to write stroke cycle boundary JSON.")
    stroke_detect_parser.add_argument("--stroke_start_frame", type=int, default=0, help="Frame index where stroke analysis begins.")
    stroke_detect_parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate.")
    stroke_detect_parser.add_argument("--width", type=int, default=1920, help="Frame width in pixels.")
    stroke_detect_parser.add_argument("--height", type=int, default=1080, help="Frame height in pixels.")
    stroke_detect_parser.add_argument(
        "--stroke_type",
        choices=["freestyle", "butterfly", "backstroke", "auto"],
        default="auto",
        help="Stroke type override (default: auto-detect).",
    )
    return parser


def _load_metadata(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to load metadata JSON '{path}': {exc}") from exc


def _write_json(payload: Dict[str, Any], path: Optional[str]) -> None:
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to write JSON output '{path}': {exc}") from exc


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        if args.command == "extract-frames":
            metadata = extract_frames(args.input, args.output, args.fps)
            _write_json(metadata, args.metadata_output)
            print(json.dumps(metadata, indent=2))
            return 0
        if args.command == "detect-strokes":
            from src.stroke.cycle_detection import detect_stroke_cycles, detect_stroke_type
            keypoints = np.load(args.keypoints)
            width = args.width
            height = args.height
            if args.stroke_type != "auto":
                stroke_type = args.stroke_type
                LOGGER.info("Using override stroke type: %s", stroke_type)
            else:
                stroke_type = detect_stroke_type(keypoints, start_frame=args.stroke_start_frame)
                LOGGER.info("Detected stroke type: %s", stroke_type)
            cycles = detect_stroke_cycles(keypoints, fps=args.fps, width=width, height=height, min_confidence=0.5)
            for cycle in cycles:
                cycle.stroke_type = stroke_type
            cycles_data = [{"cycle_index": c.cycle_index, "stroke_type": c.stroke_type, "left_entry_frame": c.left_entry_frame, "left_catch_frame": c.left_catch_frame, "left_pull_end_frame": c.left_pull_end_frame, "left_recovery_end_frame": c.left_recovery_end_frame, "right_entry_frame": c.right_entry_frame, "right_catch_frame": c.right_catch_frame, "right_pull_end_frame": c.right_pull_end_frame, "right_recovery_end_frame": c.right_recovery_end_frame, "body_roll_peak": c.body_roll_peak} for c in cycles]
            payload = {"cycles": cycles_data, "num_cycles": len(cycles), "fps": args.fps, "stroke_type": stroke_type}
            _write_json(payload, args.output)
            print(json.dumps(payload, indent=2))
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
