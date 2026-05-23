# This file extracts normalized MediaPipe pose keypoints and confidence scores from swim frames.
"""MediaPipe-based pose extraction for SwimVision."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Sequence, Tuple


os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PYTHON = PROJECT_ROOT / ".mp312" / "bin" / "python"
LEGACY_SITE_PACKAGES = PROJECT_ROOT / ".mp312" / "lib" / "python3.12" / "site-packages"
LEGACY_LITE_MODEL_PATH = (
    LEGACY_SITE_PACKAGES
    / "mediapipe"
    / "modules"
    / "pose_landmark"
    / "pose_landmark_lite.tflite"
)
PROJECT_LITE_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmark_lite.tflite"

if LEGACY_SITE_PACKAGES.exists() and str(LEGACY_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LEGACY_SITE_PACKAGES))

if not LEGACY_LITE_MODEL_PATH.exists() and PROJECT_LITE_MODEL_PATH.exists():
    LEGACY_LITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_LITE_MODEL_PATH, LEGACY_LITE_MODEL_PATH)

if LEGACY_PYTHON.exists() and Path(sys.executable).resolve() != LEGACY_PYTHON.resolve():
    os.execvpe(
        str(LEGACY_PYTHON),
        [str(LEGACY_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        os.environ,
    )

import cv2
import mediapipe.python.solutions.pose as mp_pose
import numpy as np
from mediapipe.framework.formats import landmark_pb2


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

POSE_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}
_MODEL_COMPLEXITY_OVERRIDE = 2
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mpg", ".mpeg"}


def _validate_crop(crop: Sequence[int] | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Validate crop bounds against a frame resolution.

    Args:
        crop: Optional crop tuple in ``(x, y, w, h)`` format.
        width: Source frame width in pixels.
        height: Source frame height in pixels.

    Returns:
        Validated crop tuple or ``None`` when cropping is disabled.
    """

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
    """Crop a frame to the requested region when provided.

    Args:
        frame: Input frame array.
        crop: Optional crop tuple in ``(x, y, w, h)`` format.

    Returns:
        Cropped frame or the original frame when no crop is requested.
    """

    if crop is None:
        return frame
    height, width = frame.shape[:2]
    x_coord, y_coord, crop_width, crop_height = _validate_crop(crop, width, height)
    return frame[y_coord : y_coord + crop_height, x_coord : x_coord + crop_width].copy()


def _iter_frame_paths(frames_dir: Path) -> List[Path]:
    """Collect frame image paths from a directory.

    Args:
        frames_dir: Directory containing extracted frame images.

    Returns:
        Sorted list of frame image paths.
    """

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        frame_paths = sorted(frames_dir.glob("*.jpg"))
    return frame_paths


def _iter_video_paths(raw_dir: Path) -> List[Path]:
    """Collect supported raw video files from a directory."""

    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def _extract_video_to_temp_frames(
    video_path: Path, crop: Sequence[int] | None = None
) -> Tuple[tempfile.TemporaryDirectory, List[Path]]:
    """Convert a video file into temporary frame images for extraction.

    Args:
        video_path: Input video path.
        crop: Optional crop tuple in ``(x, y, w, h)`` format.

    Returns:
        A temporary directory handle and sorted frame paths.
    """

    temp_dir = tempfile.TemporaryDirectory(prefix="swimvision_frames_")
    capture = None
    frame_paths: List[Path] = []

    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open '{video_path}'.")
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            try:
                frame = _apply_crop(frame, crop)
            except Exception as exc:
                raise RuntimeError(f"Failed to crop video frame {frame_index}: {exc}") from exc
            frame_path = Path(temp_dir.name) / f"frame_{frame_index:06d}.jpg"
            try:
                if not cv2.imwrite(str(frame_path), frame):
                    raise IOError("cv2.imwrite returned False.")
            except Exception as exc:
                raise RuntimeError(f"Failed to write temp frame '{frame_path}': {exc}") from exc
            frame_paths.append(frame_path)
            frame_index += 1
    except Exception:
        temp_dir.cleanup()
        raise
    finally:
        if capture is not None:
            capture.release()

    return temp_dir, frame_paths


def _load_frame(frame_path: Path) -> np.ndarray:
    """Load a frame image from disk.

    Args:
        frame_path: Path to the image file.

    Returns:
        Loaded BGR image array.
    """

    try:
        frame = cv2.imread(str(frame_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to read frame '{frame_path}': {exc}") from exc
    if frame is None:
        raise RuntimeError(f"OpenCV returned no data for frame '{frame_path}'.")
    return frame


def _normalize_landmarks(
    landmarks: Sequence[Any], width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert MediaPipe landmarks into normalized keypoint and confidence arrays.

    Args:
        landmarks: Sequence of 33 MediaPipe landmarks.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        A tuple of ``[33, 4]`` keypoints and ``[33]`` confidence scores.
    """

    keypoints = np.zeros((33, 4), dtype=np.float32)
    confidence = np.zeros((33,), dtype=np.float32)

    for index, landmark in enumerate(landmarks):
        x_coord = min(max(float(landmark.x), 0.0), 1.0)
        y_coord = min(max(float(landmark.y), 0.0), 1.0)
        z_coord = float(landmark.z)
        visibility = min(max(float(landmark.visibility), 0.0), 1.0)
        keypoints[index] = np.array([x_coord, y_coord, z_coord, visibility], dtype=np.float32)
        confidence[index] = visibility

    _ = width, height
    return keypoints, confidence


def _create_pose_estimator(model_complexity: int) -> mp_pose.Pose:
    """Create a legacy MediaPipe Pose estimator for a chosen complexity level.

    Args:
        model_complexity: MediaPipe model complexity level.

    Returns:
        Initialized MediaPipe Pose estimator.
    """

    try:
        return mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize MediaPipe Pose with model_complexity={model_complexity}: {exc}"
        ) from exc


def _initialize_pose_with_fallback() -> tuple[mp_pose.Pose, int]:
    """Initialize MediaPipe Pose with complexity fallback from 1 to 0.

    Args:
        None.

    Returns:
        Tuple of pose estimator and chosen complexity.
    """

    preferred_complexities = [1, 0]
    if _MODEL_COMPLEXITY_OVERRIDE == 0:
        preferred_complexities = [0]

    last_error: Exception | None = None
    for complexity in preferred_complexities:
        try:
            pose = _create_pose_estimator(complexity)
            LOGGER.info("Initialized MediaPipe Pose with model_complexity=%s.", complexity)
            return pose, complexity
        except Exception as exc:
            last_error = exc
            LOGGER.warning("MediaPipe Pose init failed at complexity %s: %s", complexity, exc)

    raise RuntimeError(f"All MediaPipe Pose initialization attempts failed: {last_error}")


def extract_keypoints(
    frames_dir: str, output_dir: str, clip_id: str, crop: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized pose keypoints and confidence scores from frames.

    Args:
        frames_dir: Directory of frame images or a video file path.
        output_dir: Directory where ``.npy`` outputs will be written.
        clip_id: Clip identifier used for output filenames.
        crop: Optional crop tuple in ``(x, y, w, h)`` format.

    Returns:
        A tuple containing keypoints ``[T, 33, 4]`` and confidence ``[T, 33]`` arrays.
    """

    source_path = Path(frames_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temp_dir: tempfile.TemporaryDirectory | None = None
    frame_paths: List[Path]
    if source_path.is_file():
        LOGGER.info("Input %s is a video file; converting to temporary frames.", source_path)
        try:
            temp_dir, frame_paths = _extract_video_to_temp_frames(source_path, crop=crop)
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare video frames from '{source_path}': {exc}") from exc
    else:
        frame_paths = _iter_frame_paths(source_path)
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in '{source_path}'.")

    keypoints_per_frame: List[np.ndarray] = []
    confidence_per_frame: List[np.ndarray] = []

    try:
        pose, _ = _initialize_pose_with_fallback()
    except Exception as exc:
        if temp_dir is not None:
            temp_dir.cleanup()
        raise RuntimeError(f"Failed to initialize MediaPipe Pose: {exc}") from exc

    try:
        for frame_index, frame_path in enumerate(frame_paths):
            frame = _load_frame(frame_path)
            try:
                frame = _apply_crop(frame, crop if not source_path.is_file() else None)
            except Exception as exc:
                raise RuntimeError(f"Failed to crop frame '{frame_path}': {exc}") from exc
            height, width = frame.shape[:2]
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to convert frame '{frame_path}' to RGB: {exc}"
                ) from exc
            try:
                result = pose.process(rgb_frame)
            except Exception as exc:
                raise RuntimeError(
                    f"MediaPipe Pose failed on frame '{frame_path}': {exc}"
                ) from exc

            if result.pose_landmarks is None:
                # If we have multiple swimmers, we could check multi_pose_landmarks
                # but legacy Solutions API only supports one.
                # However, MediaPipe can return multiple landmarks in some versions.
                # Here we stick to the most confident one.
                LOGGER.warning("No pose detected in frame %s (%s).", frame_index, frame_path.name)
                keypoints = np.zeros((33, 4), dtype=np.float32)
                confidence = np.zeros((33,), dtype=np.float32)
            else:
                keypoints, confidence = _normalize_landmarks(
                    result.pose_landmarks.landmark,
                    width,
                    height,
                )

                # Temporal consistency: if we have a previous detection, check for jumps
                if keypoints_per_frame:
                    prev_kp = keypoints_per_frame[-1]
                    # Only check if previous frame had a detection
                    if np.max(prev_kp[:, 3]) > 0.1:
                        # Compute distance for well-tracked points
                        mask = (confidence > 0.5) & (prev_kp[:, 3] > 0.5)
                        if np.any(mask):
                            dist = np.linalg.norm(keypoints[mask, :2] - prev_kp[mask, :2], axis=1).mean()
                            if dist > 0.15: # Significant jump in normalized coordinates
                                LOGGER.warning("Pose jump detected at frame %s (dist=%.3f). Likely background interference.", frame_index, dist)
                                # If confidence is lower than previous, ignore this detection
                                if np.mean(confidence) < np.mean(prev_kp[:, 3]) * 0.8:
                                    LOGGER.info("Ignoring low-confidence jump.")
                                    keypoints = np.zeros((33, 4), dtype=np.float32)
                                    confidence = np.zeros((33,), dtype=np.float32)

            mean_confidence = float(np.mean(confidence))
            LOGGER.info("Frame %s mean confidence: %.3f", frame_index, mean_confidence)
            if mean_confidence < 0.5:
                LOGGER.warning("Frame %s has low mean confidence: %.3f", frame_index, mean_confidence)

            keypoints_per_frame.append(keypoints)
            confidence_per_frame.append(confidence)
    except Exception as exc:
        raise RuntimeError(f"Keypoint extraction failed for '{frames_dir}': {exc}") from exc
    finally:
        try:
            pose.close()
        except Exception as exc:
            LOGGER.warning("Failed to close MediaPipe Pose cleanly: %s", exc)
        if temp_dir is not None:
            temp_dir.cleanup()

    keypoints_array = np.stack(keypoints_per_frame).astype(np.float32)
    confidence_array = np.stack(confidence_per_frame).astype(np.float32)

    keypoints_path = output_path / f"{clip_id}_keypoints.npy"
    confidence_path = output_path / f"{clip_id}_confidence.npy"
    try:
        np.save(keypoints_path, keypoints_array)
    except Exception as exc:
        raise RuntimeError(f"Failed to save keypoints to '{keypoints_path}': {exc}") from exc
    try:
        np.save(confidence_path, confidence_array)
    except Exception as exc:
        raise RuntimeError(f"Failed to save confidence to '{confidence_path}': {exc}") from exc

    LOGGER.info("Saved keypoints to %s", keypoints_path)
    LOGGER.info("Saved confidence to %s", confidence_path)
    return keypoints_array, confidence_array


def save_crop_preview(input_path: str, crop: Sequence[int], output_path: str = "results/crop_preview.jpg") -> str:
    """Save a preview image showing the requested crop on the first frame.

    Args:
        input_path: Video path or frame-directory path.
        crop: Crop tuple in ``(x, y, w, h)`` format.
        output_path: Path where the preview image should be written.

    Returns:
        The written preview image path.
    """

    source_path = Path(input_path)
    if source_path.is_file():
        capture = None
        try:
            capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                raise ValueError(f"OpenCV could not open '{source_path}'.")
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Failed to read the first frame from '{source_path}'.")
        except Exception as exc:
            raise RuntimeError(f"Failed to prepare crop preview from '{source_path}': {exc}") from exc
        finally:
            if capture is not None:
                capture.release()
    else:
        frame_paths = _iter_frame_paths(source_path)
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in '{source_path}'.")
        frame = _load_frame(frame_paths[0])

    try:
        cropped_frame = _apply_crop(frame, crop)
    except Exception as exc:
        raise RuntimeError(f"Failed to apply preview crop: {exc}") from exc

    preview_path = Path(output_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not cv2.imwrite(str(preview_path), cropped_frame):
            raise IOError("cv2.imwrite returned False.")
    except Exception as exc:
        raise RuntimeError(f"Failed to save crop preview to '{preview_path}': {exc}") from exc
    LOGGER.info("Saved crop preview to %s", preview_path)
    return str(preview_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for keypoint extraction.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Extract swim-start keypoints with MediaPipe.")
    parser.add_argument(
        "--input",
        help="Input frame directory or raw video path.",
    )
    parser.add_argument("--output", required=True, help="Output directory for .npy files.")
    parser.add_argument("--clip_id", help="Clip identifier for saved arrays.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process every supported raw video in data/raw/ or the provided input directory.",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional crop region in pixel coordinates.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save a single cropped preview frame to results/crop_preview.jpg without extraction.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use MediaPipe model complexity 0 instead of 1 for faster inference.",
    )
    return parser


def auto_detect_crop(video_path: Path, swimmer_index: int = 0) -> tuple[int, int, int, int] | None:
    """Heuristically detect the target swimmer and return an optimal crop.

    Args:
        video_path: Path to the raw video.

    Returns:
        Crop tuple (x, y, w, h) or None.
    """

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None

    # Sample a frame from the beginning where the swimmer is on the block
    # We skip first few frames for stability
    for _ in range(5): capture.read()
    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        return None

    height, width = frame.shape[:2]

    # Heuristic: Swimmers are usually in the center or slightly offset
    # We run MediaPipe Pose on the full frame to find the main person
    try:
        pose, _ = _initialize_pose_with_fallback()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        pose.close()

        # For legacy MediaPipe, multi-person is hard.
        # We can try to detect multiple people by running on sub-regions
        # or using a different model, but for now we improve the logic
        # to find the "swimmer-like" person.

        if result.pose_landmarks:
            # Find the bounding box of the detected landmarks
            lms = result.pose_landmarks.landmark
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Add padding
            w = (x_max - x_min) * 1.5
            h = (y_max - y_min) * 1.5
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            crop_x = int(max(0, (center_x - w/2) * width))
            crop_y = int(max(0, (center_y - h/2) * height))
            crop_w = int(min(width - crop_x, w * width))
            crop_h = int(min(height - crop_y, h * height))

            LOGGER.info("Auto-detected crop: %s", (crop_x, crop_y, crop_w, crop_h))
            return crop_x, crop_y, crop_w, crop_h

    except Exception as exc:
        LOGGER.warning("Auto-crop detection failed: %s", exc)

    return None


def main() -> int:
    """Run the keypoint extraction command-line interface.

    Args:
        None.

    Returns:
        Exit status code.
    """

    global _MODEL_COMPLEXITY_OVERRIDE

    parser = build_arg_parser()
    args = parser.parse_args()
    _MODEL_COMPLEXITY_OVERRIDE = 0 if args.fast else 1

    try:
        if args.preview:
            if args.crop is None:
                raise ValueError("--crop is required when using --preview.")
            if not args.input:
                raise ValueError("--input is required when using --preview.")
            save_crop_preview(args.input, args.crop)
            return 0
        if args.batch:
            raw_dir = Path(args.input) if args.input else PROJECT_ROOT / "data" / "raw"
            if not raw_dir.exists() or not raw_dir.is_dir():
                raise ValueError(f"Batch input directory '{raw_dir}' does not exist.")
            video_paths = _iter_video_paths(raw_dir)
            if not video_paths:
                raise FileNotFoundError(f"No supported videos found in '{raw_dir}'.")
            for video_path in video_paths:
                clip_id = video_path.stem
                LOGGER.info("Batch extracting %s with clip_id=%s", video_path, clip_id)
                extract_keypoints(str(video_path), args.output, clip_id, crop=args.crop)
            return 0
        if not args.input:
            raise ValueError("--input is required unless --batch is used.")
        if not args.clip_id:
            raise ValueError("--clip_id is required unless --batch is used.")
        extract_keypoints(args.input, args.output, args.clip_id, crop=args.crop)
        return 0
    except Exception as exc:
        LOGGER.error("Extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
