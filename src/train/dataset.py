# This file defines a PyTorch dataset for fixed-length SwimVision keypoint sequences and phase labels.
"""Dataset utilities for SwimVision phase-classification training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

PHASE_TO_INDEX = {"BLOCK": 0, "FLIGHT": 1, "ENTRY": 2, "OTHER": 3}


class SwimStartDataset(Dataset):
    """PyTorch dataset over fixed-length SwimVision keypoint sequences.

    Args:
        processed_dir: Directory containing ``*_keypoints.npy`` arrays.
        labels_csv: Metadata CSV path.
        sequence_length: Target number of frames per sample.
    """

    def __init__(self, processed_dir: str = "data/processed", labels_csv: str = "data/labels.csv", sequence_length: int = 90) -> None:
        """Initialize the dataset from processed keypoints and metadata labels.

        Args:
            processed_dir: Directory containing processed keypoint arrays.
            labels_csv: Metadata CSV path.
            sequence_length: Target number of frames per returned sequence.

        Returns:
            None.
        """

        self.processed_dir = PROJECT_ROOT / processed_dir
        self.labels_csv = PROJECT_ROOT / labels_csv
        self.sequence_length = int(sequence_length)

        try:
            self.labels_df = pd.read_csv(self.labels_csv)
        except Exception as exc:
            raise RuntimeError(f"Failed to read labels CSV '{self.labels_csv}': {exc}") from exc

        required_columns = {"clip_id", "swimmer_id", "level", "camera_angle", "notes"}
        missing_columns = required_columns.difference(self.labels_df.columns)
        if missing_columns:
            raise ValueError(
                f"labels.csv is missing required columns: {sorted(missing_columns)}."
            )

        self.records: List[Tuple[Path, int, Dict[str, str]]] = []
        for row in self.labels_df.to_dict(orient="records"):
            clip_id = str(row["clip_id"])
            keypoints_path = self.processed_dir / f"{clip_id}_keypoints.npy"
            if not keypoints_path.exists():
                LOGGER.warning("Skipping %s because %s does not exist.", clip_id, keypoints_path)
                continue
            phase_label = self._infer_phase_label(clip_id=clip_id, notes=str(row.get("notes", "")))
            self.records.append((keypoints_path, phase_label, row))

        if not self.records:
            raise RuntimeError(
                f"No training samples found in '{self.processed_dir}' matching '{self.labels_csv}'."
            )

    @staticmethod
    def _infer_phase_label(clip_id: str, notes: str) -> int:
        """Infer a phase label from available metadata when no explicit column exists.

        Args:
            clip_id: Clip identifier string.
            notes: Free-text notes column.

        Returns:
            Integer phase label in ``PHASE_TO_INDEX``.
        """

        haystack = f"{clip_id} {notes}".lower()
        if "block" in haystack:
            return PHASE_TO_INDEX["BLOCK"]
        if "flight" in haystack:
            return PHASE_TO_INDEX["FLIGHT"]
        if "entry" in haystack:
            return PHASE_TO_INDEX["ENTRY"]
        return PHASE_TO_INDEX["OTHER"]

    def __len__(self) -> int:
        """Return the number of dataset samples.

        Args:
            None.

        Returns:
            Dataset length.
        """

        return len(self.records)

    def _pad_or_truncate(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate a keypoint sequence to the configured fixed length.

        Args:
            sequence: Sequence array with shape ``[T, 33, 2]``.

        Returns:
            Fixed-length array with shape ``[sequence_length, 33, 2]``.
        """

        current_length = sequence.shape[0]
        if current_length == self.sequence_length:
            return sequence
        if current_length > self.sequence_length:
            return sequence[: self.sequence_length]

        padding = np.repeat(sequence[-1:, :, :], self.sequence_length - current_length, axis=0)
        return np.concatenate([sequence, padding], axis=0)

    @staticmethod
    def _normalize(sequence: np.ndarray) -> np.ndarray:
        """Normalize a sequence using per-sample mean and standard deviation.

        Args:
            sequence: Sequence array with shape ``[T, 33, 2]``.

        Returns:
            Normalized sequence array.
        """

        mean = sequence.mean(axis=(0, 1), keepdims=True)
        std = sequence.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (sequence - mean) / std

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a normalized fixed-length keypoint sequence and integer phase label.

        Args:
            index: Dataset index.

        Returns:
            Tuple of sequence tensor ``[T, 33, 2]`` and label tensor.
        """

        keypoints_path, label, _ = self.records[index]
        try:
            keypoints = np.load(keypoints_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load keypoints from '{keypoints_path}': {exc}") from exc

        sequence = keypoints[:, :, :2].astype(np.float32)
        sequence = self._pad_or_truncate(sequence)
        sequence = self._normalize(sequence).astype(np.float32)

        sequence_tensor = torch.from_numpy(sequence)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return sequence_tensor, label_tensor


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for dataset inspection.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Inspect the SwimVision training dataset.")
    parser.add_argument("--processed_dir", default="data/processed", help="Processed keypoint directory.")
    parser.add_argument("--labels_csv", default="data/labels.csv", help="Metadata CSV path.")
    return parser


def main() -> int:
    """Run the dataset inspection CLI.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        dataset = SwimStartDataset(args.processed_dir, args.labels_csv)
    except Exception as exc:
        LOGGER.error("Failed to initialize dataset: %s", exc)
        return 1

    LOGGER.info("Dataset size: %s samples", len(dataset))
    sample, label = dataset[0]
    LOGGER.info("First sample shape: %s, label: %s", tuple(sample.shape), int(label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
