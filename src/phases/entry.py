# This file summarizes entry-phase biomechanical metrics over the detected water-entry window.
"""Entry-phase analysis utilities for SwimVision."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def analyze(
    keypoints: np.ndarray, angles_df: pd.DataFrame, phase_boundaries: Dict[str, int], metadata: Dict[str, Any]
) -> Dict[str, float]:
    """Summarize entry-phase metrics across the detected entry window.

    Args:
        keypoints: Keypoint array for the clip.
        angles_df: Per-frame angle DataFrame.
        phase_boundaries: Detected phase boundary indices.
        metadata: Clip metadata dictionary.

    Returns:
        Averaged entry-phase biomechanical metrics.
    """

    _ = keypoints, metadata
    start_idx = int(phase_boundaries["entry_start"])
    end_idx = int(phase_boundaries["entry_end"])
    window = angles_df.loc[start_idx:end_idx]
    elbow_lock_angle = float(
        window[["left_elbow_angle", "right_elbow_angle"]].mean(axis=1).mean()
    )
    return {
        "streamline_angle": float(window["streamline_angle"].mean()),
        "elbow_lock_angle": elbow_lock_angle,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for entry-phase analysis.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(description="Analyze the SwimVision entry phase.")
    parser.add_argument("--angles", required=True, help="Angle CSV path.")
    parser.add_argument("--boundaries", required=True, help="Phase boundary JSON path.")
    return parser


def main() -> int:
    """Run the entry-phase analysis CLI.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        angles_df = pd.read_csv(args.angles, index_col=0)
    except Exception as exc:
        LOGGER.error("Failed to load angle CSV %s: %s", args.angles, exc)
        return 1
    try:
        with open(args.boundaries, "r", encoding="utf-8") as handle:
            phase_boundaries = json.load(handle)
    except Exception as exc:
        LOGGER.error("Failed to load phase boundaries %s: %s", args.boundaries, exc)
        return 1

    try:
        result = analyze(np.empty((0,)), angles_df, phase_boundaries, {})
    except Exception as exc:
        LOGGER.error("Entry analysis failed: %s", exc)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
