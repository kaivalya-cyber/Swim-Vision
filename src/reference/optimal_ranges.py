# This file stores biomechanical reference ranges and helpers for SwimVision phase metrics.
"""Reference ranges for swim start biomechanical metrics."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


OPTIMAL_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "block_phase": {
        "front_knee_angle": (90.0, 110.0),
        "rear_knee_angle": (110.0, 130.0),
        "hip_angle": (55.0, 75.0),
        "torso_lean": (15.0, 30.0),
    },
    "flight_phase": {
        "body_linearity": (0.0, 8.0),
        "entry_angle": (30.0, 45.0),
        "elbow_extension": (165.0, 180.0),
    },
    "entry_phase": {
        "streamline_angle": (0.0, 10.0),
        "elbow_lock_angle": (170.0, 180.0),
    },
    "stroke_catch_left": {
        "left_elbow_flexion": (100.0, 120.0),
        "left_shoulder_rotation": (35.0, 55.0),
    },
    "stroke_pull_left": {
        "left_elbow_extension_rate": (120.0, 160.0),
        "left_hand_speed": (0.15, 0.35),
    },
    "stroke_catch_right": {
        "right_elbow_flexion": (100.0, 120.0),
        "right_shoulder_rotation": (35.0, 55.0),
    },
    "stroke_pull_right": {
        "right_elbow_extension_rate": (120.0, 160.0),
        "right_hand_speed": (0.15, 0.35),
    },
    "stroke_cycle": {
        "stroke_rate": (30.0, 60.0),
        "body_roll": (35.0, 45.0),
        "symmetry_index": (0.0, 10.0),
    },
    "butterfly_catch": {
        "bilateral_elbow_flexion": (90.0, 110.0),
        "bilateral_entry_width": (0.1, 0.3),
    },
    "butterfly_pull": {
        "bilateral_hand_speed": (0.2, 0.4),
        "kick_amplitude": (0.15, 0.35),
    },
    "butterfly_cycle": {
        "stroke_rate": (25.0, 55.0),
        "body_undulation": (20.0, 40.0),
        "kick_count": (1.5, 2.5),
    },
    "backstroke_catch": {
        "supine_elbow_flexion": (100.0, 120.0),
        "hand_entry_depth": (0.05, 0.2),
    },
    "backstroke_pull": {
        "supine_hand_speed": (0.2, 0.4),
        "shoulder_rotation_back": (30.0, 50.0),
    },
    "backstroke_cycle": {
        "stroke_rate": (28.0, 55.0),
        "body_roll_back": (30.0, 45.0),
        "symmetry_index": (0.0, 10.0),
    },
}

DEVIATION_THRESHOLDS: Dict[str, float] = {
    "OPTIMAL": 0.0,
    "MINOR": 10.0,
    "SIGNIFICANT": 20.0,
}

# Swimmer height-based adjustment factors for optimal ranges
# Taller swimmers typically have wider optimal ranges for certain angles
HEIGHT_ADJUSTMENTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # Per-cm adjustment: (min_adjustment, max_adjustment) added to range
    "torso_lean": {"min_per_cm": 0.0, "max_per_cm": 0.05},
    "hip_angle": {"min_per_cm": -0.05, "max_per_cm": 0.05},
    "front_knee_angle": {"min_per_cm": -0.1, "max_per_cm": 0.1},
    "rear_knee_angle": {"min_per_cm": -0.1, "max_per_cm": 0.1},
    "body_linearity": {"min_per_cm": 0.0, "max_per_cm": 0.02},
}

# Experience level adjustment factors
EXPERIENCE_ADJUSTMENTS: Dict[str, Dict[str, Dict[str, float]]] = {
    # beginner: wider ranges (more forgiving), elite: tighter ranges
    "beginner": {"tolerance_multiplier": 1.5},
    "intermediate": {"tolerance_multiplier": 1.2},
    "advanced": {"tolerance_multiplier": 1.0},
    "elite": {"tolerance_multiplier": 0.85},
}


@dataclass
class SwimmerProfile:
    """Swimmer profile for personalized optimal ranges.

    Attributes:
        height_cm: Swimmer height in centimeters.
        experience: Experience level (beginner, intermediate, advanced, elite).
        gender: Optional gender for gender-specific ranges.
        age: Optional age in years.
    """

    height_cm: float | None = None
    experience: str = "intermediate"
    gender: str | None = None
    age: int | None = None

    def __post_init__(self) -> None:
        if self.experience not in EXPERIENCE_ADJUSTMENTS:
            raise ValueError(
                f"Invalid experience level '{self.experience}'. "
                f"Choose from: {sorted(EXPERIENCE_ADJUSTMENTS)}"
            )


def get_range(phase: str, metric: str, profile: SwimmerProfile | None = None) -> Tuple[float, float]:
    """Return the optimal range tuple for a phase metric, optionally personalized.

    Args:
        phase: Phase identifier such as ``block_phase``.
        metric: Metric name within the phase.
        profile: Optional swimmer profile for personalized range adjustment.

    Returns:
        The inclusive ``(min_angle, max_angle)`` range in degrees.
    """

    if phase not in OPTIMAL_RANGES:
        available_phases = ", ".join(sorted(OPTIMAL_RANGES))
        raise KeyError(
            f"Unknown phase '{phase}'. Available phases: {available_phases}."
        )
    if metric not in OPTIMAL_RANGES[phase]:
        available_metrics = ", ".join(sorted(OPTIMAL_RANGES[phase]))
        raise KeyError(
            f"Unknown metric '{metric}' for phase '{phase}'. "
            f"Available metrics: {available_metrics}."
        )

    base_min, base_max = OPTIMAL_RANGES[phase][metric]

    if profile is None:
        return base_min, base_max

    # Apply experience-level tolerance multiplier
    exp_adj = EXPERIENCE_ADJUSTMENTS.get(profile.experience, EXPERIENCE_ADJUSTMENTS["intermediate"])
    multiplier = float(exp_adj["tolerance_multiplier"])
    range_span = base_max - base_min
    expansion = range_span * (multiplier - 1.0) / 2.0
    adjusted_min = base_min - expansion
    adjusted_max = base_max + expansion

    # Apply height-based adjustments if profile has height and metric supports it
    if profile.height_cm is not None and metric in HEIGHT_ADJUSTMENTS:
        h_adj = HEIGHT_ADJUSTMENTS[metric]
        reference_height = 175.0
        height_delta = profile.height_cm - reference_height
        adjusted_min += height_delta * float(h_adj["min_per_cm"])
        adjusted_max += height_delta * float(h_adj["max_per_cm"])

    return adjusted_min, adjusted_max


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for querying optimal ranges.

    Args:
        None.

    Returns:
        A configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Query SwimVision biomechanical optimal ranges."
    )
    parser.add_argument("--phase", help="Phase name to query.")
    parser.add_argument("--metric", help="Metric name to query.")
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print the full reference dictionary as JSON.",
    )
    return parser


def main() -> int:
    """Run the command-line interface for reference range inspection.

    Args:
        None.

    Returns:
        Exit status code.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        if args.dump_json:
            print(json.dumps(OPTIMAL_RANGES, indent=2, sort_keys=True))
            return 0
        if args.phase and args.metric:
            optimal_range = get_range(args.phase, args.metric)
            LOGGER.info(
                "Optimal range for %s/%s: %s", args.phase, args.metric, optimal_range
            )
            print(json.dumps({"phase": args.phase, "metric": args.metric, "range": optimal_range}))
            return 0
        parser.error("Provide both --phase and --metric, or use --dump-json.")
    except KeyError as exc:
        LOGGER.error("Failed to query range: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
