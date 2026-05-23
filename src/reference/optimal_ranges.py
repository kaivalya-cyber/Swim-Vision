# This file stores biomechanical reference ranges and helpers for SwimVision phase metrics.
"""Reference ranges for swim start biomechanical metrics."""

from __future__ import annotations

import argparse
import json
import logging
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
        "velocity": (4.0, 6.0),
        "stability_score": (85.0, 100.0),
    },
    "entry_phase": {
        "streamline_angle": (0.0, 10.0),
        "elbow_lock_angle": (170.0, 180.0),
        "velocity": (1.5, 3.0),
        "splash_score": (0.0, 15.0),
        "angle_of_attack": (35.0, 45.0),
    },
}

DEVIATION_THRESHOLDS: Dict[str, float] = {
    "OPTIMAL": 0.0,
    "MINOR": 10.0,
    "SIGNIFICANT": 20.0,
}


def get_range(phase: str, metric: str) -> Tuple[float, float]:
    """Return the optimal range tuple for a phase metric.

    Args:
        phase: Phase identifier such as ``block_phase``.
        metric: Metric name within the phase.

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
    return OPTIMAL_RANGES[phase][metric]


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
