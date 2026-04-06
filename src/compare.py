# This file compares two SwimVision clips by aligning their phase deviation tables side by side.
"""Clip-to-clip deviation comparison utilities for SwimVision."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


def _flatten_deviations(payload: Dict[str, Any], column_prefix: str) -> pd.DataFrame:
    """Flatten a deviation JSON payload into a tabular format."""

    rows: List[Dict[str, Any]] = []
    for phase_name in ("block_phase", "flight_phase", "entry_phase"):
        for row in payload.get(phase_name, []):
            if isinstance(row, dict):
                rows.append(
                    {
                        "phase": phase_name,
                        "metric": row.get("metric"),
                        f"{column_prefix}_measured": row.get("measured"),
                        f"{column_prefix}_deviation": row.get("deviation"),
                        f"{column_prefix}_flag": row.get("flag"),
                    }
                )
    return pd.DataFrame(rows)


def _load_deviation_payload(results_dir: Path, clip_id: str) -> Dict[str, Any]:
    """Load a saved deviation JSON for a clip."""

    deviations_path = results_dir / f"{clip_id}_deviations.json"
    if not deviations_path.exists():
        raise FileNotFoundError(f"Deviation JSON not found: '{deviations_path}'.")
    with open(deviations_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_clips(clip_a: str, clip_b: str, results_dir: str) -> Path:
    """Compare two clips and save a side-by-side deviation CSV."""

    results_path = Path(results_dir)
    payload_a = _load_deviation_payload(results_path, clip_a)
    payload_b = _load_deviation_payload(results_path, clip_b)

    prefix_a = clip_a if clip_a != clip_b else f"{clip_a}_a"
    prefix_b = clip_b if clip_a != clip_b else f"{clip_b}_b"

    frame_a = _flatten_deviations(payload_a, prefix_a)
    frame_b = _flatten_deviations(payload_b, prefix_b)
    comparison = frame_a.merge(frame_b, on=["phase", "metric"], how="outer")
    comparison[f"{prefix_a}_measured"] = pd.to_numeric(comparison[f"{prefix_a}_measured"], errors="coerce")
    comparison[f"{prefix_b}_measured"] = pd.to_numeric(comparison[f"{prefix_b}_measured"], errors="coerce")
    comparison["measured_delta"] = comparison[f"{prefix_a}_measured"] - comparison[f"{prefix_b}_measured"]

    output_path = results_path / f"{clip_a}_vs_{clip_b}_comparison.csv"
    comparison.to_csv(output_path, index=False)
    LOGGER.info("Saved comparison CSV to %s", output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for clip comparison."""

    parser = argparse.ArgumentParser(description="Compare SwimVision deviations for two clips.")
    parser.add_argument("--clip_a", required=True, help="First clip identifier.")
    parser.add_argument("--clip_b", required=True, help="Second clip identifier.")
    parser.add_argument("--results_dir", default="results", help="Directory containing deviation JSON files.")
    return parser


def main() -> int:
    """Run the command-line interface for clip comparison."""

    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        output_path = compare_clips(args.clip_a, args.clip_b, args.results_dir)
    except Exception as exc:
        LOGGER.error("Clip comparison failed: %s", exc)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
