# This file exposes SwimVision stroke-analysis modules as a package.
"""Stroke-analysis package for SwimVision."""

from src.stroke.cycle_detection import detect_stroke_cycles, StrokeCycle, angle_between, detect_stroke_type

__all__ = ["detect_stroke_cycles", "StrokeCycle", "angle_between", "detect_stroke_type"]
