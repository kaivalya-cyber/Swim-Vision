"""Tests for stroke metrics computation."""

import numpy as np
import pytest

from src.stroke.cycle_detection import StrokeCycle
from src.metrics.stroke_metrics import (
    StrokeMetrics,
    compute_stroke_metrics,
    aggregate_stroke_metrics,
    _compute_mean_elbow_angle,
    _compute_hand_speed,
    _compute_extension_rate,
    _compute_shoulder_rotation,
)


def _make_keypoints(num_frames: int = 120) -> np.ndarray:
    """Create simple synthetic keypoints with known structure."""
    keypoints = np.zeros((num_frames, 33, 4), dtype=np.float32)
    keypoints[:, :, 3] = 0.9
    # Shoulders
    keypoints[:, 11] = [0.3, 0.4, 0.0, 0.9]
    keypoints[:, 12] = [0.7, 0.4, 0.0, 0.9]
    # Elbows
    keypoints[:, 13] = [0.25, 0.5, 0.0, 0.9]
    keypoints[:, 14] = [0.75, 0.5, 0.0, 0.9]
    # Wrists
    keypoints[:, 15] = [0.2, 0.6, 0.0, 0.9]
    keypoints[:, 16] = [0.8, 0.6, 0.0, 0.9]
    # Hips
    keypoints[:, 23] = [0.35, 0.7, 0.0, 0.9]
    keypoints[:, 24] = [0.65, 0.7, 0.0, 0.9]
    return keypoints


def _make_cycle(
    cycle_index: int = 0,
    stroke_type: str = "freestyle",
    left_entry: int = 0,
    left_catch: int = 10,
    left_pull_end: int = 25,
    left_recovery_end: int = 50,
    right_entry: int = 15,
    right_catch: int = 30,
    right_pull_end: int = 45,
    right_recovery_end: int = 70,
) -> StrokeCycle:
    """Create a StrokeCycle with specified boundaries."""
    return StrokeCycle(
        cycle_index=cycle_index,
        stroke_type=stroke_type,
        left_entry_frame=left_entry,
        left_catch_frame=left_catch,
        left_pull_end_frame=left_pull_end,
        left_recovery_end_frame=left_recovery_end,
        right_entry_frame=right_entry,
        right_catch_frame=right_catch,
        right_pull_end_frame=right_pull_end,
        right_recovery_end_frame=right_recovery_end,
        body_roll_peak=30.0,
    )


class TestComputeMeanElbowAngle:
    """Tests for _compute_mean_elbow_angle."""

    def test_basic_angle(self):
        keypoints = _make_keypoints(60)
        angle = _compute_mean_elbow_angle(keypoints, 0, 30, side="left")
        assert isinstance(angle, float)
        assert 0.0 <= angle <= 180.0

    def test_right_side(self):
        keypoints = _make_keypoints(60)
        angle = _compute_mean_elbow_angle(keypoints, 0, 30, side="right")
        assert isinstance(angle, float)

    def test_empty_window(self):
        keypoints = _make_keypoints(60)
        angle = _compute_mean_elbow_angle(keypoints, 30, 0, side="left")
        assert angle == 0.0


class TestComputeHandSpeed:
    """Tests for _compute_hand_speed."""

    def test_stationary_wrist(self):
        keypoints = _make_keypoints(60)
        speed = _compute_hand_speed(keypoints, 0, 30, side="left")
        assert speed == 0.0  # All wrist positions identical

    def test_moving_wrist(self):
        keypoints = _make_keypoints(60)
        # Make left wrist move
        for f in range(30, 60):
            keypoints[f, 15, 0] = 0.2 + 0.01 * (f - 30)
        speed = _compute_hand_speed(keypoints, 30, 59, side="left")
        assert speed > 0.0


class TestComputeExtensionRate:
    """Tests for _compute_extension_rate."""

    def test_constant_angle(self):
        keypoints = _make_keypoints(60)
        rate = _compute_extension_rate(keypoints, 0, 30, side="left")
        assert abs(rate) < 0.1  # Very small since static

    def test_short_window_returns_zero(self):
        keypoints = _make_keypoints(60)
        rate = _compute_extension_rate(keypoints, 30, 30, side="left")
        assert rate == 0.0


class TestComputeShoulderRotation:
    """Tests for _compute_shoulder_rotation."""

    def test_returns_float(self):
        keypoints = _make_keypoints(60)
        rot = _compute_shoulder_rotation(keypoints, 0, 30, side="left")
        assert isinstance(rot, float)

    def test_returns_float_right(self):
        keypoints = _make_keypoints(60)
        rot = _compute_shoulder_rotation(keypoints, 0, 30, side="right")
        assert isinstance(rot, float)


class TestComputeStrokeMetrics:
    """Tests for compute_stroke_metrics."""

    def test_empty_cycles(self):
        keypoints = _make_keypoints(120)
        metrics = compute_stroke_metrics(keypoints, [], fps=30.0, width=1920, height=1080)
        assert metrics == []

    def test_single_cycle(self):
        keypoints = _make_keypoints(120)
        cycles = [_make_cycle()]
        metrics = compute_stroke_metrics(keypoints, cycles, fps=30.0, width=1920, height=1080)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.cycle_index == 0
        assert isinstance(m.stroke_rate, float)
        assert isinstance(m.symmetry_index, float)
        assert m.stroke_rate > 0
        assert m.cycle_duration_seconds > 0

    def test_symmetry_perfect(self):
        """Symmetric arm motion should give low symmetry index."""
        keypoints = _make_keypoints(120)
        cycles = [_make_cycle()]
        metrics = compute_stroke_metrics(keypoints, cycles, fps=30.0, width=1920, height=1080)
        assert len(metrics) == 1


class TestAggregateStrokeMetrics:
    """Tests for aggregate_stroke_metrics."""

    def test_empty_list(self):
        result = aggregate_stroke_metrics([])
        assert result == {}

    def test_single_metric(self):
        m = StrokeMetrics(cycle_index=0, stroke_rate=45.0, body_roll=38.0, symmetry_index=5.0)
        result = aggregate_stroke_metrics([m])
        assert result["stroke_rate"] == 45.0
        assert result["body_roll"] == 38.0
        assert result["num_cycles"] == 1.0

    def test_multiple_metrics(self):
        metrics = [
            StrokeMetrics(cycle_index=0, stroke_rate=40.0, left_elbow_flexion=110.0, right_elbow_flexion=105.0),
            StrokeMetrics(cycle_index=1, stroke_rate=50.0, left_elbow_flexion=115.0, right_elbow_flexion=110.0),
        ]
        result = aggregate_stroke_metrics(metrics)
        assert result["stroke_rate"] == 45.0
        assert result["num_cycles"] == 2.0
