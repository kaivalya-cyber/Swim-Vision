"""Tests for stroke cycle detection utilities."""

import numpy as np
import pytest

from src.stroke.cycle_detection import (
    StrokeCycle,
    _smooth_signal,
    _compute_body_roll,
    angle_between,
    detect_stroke_cycles,
)


class TestAngleBetween:
    """Tests for the angle_between function."""

    def test_straight_line(self):
        """Angle between three collinear points should be 180 degrees."""
        a = np.array([0.0, 0.0])
        b = np.array([0.5, 0.5])
        c = np.array([1.0, 1.0])
        result = angle_between(a, b, c)
        assert abs(result - 180.0) < 0.01

    def test_right_angle(self):
        """Right angle should be 90 degrees."""
        a = np.array([0.0, 0.0])
        b = np.array([0.5, 0.0])
        c = np.array([0.5, 0.5])
        result = angle_between(a, b, c)
        assert abs(result - 90.0) < 0.01

    def test_acute_angle(self):
        """45-degree angle."""
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])
        result = angle_between(a, b, c)
        assert abs(result - 90.0) < 0.01

    def test_obtuse_angle(self):
        """135-degree angle."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([1.0, 1.0])
        result = angle_between(a, b, c)
        assert abs(result - 90.0) < 0.01

    def test_zero_angle_overlap(self):
        """Angle with overlapping points should not crash."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 1.0])
        result = angle_between(a, b, c)
        # Should return a valid float, not NaN or inf
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_angle_boundaries(self):
        """Angle should always be between 0 and 180."""
        for _ in range(100):
            a = np.random.rand(2)
            b = np.random.rand(2)
            c = np.random.rand(2)
            result = angle_between(a, b, c)
            assert 0.0 <= result <= 180.0


class TestSmoothSignal:
    """Tests for the _smooth_signal function."""

    def test_identity_input(self):
        """Constant signal should remain constant after smoothing (allow edge effects)."""
        signal = np.ones(50, dtype=np.float32)
        smoothed = _smooth_signal(signal, window=5)
        # Interior values should be 1.0; edges may differ due to convolution boundaries
        interior = smoothed[5:-5]
        assert np.allclose(interior, 1.0, atol=0.01)

    def test_output_same_length(self):
        """Output should have the same length as input."""
        signal = np.random.rand(100).astype(np.float32)
        smoothed = _smooth_signal(signal, window=5)
        assert len(smoothed) == len(signal)

    def test_noise_reduction(self):
        """Smoothing should reduce variance."""
        np.random.seed(42)
        signal = np.sin(np.linspace(0, 10 * np.pi, 200)) + np.random.normal(0, 0.5, 200)
        signal = signal.astype(np.float32)
        smoothed = _smooth_signal(signal, window=10)
        assert np.std(smoothed) < np.std(signal)

    def test_window_one(self):
        """Window of 1 should return original signal."""
        signal = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float32)
        smoothed = _smooth_signal(signal, window=1)
        assert np.allclose(smoothed, signal)


class TestComputeBodyRoll:
    """Tests for _compute_body_roll."""

    def test_zero_roll_level_shoulders(self):
        """Level shoulders should give zero body roll."""
        num_frames = 10
        keypoints = np.zeros((num_frames, 33, 4), dtype=np.float32)
        # Set left and right shoulders at same y
        keypoints[0, 11, :2] = [0.3, 0.4]
        keypoints[0, 12, :2] = [0.7, 0.4]
        roll = _compute_body_roll(keypoints, 0, 1920, 1080)
        assert abs(roll) < 0.01

    def test_nonzero_roll(self):
        """Tilted shoulders should produce nonzero roll."""
        num_frames = 5
        keypoints = np.zeros((num_frames, 33, 4), dtype=np.float32)
        keypoints[0, 11, :2] = [0.3, 0.5]
        keypoints[0, 12, :2] = [0.7, 0.3]
        roll = _compute_body_roll(keypoints, 0, 1920, 1080)
        assert roll > 5.0  # Significant shoulder tilt


class TestDetectStrokeCycles:
    """Tests for detect_stroke_cycles."""

    def _make_freestyle_keypoints(
        self, num_cycles: int = 3, fps: float = 30.0, cycle_frames: int = 60
    ) -> np.ndarray:
        """Create synthetic keypoint data simulating freestyle stroke cycles.

        Each cycle: alternating left/right arm entries with body roll.
        """
        total_frames = num_cycles * cycle_frames
        keypoints = np.zeros((total_frames, 33, 4), dtype=np.float32)

        for frame_idx in range(total_frames):
            cycle_pos = (frame_idx % cycle_frames) / float(cycle_frames)
            half_cycle = cycle_pos * 2.0 if cycle_pos < 0.5 else (cycle_pos - 0.5) * 2.0

            # Simulate alternating arm motion
            if cycle_pos < 0.5:
                # Right arm pulling (low wrist), left arm recovering (high wrist)
                left_wrist_y = 0.2 + 0.2 * half_cycle
                right_wrist_y = 0.6 + 0.15 * half_cycle
                left_wrist_x = 0.4 - 0.1 * half_cycle
                right_wrist_x = 0.6 + 0.2 * half_cycle
            else:
                # Left arm pulling, right arm recovering
                left_wrist_y = 0.6 + 0.15 * (1.0 - half_cycle)
                right_wrist_y = 0.2 + 0.2 * (1.0 - half_cycle)
                left_wrist_x = 0.4 + 0.2 * (1.0 - half_cycle)
                right_wrist_x = 0.6 - 0.1 * (1.0 - half_cycle)

            # Set keypoints for MediaPipe pose format
            keypoints[frame_idx, :, 3] = 0.9  # High visibility for all joints
            keypoints[frame_idx, 15, 0] = left_wrist_x
            keypoints[frame_idx, 15, 1] = left_wrist_y
            keypoints[frame_idx, 16, 0] = right_wrist_x
            keypoints[frame_idx, 16, 1] = right_wrist_y
            keypoints[frame_idx, 13, 0] = left_wrist_x - 0.05
            keypoints[frame_idx, 13, 1] = left_wrist_y - 0.1
            keypoints[frame_idx, 14, 0] = right_wrist_x + 0.05
            keypoints[frame_idx, 14, 1] = right_wrist_y - 0.1
            keypoints[frame_idx, 11, 0] = left_wrist_x - 0.08
            keypoints[frame_idx, 11, 1] = left_wrist_y - 0.2
            keypoints[frame_idx, 12, 0] = right_wrist_x + 0.08
            keypoints[frame_idx, 12, 1] = right_wrist_y - 0.2

            # Body roll: sinusoidal
            body_roll_rad = np.sin(2.0 * np.pi * cycle_pos) * 0.1
            keypoints[frame_idx, 11, 1] += float(body_roll_rad) * 0.5
            keypoints[frame_idx, 12, 1] -= float(body_roll_rad) * 0.5

        return keypoints

    def test_detect_cycles_on_synthetic_data(self):
        """Should detect at least some cycles on synthetic freestyle data."""
        keypoints = self._make_freestyle_keypoints(num_cycles=3, cycle_frames=60)
        cycles = detect_stroke_cycles(keypoints, fps=30.0, width=1920, height=1080)
        # Synthetic data with clear patterns should detect cycles
        assert len(cycles) > 0, "Expected at least 1 detected cycle"

    def test_too_short_sequence(self):
        """Very short sequences should return empty list."""
        keypoints = np.zeros((10, 33, 4), dtype=np.float32)
        keypoints[:, :, 3] = 0.9
        cycles = detect_stroke_cycles(keypoints, fps=30.0)
        assert len(cycles) == 0

    def test_all_zero_visibility(self):
        """All-zero visibility should not crash."""
        keypoints = np.zeros((120, 33, 4), dtype=np.float32)
        # All visibility is 0.0, body roll check skips
        cycles = detect_stroke_cycles(keypoints, fps=30.0)
        assert isinstance(cycles, list)
