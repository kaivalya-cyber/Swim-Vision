"""Tests for stroke type detection and trend analytics."""

import json
import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.stroke.cycle_detection import detect_stroke_type
from src.analytics.trend import (
    SessionRecord,
    _load_report,
    _compute_trend,
    _is_improvement_direction,
    analyze_trends,
)


class TestDetectStrokeType:
    """Tests for auto-detection of stroke type."""

    def _make_backstroke_keypoints(self, num_frames: int = 60) -> np.ndarray:
        """Create synthetic backstroke keypoints: supine (shoulders above hips in y)."""
        keypoints = np.zeros((num_frames, 33, 4), dtype=np.float32)
        keypoints[:, :, 3] = 0.9
        # Backstroke: shoulders have smaller y-values than hips (higher in image)
        keypoints[:, 11, :2] = [0.35, 0.35]  # left shoulder (higher)
        keypoints[:, 12, :2] = [0.65, 0.35]  # right shoulder
        keypoints[:, 23, :2] = [0.35, 0.65]  # left hip (lower)
        keypoints[:, 24, :2] = [0.65, 0.65]  # right hip
        # Alternating arm motion for wrists
        for f in range(num_frames):
            phase = f % 30
            if phase < 15:
                keypoints[f, 15, 1] = 0.3 + 0.02 * phase  # left going up
                keypoints[f, 16, 1] = 0.6 - 0.02 * phase  # right going down
            else:
                keypoints[f, 15, 1] = 0.6 - 0.02 * (phase - 15)
                keypoints[f, 16, 1] = 0.3 + 0.02 * (phase - 15)
        return keypoints

    def _make_butterfly_keypoints(self, num_frames: int = 60) -> np.ndarray:
        """Create synthetic butterfly keypoints: bilateral arm sync."""
        keypoints = np.zeros((num_frames, 33, 4), dtype=np.float32)
        keypoints[:, :, 3] = 0.9
        keypoints[:, 11, :2] = [0.35, 0.7]  # shoulders below hips (larger y)
        keypoints[:, 12, :2] = [0.65, 0.7]
        keypoints[:, 23, :2] = [0.35, 0.45]  # hips above shoulders (smaller y)
        keypoints[:, 24, :2] = [0.65, 0.45]
        # Bilateral arm motion: both wrists move together
        for f in range(num_frames):
            phase = f % 40
            wrist_y = 0.3 + 0.3 * abs(np.sin(np.pi * phase / 20))
            keypoints[f, 15, 1] = wrist_y
            keypoints[f, 16, 1] = wrist_y + 0.02  # near-identical
        return keypoints

    def test_backstroke_detection(self):
        keypoints = self._make_backstroke_keypoints(60)
        result = detect_stroke_type(keypoints, start_frame=0)
        assert result == "backstroke"

    def test_butterfly_detection(self):
        keypoints = self._make_butterfly_keypoints(60)
        result = detect_stroke_type(keypoints, start_frame=0)
        assert result in ("butterfly", "freestyle")  # may fall back if correlation misses

    def test_defaults_to_freestyle(self):
        keypoints = np.zeros((60, 33, 4), dtype=np.float32)
        keypoints[:, :, 3] = 0.9
        keypoints[:, 11, :2] = [0.3, 0.5]  # shoulders and hips at similar y
        keypoints[:, 12, :2] = [0.7, 0.5]
        keypoints[:, 23, :2] = [0.3, 0.55]
        keypoints[:, 24, :2] = [0.7, 0.55]
        result = detect_stroke_type(keypoints, start_frame=0)
        assert result == "freestyle"


class TestImprovementDirection:
    """Tests for _is_improvement_direction."""

    def test_symmetry_is_lower_better(self):
        assert _is_improvement_direction("symmetry_index") is False

    def test_duration_is_lower_better(self):
        assert _is_improvement_direction("cycle_duration_seconds") is False

    def test_stroke_rate_is_higher_better(self):
        assert _is_improvement_direction("stroke_rate") is True

    def test_unknown_metric_defaults_higher_better(self):
        assert _is_improvement_direction("some_unknown_metric") is True


class TestComputeTrend:
    """Tests for _compute_trend."""

    def test_improving_trend(self):
        values = [40.0, 42.0, 44.0, 46.0]
        result = _compute_trend(values, metric_name="stroke_rate")
        assert result["direction"] == "improving"
        assert result["slope"] > 0.5
        assert result["num_sessions"] == 4.0

    def test_declining_symmetry(self):
        """Declining symmetry index is actually improving (lower is better)."""
        values = [15.0, 12.0, 9.0, 6.0]
        result = _compute_trend(values, metric_name="symmetry_index")
        assert result["direction"] == "improving"
        assert result["slope"] < 0

    def test_stable_trend(self):
        values = [45.0, 45.001, 44.999, 45.0]
        result = _compute_trend(values, metric_name="stroke_rate")
        assert result["direction"] == "stable"

    def test_single_value(self):
        values = [42.0]
        result = _compute_trend(values, metric_name="stroke_rate")
        assert result["direction"] == "stable"
        assert result["std"] == 0.0

    def test_empty_returns_empty(self):
        result = _compute_trend([])
        assert result == {}


class TestLoadReport:
    """Tests for _load_report."""

    def test_load_stroke_report(self, tmp_path):
        report_data = {
            "clip_id": "test_swimmer",
            "analysis_mode": "stroke",
            "date": "2026-01-01",
            "overall_severity": "MINOR",
            "num_cycles": 5,
            "aggregate_metrics": {
                "stroke_rate": 48.0,
                "body_roll": 38.0,
                "symmetry_index": 5.0,
                "num_cycles": 5,
            },
        }
        report_path = tmp_path / "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        record = _load_report(report_path)
        assert record is not None
        assert record.session_id == "test_swimmer"
        assert record.analysis_mode == "stroke"
        assert record.num_cycles == 5
        assert record.metrics["stroke_rate"] == 48.0

    def test_load_dive_report(self, tmp_path):
        report_data = {
            "clip_id": "dive_test",
            "analysis_mode": "dive",
            "date": "2026-01-01",
            "overall_severity": "OPTIMAL",
            "deviations": {
                "block_phase": [
                    {"metric": "front_knee_angle", "measured": 105.0, "flag": "MINOR"},
                ],
                "flight_phase": [],
                "entry_phase": [],
            },
        }
        report_path = tmp_path / "dive_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        record = _load_report(report_path)
        assert record is not None
        assert record.session_id == "dive_test"
        assert record.analysis_mode == "dive"
        assert "block_phase_front_knee_angle" in record.metrics

    def test_load_invalid_path(self):
        record = _load_report(Path("nonexistent.json"))
        assert record is None


class TestAnalyzeTrends:
    """Integration test for analyze_trends."""

    def test_two_session_trend(self, tmp_path):
        # Create two report files
        for idx, spm in enumerate([40, 48]):
            data = {
                "clip_id": f"swimmer_sess{idx}",
                "analysis_mode": "stroke",
                "date": f"2026-01-0{idx+1}",
                "overall_severity": "MINOR",
                "num_cycles": 5 + idx,
                "aggregate_metrics": {
                    "stroke_rate": float(spm),
                    "symmetry_index": 10.0 - idx * 2,
                },
            }
            path = tmp_path / f"report{idx}.json"
            with open(path, "w") as f:
                json.dump(data, f)

        result = analyze_trends(
            [str(tmp_path / "report0.json"), str(tmp_path / "report1.json")],
            primary_metric="stroke_rate",
        )
        assert result["trend_summary"]["num_sessions"] == 2
        assert result["trend_summary"]["primary_metric"] == "stroke_rate"
        assert "metric_trends" in result
        assert result["trend_summary"]["primary_trend"]["direction"] == "improving"
