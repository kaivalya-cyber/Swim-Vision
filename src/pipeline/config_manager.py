# This file manages SwimVision pipeline configuration with feature flags and presets.
"""Pipeline configuration manager for SwimVision — feature flags, presets, and tuning."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "label": "Quick Analysis",
        "description": "Fast processing with core metrics only. Best for rapid feedback.",
        "features": {
            "extract_keypoints": True,
            "detect_phases": True,
            "compute_angles": True,
            "compute_velocity_acceleration": False,
            "compute_dynamic_estimates": False,
            "compute_deviations": True,
            "analyze_symmetry": False,
            "assess_injury_risk": False,
            "render_overlay": False,
            "generate_report": True,
        },
        "params": {
            "extraction_model_complexity": 0,
            "angle_smoothing_window": 3,
            "deviation_thresholds": {"OPTIMAL": 0, "MINOR": 12, "SIGNIFICANT": 25},
        },
    },
    "standard": {
        "label": "Standard Analysis",
        "description": "Balanced processing with advanced metrics. Recommended for most uses.",
        "features": {
            "extract_keypoints": True,
            "detect_phases": True,
            "compute_angles": True,
            "compute_velocity_acceleration": True,
            "compute_dynamic_estimates": True,
            "compute_deviations": True,
            "analyze_symmetry": True,
            "assess_injury_risk": True,
            "render_overlay": True,
            "generate_report": True,
        },
        "params": {
            "extraction_model_complexity": 1,
            "angle_smoothing_window": 5,
            "deviation_thresholds": {"OPTIMAL": 0, "MINOR": 10, "SIGNIFICANT": 20},
        },
    },
    "research": {
        "label": "Research-Grade Analysis",
        "description": "Maximum detail with all metrics enabled.",
        "features": {
            "extract_keypoints": True,
            "detect_phases": True,
            "compute_angles": True,
            "compute_velocity_acceleration": True,
            "compute_dynamic_estimates": True,
            "compute_deviations": True,
            "analyze_symmetry": True,
            "assess_injury_risk": True,
            "render_overlay": True,
            "generate_report": True,
        },
        "params": {
            "extraction_model_complexity": 2,
            "angle_smoothing_window": 7,
            "deviation_thresholds": {"OPTIMAL": 0, "MINOR": 8, "SIGNIFICANT": 18},
        },
    },
}


@dataclass
class PipelineConfig:
    """Active pipeline configuration."""

    preset: str = "standard"
    features: Dict[str, bool] = field(default_factory=lambda: PRESETS["standard"]["features"].copy())
    params: Dict[str, Any] = field(default_factory=lambda: PRESETS["standard"]["params"].copy())
    analysis_mode: str = "dive"
    swimmer_id: str = ""

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset '{self.preset}'. Choose from: {sorted(PRESETS)}")
        # Apply the selected preset's features and params, merging any explicitly-passed overrides
        preset_data = PRESETS[self.preset]
        if self.features == PRESETS["standard"]["features"]:
            self.features = preset_data["features"].copy()
        else:
            self.features = {**preset_data["features"], **self.features}
        if self.params == PRESETS["standard"]["params"]:
            self.params = preset_data["params"].copy()
        else:
            self.params = {**preset_data["params"], **self.params}

    def is_enabled(self, feature: str) -> bool:
        return self.features.get(feature, False)

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        preset = data.get("preset", "standard")
        base = PRESETS.get(preset, PRESETS["standard"])
        features = {**base["features"], **data.get("features", {})}
        params = {**base["params"], **data.get("params", {})}
        return cls(
            preset=preset,
            features=features,
            params=params,
            analysis_mode=data.get("analysis_mode", "dive"),
            swimmer_id=data.get("swimmer_id", ""),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path)
        if not config_path.exists():
            LOGGER.warning("Config file %s not found, using defaults.", config_path)
            return cls()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except Exception as exc:
            LOGGER.error("Failed to load config from %s: %s", config_path, exc)
            return cls()

    def save(self, path: str | Path) -> None:
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            LOGGER.info("Saved config to %s", config_path)
        except Exception as exc:
            LOGGER.error("Failed to save config to %s: %s", config_path, exc)


def create_default_config(analysis_mode: str = "dive") -> PipelineConfig:
    return PipelineConfig(preset="standard", analysis_mode=analysis_mode)


def get_preset_info() -> Dict[str, Dict[str, str]]:
    return {
        name: {"label": data["label"], "description": data["description"]}
        for name, data in PRESETS.items()
    }
