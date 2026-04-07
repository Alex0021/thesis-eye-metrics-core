"""EyeMetricsConfig — hierarchical configuration for all preprocessing and feature parameters.

Load from a YAML file or use the bundled defaults::

    cfg = EyeMetricsConfig.default()          # loads config/default.yaml
    cfg = EyeMetricsConfig.from_yaml("my.yaml")
    cfg = EyeMetricsConfig()                  # pure Python defaults (no file I/O)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_type_hints

import yaml


@dataclass
class EyeSelectionConfig:
    validity_difference_threshold: float = 0.05


@dataclass
class GapsConfig:
    confidence_threshold: float = 0.5
    openness_threshold: float = 0.5
    blink_duration_min_ms: int = 100
    blink_duration_max_ms: int = 300
    blink_margin_ms: int = 100


@dataclass
class OutlierConfig:
    ema_alpha: float = 0.3
    n_mad_multiplier: float = 10.0
    n_passes: int = 1


@dataclass
class InterpolationConfig:
    max_gap_ms: int = 300
    min_samples: int = 20
    resample_period_ms: float = 16.67
    method: str = "slinear"


@dataclass
class ValidationConfig:
    min_non_blink_gap_ratio: float = 0.5


@dataclass
class PreprocessingConfig:
    eye_selection: EyeSelectionConfig = field(default_factory=EyeSelectionConfig)
    gaps_and_blinks: GapsConfig = field(default_factory=GapsConfig)
    outlier_rejection: OutlierConfig = field(default_factory=OutlierConfig)
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    skip_edge_samples: int = 60


@dataclass
class GazeFeatureConfig:
    idt_duration_threshold_ms: int = 100
    idt_dispersion_threshold: float = 0.05
    ivt_threshold_deg_per_sec: float = 45.0
    ivt_min_fixation_duration_ms: int = 55
    ivt_min_datapoints: int = 2
    velocity_limit_min: float = -1000.0
    velocity_limit_max: float = 1000.0
    velocity_filter_cutoff_hz: float = 10.0
    velocity_filter_taps: int = 5


@dataclass
class RIPA2Config:
    m_vlf: int = 98
    n_vlf: int = 2
    m_lf: int = 13
    n_lf: int = 4
    clip_min: float = 0.0
    clip_max: float = 1.5
    smoothing_sec: float = 1.0


@dataclass
class LHIPAConfig:
    wavelet: str = "sym16"


@dataclass
class WaveletConfig:
    wavelet: str = "db8"
    level: int = 4
    mode: str = "periodization"
    smoothing_sec: float = 1.0


@dataclass
class PupilFeatureConfig:
    ripa2: RIPA2Config = field(default_factory=RIPA2Config)
    lhipa: LHIPAConfig = field(default_factory=LHIPAConfig)
    wavelets: WaveletConfig = field(default_factory=WaveletConfig)


@dataclass
class WindowConfig:
    size_samples: int = 300
    interval_samples: int = 60
    min_samples: int = 100


@dataclass
class FeaturesConfig:
    gaze: GazeFeatureConfig = field(default_factory=GazeFeatureConfig)
    pupil: PupilFeatureConfig = field(default_factory=PupilFeatureConfig)
    feature_set: str = "ipa_wavelets"
    window: WindowConfig = field(default_factory=WindowConfig)
    rolling_buffer_size: int = 1000


@dataclass
class NormalizationConfig:
    min_std: float = 1e-10
    min_observations: int = 3


def _load_dataclass(cls, data: dict):
    """Recursively populate a dataclass from a nested dict."""
    hints = get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        hint = hints.get(f.name)
        if hint is not None and dataclasses.is_dataclass(hint) and isinstance(val, dict):
            kwargs[f.name] = _load_dataclass(hint, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


@dataclass
class EyeMetricsConfig:
    sample_rate: float = 60.0
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EyeMetricsConfig":
        """Load configuration from a YAML file. Unknown keys are ignored."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return _load_dataclass(cls, raw)

    @classmethod
    def default(cls) -> "EyeMetricsConfig":
        """Load the bundled default configuration."""
        default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls()
