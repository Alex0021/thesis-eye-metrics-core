"""eye-metrics-core — shared eye-tracking preprocessing and feature extraction."""

from .config import EyeMetricsConfig
from .features.definitions import FEATURE_SETS
from .features.extraction import extract_window_features
from .features.gaze import (
    calculate_angular_velocity,
    calculate_fixations_saccades_idt,
    calculate_fixations_saccades_ivt,
    calculate_gaze_angular_delta,
)
from .features.normalization import WelfordNormalizer
from .features.pupil import LHIPA, RIPA2, WaveletFeature
from .preprocessing.eye_selection import select_best_eye
from .preprocessing.gaps import detect_gaps_and_blinks
from .preprocessing.interpolation import interpolate_eye_data, interpolate_gaze, interpolate_pupil_data
from .preprocessing.outliers import OnlinePupilStats, detect_outliers
from .preprocessing.pipeline import PreprocessedResult, preprocess
from .types import FixationRow, GapInfo, SaccadeRow

__all__ = [
    "EyeMetricsConfig",
    "FEATURE_SETS",
    "FixationRow",
    "GapInfo",
    "LHIPA",
    "OnlinePupilStats",
    "detect_outliers",
    "PreprocessedResult",
    "RIPA2",
    "SaccadeRow",
    "WaveletFeature",
    "WelfordNormalizer",
    "calculate_angular_velocity",
    "calculate_fixations_saccades_idt",
    "calculate_fixations_saccades_ivt",
    "calculate_gaze_angular_delta",
    "detect_gaps_and_blinks",
    "extract_window_features",
    "interpolate_eye_data",
    "interpolate_gaze",
    "interpolate_pupil_data",
    "preprocess",
    "select_best_eye",
]
