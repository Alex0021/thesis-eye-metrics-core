"""Eye-tracking preprocessing utilities."""

from .eye_selection import select_best_eye
from .gaps import detect_gaps_and_blinks
from .interpolation import interpolate_eye_data, interpolate_gaze, interpolate_pupil_data
from .outliers import OnlinePupilStats
from .pipeline import PreprocessedResult, preprocess

__all__ = [
    "OnlinePupilStats",
    "PreprocessedResult",
    "detect_gaps_and_blinks",
    "interpolate_eye_data",
    "interpolate_gaze",
    "interpolate_pupil_data",
    "preprocess",
    "select_best_eye",
]
