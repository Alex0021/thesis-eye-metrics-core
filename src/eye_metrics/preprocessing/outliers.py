"""Pupil outlier detection — offline (training) and online (inference) variants."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_outliers(
    eye_df: pd.DataFrame, column: str, n_multiplier: float = 3.0, n_passes: int = 1
) -> pd.DataFrame:
    """Detect outliers in a given column of a DataFrame using the MAD method.

    Each pass recomputes median and MAD only on the surviving samples, progressively
    tightening the threshold.  Use this during offline training where the full window
    is available.

    :param eye_df: The DataFrame to analyze.
    :param column: The name of the column to analyze for outliers.
    :param n_multiplier: The multiplier used for the MAD threshold.
    :param n_passes: The number of passes to perform outlier detection.
    :return: A DataFrame containing the detected outliers.
    """
    def dilation_speed(data: pd.DataFrame) -> pd.Series:
        speed_before = data[column].diff(1) / data["timestamp_sec"].diff(1)
        speed_after = data[column].diff(-1) / data["timestamp_sec"].diff(-1)
        return pd.concat([speed_before.abs(), speed_after.abs()], axis=1).max(axis=1)

    pupil_df = eye_df[[column, "timestamp_sec"]].copy()
    pupil_df["is_outlier"] = False
    for _ in range(n_passes):
        non_outlier_idx = ~pupil_df["is_outlier"]
        speed = dilation_speed(pupil_df[non_outlier_idx])
        MAD = (speed - speed.median()).abs().median()
        threshold = speed.median() + n_multiplier * MAD
        outlier_idx = speed[speed > threshold].index
        pupil_df.loc[outlier_idx, "is_outlier"] = True

    return pupil_df[pupil_df["is_outlier"]]


class OnlinePupilStats:
    """Running statistics for pupil diameter used for fast outlier rejection.

    Tracks an EMA of dilation speed median and MAD so that each inference window
    can reject outliers without recomputing global statistics.

    Mirrors :func:`detect_outliers` behaviour: ``update_from_speeds`` performs
    ``n_passes`` of iterative MAD refinement on the current window before
    blending the cleaned stats into the EMA, so the running distribution is
    estimated from clean data just as in the offline case.
    """

    def __init__(
        self,
        ema_alpha: float,
        n_mad_multiplier: float = 3.0,
        n_passes: int = 1,
    ):
        self._alpha = ema_alpha
        self._n_multiplier = n_mad_multiplier
        self._n_passes = n_passes
        self.median_speed: float | None = None
        self.mad_speed: float | None = None

    def update_from_speeds(self, speeds: np.ndarray) -> None:
        """Update running stats from a window of dilation speeds.

        Performs ``n_passes`` of iterative MAD refinement on the current window
        (excluding detected outliers from each successive stat estimate) before
        EMA-blending the result into the running statistics.
        """
        speeds = speeds[np.isfinite(speeds)]
        if len(speeds) < 5:
            return

        mask = np.zeros(len(speeds), dtype=bool)
        med, mad = float(np.median(speeds)), 0.0
        for _ in range(self._n_passes):
            clean = speeds[~mask]
            if len(clean) < 5:
                break
            med = float(np.median(clean))
            mad = float(np.median(np.abs(clean - med)))
            new_mask = speeds > med + self._n_multiplier * mad
            if np.array_equal(new_mask, mask):
                break
            mask = new_mask

        if self.median_speed is None:
            self.median_speed = med
            self.mad_speed = mad
        else:
            a = self._alpha
            self.median_speed = a * med + (1 - a) * self.median_speed
            self.mad_speed = a * mad + (1 - a) * self.mad_speed

    def outlier_mask(self, speeds: np.ndarray) -> np.ndarray:
        if self.median_speed is None or self.mad_speed is None:
            return np.zeros(len(speeds), dtype=bool)
        threshold = self.median_speed + self._n_multiplier * self.mad_speed
        return speeds > threshold
