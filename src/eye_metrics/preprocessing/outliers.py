"""Online pupil statistics for fast outlier rejection."""

import numpy as np


class OnlinePupilStats:
    """Running statistics for pupil diameter used for fast outlier rejection.

    Tracks an EMA of dilation speed median and MAD so that each inference
    window can reject outliers without recomputing global statistics.
    """

    def __init__(self, ema_alpha: float):
        self._alpha = ema_alpha
        self.median_speed: float | None = None
        self.mad_speed: float | None = None

    def update_from_speeds(self, speeds: np.ndarray) -> None:
        speeds = speeds[np.isfinite(speeds)]
        if len(speeds) < 5:
            return
        med = float(np.median(speeds))
        mad = float(np.median(np.abs(speeds - med)))
        if self.median_speed is None:
            self.median_speed = med
            self.mad_speed = mad
        else:
            a = self._alpha
            self.median_speed = a * med + (1 - a) * self.median_speed
            self.mad_speed = a * mad + (1 - a) * self.mad_speed

    def outlier_mask(
        self, speeds: np.ndarray, n_multiplier: float = 10.0
    ) -> np.ndarray:
        if self.median_speed is None or self.mad_speed is None:
            return np.zeros(len(speeds), dtype=bool)
        threshold = self.median_speed + n_multiplier * self.mad_speed
        return speeds > threshold
