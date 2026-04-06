"""Online z-normalization using Welford's algorithm."""

import numpy as np


class WelfordNormalizer:
    """Online z-normalization using Welford's algorithm."""

    def __init__(self, n_features: int):
        self.n = 0
        self.mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)

    def update(self, x: np.ndarray) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self._m2 / (self.n - 1))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        s = self.std.copy()
        s[s < 1e-10] = 1.0
        return (x - self.mean) / s
