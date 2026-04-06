"""Unified preprocessing pipeline for eye-tracking data.

Both online and offline paths call :func:`preprocess` with the same interface.
For online use, pass a persistent :class:`OnlinePupilStats` instance so the
running speed statistics stay warm across windows.  For offline use, omit it
and a fresh tracker is created internally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PreprocessingConfig
from .eye_selection import select_best_eye
from .gaps import detect_gaps_and_blinks
from .interpolation import interpolate_pupil_data
from .outliers import OnlinePupilStats


@dataclass
class PreprocessedResult:
    """Output of :func:`preprocess`."""

    pupil_df: pd.DataFrame
    gaps_df: pd.DataFrame
    eye_method: str

    @property
    def is_valid(self) -> bool:
        return not self.pupil_df.empty


def preprocess(
    df: pd.DataFrame,
    cfg: PreprocessingConfig,
    outlier_tracker: OnlinePupilStats | None = None,
) -> PreprocessedResult:
    """Unified preprocessing: eye selection → gaps → outliers → blink removal → interpolation.

    :param df: Raw eye-tracking DataFrame with left_/right_ prefixed columns.
    :param cfg: Preprocessing configuration.
    :param outlier_tracker: Persistent :class:`OnlinePupilStats` for online use.
                            If ``None``, a fresh tracker is created and discarded.
    :returns: :class:`PreprocessedResult` with ``pupil_df`` and ``gaps_df``.
    """
    empty = pd.DataFrame()

    # 1. Eye selection
    df, eye_method = select_best_eye(
        df.copy(),
        threshold=cfg.eye_selection.validity_difference_threshold,
    )

    # 2. Gap and blink detection
    gaps_df = detect_gaps_and_blinks(
        df,
        confidence_threshold=cfg.gaps_and_blinks.confidence_threshold,
        blink_threshold_range=(
            cfg.gaps_and_blinks.blink_duration_min_ms,
            cfg.gaps_and_blinks.blink_duration_max_ms,
        ),
        eye_openness_column="openness",
        openness_threshold=cfg.gaps_and_blinks.openness_threshold,
    )

    df.rename(columns={"pupil_diameter_mm": "pupil_diameter"}, inplace=True)

    # 3. Validity check: reject window if too many non-blink gaps
    total_samples = len(df)
    if total_samples == 0:
        return PreprocessedResult(empty, gaps_df, eye_method)

    if not gaps_df.empty:
        non_blink = gaps_df[~gaps_df["is_blink"]]
        if not non_blink.empty:
            lc_count = (non_blink["stop_id"] - non_blink["start_id"] + 1).sum()
            if lc_count / total_samples > cfg.validation.min_non_blink_gap_ratio:
                return PreprocessedResult(empty, gaps_df, eye_method)

    # 4. Filter low-confidence samples
    df = df[df["confidence"] >= cfg.gaps_and_blinks.confidence_threshold]

    # 5. Outlier rejection on pupil dilation speed
    if "pupil_diameter" in df.columns:
        ts = df["timestamp_sec"].values
        pd_vals = df["pupil_diameter"].values
        dt = np.diff(ts)
        dt[dt == 0] = 1e-6
        speed_fwd = np.abs(np.diff(pd_vals) / dt)
        speeds = np.empty(len(pd_vals))
        speeds[0] = speed_fwd[0] if len(speed_fwd) > 0 else 0.0
        speeds[-1] = speed_fwd[-1] if len(speed_fwd) > 0 else 0.0
        speeds[1:-1] = np.maximum(speed_fwd[:-1], speed_fwd[1:])

        if outlier_tracker is None:
            outlier_tracker = OnlinePupilStats(
                ema_alpha=cfg.outlier_rejection.ema_alpha
            )
        outlier_tracker.update_from_speeds(speeds)
        outlier_mask = outlier_tracker.outlier_mask(
            speeds, n_multiplier=cfg.outlier_rejection.n_mad_multiplier
        )
        df = df[~outlier_mask]

    # 6. Remove blinks with a margin on either side to avoid edge artefacts
    margins = cfg.gaps_and_blinks.blink_margin_ms / 1000.0
    for _, row in gaps_df[
        gaps_df["duration_ms"] >= cfg.gaps_and_blinks.blink_duration_min_ms
    ].iterrows():
        idx_to_drop = df[
            (df["timestamp_sec"] >= row["start_timestamp"] - margins)
            & (df["timestamp_sec"] <= row["stop_timestamp"] + margins)
        ].index
        df.drop(idx_to_drop, inplace=True)

    # 7. Skip if too few valid samples remain
    if len(df) < cfg.interpolation.min_samples:
        return PreprocessedResult(empty, gaps_df, eye_method)

    # 8. Interpolate pupil diameter over short gaps
    pupil_df = interpolate_pupil_data(
        df,
        gaps_df,
        max_gap_ms=cfg.interpolation.max_gap_ms,
    )

    return PreprocessedResult(pupil_df, gaps_df, eye_method)
