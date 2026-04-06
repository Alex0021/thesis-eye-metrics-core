import numpy as np
import pandas as pd


def detect_gaps_and_blinks(
    df: pd.DataFrame,
    confidence_threshold: float,
    blink_threshold_range: tuple[int, int],
    openness_threshold: float,
    eye_openness_column: str = None,
) -> pd.DataFrame:
    """
    Detect low confidence gaps and blinks in the eye-tracking data.

    :param df: DataFrame containing eye-tracking data with 'timestamp_sec' and 'confidence' columns.
    :param confidence_threshold: The threshold below which confidence is considered low.
    :param blink_threshold_range: The range of durations (in milliseconds) that are considered blinks.
    :param eye_openness_column: Optional column name for eye openness to help identify blinks.
    :raises ValueError: If the DataFrame does not contain the required columns.
    :return gaps: A DataFrame with information about detected gaps and blinks.
    """
    if not all([col in df.columns for col in ["timestamp_sec", "confidence"]]):
        raise ValueError("DataFrame must contain 'confidence' column")

    low_confidence_df = df[["timestamp_sec", "confidence"]].copy()
    low_confidence_df["openness"] = 0  # Initialize openness column with NaN
    normal_openness_value = np.inf
    if eye_openness_column and eye_openness_column in df.columns:
        normal_openness_value = df[eye_openness_column].mean(skipna=True)
        low_confidence_df["openness"] = df[eye_openness_column].copy().fillna(0)

    low_confidence_df["low_confidence"] = (
        low_confidence_df["confidence"] < confidence_threshold
    )
    low_confidence_df["transition"] = low_confidence_df[
        "low_confidence"
    ] != low_confidence_df["low_confidence"].shift(1)
    low_confidence_df["group"] = low_confidence_df["transition"].cumsum()
    low_confidence_df["id"] = low_confidence_df.index
    low_confidence_df_group = low_confidence_df.groupby("group").agg(
        {
            "low_confidence": ["first", "count"],
            "timestamp_sec": ["min", "max"],
            "id": ["min", "max"],
            "openness": "mean",
        }
    )
    low_confidence_df_group["duration_ms"] = (
        low_confidence_df_group["timestamp_sec"]["max"]
        - low_confidence_df_group["timestamp_sec"]["min"]
    ) * 1000

    # Gaps
    gaps_to_fill_df = low_confidence_df_group[
        low_confidence_df_group["low_confidence"]["first"]
    ]
    gaps_to_fill_df["start_timestamp"] = gaps_to_fill_df["timestamp_sec"]["min"]
    gaps_to_fill_df["stop_timestamp"] = gaps_to_fill_df["timestamp_sec"]["max"]
    gaps_to_fill_df["start_id"] = gaps_to_fill_df["id"]["min"].astype(int)
    gaps_to_fill_df["stop_id"] = gaps_to_fill_df["id"]["max"].astype(int)
    gaps_to_fill_df["openness_mean"] = gaps_to_fill_df["openness"]["mean"]
    gaps_to_fill_df = gaps_to_fill_df[
        [
            "start_id",
            "stop_id",
            "start_timestamp",
            "stop_timestamp",
            "duration_ms",
            "openness_mean",
        ]
    ].reset_index(drop=True)
    gaps_to_fill_df = gaps_to_fill_df.droplevel(level=1, axis=1)

    # Blinks
    gaps_to_fill_df["is_blink"] = (
        gaps_to_fill_df["duration_ms"] >= blink_threshold_range[0]
    )
    gaps_to_fill_df["is_blink"] &= (
        gaps_to_fill_df["duration_ms"] <= blink_threshold_range[1]
    )
    gaps_to_fill_df["is_blink"] &= gaps_to_fill_df["openness_mean"] < (
        normal_openness_value * openness_threshold
    )
    gaps_to_fill_df.drop(columns=["openness_mean"], inplace=True)

    return gaps_to_fill_df
