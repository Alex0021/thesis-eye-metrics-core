import numpy as np
import pandas as pd
from scipy.signal import filtfilt, firwin


def calculate_gaze_angular_delta(
    df: pd.DataFrame,
    gaze_point_columns_prefix: str,
) -> pd.Series:
    """
    Compute the frame-to-frame angular delta (degrees) between consecutive 3-D gaze vectors.

    Requires columns: 'timestamp_sec', '{prefix}_x', '{prefix}_y', '{prefix}_z'
    """
    gaze_columns = [
        f"{gaze_point_columns_prefix}_x",
        f"{gaze_point_columns_prefix}_y",
        f"{gaze_point_columns_prefix}_z",
    ]
    if not all(col in df.columns for col in ["timestamp_sec"] + gaze_columns):
        raise ValueError(
            f"DataFrame must contain the following columns: "
            f"'timestamp_sec', {', '.join(gaze_columns)}"
        )
    gaze_angular_data = df[["timestamp_sec"] + gaze_columns].copy()
    pre_gaze_columns = [f"prev_{col}" for col in gaze_columns]
    gaze_angular_data[pre_gaze_columns] = gaze_angular_data[gaze_columns].shift(1)

    def calculate_angle(row):
        if any(pd.isna(row[col]) for col in pre_gaze_columns):
            return np.nan
        v1 = np.array([row[col] for col in gaze_columns])
        v2 = np.array([row[col] for col in pre_gaze_columns])
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return np.nan
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    return gaze_angular_data.apply(calculate_angle, axis=1)


def calculate_angular_velocity(
    gaze_df: pd.DataFrame,
    sample_rate: float,
    velocity_limit_min: float,
    velocity_limit_max: float,
    filter_cutoff_hz: float,
    filter_taps: int,
    absolute: bool = True,
    filtered: bool = True,
) -> pd.Series:
    """
    Compute angular velocity (deg/s) from frame-to-frame gaze angle deltas.

    Requires columns: 'timestamp_sec', 'gaze_angle_delta_deg'

    :param gaze_df: DataFrame with gaze angle deltas.
    :param sample_rate: Sampling rate in Hz (used for FIR filter design).
    :param velocity_limit_min: Lower clip limit (deg/s).
    :param velocity_limit_max: Upper clip limit (deg/s).
    :param filter_cutoff_hz: FIR low-pass cutoff frequency (Hz).
    :param filter_taps: Number of FIR filter taps.
    :param absolute: If True, return absolute values.
    :param filtered: If True, apply FIR low-pass filter.
    """
    if "gaze_angle_delta_deg" not in gaze_df.columns:
        raise ValueError("gaze_df must contain 'gaze_angle_delta_deg' column")

    data = gaze_df[["timestamp_sec", "gaze_angle_delta_deg"]].copy()
    data["prev_angle"] = data["gaze_angle_delta_deg"].shift(1)
    data["delta_time"] = data["timestamp_sec"] - data["timestamp_sec"].shift(1)

    def _velocity(row):
        if pd.isna(row["prev_angle"]) or pd.isna(row["gaze_angle_delta_deg"]):
            return np.nan
        if row["delta_time"] == 0:
            return np.nan
        v = (row["gaze_angle_delta_deg"] - row["prev_angle"]) / row["delta_time"]
        v = np.clip(v, velocity_limit_min, velocity_limit_max)
        return abs(v) if absolute else v

    data["gaze_angular_velocity"] = data.apply(_velocity, axis=1)

    if filtered:
        fir = firwin(filter_taps, filter_cutoff_hz, fs=sample_rate)
        data["gaze_angular_velocity"] = filtfilt(
            fir, [1.0], data["gaze_angular_velocity"]
        )

    return data["gaze_angular_velocity"]


def calculate_fixations_saccades_idt(
    eye_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    idt_duration_threshold: int,
    idt_dispersion_threshold: float,
    sample_rate: float,
    verbose: bool = True,
):
    """
    Detect fixations and saccades using the I-DT (dispersion-threshold) algorithm.

    Requires columns: 'timestamp_sec', 'gaze_point_screen_x', 'gaze_point_screen_y'

    :param eye_df: Eye-tracking data.
    :param gaps_df: Detected gaps/blinks.
    :param idt_duration_threshold: Minimum fixation duration in ms.
    :param idt_dispersion_threshold: Maximum spatial dispersion for a fixation.
    :param sample_rate: Sampling rate in Hz.
    :param verbose: Print detection summary.
    :returns: (fixations_df, saccades_df)
    """

    def dispersion(x_points: pd.Series, y_points: pd.Series):
        return np.sqrt(
            (x_points.max() - x_points.min()) ** 2
            + (y_points.max() - y_points.min()) ** 2
        )

    window_size = int(idt_duration_threshold / 1000 * sample_rate)
    sliding_window_start = 0
    gaps_to_skip = gaps_df[gaps_df["duration_ms"] >= idt_duration_threshold]
    fixations_rows = []
    if (gaps_to_skip["start_timestamp"] < idt_duration_threshold / 1000).any():
        sliding_window_start = window_size

    while sliding_window_start < eye_df.shape[0]:
        end_idx = min(sliding_window_start + window_size, eye_df.shape[0])
        sliding_window = eye_df.iloc[sliding_window_start:end_idx]
        disp_value = dispersion(
            sliding_window["gaze_point_screen_x"], sliding_window["gaze_point_screen_y"]
        )
        if disp_value <= idt_dispersion_threshold:
            while disp_value <= idt_dispersion_threshold:
                end_idx += 1
                if end_idx >= eye_df.shape[0]:
                    break
                sliding_window = eye_df.iloc[sliding_window_start:end_idx]
                disp_value = dispersion(
                    sliding_window["gaze_point_screen_x"],
                    sliding_window["gaze_point_screen_y"],
                )
                if disp_value > idt_dispersion_threshold:
                    break
            mean_x = sliding_window["gaze_point_screen_x"].mean()
            mean_y = sliding_window["gaze_point_screen_y"].mean()
            radius = np.sqrt(
                (sliding_window["gaze_point_screen_x"] - mean_x) ** 2
                + (sliding_window["gaze_point_screen_y"] - mean_y) ** 2
            ).max()
            duration = (
                sliding_window["timestamp_sec"].iloc[-1]
                - sliding_window["timestamp_sec"].iloc[0]
            ) * 1000
            fixations_rows.append(
                {
                    "start_timestamp": sliding_window["timestamp_sec"].iloc[0],
                    "stop_timestamp": sliding_window["timestamp_sec"].iloc[-1],
                    "duration_ms": duration,
                    "x": mean_x,
                    "y": mean_y,
                    "radius": radius,
                    "start_idx": sliding_window.index[0],
                    "stop_idx": sliding_window.index[-1],
                }
            )
            sliding_window_start = end_idx + 1
        else:
            sliding_window_start += 1

    fixations_df_idt = pd.DataFrame(fixations_rows)

    saccades_rows = []
    gaps_too_long = gaps_df[gaps_df["duration_ms"] >= idt_duration_threshold]
    for i in range(len(fixations_df_idt) - 1):
        saccade_start_time = fixations_df_idt["stop_timestamp"].iloc[i]
        saccade_end_time = fixations_df_idt["start_timestamp"].iloc[i + 1]
        overlapping_gaps = gaps_too_long[
            (gaps_too_long["start_timestamp"] < saccade_end_time)
            & (gaps_too_long["stop_timestamp"] > saccade_start_time)
        ]
        if not overlapping_gaps.empty:
            for _, gap in overlapping_gaps.iterrows():
                if gap["start_timestamp"] > saccade_start_time:
                    saccade_end_time = gap["start_timestamp"]
                if gap["stop_timestamp"] < saccade_end_time:
                    saccade_start_time = gap["stop_timestamp"]
                start_idx = eye_df[eye_df["timestamp_sec"] >= saccade_start_time][
                    "timestamp_sec"
                ].idxmin()
                stop_idx = eye_df[eye_df["timestamp_sec"] <= saccade_end_time][
                    "timestamp_sec"
                ].idxmax()
                amplitude = dispersion(
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_x"],
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_y"],
                )
                saccades_rows.append(
                    {
                        "start_timestamp": saccade_start_time,
                        "stop_timestamp": saccade_end_time,
                        "duration_ms": (saccade_end_time - saccade_start_time) * 1000,
                        "start_idx": start_idx,
                        "stop_idx": stop_idx,
                        "amplitude": amplitude,
                        "peak_velocity": eye_df["gaze_angular_velocity"]
                        .iloc[start_idx:stop_idx]
                        .max(),
                        "velocity": eye_df["gaze_angular_velocity"]
                        .iloc[start_idx:stop_idx]
                        .mean(),
                    }
                )
        else:
            start_idx = fixations_df_idt["stop_idx"].iloc[i]
            stop_idx = fixations_df_idt["start_idx"].iloc[i + 1]
            amplitude = np.sqrt(
                (
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_x"].max()
                    - eye_df.loc[start_idx:stop_idx, "gaze_point_screen_x"].min()
                )
                ** 2
                + (
                    eye_df.loc[start_idx:stop_idx, "gaze_point_screen_y"].max()
                    - eye_df.loc[start_idx:stop_idx, "gaze_point_screen_y"].min()
                )
                ** 2
            )
            saccades_rows.append(
                {
                    "start_timestamp": saccade_start_time,
                    "stop_timestamp": saccade_end_time,
                    "duration_ms": (saccade_end_time - saccade_start_time) * 1000,
                    "start_idx": start_idx,
                    "stop_idx": stop_idx,
                    "amplitude": amplitude,
                    "peak_velocity": eye_df["gaze_angular_velocity"]
                    .iloc[start_idx:stop_idx]
                    .max(),
                    "velocity": eye_df["gaze_angular_velocity"]
                    .iloc[start_idx:stop_idx]
                    .mean(),
                }
            )
    saccades_df_idt = pd.DataFrame(saccades_rows)

    if verbose:
        print(
            f"Detected {len(fixations_df_idt)} fixations and {len(saccades_df_idt)} saccades with IDT algorithm"
        )

    return fixations_df_idt, saccades_df_idt


def calculate_fixations_saccades_ivt(
    eye_df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    ivt_threshold: float,
    min_fixation_duration_ms: int,
    sample_rate: float,
    min_datapoints: int = 2,
    verbose: bool = True,
):
    """
    Detect fixations and saccades using the I-VT (velocity-threshold) algorithm.

    Requires columns: 'timestamp_sec', 'gaze_angular_velocity',
                      'gaze_point_screen_x', 'gaze_point_screen_y',
                      'gaze_angle_delta_deg'

    :param eye_df: Eye-tracking data.
    :param gaps_df: Detected gaps/blinks (currently unused but kept for API consistency).
    :param ivt_threshold: Angular velocity threshold (deg/s) above which a sample is a saccade.
    :param min_fixation_duration_ms: Fixations shorter than this are reclassified as saccades.
    :param sample_rate: Sampling rate in Hz.
    :param min_datapoints: Isolated segments shorter than this are interpolated away.
    :param verbose: Print detection summary.
    :returns: (eye_df_annotated, fixations_df, saccades_df)
    """
    eye_df = eye_df.copy()
    eye_df["saccade"] = eye_df["gaze_angular_velocity"] > ivt_threshold
    eye_df["fixation"] = ~eye_df["saccade"]
    eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)

    if verbose:
        print(f"Found {eye_df['transition'].sum()} transitions between saccades and fixations")

    # Reclassify fixations shorter than min_fixation_duration_ms as saccades
    min_fixation_samples = int(np.ceil(min_fixation_duration_ms / (1000 / sample_rate)))
    eye_df["id"] = eye_df.index
    grouped = eye_df.groupby(eye_df["transition"].cumsum())
    fixation_groups = grouped.agg(
        {"fixation": ["first", "count"], "id": ["min", "max"]}
    )

    if verbose:
        n_short = (
            (fixation_groups[("fixation", "count")] < min_fixation_samples)
            & (fixation_groups[("fixation", "first")])
        ).sum()
        print(f"Will mark {n_short} short fixations as saccades")

    for _, group in fixation_groups.iterrows():
        if group["fixation"]["count"] < min_fixation_samples and group["fixation"]["first"]:
            eye_df.loc[group["id"]["min"]: group["id"]["max"], "saccade"] = True
            eye_df.loc[group["id"]["min"]: group["id"]["max"], "fixation"] = False

    # Remove isolated single samples (noise)
    if min_datapoints > 1:
        eye_df["saccade"] = eye_df["saccade"].astype(np.int32)
        eye_df["fixation"] = eye_df["fixation"].astype(np.int32)
        eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)
        grouped = eye_df.groupby(eye_df["transition"].cumsum())
        single_samples = grouped.agg(
            {"saccade": "count", "id": "first", "fixation": "count"}
        )
        single_samples = single_samples[
            (single_samples["saccade"] <= min_datapoints - 1)
            | (single_samples["fixation"] <= min_datapoints - 1)
        ]["id"]
        eye_df.loc[single_samples, "saccade"] = pd.NA
        eye_df.loc[single_samples, "fixation"] = pd.NA
        eye_df["saccade"] = eye_df["saccade"].interpolate(method="nearest")
        eye_df["fixation"] = eye_df["fixation"].interpolate(method="nearest")
        eye_df["saccade"] = eye_df["saccade"].astype(bool)
        eye_df["fixation"] = eye_df["fixation"].astype(bool)
        eye_df["transition"] = eye_df["saccade"] != eye_df["saccade"].shift(1)

    # Build fixation and saccade event DataFrames
    rows_saccades = []
    rows_fixations = []

    grouped = eye_df.groupby(eye_df["transition"].cumsum())
    for _, group in grouped:
        if group["saccade"].iloc[0]:
            rows_saccades.append(
                {
                    "start_timestamp": group["timestamp_sec"].iloc[0],
                    "stop_timestamp": group["timestamp_sec"].iloc[-1],
                    "duration_ms": (
                        group["timestamp_sec"].iloc[-1] - group["timestamp_sec"].iloc[0]
                    ) * 1000,
                    "start_idx": group["id"].iloc[0],
                    "stop_idx": group["id"].iloc[-1],
                    "amplitude": abs(
                        group["gaze_angle_delta_deg"].iloc[-1]
                        - group["gaze_angle_delta_deg"].iloc[0]
                    ),
                    "peak_velocity": group["gaze_angular_velocity"].max(),
                    "velocity": group["gaze_angular_velocity"].mean(),
                }
            )
        else:
            mean_x = group["gaze_point_screen_x"].mean()
            mean_y = group["gaze_point_screen_y"].mean()
            radius = np.sqrt(
                (
                    (group["gaze_point_screen_x"] - mean_x) ** 2
                    + (group["gaze_point_screen_y"] - mean_y) ** 2
                ).max()
            )
            rows_fixations.append(
                {
                    "start_timestamp": group["timestamp_sec"].iloc[0],
                    "stop_timestamp": group["timestamp_sec"].iloc[-1],
                    "duration_ms": (
                        group["timestamp_sec"].iloc[-1] - group["timestamp_sec"].iloc[0]
                    ) * 1000,
                    "start_idx": group["id"].iloc[0],
                    "stop_idx": group["id"].iloc[-1],
                    "x": mean_x,
                    "y": mean_y,
                    "radius": radius,
                }
            )

    fixations_df = pd.DataFrame(rows_fixations)
    saccades_df = pd.DataFrame(rows_saccades)

    if verbose:
        print(
            f"Detected {len(fixations_df)} fixations and {len(saccades_df)} saccades with IVT algorithm"
        )

    return eye_df, fixations_df, saccades_df
