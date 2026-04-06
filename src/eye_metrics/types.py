"""Shared data structures for eye-metrics-core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GapInfo:
    """Represents a detected gap or blink interval in eye-tracking data."""

    start_id: int
    stop_id: int
    start_timestamp: float
    stop_timestamp: float
    duration_ms: float
    is_blink: bool


@dataclass
class FixationRow:
    """One fixation event — compatible with both IDT and IVT output DataFrames.

    IDT columns: start_timestamp, stop_timestamp, duration_ms, x, y, radius, start_idx, stop_idx
    IVT columns: start_id, stop_id, start_timestamp, stop_timestamp, duration_ms, x, y, radius
    """

    start_timestamp: float
    stop_timestamp: float
    duration_ms: float
    x: float        # mean screen X position
    y: float        # mean screen Y position
    radius: float   # max distance from mean position (dispersion radius)
    start_idx: int  # row index in the source DataFrame
    stop_idx: int


@dataclass
class SaccadeRow:
    """One saccade event — compatible with both IDT and IVT output DataFrames.

    IDT columns: start_timestamp, stop_timestamp, duration_ms, start_idx, stop_idx,
                 amplitude, peak_velocity, velocity
    IVT columns: start_id, stop_id, start_timestamp, stop_timestamp, duration_ms,
                 amplitude_deg, peak_velocity, velocity
    """

    start_timestamp: float
    stop_timestamp: float
    duration_ms: float
    amplitude: float      # spatial amplitude (screen units for IDT; degrees for IVT)
    peak_velocity: float  # deg/s — maximum angular velocity during saccade
    velocity: float       # deg/s — mean angular velocity during saccade
    start_idx: int        # row index in the source DataFrame
    stop_idx: int
