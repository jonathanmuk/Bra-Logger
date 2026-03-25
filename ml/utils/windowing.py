"""
ml/utils/windowing.py
======================
Sliding window utilities for converting raw time-series DataFrames
into fixed-length feature windows.

A "window" is a contiguous slice of sensor readings (e.g., 30 s at 2 Hz
= 60 rows) that we treat as one sample for the ML model.

Usage
-----
  from ml.utils.windowing import create_windows

  df        # pandas DataFrame, one row per sensor reading
  windows = create_windows(df, window_s=30, step_s=5, hz=2.0)
  # windows: list of DataFrames, each 60 rows long
"""

from __future__ import annotations

import pandas as pd


def create_windows(
    df: pd.DataFrame,
    window_s: float = 30.0,
    step_s: float = 5.0,
    hz: float = 2.0,
) -> list[pd.DataFrame]:
    """
    Split a session DataFrame into overlapping fixed-length windows.

    Parameters
    ----------
    df       : Session DataFrame with one row per sample.
    window_s : Window length in seconds (default 30 s → 60 rows at 2 Hz).
    step_s   : Slide step in seconds   (default  5 s → 10 rows at 2 Hz).
    hz       : Sampling rate in Hz.

    Returns
    -------
    List of DataFrames, each ``window_rows`` long.
    Incomplete trailing windows are discarded.
    """
    window_rows = int(window_s * hz)
    step_rows   = int(step_s   * hz)

    windows: list[pd.DataFrame] = []
    start = 0
    while start + window_rows <= len(df):
        windows.append(df.iloc[start : start + window_rows].copy())
        start += step_rows

    return windows


def windows_from_sessions(
    sessions: list[pd.DataFrame],
    window_s: float = 30.0,
    step_s: float = 5.0,
    hz: float = 2.0,
) -> list[pd.DataFrame]:
    """
    Apply ``create_windows`` to a list of session DataFrames.

    Returns a flat list of window DataFrames across all sessions.
    """
    all_windows: list[pd.DataFrame] = []
    for session_df in sessions:
        all_windows.extend(create_windows(session_df, window_s, step_s, hz))
    return all_windows
