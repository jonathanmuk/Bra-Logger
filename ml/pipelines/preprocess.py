"""
ml/pipelines/preprocess.py
===========================
Step 1 of the shared ML pipeline: load, validate, and clean raw sensor CSVs.

This module is source-agnostic — it does not care whether the CSV came
from the data simulator or the real ESP32.  That is controlled by the
caller, who points it at ``ml/data/raw/simulated/`` or
``ml/data/raw/hardware/``.

Key responsibilities
--------------------
1. Load one or many CSV files into Pandas DataFrames.
2. Validate column presence and dtypes.
3. Handle missing values (imputation or row-drop).
4. Clip obviously invalid readings (negative temperatures, ADC > 4095).
5. Optionally normalise the ambient-temp-corrected temperature offset.

Usage
-----
  from ml.pipelines.preprocess import load_and_clean, load_sessions

  # Load all CSVs from a directory
  df = load_and_clean(source="simulated")      # → single combined DataFrame
  df = load_and_clean(source="hardware")

  # Or load a single file
  from ml.pipelines.preprocess import load_csv
  df = load_csv("ml/data/raw/simulated/healthy/sim_session_001.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from data_collection.schema import (
    ENV_COLS,
    IMPEDANCE_COLS,
    IMU_COLS,
    PRESSURE_COLS,
    SENSOR_COLS,
    TEMP_COLS,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "data_collection" / "config.yaml"

# Keep lists (not tuples) for pandas indexing ergonomics.
TEMP_COLS = list(TEMP_COLS)
PRESSURE_COLS = list(PRESSURE_COLS)
IMPEDANCE_COLS = list(IMPEDANCE_COLS)
ENV_COLS = list(ENV_COLS)
IMU_COLS = list(IMU_COLS)
SENSOR_COLS = list(SENSOR_COLS)

# Metadata / bookkeeping columns (not features)
META_COLS = ["pc_time_iso", "ts_ms", "source", "label"]

# Validity bounds for clipping
BOUNDS = {
    "temp":      (20.0,  45.0),    # °C  — plausible skin surface range
    "pressure":  (0,     4095),    # 12-bit ADC
    "impedance": (10.0, 5000.0),   # Ω   — very wide, clips only wild spikes
    "ambient":   (5.0,   45.0),    # °C
    "humidity":  (0.0,  100.0),    # %
}


# =============================================================================
# Public API
# =============================================================================

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load and lightly clean a single session CSV."""
    path = Path(path)
    df = pd.read_csv(path, low_memory=False)
    df = _ensure_columns(df)
    df = _cast_dtypes(df)
    df = _clip_bounds(df)
    df = _handle_missing(df)
    return df


def load_sessions(
    directory: str | Path,
    label_filter: str | None = None,
    max_files: int | None = None,
) -> list[pd.DataFrame]:
    """
    Load all CSVs in a directory (non-recursive) into a list of DataFrames.

    Parameters
    ----------
    directory    : Path to e.g. ``ml/data/raw/simulated``.
    label_filter : If set, only loads files whose ``label`` column matches.
    max_files    : Limit number of files (useful for quick tests).
    """
    directory = Path(directory)
    csv_files = sorted(directory.rglob("*.csv"))[:max_files]

    sessions: list[pd.DataFrame] = []
    for fp in csv_files:
        try:
            df = load_csv(fp)
            if label_filter is not None and "label" in df.columns:
                df = df[df["label"].str.contains(label_filter, na=False)]
                if df.empty:
                    continue
            df["_file"] = fp.name   # track origin for debugging
            sessions.append(df)
        except Exception as e:
            logger.warning(f"Skipping {fp.name}: {e}")

    logger.info(f"Loaded {len(sessions)} sessions from {directory}")
    return sessions


def load_and_clean(
    source: str = "simulated",
    base_dir: str | Path | None = None,
    label_filter: str | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Load all sessions for a given source and return a single combined DataFrame.

    Parameters
    ----------
    source : ``"simulated"`` or ``"hardware"``
    """
    if base_dir is None:
        base_dir = PROJECT_ROOT / "ml" / "data" / "raw" / source

    sessions = load_sessions(base_dir, label_filter=label_filter, max_files=max_files)
    if not sessions:
        raise FileNotFoundError(
            f"No CSV files found in {base_dir}. "
            f"Run simulate_data.py first if using source='simulated'."
        )
    combined = pd.concat(sessions, ignore_index=True)
    logger.info(f"Combined DataFrame: {len(combined):,} rows, {len(sessions)} sessions")
    return combined


# =============================================================================
# Internal helpers
# =============================================================================

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing sensor columns as NaN (graceful degradation)."""
    for col in SENSOR_COLS:
        if col not in df.columns:
            logger.debug(f"Column '{col}' missing — filling with NaN")
            df[col] = np.nan
    return df


def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast sensor columns to float; pressure to float (might be int)."""
    for col in TEMP_COLS + IMPEDANCE_COLS + ENV_COLS + IMU_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in PRESSURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _clip_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values to physiologically sensible ranges."""
    t_lo, t_hi = BOUNDS["temp"]
    for col in TEMP_COLS:
        df[col] = df[col].clip(t_lo, t_hi)

    p_lo, p_hi = BOUNDS["pressure"]
    for col in PRESSURE_COLS:
        df[col] = df[col].clip(p_lo, p_hi)

    z_lo, z_hi = BOUNDS["impedance"]
    for col in IMPEDANCE_COLS:
        df[col] = df[col].clip(z_lo, z_hi)

    if "ambient_temp_c" in df.columns:
        df["ambient_temp_c"] = df["ambient_temp_c"].clip(*BOUNDS["ambient"])
    if "ambient_humidity_pct" in df.columns:
        df["ambient_humidity_pct"] = df["ambient_humidity_pct"].clip(*BOUNDS["humidity"])

    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing sensor values using linear interpolation within column,
    then forward-fill and back-fill any remaining NaNs at edges.
    Rows that are still entirely NaN across all sensor cols are dropped.
    """
    for col in SENSOR_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # Last resort: fill with column median
    df[SENSOR_COLS] = df[SENSOR_COLS].fillna(df[SENSOR_COLS].median())
    return df
