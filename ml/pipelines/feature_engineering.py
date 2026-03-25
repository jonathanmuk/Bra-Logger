"""
ml/pipelines/feature_engineering.py
====================================
Orchestrates window-level feature extraction across all sensor channels.

Takes a list of windowed DataFrames (from ``ml.utils.windowing``) and produces
a flat feature matrix (N_windows × ~150 features) suitable for ML models.

Uses the low-level feature functions in ``ml.utils.features`` as building blocks.

Usage
-----
  from ml.pipelines.feature_engineering import extract_features

  feature_df, labels = extract_features(windows)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.utils.features import (
    cross_modal_correlation,
    hotspot_features,
    left_right_diff_features,
    spectral_features,
    time_domain_features,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "ml_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Sensor column lists (kept here for direct numpy indexing)
TEMP_COLS = ["t1_c", "t2_c", "t3_c", "t4_c"]
PRESSURE_COLS = ["p1_raw", "p2_raw", "p3_raw", "p4_raw",
                 "p5_raw", "p6_raw", "p7_raw", "p8_raw"]
IMPEDANCE_COLS = ["z1_ohm", "z2_ohm", "z3_ohm", "z4_ohm"]
ENV_COLS = ["ambient_temp_c", "ambient_humidity_pct"]
IMU_COLS = ["accel_x", "accel_y", "accel_z"]

ALL_SENSOR_COLS = TEMP_COLS + PRESSURE_COLS + IMPEDANCE_COLS + ENV_COLS + IMU_COLS

# Left / right channel mapping
LR_MAP = {
    "temp": {"left": ["t1_c", "t2_c"], "right": ["t3_c", "t4_c"]},
    "press": {"left": ["p1_raw", "p2_raw", "p3_raw", "p4_raw"],
              "right": ["p5_raw", "p6_raw", "p7_raw", "p8_raw"]},
    "imp": {"left": ["z1_ohm", "z2_ohm"], "right": ["z3_ohm", "z4_ohm"]},
}


def _extract_window_features(
    window_df: pd.DataFrame,
    cfg: dict,
) -> dict[str, float]:
    """Compute all features for a single window DataFrame."""
    feat = cfg.get("features", {})
    feats: dict[str, float] = {}

    # ── Per-channel time-domain + spectral features ─────────────────
    hz = cfg.get("windowing", {}).get("hz", 2.0)

    for col in ALL_SENSOR_COLS:
        if col not in window_df.columns:
            continue
        x = window_df[col].values.astype(float)

        if feat.get("time_domain", True):
            feats.update(time_domain_features(x, prefix=f"{col}_"))

        if feat.get("spectral", True):
            feats.update(spectral_features(x, hz=hz, prefix=f"{col}_"))

    # ── Left-right difference features ──────────────────────────────
    if feat.get("left_right_diff", True):
        for modality, sides in LR_MAP.items():
            left_cols = [c for c in sides["left"] if c in window_df.columns]
            right_cols = [c for c in sides["right"] if c in window_df.columns]
            if left_cols and right_cols:
                left_vals = window_df[left_cols].values.astype(float)
                right_vals = window_df[right_cols].values.astype(float)
                feats.update(left_right_diff_features(
                    left_vals, right_vals, modality=modality,
                ))

    # ── Hotspot features ────────────────────────────────────────────
    if feat.get("hotspot", True):
        for modality, cols in [
            ("temp", TEMP_COLS),
            ("press", PRESSURE_COLS),
            ("imp", IMPEDANCE_COLS),
        ]:
            present = [c for c in cols if c in window_df.columns]
            if present:
                vals = window_df[present].values.astype(float)
                feats.update(hotspot_features(vals, modality=modality))

    # ── Cross-modal correlation ─────────────────────────────────────
    if feat.get("cross_modal", True):
        if all(c in window_df.columns for c in TEMP_COLS + PRESSURE_COLS):
            temp_mean = window_df[TEMP_COLS].mean(axis=1).values
            press_mean = window_df[PRESSURE_COLS].mean(axis=1).values
            feats.update(cross_modal_correlation(temp_mean, press_mean,
                                                  prefix="temp_press"))

        if all(c in window_df.columns for c in TEMP_COLS + IMPEDANCE_COLS):
            temp_mean = window_df[TEMP_COLS].mean(axis=1).values
            imp_mean = window_df[IMPEDANCE_COLS].mean(axis=1).values
            feats.update(cross_modal_correlation(temp_mean, imp_mean,
                                                  prefix="temp_imp"))

    return feats


def _extract_label(window_df: pd.DataFrame) -> int:
    """
    Determine the binary label for a window.
    0 = healthy, 1 = anomaly.
    """
    if "label" not in window_df.columns:
        return 0
    labels = window_df["label"].dropna().unique()
    for lab in labels:
        if "anomaly" in str(lab).lower():
            return 1
    return 0


def extract_features(
    windows: list[pd.DataFrame],
    config: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features from a list of window DataFrames.

    Parameters
    ----------
    windows : List of DataFrames (one per window), as produced by
              ``ml.utils.windowing.create_windows``.
    config  : ML config dict. If None, loaded from ml_config.yaml.

    Returns
    -------
    feature_df : DataFrame with shape (N_windows, N_features).
    labels     : 1-D int array with shape (N_windows,).
    """
    if config is None:
        config = _load_config()

    rows: list[dict[str, float]] = []
    labels: list[int] = []

    for i, wdf in enumerate(windows):
        feats = _extract_window_features(wdf, config)
        rows.append(feats)
        labels.append(_extract_label(wdf))

    feature_df = pd.DataFrame(rows)
    # Replace any remaining NaN/Inf with 0
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(0.0, inplace=True)

    logger.info(
        f"Extracted {feature_df.shape[1]} features from "
        f"{feature_df.shape[0]} windows "
        f"({sum(labels)} anomaly, {len(labels) - sum(labels)} healthy)"
    )
    return feature_df, np.array(labels, dtype=int)
