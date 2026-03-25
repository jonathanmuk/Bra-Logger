"""
ml/utils/features.py
=====================
Low-level statistical and spectral feature extraction functions.
These are the building blocks called by ml/pipelines/feature_engineering.py.

Each function takes a 1-D numpy array (one channel, one window) and returns
a dict of computed feature scalars.  This keeps the functions pure and testable.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis


# =============================================================================
# Per-channel time-domain features
# =============================================================================

def time_domain_features(x: np.ndarray, prefix: str = "") -> dict[str, float]:
    """
    Compute standard time-domain statistics for one channel window.

    Parameters
    ----------
    x      : 1-D array of sensor values.
    prefix : String to prepend to all feature keys.

    Returns
    -------
    Dict with keys: {prefix}mean, {prefix}std, {prefix}min, {prefix}max,
                    {prefix}range, {prefix}median, {prefix}iqr,
                    {prefix}skew, {prefix}kurtosis,
                    {prefix}slope, {prefix}energy,
                    {prefix}d1_mean, {prefix}d1_std
    """
    p = prefix
    n = len(x)
    if n == 0:
        return {}

    d1 = np.diff(x)
    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, x, 1)[0]) if n > 1 else 0.0

    return {
        f"{p}mean":      float(np.mean(x)),
        f"{p}std":       float(np.std(x)),
        f"{p}min":       float(np.min(x)),
        f"{p}max":       float(np.max(x)),
        f"{p}range":     float(np.max(x) - np.min(x)),
        f"{p}median":    float(np.median(x)),
        f"{p}iqr":       float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{p}skew":      float(skew(x)) if n >= 3 else 0.0,
        f"{p}kurtosis":  float(kurtosis(x)) if n >= 4 else 0.0,
        f"{p}slope":     slope,
        f"{p}energy":    float(np.mean(x ** 2)),
        f"{p}d1_mean":   float(np.mean(d1)) if len(d1) > 0 else 0.0,
        f"{p}d1_std":    float(np.std(d1))  if len(d1) > 0 else 0.0,
    }


# =============================================================================
# Spectral features
# =============================================================================

def spectral_features(x: np.ndarray, hz: float = 2.0, prefix: str = "") -> dict[str, float]:
    """
    Compute frequency-domain features for one channel window.

    Returns
    -------
    Dict with keys: {prefix}dominant_freq, {prefix}spectral_entropy,
                    {prefix}spectral_energy, {prefix}spectral_centroid
    """
    p = prefix
    n = len(x)
    if n < 4:
        return {f"{p}dominant_freq": 0.0, f"{p}spectral_entropy": 0.0,
                f"{p}spectral_energy": 0.0, f"{p}spectral_centroid": 0.0}

    freqs = np.fft.rfftfreq(n, d=1.0 / hz)
    fft_mag = np.abs(np.fft.rfft(x - np.mean(x)))
    # Avoid log(0)
    power = fft_mag ** 2 + 1e-12
    power_norm = power / power.sum()

    dominant_freq  = float(freqs[np.argmax(fft_mag)])
    spectral_entropy = float(-np.sum(power_norm * np.log(power_norm)))
    spectral_energy  = float(np.sum(power))
    spectral_centroid = float(np.sum(freqs * power_norm))

    return {
        f"{p}dominant_freq":    dominant_freq,
        f"{p}spectral_entropy": spectral_entropy,
        f"{p}spectral_energy":  spectral_energy,
        f"{p}spectral_centroid":spectral_centroid,
    }


# =============================================================================
# Cross-channel / spatial features
# =============================================================================

def left_right_diff_features(
    left_vals: np.ndarray,  # shape (N, n_left)
    right_vals: np.ndarray, # shape (N, n_right)
    modality: str = "",
) -> dict[str, float]:
    """
    Compute left-breast vs right-breast difference features.
    High asymmetry is a key breast thermography anomaly indicator.
    """
    p = f"{modality}_lr_" if modality else "lr_"
    left_mean  = left_vals.mean(axis=1)   # (N,) mean across left channels
    right_mean = right_vals.mean(axis=1)  # (N,) mean across right channels
    diff = left_mean - right_mean

    return {
        f"{p}diff_mean": float(diff.mean()),
        f"{p}diff_std":  float(diff.std()),
        f"{p}diff_max":  float(np.abs(diff).max()),
        f"{p}diff_min":  float(diff.min()),
        # Sustained asymmetry: fraction of samples where |diff| > threshold
        f"{p}asym_frac_05": float((np.abs(diff) > 0.5).mean()),
        f"{p}asym_frac_1":  float((np.abs(diff) > 1.0).mean()),
    }


def hotspot_features(values: np.ndarray, modality: str = "") -> dict[str, float]:
    """
    Detect localised extremes across sensor channels.

    Parameters
    ----------
    values  : shape (N, C) — N samples, C channels.
    modality: prefix string.
    """
    p = f"{modality}_hotspot_" if modality else "hotspot_"
    channel_means = values.mean(axis=0)   # (C,)
    overall_mean  = channel_means.mean()
    max_channel   = channel_means.max()
    min_channel   = channel_means.min()

    return {
        f"{p}max_deviation": float(max_channel - overall_mean),
        f"{p}min_deviation": float(min_channel - overall_mean),
        f"{p}range":         float(max_channel - min_channel),
        f"{p}cv":            float(channel_means.std() / (overall_mean + 1e-9)),
    }


def cross_modal_correlation(
    x: np.ndarray, y: np.ndarray, prefix: str = "cross"
) -> dict[str, float]:
    """Pearson correlation between mean of two modality arrays (N,)."""
    if len(x) < 3 or len(y) < 3:
        return {f"{prefix}_corr": 0.0}
    corr = float(np.corrcoef(x, y)[0, 1])
    return {f"{prefix}_corr": corr if not np.isnan(corr) else 0.0}
