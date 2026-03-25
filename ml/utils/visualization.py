"""
ml/utils/visualization.py
==========================
Reusable plotting helpers used by notebooks and the evaluate pipeline.
All functions return matplotlib Figure objects so callers can save or show them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


# ── Colours ───────────────────────────────────────────────────────────────────
RISK_COLOURS = {
    "low":      "#27ae60",
    "moderate": "#f39c12",
    "high":     "#e74c3c",
    "critical": "#8e44ad",
}


# =============================================================================
# Time-series plots
# =============================================================================

def plot_session(
    df: pd.DataFrame,
    title: str = "Session",
    figsize: tuple = (14, 10),
) -> Figure:
    """
    Plot all sensor channels for one session DataFrame.

    Produces four sub-plots: Temperature | Pressure | Impedance | Environment + IMU.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    t_cols = [c for c in df.columns if c.startswith("t") and c.endswith("_c")]
    p_cols = [c for c in df.columns if c.startswith("p") and c.endswith("_raw")]
    z_cols = [c for c in df.columns if c.startswith("z") and c.endswith("_ohm")]
    env_cols = ["ambient_temp_c", "ambient_humidity_pct"]

    for col in t_cols:
        axes[0].plot(df[col].values, label=col)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for col in p_cols:
        axes[1].plot(df[col].values, label=col, alpha=0.7)
    axes[1].set_ylabel("Pressure (ADC)")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    for col in z_cols:
        axes[2].plot(df[col].values, label=col)
    axes[2].set_ylabel("Impedance (Ω)")
    axes[2].legend(ncol=4, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    for col in env_cols:
        if col in df.columns:
            axes[3].plot(df[col].values, label=col)
    axes[3].set_ylabel("Environment")
    axes[3].set_xlabel("Sample index")
    axes[3].legend(ncol=2, fontsize=8)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# Anomaly score distribution
# =============================================================================

def plot_anomaly_scores(
    scores: np.ndarray | list,
    labels: np.ndarray | list | None = None,
    threshold: float | None = None,
    title: str = "Anomaly Score Distribution",
    figsize: tuple = (10, 5),
) -> Figure:
    """
    Histogram of anomaly scores (0–1 range).

    Parameters
    ----------
    scores    : 1-D array of per-window anomaly scores.
    labels    : Optional 1-D array of true labels (0=healthy, 1=anomaly).
                If provided, plots two overlaid histograms.
    threshold : Decision threshold to overlay as a vertical line.
    """
    fig, ax = plt.subplots(figsize=figsize)
    scores = np.asarray(scores)

    if labels is not None:
        labels = np.asarray(labels)
        ax.hist(scores[labels == 0], bins=50, alpha=0.6, color="#27ae60", label="Healthy")
        ax.hist(scores[labels == 1], bins=50, alpha=0.6, color="#e74c3c", label="Anomaly")
        ax.legend()
    else:
        ax.hist(scores, bins=50, color="#3498db", alpha=0.8)

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                   label=f"Threshold = {threshold:.2f}")
        ax.legend()

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# Breast heatmap
# =============================================================================

def plot_breast_heatmap(
    temp_values: list[float],
    pressure_values: list[float],
    title: str = "Sensor Spatial Map",
    figsize: tuple = (10, 5),
) -> Figure:
    """
    Render a schematic breast heatmap for two groups of sensors.

    The layout maps to a simplified front-view of the breast:
      Left breast → sensors t1, t2 / p1-p4
      Right breast → sensors t3, t4 / p5-p8

    Sensor positions (quadrants):
      Temp:     T[0]=upper-left  T[1]=lower-left  T[2]=upper-right  T[3]=lower-right
      Pressure: P[0-3] = left breast (12 o'clock, 3, 6, 9)
                P[4-7] = right breast (12, 3, 6, 9)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _draw_breast(ax, temps, pressures, side: str):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(f"{side} Breast", fontsize=11)
        ax.axis("off")

        # Draw circle outline
        circle = plt.Circle((5, 5), 4.2, fill=False, color="gray", linewidth=1.5)
        ax.add_patch(circle)

        # Temperature zones (quadrant polygons) — simplified as scatter
        temp_pos = [(3.5, 7), (3.5, 3)]     # UL, LL
        cmap = plt.get_cmap("RdYlBu_r")
        t_min, t_max = 32, 38
        for (x, y), val in zip(temp_pos, temps):
            norm_val = (val - t_min) / (t_max - t_min)
            colour = cmap(np.clip(norm_val, 0, 1))
            ax.scatter(x, y, s=800, c=[colour], marker="s", zorder=3)
            ax.text(x, y, f"{val:.1f}°", ha="center", va="center", fontsize=9, color="white")

        # Pressure zones
        press_pos = [(5, 8.5), (8.5, 5), (5, 1.5), (1.5, 5)]   # 12, 3, 6, 9 o'clock
        p_min, p_max = 800, 2200
        for (x, y), val in zip(press_pos, pressures):
            norm_val = (val - p_min) / (p_max - p_min)
            size = 200 + int(norm_val * 600)
            colour = plt.get_cmap("Oranges")(np.clip(norm_val, 0, 1))
            ax.scatter(x, y, s=size, c=[colour], marker="o", alpha=0.7, zorder=3)
            ax.text(x, y, f"{int(val)}", ha="center", va="center", fontsize=7, color="black")

    _draw_breast(axes[0], temp_values[:2], pressure_values[:4], "Left")
    _draw_breast(axes[1], temp_values[2:], pressure_values[4:], "Right")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.01,
             "Squares = Temperature  |  Circles = Pressure (size ∝ stiffness)",
             ha="center", fontsize=9, fontstyle="italic")
    plt.tight_layout()
    return fig


# =============================================================================
# Feature importance bar chart
# =============================================================================

def plot_feature_importance(
    feature_names: list[str],
    importances: list[float],
    top_n: int = 20,
    title: str = "Top Feature Importances",
    figsize: tuple = (10, 6),
) -> Figure:
    """Horizontal bar chart of the top-N most important features."""
    pairs = sorted(zip(importances, feature_names), reverse=True)[:top_n]
    imp_sorted, names_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=figsize)
    colours = ["#e74c3c" if "temp" in n else ("#3498db" if "press" in n or "raw" in n else "#9b59b6")
               for n in names_sorted]
    ax.barh(names_sorted[::-1], imp_sorted[::-1], color=colours[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig
