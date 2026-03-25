"""
ml/pipelines/explain.py
========================
SHAP-based explainability for the trained models.

Generates feature importance rankings and summary plots that show which
sensor features contribute most to anomaly predictions.  This is critical
for a medical-adjacent system — clinicians need to know *why* the model
flagged something, not just that it did.

Outputs
-------
- ``{eval_dir}/shap_summary.png``           — SHAP beeswarm summary plot
- ``{eval_dir}/shap_bar.png``               — global feature importance bar plot
- ``{eval_dir}/feature_importance.json``     — ranked feature importances

Usage
-----
  from ml.pipelines.explain import explain_model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "ml_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def explain_model(
    model,
    scaler,
    X: pd.DataFrame,
    feature_names: list[str],
    config: dict | None = None,
    label: str = "anomaly",
) -> dict[str, float]:
    """
    Compute SHAP values and generate explanation plots.

    Parameters
    ----------
    model         : Trained model (IsolationForest, XGBClassifier, etc.)
    scaler        : Fitted scaler used during training.
    X             : Feature DataFrame (test set or sample).
    feature_names : Ordered feature column names.
    config        : ML config dict.
    label         : Prefix for saved files.

    Returns
    -------
    Dict mapping feature name → mean |SHAP value| (importance).
    """
    if config is None:
        config = _load_config()

    try:
        import shap
    except ImportError:
        logger.warning("shap package not installed — skipping explanations")
        return {}

    explain_cfg = config.get("explain", {})
    max_samples = explain_cfg.get("max_samples", 200)
    top_n = explain_cfg.get("top_n_features", 20)

    eval_dir = PROJECT_ROOT / config.get("output", {}).get(
        "evaluation_dir", "ml/models/simulation/evaluation"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for speed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X.copy()

    X_scaled = pd.DataFrame(
        scaler.transform(X_sample),
        columns=feature_names,
        index=X_sample.index,
    )

    # Use TreeExplainer for tree-based models, KernelExplainer as fallback
    model_type = type(model).__name__
    if model_type in ("IsolationForest", "XGBClassifier",
                       "RandomForestClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
    else:
        explainer = shap.KernelExplainer(model.predict, X_scaled.iloc[:50])
        shap_values = explainer.shap_values(X_scaled)

    # For multi-output SHAP (e.g., IsolationForest), take the relevant array
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]  # last class for classifiers; only one for IF

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs_shap.tolist()))
    importance_sorted = dict(
        sorted(importance.items(), key=lambda t: t[1], reverse=True)
    )

    # Save JSON
    with open(eval_dir / f"{label}_feature_importance.json", "w") as f:
        json.dump(importance_sorted, f, indent=2)

    # ── SHAP summary plot ───────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_scaled,
            feature_names=feature_names,
            max_display=top_n,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(eval_dir / f"{label}_shap_summary.png", dpi=150,
                    bbox_inches="tight")
        plt.close("all")
    except Exception as e:
        logger.warning(f"Could not generate SHAP summary plot: {e}")

    # ── SHAP bar plot ───────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Manual bar plot of top-N features
        top_features = list(importance_sorted.keys())[:top_n]
        top_values = [importance_sorted[f] for f in top_features]
        top_features.reverse()
        top_values.reverse()

        ax.barh(top_features, top_values, color="#3498db", alpha=0.8)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Feature Importances — {label}")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(eval_dir / f"{label}_shap_bar.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Could not generate SHAP bar plot: {e}")

    logger.info(
        f"[{label}] SHAP explanations computed for {len(X_sample)} samples, "
        f"top feature: {list(importance_sorted.keys())[0]}"
    )
    return importance_sorted
