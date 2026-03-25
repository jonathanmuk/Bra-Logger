"""
ml/pipelines/train_anomaly.py
==============================
Train an anomaly detector on healthy-only data.

Uses the model registry for pluggable algorithm support — the algorithm
is selected from ml_config.yaml (e.g. IsolationForest, OneClassSVM,
LocalOutlierFactor, Autoencoder).  Optionally integrates Optuna tuning
and MLflow tracking.

Outputs
-------
- ``ml/models/simulation/anomaly_model.joblib``  — trained model
- ``ml/models/simulation/scaler.joblib``          — fitted StandardScaler
- ``ml/models/simulation/feature_names.json``     — ordered feature column names

Usage
-----
  python -m ml.pipelines.train_anomaly          # uses defaults from ml_config.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from ml.pipelines.model_registry import get_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "ml_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def train_anomaly_detector(
    X_train: pd.DataFrame,
    config: dict | None = None,
    params_override: dict | None = None,
) -> tuple:
    """
    Fit a StandardScaler + anomaly model on the training feature matrix.

    The algorithm is resolved from the model registry using the name in
    ``config['anomaly_model']['algorithm']``.

    Parameters
    ----------
    X_train         : Feature DataFrame (healthy windows only for semi-supervised mode,
                      or all windows for unsupervised).
    config          : ML config dict. If None, loaded from ml_config.yaml.
    params_override : If provided (e.g. from Optuna tuning), these params replace
                      the config defaults.

    Returns
    -------
    model          : Fitted anomaly model.
    scaler         : Fitted StandardScaler.
    feature_names  : Ordered list of feature column names.
    """
    if config is None:
        config = _load_config()

    model_cfg = config.get("anomaly_model", {})
    algorithm = model_cfg.get("algorithm", "IsolationForest")
    params = params_override if params_override else model_cfg.get("params", {})

    feature_names = list(X_train.columns)

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Fit model from registry
    model = get_model(algorithm, params)
    model.fit(X_scaled)

    logger.info(
        f"{algorithm} trained on {X_scaled.shape[0]} samples × "
        f"{X_scaled.shape[1]} features"
    )
    return model, scaler, feature_names


def score_anomalies(
    model,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Compute anomaly scores in [0, 1] range.

    Isolation Forest's ``decision_function`` returns negative scores for
    anomalies.  We negate and min-max normalise to [0, 1] where
    higher = more anomalous.
    """
    X_scaled = scaler.transform(X)

    # Use decision_function if available, else score_samples
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X_scaled)
        negated = -raw_scores
    elif hasattr(model, "score_samples"):
        raw_scores = model.score_samples(X_scaled)
        negated = raw_scores  # higher = more anomalous for autoencoders
    else:
        # Fallback: predict returns -1 for anomaly
        preds = model.predict(X_scaled)
        return (np.array(preds) == -1).astype(float)

    s_min, s_max = negated.min(), negated.max()
    if s_max - s_min < 1e-12:
        return np.zeros(len(negated))
    normalised = (negated - s_min) / (s_max - s_min)
    return normalised


def save_artifacts(
    model,
    scaler: StandardScaler,
    feature_names: list[str],
    config: dict | None = None,
) -> Path:
    """Save model, scaler, and feature names to the configured output directory."""
    if config is None:
        config = _load_config()

    out_dir = PROJECT_ROOT / config.get("output", {}).get("model_dir", "ml/models/simulation")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "anomaly_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")

    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(f"Artifacts saved to {out_dir}")
    return out_dir


def load_artifacts(
    config: dict | None = None,
) -> tuple:
    """Load previously saved model artifacts."""
    if config is None:
        config = _load_config()

    model_dir = PROJECT_ROOT / config.get("output", {}).get("model_dir", "ml/models/simulation")

    model = joblib.load(model_dir / "anomaly_model.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")

    with open(model_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    return model, scaler, feature_names
