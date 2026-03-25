"""
ml/pipelines/train_classifier.py
==================================
Train supervised classifiers for anomaly classification.

Uses the model registry for pluggable algorithm support — the algorithm
is selected from ml_config.yaml (e.g. XGBClassifier, RandomForestClassifier,
GradientBoostingClassifier, SVC, MLPClassifier).

Outputs
-------
- ``ml/models/simulation/classifier_model.joblib`` — trained classifier
- ``ml/models/simulation/classifier_scaler.joblib`` — fitted scaler

Usage
-----
  python -m ml.pipelines.train_classifier
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


def train_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: dict | None = None,
    params_override: dict | None = None,
) -> tuple:
    """
    Fit scaler + classifier.

    The algorithm is resolved from the model registry using the name in
    ``config['classifier']['algorithm']``.

    Parameters
    ----------
    X_train         : Feature matrix.
    y_train         : Binary labels (0=healthy, 1=anomaly).
    config          : ML config dict.
    params_override : If provided (e.g. from Optuna tuning), these params replace
                      the config defaults.

    Returns
    -------
    clf            : Fitted classifier.
    scaler         : Fitted StandardScaler.
    feature_names  : Ordered feature column names.
    """
    if config is None:
        config = _load_config()

    clf_cfg = config.get("classifier", {})
    algorithm = clf_cfg.get("algorithm", "XGBClassifier")
    params = params_override if params_override else clf_cfg.get("params", {})

    feature_names = list(X_train.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = get_model(algorithm, params)
    clf.fit(X_scaled, y_train)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    logger.info(
        f"Classifier trained: {algorithm} on "
        f"{len(y_train)} samples ({n_neg} healthy, {n_pos} anomaly)"
    )
    return clf, scaler, feature_names


def predict_proba(
    clf,
    scaler: StandardScaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return probability of anomaly class (class 1)."""
    X_scaled = scaler.transform(X)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_scaled)[:, 1]
    return clf.predict(X_scaled).astype(float)


def save_artifacts(
    clf,
    scaler: StandardScaler,
    feature_names: list[str],
    config: dict | None = None,
) -> Path:
    """Save classifier artifacts."""
    if config is None:
        config = _load_config()

    out_dir = PROJECT_ROOT / config.get("output", {}).get("model_dir", "ml/models/simulation")
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, out_dir / "classifier_model.joblib")
    joblib.dump(scaler, out_dir / "classifier_scaler.joblib")

    with open(out_dir / "classifier_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(f"Classifier artifacts saved to {out_dir}")
    return out_dir
