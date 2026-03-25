"""
ml/pipelines/tuning.py
========================
Optuna hyperparameter tuning wrapper.

Provides a unified interface for tuning any registered model, regardless
of whether it's an anomaly detector or classifier.

Usage
-----
  from ml.pipelines.tuning import tune_model

  best_params = tune_model(
      model_name="XGBClassifier",
      X_train=X_train, y_train=y_train,
      X_val=X_val, y_val=y_val,
      config=config,
  )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

from ml.pipelines.model_registry import get_model, get_search_space, get_spec

logger = logging.getLogger(__name__)

# Silence Optuna's verbose default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    X_val: np.ndarray,
    y_val: np.ndarray | None,
    config: dict,
) -> dict[str, Any]:
    """
    Run Optuna hyperparameter optimization for the named model.

    Parameters
    ----------
    model_name : str
        Registered model name (e.g. "XGBClassifier", "IsolationForest").
    X_train, y_train : training data (y_train is None for anomaly models).
    X_val, y_val : validation data.
    config : dict
        Full pipeline config; reads the 'optuna' section.

    Returns
    -------
    dict : Best hyperparameters found.
    """
    spec = get_spec(model_name)
    tuning_cfg = config.get("optuna", {})
    n_trials = tuning_cfg.get("n_trials", 50)
    timeout = tuning_cfg.get("timeout", 300)

    if spec.category == "anomaly":
        direction = tuning_cfg.get("anomaly_direction", "maximize")
        objective_fn = _make_anomaly_objective(model_name, X_train, X_val, y_val, config)
    else:
        direction = tuning_cfg.get("classifier_direction", "maximize")
        objective_fn = _make_classifier_objective(model_name, X_train, y_train, X_val, y_val, config)

    study = optuna.create_study(direction=direction)
    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)

    logger.info(f"Tuning complete — best {direction} value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params


def _make_anomaly_objective(model_name, X_train, X_val, y_val, config):
    """Create Optuna objective for anomaly detection models."""

    def objective(trial):
        params = get_search_space(model_name, trial)
        model = get_model(model_name, params)
        model.fit(X_train)

        # Score: use F1 if labels available, else silhouette-like metric
        if y_val is not None:
            preds = model.predict(X_val)
            # Convert sklearn convention (-1=anomaly, 1=normal) to binary
            pred_labels = (np.array(preds) == -1).astype(int)
            true_labels = np.array(y_val).astype(int)
            from sklearn.metrics import f1_score
            return f1_score(true_labels, pred_labels, zero_division=0)
        else:
            # Unsupervised: use mean negative decision function as proxy
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_val)
                return -np.mean(np.abs(scores))
            return 0.0

    return objective


def _make_classifier_objective(model_name, X_train, y_train, X_val, y_val, config):
    """Create Optuna objective for classifier models."""

    metric = config.get("optuna", {}).get("classifier_metric", "f1_weighted")

    def objective(trial):
        params = get_search_space(model_name, trial)
        model = get_model(model_name, params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

        if metric == "f1_weighted":
            return f1_score(y_val, preds, average="weighted", zero_division=0)
        elif metric == "f1_macro":
            return f1_score(y_val, preds, average="macro", zero_division=0)
        elif metric == "accuracy":
            return accuracy_score(y_val, preds)
        elif metric == "roc_auc" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)
            if proba.shape[1] == 2:
                return roc_auc_score(y_val, proba[:, 1])
            return roc_auc_score(y_val, proba, multi_class="ovr", average="weighted")
        else:
            return f1_score(y_val, preds, average="weighted", zero_division=0)

    return objective
