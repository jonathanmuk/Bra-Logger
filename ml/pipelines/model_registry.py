"""
ml/pipelines/model_registry.py
================================
Pluggable model registry with factory pattern.

Allows swapping algorithms (ML or DL) via config without touching pipeline code.
Each algorithm is registered with its class, default params, and Optuna search space.

Usage
-----
  from ml.pipelines.model_registry import get_model, list_models

  # Get a model instance from config
  model = get_model("IsolationForest", params={"n_estimators": 200})

  # Get Optuna search space for tuning
  space = get_search_space("XGBClassifier", trial)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification for a registered model algorithm."""
    name: str
    category: str                          # "anomaly" | "classifier"
    builder: Callable[..., Any]            # factory function → model instance
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: Callable | None = None   # fn(trial) → dict of params


# Global registry
_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    """Register a model spec."""
    _REGISTRY[spec.name] = spec


def get_spec(name: str) -> ModelSpec:
    """Look up a registered model spec by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Model '{name}' not in registry. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_model(name: str, params: dict | None = None):
    """Instantiate a model by name with merged params."""
    spec = get_spec(name)
    merged = {**spec.default_params, **(params or {})}
    return spec.builder(**merged)


def get_search_space(name: str, trial) -> dict:
    """Get Optuna search space for a registered model."""
    spec = get_spec(name)
    if spec.search_space is None:
        logger.warning(f"No search space defined for '{name}', using defaults")
        return spec.default_params
    return spec.search_space(trial)


def list_models(category: str | None = None) -> list[str]:
    """List registered model names, optionally filtered by category."""
    if category is None:
        return list(_REGISTRY.keys())
    return [name for name, spec in _REGISTRY.items() if spec.category == category]


# =========================================================================
# Built-in registrations — ML models
# =========================================================================

def _register_builtins() -> None:
    """Register all built-in ML/DL algorithms."""

    # ── Isolation Forest (anomaly) ──────────────────────────────────
    def _build_iforest(**kw):
        from sklearn.ensemble import IsolationForest
        return IsolationForest(**kw)

    def _iforest_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "contamination": trial.suggest_float("contamination", 0.01, 0.15),
            "max_samples": trial.suggest_categorical("max_samples", ["auto", 0.5, 0.8, 1.0]),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "random_state": 42,
        }

    register(ModelSpec(
        name="IsolationForest",
        category="anomaly",
        builder=_build_iforest,
        default_params={"n_estimators": 200, "contamination": 0.05,
                        "max_samples": "auto", "random_state": 42, "n_jobs": -1},
        search_space=_iforest_space,
    ))

    # ── One-Class SVM (anomaly) ─────────────────────────────────────
    def _build_ocsvm(**kw):
        from sklearn.svm import OneClassSVM
        return OneClassSVM(**kw)

    def _ocsvm_space(trial):
        return {
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "nu": trial.suggest_float("nu", 0.01, 0.5),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

    register(ModelSpec(
        name="OneClassSVM",
        category="anomaly",
        builder=_build_ocsvm,
        default_params={"kernel": "rbf", "nu": 0.05, "gamma": "scale"},
        search_space=_ocsvm_space,
    ))

    # ── Local Outlier Factor (anomaly) ──────────────────────────────
    def _build_lof(**kw):
        from sklearn.neighbors import LocalOutlierFactor
        kw.setdefault("novelty", True)
        return LocalOutlierFactor(**kw)

    def _lof_space(trial):
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 10, 100),
            "contamination": trial.suggest_float("contamination", 0.01, 0.15),
            "novelty": True,
        }

    register(ModelSpec(
        name="LocalOutlierFactor",
        category="anomaly",
        builder=_build_lof,
        default_params={"n_neighbors": 20, "contamination": 0.05, "novelty": True},
        search_space=_lof_space,
    ))

    # ── XGBoost Classifier ──────────────────────────────────────────
    def _build_xgb(**kw):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(**kw)
        except ImportError:
            logger.warning("xgboost not installed, falling back to RandomForest")
            safe = {k: v for k, v in kw.items()
                    if k in ("n_estimators", "max_depth", "random_state", "n_jobs")}
            safe.setdefault("n_estimators", 200)
            safe.setdefault("random_state", 42)
            safe.setdefault("n_jobs", -1)
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**safe)

    def _xgb_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }

    register(ModelSpec(
        name="XGBClassifier",
        category="classifier",
        builder=_build_xgb,
        default_params={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                        "subsample": 0.8, "colsample_bytree": 0.8,
                        "use_label_encoder": False, "eval_metric": "logloss",
                        "random_state": 42},
        search_space=_xgb_space,
    ))

    # ── Random Forest Classifier ────────────────────────────────────
    def _build_rf(**kw):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**kw)

    def _rf_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": 42,
            "n_jobs": -1,
        }

    register(ModelSpec(
        name="RandomForestClassifier",
        category="classifier",
        builder=_build_rf,
        default_params={"n_estimators": 200, "max_depth": None,
                        "random_state": 42, "n_jobs": -1},
        search_space=_rf_space,
    ))

    # ── Gradient Boosting Classifier ────────────────────────────────
    def _build_gbc(**kw):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**kw)

    def _gbc_space(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": 42,
        }

    register(ModelSpec(
        name="GradientBoostingClassifier",
        category="classifier",
        builder=_build_gbc,
        default_params={"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
                        "random_state": 42},
        search_space=_gbc_space,
    ))

    # ── SVM Classifier ──────────────────────────────────────────────
    def _build_svc(**kw):
        from sklearn.svm import SVC
        kw.setdefault("probability", True)
        return SVC(**kw)

    def _svc_space(trial):
        return {
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "probability": True,
            "random_state": 42,
        }

    register(ModelSpec(
        name="SVC",
        category="classifier",
        builder=_build_svc,
        default_params={"kernel": "rbf", "probability": True, "random_state": 42},
        search_space=_svc_space,
    ))

    # ── PyTorch Autoencoder (anomaly, deep learning) ────────────────
    def _build_autoencoder(**kw):
        from ml.pipelines.dl_models import AnomalyAutoencoder
        return AnomalyAutoencoder(**kw)

    def _autoencoder_space(trial):
        return {
            "hidden_dims": trial.suggest_categorical(
                "hidden_dims",
                ["[128,64,32]", "[256,128,64]", "[128,64]", "[256,128,64,32]"],
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 20, 100),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

    register(ModelSpec(
        name="Autoencoder",
        category="anomaly",
        builder=_build_autoencoder,
        default_params={"hidden_dims": [128, 64, 32], "dropout": 0.2,
                        "lr": 1e-3, "epochs": 50, "batch_size": 64},
        search_space=_autoencoder_space,
    ))

    # ── PyTorch MLP Classifier (deep learning) ──────────────────────
    def _build_mlp(**kw):
        from ml.pipelines.dl_models import MLPClassifier
        return MLPClassifier(**kw)

    def _mlp_space(trial):
        return {
            "hidden_dims": trial.suggest_categorical(
                "hidden_dims",
                ["[256,128,64]", "[128,64]", "[256,128]", "[512,256,128]"],
            ),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 20, 100),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

    register(ModelSpec(
        name="MLPClassifier",
        category="classifier",
        builder=_build_mlp,
        default_params={"hidden_dims": [256, 128, 64], "dropout": 0.3,
                        "lr": 1e-3, "epochs": 50, "batch_size": 64},
        search_space=_mlp_space,
    ))


# Auto-register on import
_register_builtins()
