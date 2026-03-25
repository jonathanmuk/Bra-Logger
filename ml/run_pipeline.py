"""
ml/run_pipeline.py
===================
CLI entry point for the end-to-end ML pipeline.

Orchestrates: load data → preprocess → window → feature engineer →
              (optional: tune) → train → evaluate → explain.

Supports pluggable algorithms via model registry, Optuna hyperparameter
tuning (--tune), and MLflow experiment tracking (--track).

Usage
-----
  # Full pipeline (train + evaluate both models)
  python -m ml.run_pipeline

  # Anomaly detector only
  python -m ml.run_pipeline --mode anomaly

  # Classifier only (requires labels)
  python -m ml.run_pipeline --mode classifier

  # With Optuna hyperparameter tuning
  python -m ml.run_pipeline --tune

  # With MLflow experiment tracking
  python -m ml.run_pipeline --track

  # All bells and whistles
  python -m ml.run_pipeline --tune --track

  # Skip SHAP explanations (faster)
  python -m ml.run_pipeline --no-explain
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Project root so imports work ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipelines.preprocess import load_sessions
from ml.utils.windowing import windows_from_sessions
from ml.pipelines.feature_engineering import extract_features
from ml.pipelines.train_anomaly import (
    train_anomaly_detector,
    score_anomalies,
    save_artifacts as save_anomaly_artifacts,
)
from ml.pipelines.train_classifier import (
    train_classifier,
    predict_proba,
    save_artifacts as save_classifier_artifacts,
)
from ml.pipelines.evaluate import evaluate_anomaly, evaluate_classifier
from ml.pipelines.explain import explain_model
from ml.pipelines.tracking import ExperimentTracker

CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "ml_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml.run_pipeline")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _split_train_test(
    feature_df, labels, config,
):
    """Stratified train/test split preserving class proportions."""
    from sklearn.model_selection import train_test_split

    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    logger.info(
        f"Split: train={len(X_train)} ({sum(y_train)} anomaly) | "
        f"test={len(X_test)} ({sum(y_test)} anomaly)"
    )
    return X_train, X_test, y_train, y_test


def _maybe_tune(model_name, category, X_train, y_train, X_val, y_val, config):
    """Run Optuna tuning if available, return best params or None."""
    from ml.pipelines.tuning import tune_model

    logger.info(f"Tuning {model_name} with Optuna...")
    if category == "anomaly":
        best_params = tune_model(
            model_name=model_name,
            X_train=X_train, y_train=None,
            X_val=X_val, y_val=y_val,
            config=config,
        )
    else:
        best_params = tune_model(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            config=config,
        )
    return best_params


def run(mode: str = "all", do_explain: bool = True,
        do_tune: bool = False, do_track: bool = False):
    """Execute the ML pipeline."""
    t0 = time.time()
    config = _load_config()

    # Enable tracking if --track flag was passed
    if do_track:
        config.setdefault("mlflow", {})["enabled"] = True

    tracker = ExperimentTracker(config)

    # ── 1. Load data ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)

    data_root = PROJECT_ROOT / "ml" / "data" / "raw" / "simulated"
    healthy_dir = data_root / "healthy"
    anomaly_dir = data_root / "anomaly"

    healthy_sessions = load_sessions(healthy_dir)
    anomaly_sessions = load_sessions(anomaly_dir)
    all_sessions = healthy_sessions + anomaly_sessions

    logger.info(
        f"Loaded {len(healthy_sessions)} healthy + "
        f"{len(anomaly_sessions)} anomaly sessions"
    )

    # ── 2. Window ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Windowing sessions")
    logger.info("=" * 60)

    w_cfg = config.get("windowing", {})
    windows = windows_from_sessions(
        all_sessions,
        window_s=w_cfg.get("window_s", 30.0),
        step_s=w_cfg.get("step_s", 5.0),
        hz=w_cfg.get("hz", 2.0),
    )
    logger.info(f"Created {len(windows)} windows")

    # ── 3. Feature engineering ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting features")
    logger.info("=" * 60)

    feature_df, labels = extract_features(windows, config=config)
    logger.info(f"Feature matrix: {feature_df.shape}")

    # ── 4. Train/test split ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Train/test split")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = _split_train_test(
        feature_df, labels, config,
    )

    # ── 5. Train & evaluate models ──────────────────────────────────
    if mode in ("all", "anomaly"):
        anomaly_algo = config.get("anomaly_model", {}).get("algorithm", "IsolationForest")

        with tracker.start_run(run_name=f"anomaly_{anomaly_algo}"):
            logger.info("=" * 60)
            logger.info(f"STEP 5a: Training {anomaly_algo}")
            logger.info("=" * 60)

            # Train on healthy-only subset from training set
            healthy_mask = y_train == 0
            X_train_healthy = X_train[healthy_mask]
            logger.info(f"Training anomaly detector on {len(X_train_healthy)} healthy windows")

            # Optional tuning
            tuned_params = None
            if do_tune:
                from sklearn.preprocessing import StandardScaler
                scaler_tmp = StandardScaler()
                X_h_scaled = scaler_tmp.fit_transform(X_train_healthy)
                X_t_scaled = scaler_tmp.transform(X_test)
                tuned_params = _maybe_tune(
                    anomaly_algo, "anomaly",
                    X_h_scaled, None, X_t_scaled, y_test, config,
                )
                tracker.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})

            model_if, scaler_if, feat_names_if = train_anomaly_detector(
                X_train_healthy, config=config, params_override=tuned_params,
            )
            save_anomaly_artifacts(model_if, scaler_if, feat_names_if, config=config)

            # Log model params
            tracker.log_params({
                "anomaly_algorithm": anomaly_algo,
                "train_samples": len(X_train_healthy),
                "n_features": len(feat_names_if),
            })

            # Score test set
            scores_if = score_anomalies(model_if, scaler_if, X_test)

            logger.info("=" * 60)
            logger.info(f"STEP 6a: Evaluating {anomaly_algo}")
            logger.info("=" * 60)
            metrics_if = evaluate_anomaly(scores_if, y_test, config=config, label="anomaly")
            tracker.log_metrics(metrics_if)

            if do_explain:
                logger.info("=" * 60)
                logger.info(f"STEP 7a: SHAP explanations ({anomaly_algo})")
                logger.info("=" * 60)
                explain_model(model_if, scaler_if, X_test, feat_names_if,
                              config=config, label="anomaly")

            # Log evaluation artifacts
            eval_dir = PROJECT_ROOT / config.get("output", {}).get(
                "evaluation_dir", "ml/models/simulation/evaluation"
            )
            for artifact in eval_dir.glob("anomaly_*"):
                tracker.log_artifact(artifact)

    if mode in ("all", "classifier"):
        clf_algo = config.get("classifier", {}).get("algorithm", "XGBClassifier")

        with tracker.start_run(run_name=f"classifier_{clf_algo}"):
            logger.info("=" * 60)
            logger.info(f"STEP 5b: Training {clf_algo}")
            logger.info("=" * 60)

            # Optional tuning
            tuned_params = None
            if do_tune:
                from sklearn.preprocessing import StandardScaler
                scaler_tmp = StandardScaler()
                X_tr_scaled = scaler_tmp.fit_transform(X_train)
                X_te_scaled = scaler_tmp.transform(X_test)
                tuned_params = _maybe_tune(
                    clf_algo, "classifier",
                    X_tr_scaled, y_train, X_te_scaled, y_test, config,
                )
                tracker.log_params({f"tuned_{k}": v for k, v in tuned_params.items()})

            clf, scaler_clf, feat_names_clf = train_classifier(
                X_train, y_train, config=config, params_override=tuned_params,
            )
            save_classifier_artifacts(clf, scaler_clf, feat_names_clf, config=config)

            # Log model params
            tracker.log_params({
                "classifier_algorithm": clf_algo,
                "train_samples": len(X_train),
                "n_features": len(feat_names_clf),
                "class_balance": f"{int(y_train.sum())}/{len(y_train)}",
            })

            # Predict on test set
            y_prob_clf = predict_proba(clf, scaler_clf, X_test)

            logger.info("=" * 60)
            logger.info(f"STEP 6b: Evaluating {clf_algo}")
            logger.info("=" * 60)
            metrics_clf = evaluate_classifier(y_prob_clf, y_test, config=config,
                                              label="classifier")
            tracker.log_metrics(metrics_clf)

            if do_explain:
                logger.info("=" * 60)
                logger.info(f"STEP 7b: SHAP explanations ({clf_algo})")
                logger.info("=" * 60)
                explain_model(clf, scaler_clf, X_test, feat_names_clf,
                              config=config, label="classifier")

            # Log evaluation artifacts
            eval_dir = PROJECT_ROOT / config.get("output", {}).get(
                "evaluation_dir", "ml/models/simulation/evaluation"
            )
            for artifact in eval_dir.glob("classifier_*"):
                tracker.log_artifact(artifact)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="BRA Logger ML Pipeline — train, evaluate, explain",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "anomaly", "classifier"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip SHAP explanations (faster)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable Optuna hyperparameter tuning",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable MLflow experiment tracking",
    )
    args = parser.parse_args()

    run(
        mode=args.mode,
        do_explain=not args.no_explain,
        do_tune=args.tune,
        do_track=args.track,
    )


if __name__ == "__main__":
    main()
