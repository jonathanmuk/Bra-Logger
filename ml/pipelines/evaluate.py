"""
ml/pipelines/evaluate.py
=========================
Evaluation pipeline for both the anomaly detector and the supervised classifier.

Computes standard metrics (accuracy, precision, recall, F1, AUC-ROC, AUC-PR)
and generates diagnostic plots saved to the evaluation output directory.

Usage
-----
  from ml.pipelines.evaluate import evaluate_anomaly, evaluate_classifier
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.utils.visualization import plot_anomaly_scores

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "ml" / "config" / "ml_config.yaml"


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _ensure_eval_dir(config: dict) -> Path:
    eval_dir = PROJECT_ROOT / config.get("output", {}).get(
        "evaluation_dir", "ml/models/simulation/evaluation"
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


# =========================================================================
# Anomaly detector evaluation
# =========================================================================

def evaluate_anomaly(
    scores: np.ndarray,
    y_true: np.ndarray,
    config: dict | None = None,
    label: str = "anomaly",
) -> dict:
    """
    Evaluate anomaly scores against ground-truth labels.

    Parameters
    ----------
    scores : Anomaly scores in [0, 1] (higher = more anomalous).
    y_true : Binary labels (0=healthy, 1=anomaly).
    config : ML config dict.
    label  : Prefix for saved artifact filenames.

    Returns
    -------
    Dict of metrics.
    """
    if config is None:
        config = _load_config()

    eval_dir = _ensure_eval_dir(config)
    threshold = config.get("evaluation", {}).get("score_threshold", 0.5)
    y_pred = (scores >= threshold).astype(int)

    metrics = _compute_metrics(y_true, y_pred, scores)
    metrics["threshold"] = threshold

    # Save metrics JSON
    with open(eval_dir / f"{label}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    _plot_roc(scores, y_true, eval_dir, label, config)
    _plot_precision_recall(scores, y_true, eval_dir, label, config)
    _plot_confusion(y_true, y_pred, eval_dir, label, config)
    _plot_score_distribution(scores, y_true, threshold, eval_dir, label, config)

    logger.info(
        f"[{label}] AUC-ROC={metrics['auc_roc']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}"
    )
    return metrics


# =========================================================================
# Classifier evaluation
# =========================================================================

def evaluate_classifier(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    config: dict | None = None,
    label: str = "classifier",
) -> dict:
    """
    Evaluate a supervised classifier's probability predictions.

    Parameters
    ----------
    y_prob : Predicted probability of class 1 (anomaly).
    y_true : Binary labels.
    """
    if config is None:
        config = _load_config()

    eval_dir = _ensure_eval_dir(config)
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    metrics = _compute_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = threshold

    with open(eval_dir / f"{label}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_roc(y_prob, y_true, eval_dir, label, config)
    _plot_precision_recall(y_prob, y_true, eval_dir, label, config)
    _plot_confusion(y_true, y_pred, eval_dir, label, config)
    _plot_score_distribution(y_prob, y_true, threshold, eval_dir, label, config)

    logger.info(
        f"[{label}] AUC-ROC={metrics['auc_roc']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}"
    )
    return metrics


# =========================================================================
# Shared helpers
# =========================================================================

def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """Compute standard binary classification metrics."""
    metrics: dict = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # AUC requires both classes present
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
        metrics["auc_pr"] = float(auc(rec_arr, prec_arr))
    else:
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0

    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=["healthy", "anomaly"], output_dict=True,
        zero_division=0,
    )

    return metrics


def _plot_roc(
    scores: np.ndarray,
    y_true: np.ndarray,
    eval_dir: Path,
    label: str,
    config: dict,
) -> None:
    if not config.get("evaluation", {}).get("plots", {}).get("roc_curve", True):
        return
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {label}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(eval_dir / f"{label}_roc_curve.png", dpi=150)
    plt.close(fig)


def _plot_precision_recall(
    scores: np.ndarray,
    y_true: np.ndarray,
    eval_dir: Path,
    label: str,
    config: dict,
) -> None:
    if not config.get("evaluation", {}).get("plots", {}).get("precision_recall_curve", True):
        return
    if len(np.unique(y_true)) < 2:
        return

    prec, rec, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="#e74c3c", lw=2, label=f"AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {label}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(eval_dir / f"{label}_pr_curve.png", dpi=150)
    plt.close(fig)


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_dir: Path,
    label: str,
    config: dict,
) -> None:
    if not config.get("evaluation", {}).get("plots", {}).get("confusion_matrix", True):
        return

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    classes = ["healthy", "anomaly"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix — {label}")
    fig.tight_layout()
    fig.savefig(eval_dir / f"{label}_confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    eval_dir: Path,
    label: str,
    config: dict,
) -> None:
    if not config.get("evaluation", {}).get("plots", {}).get("score_distribution", True):
        return

    fig = plot_anomaly_scores(scores, labels=y_true, threshold=threshold,
                              title=f"Score Distribution — {label}")
    fig.savefig(eval_dir / f"{label}_score_distribution.png", dpi=150)
    plt.close(fig)
