"""
ml/pipelines/tracking.py
===========================
MLflow experiment tracking wrapper.

Provides a clean context-manager interface for tracking experiments,
logging parameters, metrics, artifacts, and models.

Designed to be optional — if MLflow is not installed or tracking is
disabled in config, operations gracefully become no-ops.

Usage
-----
  from ml.pipelines.tracking import ExperimentTracker

  tracker = ExperimentTracker(config)
  with tracker.start_run(run_name="anomaly_v1"):
      tracker.log_params({"n_estimators": 200})
      tracker.log_metrics({"f1": 0.92, "precision": 0.89})
      tracker.log_artifact("outputs/model.joblib")
      tracker.log_model(model, "anomaly_model")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


class ExperimentTracker:
    """
    MLflow tracking wrapper with graceful degradation.

    If MLflow is not installed or tracking is disabled in config,
    all tracking calls become silent no-ops.
    """

    def __init__(self, config: dict):
        tracking_cfg = config.get("mlflow", {})
        self._enabled = tracking_cfg.get("enabled", False)
        self._tracking_uri = tracking_cfg.get("tracking_uri", "mlruns")
        self._experiment_name = tracking_cfg.get("experiment_name", "bra_pipeline")
        self._run = None
        self._mlflow = None

        if self._enabled:
            if not _mlflow_available():
                logger.warning(
                    "MLflow tracking enabled in config but mlflow is not installed. "
                    "Install with: pip install mlflow. Tracking disabled."
                )
                self._enabled = False
            else:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_tracking_uri(self._tracking_uri)
                mlflow.set_experiment(self._experiment_name)
                logger.info(f"MLflow tracking to '{self._tracking_uri}', "
                            f"experiment '{self._experiment_name}'")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def start_run(self, run_name: str | None = None):
        """Context manager to start/end an MLflow run."""
        if not self._enabled:
            yield self
            return

        self._run = self._mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run_name}")
        try:
            yield self
        finally:
            self._mlflow.end_run()
            self._run = None
            logger.info(f"MLflow run ended: {run_name}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of parameters."""
        if not self._enabled:
            return
        # MLflow params must be strings; flatten nested values
        flat = {}
        for k, v in params.items():
            flat[str(k)] = str(v)
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of metrics."""
        if not self._enabled:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str | Path) -> None:
        """Log a file or directory as an artifact."""
        if not self._enabled:
            return
        path = str(path)
        if Path(path).exists():
            self._mlflow.log_artifact(path)
        else:
            logger.warning(f"Artifact path not found: {path}")

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a model — auto-detects sklearn vs pytorch."""
        if not self._enabled:
            return

        model_type = type(model).__module__
        if "torch" in model_type or hasattr(model, "_model"):
            # DL model — log with sklearn flavor as wrapper
            self._mlflow.sklearn.log_model(model, artifact_path)
        else:
            self._mlflow.sklearn.log_model(model, artifact_path)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib figure."""
        if not self._enabled:
            return
        self._mlflow.log_figure(figure, artifact_file)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set tags on the current run."""
        if not self._enabled:
            return
        self._mlflow.set_tags(tags)

    def log_dict(self, data: dict, artifact_file: str) -> None:
        """Log a dictionary as a JSON/YAML artifact."""
        if not self._enabled:
            return
        self._mlflow.log_dict(data, artifact_file)
