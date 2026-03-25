"""
ml/pipelines/dl_models.py
===========================
Deep learning model wrappers for the pipeline.

These wrappers provide a scikit-learn-compatible interface (fit/predict/score)
so they integrate seamlessly with the rest of the pipeline (tuning, evaluation,
tracking, explainability).

Requires: torch (optional dependency — imports are lazy).
"""

from __future__ import annotations

import logging
import json
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _import_torch():
    """Lazy import of torch with helpful error."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install with: pip install torch"
        )


# =========================================================================
# Autoencoder for anomaly detection
# =========================================================================

class AnomalyAutoencoder:
    """
    Autoencoder-based anomaly detector with sklearn-compatible interface.

    Anomalies are detected by high reconstruction error: the model learns to
    reconstruct 'normal' patterns, so abnormal inputs produce larger errors.
    """

    def __init__(
        self,
        hidden_dims: list[int] | str = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        threshold_percentile: float = 95.0,
        **kwargs,
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        if isinstance(hidden_dims, str):
            hidden_dims = json.loads(hidden_dims)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        self._model = None
        self._threshold = None
        self._input_dim = None

    def _build_network(self, input_dim: int):
        torch, nn = _import_torch()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = dim

        # Decoder (reverse)
        decoder_layers = []
        reversed_dims = list(reversed(self.hidden_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.Linear(reversed_dims[i], reversed_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = reversed_dims[i + 1]
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        model = nn.Sequential(
            *encoder_layers,
            *decoder_layers,
        )
        return model

    def fit(self, X, y=None):
        """Train the autoencoder on normal data."""
        torch, nn = _import_torch()

        self._input_dim = X.shape[1]
        self._model = self._build_network(self._input_dim)

        X_tensor = torch.FloatTensor(np.asarray(X))
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                output = self._model(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Autoencoder epoch {epoch + 1}/{self.epochs}, "
                            f"loss: {epoch_loss / len(loader):.6f}")

        # Compute anomaly threshold from training reconstruction errors
        self._model.eval()
        with torch.no_grad():
            recon = self._model(X_tensor)
            errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()
        self._threshold = np.percentile(errors, self.threshold_percentile)
        return self

    def predict(self, X):
        """Return -1 for anomalies, 1 for normal (sklearn convention)."""
        scores = self.score_samples(X)
        return np.where(scores > self._threshold, -1, 1)

    def score_samples(self, X):
        """Return reconstruction error per sample (higher = more anomalous)."""
        torch, _ = _import_torch()
        self._model.eval()
        X_tensor = torch.FloatTensor(np.asarray(X))
        with torch.no_grad():
            recon = self._model(X_tensor)
            errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()
        return errors

    def decision_function(self, X):
        """Negative reconstruction error (sklearn convention: more negative = anomaly)."""
        return -self.score_samples(X)


# =========================================================================
# MLP Classifier
# =========================================================================

class MLPClassifier:
    """
    PyTorch MLP classifier with sklearn-compatible interface.
    """

    def __init__(
        self,
        hidden_dims: list[int] | str = None,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        **kwargs,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        if isinstance(hidden_dims, str):
            hidden_dims = json.loads(hidden_dims)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._model = None
        self._classes = None
        self._input_dim = None

    def _build_network(self, input_dim: int, n_classes: int):
        torch, nn = _import_torch()

        layers = []
        prev_dim = input_dim
        for dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(self.dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        """Train the MLP classifier."""
        torch, nn = _import_torch()
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self._classes = le.classes_
        n_classes = len(self._classes)

        self._input_dim = X.shape[1]
        self._model = self._build_network(self._input_dim, n_classes)

        X_tensor = torch.FloatTensor(np.asarray(X))
        y_tensor = torch.LongTensor(y_enc)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self._model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"MLP epoch {epoch + 1}/{self.epochs}, "
                            f"loss: {epoch_loss / len(loader):.6f}")
        return self

    def predict(self, X):
        """Predict class labels."""
        torch, _ = _import_torch()
        self._model.eval()
        X_tensor = torch.FloatTensor(np.asarray(X))
        with torch.no_grad():
            logits = self._model(X_tensor)
            preds = logits.argmax(dim=1).numpy()
        return self._classes[preds]

    def predict_proba(self, X):
        """Predict class probabilities."""
        torch, nn = _import_torch()
        self._model.eval()
        X_tensor = torch.FloatTensor(np.asarray(X))
        with torch.no_grad():
            logits = self._model(X_tensor)
            probs = nn.functional.softmax(logits, dim=1).numpy()
        return probs

    @property
    def classes_(self):
        return self._classes
