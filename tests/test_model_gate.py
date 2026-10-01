"""
test_model_gate.py — Quality-gate tests for the champion model.

Three mandatory checks (assignment rubric):
  1. Validation Macro-F1 >= 0.88
  2. Batch inference latency <= 30 ms
  3. Predictions contain only classes {0, 1, 2}
"""

import time
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score

from src.data import load_data, validate_data, split_data

REGISTERED_MODEL_NAME = "WineClassifier"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@champion"


_CACHED_MODEL_DATA = None


def _load_champion_and_test_data():
    """Helper: load champion model + test split (with auto-train fallback)."""
    global _CACHED_MODEL_DATA
    if _CACHED_MODEL_DATA is not None:
        return _CACHED_MODEL_DATA

    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception:
        # Auto-train if champion model is not yet registered in MLflow
        from src.train import main as train_main
        train_main()
        model = mlflow.sklearn.load_model(MODEL_URI)

    X, y = load_data()
    validate_data(X, y)
    _, X_test, _, y_test = split_data(X, y)
    _CACHED_MODEL_DATA = (model, X_test, y_test)
    return _CACHED_MODEL_DATA


class TestModelQualityGate:
    """Quality-gate tests executed in CI."""

    def test_f1_threshold(self):
        """Check 1 — Validation Macro F1 >= 0.88."""
        model, X_test, y_test = _load_champion_and_test_data()
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        assert f1 >= 0.88, (
            f"Champion F1 = {f1:.4f}, below threshold 0.88"
        )

    def test_inference_latency(self):
        """Check 2 — Batch inference <= 30 ms."""
        model, X_test, _ = _load_champion_and_test_data()
        start = time.perf_counter()
        model.predict(X_test)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms <= 30, (
            f"Inference took {elapsed_ms:.1f} ms, "
            f"exceeds 30 ms limit"
        )

    def test_output_classes(self):
        """Check 3 — Predictions contain only {0, 1, 2}."""
        model, X_test, _ = _load_champion_and_test_data()
        y_pred = model.predict(X_test)
        unique = set(np.unique(y_pred))
        assert unique.issubset({0, 1, 2}), (
            f"Unexpected prediction classes: {unique}"
        )
