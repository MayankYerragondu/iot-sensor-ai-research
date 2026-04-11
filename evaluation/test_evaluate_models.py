import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from sklearn.base import BaseEstimator, TransformerMixin
from evaluation import evaluate_models

class DummyScaler(TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

class DummyModel(BaseEstimator):
    def __init__(self, preds):
        self._preds = preds
    def predict(self, X):
        return self._preds

@pytest.mark.parametrize(
    "y_true, model_preds, expected",
    [
        # All correct, no anomalies
        (np.array([0, 0, 0]), np.array([1, 1, 1]), {"precision": 0, "recall": 0, "f1": 0, "false_alarm_rate": 0}),
        # All anomalies detected
        (np.array([1, 1, 1]), np.array([-1, -1, -1]), {"precision": 1, "recall": 1, "f1": 1, "false_alarm_rate": 0}),
        # Mixed
        (np.array([0, 1, 0, 1]), np.array([1, -1, 1, 1]), None),  # We'll check values
        # False alarms
        (np.array([0, 0, 1, 1]), np.array([-1, -1, -1, 1]), None),
    ]
)
def test_evaluate_model(y_true, model_preds, expected):
    X_test = np.zeros((len(y_true), 2))
    scaler = DummyScaler()
    model = DummyModel(model_preds)
    metrics = evaluate_models.evaluate_model(model, scaler, X_test, y_true)
    assert set(metrics.keys()) == {"precision", "recall", "f1", "false_alarm_rate"}
    if expected is not None:
        for k, v in expected.items():
            assert np.isclose(metrics[k], v, atol=1e-5)
    else:
        # For mixed/false alarm, check value ranges
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["false_alarm_rate"] <= 1

def test_evaluate_model_handles_zero_division():
    # No positive predictions, but y_true has positives
    y_true = np.array([1, 1, 1])
    model_preds = np.array([1, 1, 1])  # All normal
    X_test = np.zeros((3, 2))
    scaler = DummyScaler()
    model = DummyModel(model_preds)
    metrics = evaluate_models.evaluate_model(model, scaler, X_test, y_true)
    assert metrics["precision"] == 0
    assert metrics["recall"] == 0
    assert metrics["f1"] == 0

def test_main_runs(monkeypatch, tmp_path):
    # Patch S3, joblib, pd.read_csv, and OUTPUT_DIR
    test_df = pd.DataFrame({
        "hour": [1, 2],
        "minute": [30, 45],
        "day_of_week": [0, 1],
        "time_diff": [10, 20],
        "label": [0, 1]
    })
    monkeypatch.setattr(evaluate_models.pd, "read_csv", lambda _: test_df)
    monkeypatch.setattr(evaluate_models, "BUCKET_NAME", "dummy-bucket")
    monkeypatch.setattr(evaluate_models, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(evaluate_models.os, "makedirs", lambda *a, **k: None)
    # Dummy S3 client
    dummy_s3 = MagicMock()
    monkeypatch.setattr(evaluate_models, "s3", dummy_s3)
    # Dummy model/scaler
    class DummyScaler2:
        def transform(self, X): return X
    class DummyModel2:
        def predict(self, X): return np.array([1]*len(X))
    def dummy_load(path):
        if "scaler" in path:
            return DummyScaler2()
        else:
            return DummyModel2()
    monkeypatch.setattr(evaluate_models.joblib, "load", dummy_load)
    # Run main
    evaluate_models.main()
    out_path = tmp_path / "pir_metrics.csv"
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert set(df.columns) == {"precision", "recall", "f1", "false_alarm_rate", "model"}
    assert len(df) == 3  # 3 models
