"""Tests for train.py — data generation, model training, evaluation, and saving."""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train import (
    generate_normal_data,
    generate_anomalous_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

class TestGenerateNormalData:
    def test_row_count(self):
        df = generate_normal_data(n_samples=100)
        assert len(df) == 100

    def test_columns_present(self):
        df = generate_normal_data(n_samples=50)
        for col in ["timestamp", "cpu_usage", "memory_usage", "network_io", "disk_io", "error_count"]:
            assert col in df.columns

    def test_cpu_in_range(self):
        df = generate_normal_data(n_samples=500)
        assert df["cpu_usage"].between(0, 100).all(), "CPU values out of 0-100 bounds"

    def test_memory_in_range(self):
        df = generate_normal_data(n_samples=500)
        assert df["memory_usage"].between(0, 100).all()

    def test_network_io_positive(self):
        df = generate_normal_data(n_samples=500)
        assert (df["network_io"] >= 0).all()

    def test_disk_io_positive(self):
        df = generate_normal_data(n_samples=500)
        assert (df["disk_io"] >= 0).all()

    def test_error_count_non_negative(self):
        df = generate_normal_data(n_samples=200)
        assert (df["error_count"] >= 0).all()

    def test_reproducible_with_same_seed(self):
        """Numeric columns must be identical; timestamp uses datetime.now() so is excluded."""
        df1 = generate_normal_data(n_samples=50, random_state=0)
        df2 = generate_normal_data(n_samples=50, random_state=0)
        numeric_cols = ["cpu_usage", "memory_usage", "network_io", "disk_io", "error_count"]
        pd.testing.assert_frame_equal(df1[numeric_cols], df2[numeric_cols])

    def test_different_with_different_seeds(self):
        df1 = generate_normal_data(n_samples=50, random_state=0)
        df2 = generate_normal_data(n_samples=50, random_state=99)
        assert not df1["cpu_usage"].equals(df2["cpu_usage"])


class TestGenerateAnomalousData:
    def test_row_count(self):
        df = generate_anomalous_data(n_samples=50)
        assert len(df) == 50

    def test_cpu_high(self):
        df = generate_anomalous_data(n_samples=200)
        assert (df["cpu_usage"] >= 90).all(), "Anomalous CPU should be ≥ 90%"

    def test_memory_high(self):
        df = generate_anomalous_data(n_samples=200)
        assert (df["memory_usage"] >= 95).all(), "Anomalous memory should be ≥ 95%"

    def test_error_count_high(self):
        df = generate_anomalous_data(n_samples=200)
        assert (df["error_count"] >= 10).all(), "Anomalous error_count should be ≥ 10"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

class TestPrepareData:
    def setup_method(self):
        self.normal_df = generate_normal_data(n_samples=100, random_state=1)
        self.anomaly_df = generate_anomalous_data(n_samples=20, random_state=1)

    def test_total_rows(self):
        X, y, _, _ = prepare_data(self.normal_df, self.anomaly_df)
        assert len(X) == 120

    def test_labels_binary(self):
        _, y, _, _ = prepare_data(self.normal_df, self.anomaly_df)
        assert set(y).issubset({1, -1})

    def test_label_counts(self):
        _, y, _, _ = prepare_data(self.normal_df, self.anomaly_df)
        assert (y == 1).sum() == 100
        assert (y == -1).sum() == 20

    def test_feature_cols_correct(self):
        _, _, feature_cols, _ = prepare_data(self.normal_df, self.anomaly_df)
        assert feature_cols == ["cpu_usage", "memory_usage", "network_io", "disk_io", "error_count"]

    def test_no_timestamp_in_features(self):
        X, _, feature_cols, _ = prepare_data(self.normal_df, self.anomaly_df)
        assert "timestamp" not in feature_cols
        assert X.shape[1] == 5

    def test_no_nan_in_features(self):
        X, _, _, _ = prepare_data(self.normal_df, self.anomaly_df)
        assert not np.isnan(X).any()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

class TestTrainModel:
    def setup_method(self):
        normal_df = generate_normal_data(n_samples=200, random_state=42)
        anomaly_df = generate_anomalous_data(n_samples=20, random_state=42)
        X, y, _, _ = prepare_data(normal_df, anomaly_df)
        self.X = X
        self.y = y

    def test_model_trains_without_error(self):
        model = train_model(self.X, self.y)
        assert model is not None

    def test_model_predict_returns_1_or_minus1(self):
        model = train_model(self.X, self.y)
        preds = model.predict(self.X)
        assert set(preds).issubset({1, -1})

    def test_model_decision_function_shape(self):
        model = train_model(self.X, self.y)
        scores = model.decision_function(self.X)
        assert scores.shape == (len(self.X),)

    def test_anomalies_score_lower_than_normals(self):
        """Anomaly rows should on average score lower than normal rows."""
        model = train_model(self.X, self.y)
        scores = model.decision_function(self.X)
        normal_mean = scores[self.y == 1].mean()
        anomaly_mean = scores[self.y == -1].mean()
        assert anomaly_mean < normal_mean, (
            f"Anomaly mean score {anomaly_mean:.4f} should be < normal mean {normal_mean:.4f}"
        )

    def test_extreme_spike_detected(self):
        """A maxed-out spike should be scored below threshold -0.05."""
        model = train_model(self.X, self.y)
        spike = np.array([[95, 98, 8000, 600, 25]])
        score = model.decision_function(spike)[0]
        assert score < -0.05, f"Extreme spike should score below -0.05, got {score:.4f}"

    def test_normal_reading_not_anomalous(self):
        """A typical idle reading should score above threshold -0.05."""
        model = train_model(self.X, self.y)
        normal = np.array([[15, 20, 300, 5, 0]])
        score = model.decision_function(normal)[0]
        assert score > -0.05, f"Normal reading scored {score:.4f}, expected > -0.05"


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def setup_method(self):
        normal_df = generate_normal_data(n_samples=500, random_state=42)
        anomaly_df = generate_anomalous_data(n_samples=50, random_state=42)
        X, y, _, _ = prepare_data(normal_df, anomaly_df)
        self.model = train_model(X, y)
        self.X_test = X
        self.y_test = y

    def test_returns_dict_with_required_keys(self):
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

    def test_recall_is_high(self):
        """Recall should be ≥ 0.8 on the training distribution."""
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        assert metrics["recall"] >= 0.8, f"Recall too low: {metrics['recall']:.4f}"

    def test_metrics_in_valid_range(self):
        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        for key, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"


# ---------------------------------------------------------------------------
# Model saving / loading
# ---------------------------------------------------------------------------

class TestSaveModel:
    def setup_method(self):
        normal_df = generate_normal_data(n_samples=100, random_state=42)
        anomaly_df = generate_anomalous_data(n_samples=10, random_state=42)
        X, y, feature_cols, _ = prepare_data(normal_df, anomaly_df)
        self.model = train_model(X, y)
        self.feature_cols = feature_cols

    def test_file_created(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_model(self.model, self.feature_cols, filepath=path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_file_loadable_with_joblib(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_model(self.model, self.feature_cols, filepath=path)
            data = joblib.load(path)
            assert "model" in data
            assert "feature_cols" in data
            assert "created_at" in data
        finally:
            os.unlink(path)

    def test_loaded_model_predicts(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_model(self.model, self.feature_cols, filepath=path)
            data = joblib.load(path)
            pred = data["model"].predict([[15, 20, 300, 5, 0]])
            assert pred[0] in (1, -1)
        finally:
            os.unlink(path)

    def test_does_not_use_pickle_module(self):
        """Ensure save_model uses joblib, not pickle."""
        import inspect
        import train as train_module
        source = inspect.getsource(train_module.save_model)
        assert "pickle" not in source, "save_model should use joblib, not pickle"
