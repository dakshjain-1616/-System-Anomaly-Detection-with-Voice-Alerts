"""Tests for monitor.py — metrics collection, anomaly detection, alerting, logging, cooldown."""

import os
import sys
import time
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ensure model.pkl exists before importing monitor
import train as train_module
from train import generate_normal_data, generate_anomalous_data, prepare_data, train_model, save_model

_TMP_MODEL = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False).name

def _build_model():
    normal_df = generate_normal_data(n_samples=300, random_state=42)
    anomaly_df = generate_anomalous_data(n_samples=30, random_state=42)
    X, y, feature_cols, _ = prepare_data(normal_df, anomaly_df)
    model = train_model(X, y)
    save_model(model, feature_cols, filepath=_TMP_MODEL)

_build_model()

from monitor import AnomalyMonitor


def _make_monitor(tmp_path, **kwargs) -> AnomalyMonitor:
    log = str(tmp_path / "log.csv")
    return AnomalyMonitor(model_path=_TMP_MODEL, log_file=log, **kwargs)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_loads_model(self, tmp_path):
        m = _make_monitor(tmp_path)
        assert m.model is not None

    def test_feature_cols_loaded(self, tmp_path):
        m = _make_monitor(tmp_path)
        assert m.feature_cols == ["cpu_usage", "memory_usage", "network_io", "disk_io", "error_count"]

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AnomalyMonitor(model_path="/nonexistent/model.pkl",
                           log_file=str(tmp_path / "log.csv"))

    def test_log_file_created_if_missing(self, tmp_path):
        log = str(tmp_path / "new_log.csv")
        assert not os.path.exists(log)
        AnomalyMonitor(model_path=_TMP_MODEL, log_file=log)
        assert os.path.exists(log)

    def test_log_file_has_correct_headers(self, tmp_path):
        log = str(tmp_path / "header_test.csv")
        AnomalyMonitor(model_path=_TMP_MODEL, log_file=log)
        df = pd.read_csv(log)
        expected = {"timestamp", "cpu_usage", "memory_usage", "network_io",
                    "disk_io", "error_count", "anomaly_score", "is_anomaly"}
        assert expected == set(df.columns)

    def test_rcli_unavailable_does_not_crash(self, tmp_path):
        m = _make_monitor(tmp_path)
        assert isinstance(m.rcli_available, bool)

    def test_uses_joblib_not_pickle(self):
        import inspect
        source = inspect.getsource(AnomalyMonitor._load_model)
        assert "pickle" not in source, "_load_model should use joblib"


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

class TestCollectMetrics:
    def test_returns_dict(self, tmp_path):
        m = _make_monitor(tmp_path)
        metrics = m.collect_metrics()
        assert isinstance(metrics, dict)

    def test_required_keys(self, tmp_path):
        m = _make_monitor(tmp_path)
        metrics = m.collect_metrics()
        for key in ["timestamp", "cpu_usage", "memory_usage", "network_io", "disk_io", "error_count"]:
            assert key in metrics, f"Missing key: {key}"

    def test_cpu_in_valid_range(self, tmp_path):
        m = _make_monitor(tmp_path)
        metrics = m.collect_metrics()
        assert 0 <= metrics["cpu_usage"] <= 100

    def test_memory_in_valid_range(self, tmp_path):
        m = _make_monitor(tmp_path)
        metrics = m.collect_metrics()
        assert 0 <= metrics["memory_usage"] <= 100

    def test_first_sample_io_is_zero(self, tmp_path):
        """First sample should return 0 for I/O (no previous counter to diff)."""
        m = _make_monitor(tmp_path)
        metrics = m.collect_metrics()
        assert metrics["network_io"] == 0.0
        assert metrics["disk_io"] == 0.0

    def test_second_sample_io_non_negative(self, tmp_path):
        m = _make_monitor(tmp_path)
        m.collect_metrics()          # seed previous counters
        time.sleep(1)
        metrics = m.collect_metrics()
        assert metrics["network_io"] >= 0
        assert metrics["disk_io"] >= 0

    def test_timestamp_is_datetime(self, tmp_path):
        m = _make_monitor(tmp_path)
        metrics = m.collect_metrics()
        assert isinstance(metrics["timestamp"], datetime)

    def test_error_count_non_negative(self, tmp_path):
        m = _make_monitor(tmp_path)
        metrics = m.collect_metrics()
        assert metrics["error_count"] >= 0


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

class TestDetectAnomaly:
    def setup_method(self, tmp_path=None):
        self.tmp = tempfile.mkdtemp()
        self.m = AnomalyMonitor(model_path=_TMP_MODEL,
                                log_file=os.path.join(self.tmp, "log.csv"),
                                anomaly_threshold=-0.05)

    def test_returns_tuple(self):
        result = self.m.detect_anomaly({"cpu_usage": 15, "memory_usage": 20,
                                        "network_io": 300, "disk_io": 5, "error_count": 0})
        assert isinstance(result, tuple) and len(result) == 2

    def test_normal_reading_not_anomaly(self):
        is_anom, score = self.m.detect_anomaly(
            {"cpu_usage": 15, "memory_usage": 20, "network_io": 300, "disk_io": 5, "error_count": 0}
        )
        assert not is_anom, f"Normal reading wrongly flagged (score={score:.4f})"

    def test_extreme_spike_is_anomaly(self):
        is_anom, score = self.m.detect_anomaly(
            {"cpu_usage": 95, "memory_usage": 98, "network_io": 8000, "disk_io": 600, "error_count": 25}
        )
        assert is_anom, f"Extreme spike not detected (score={score:.4f})"

    def test_score_is_float(self):
        _, score = self.m.detect_anomaly(
            {"cpu_usage": 15, "memory_usage": 20, "network_io": 300, "disk_io": 5, "error_count": 0}
        )
        assert isinstance(score, float)

    def test_anomaly_score_lower_for_spike(self):
        _, normal_score = self.m.detect_anomaly(
            {"cpu_usage": 15, "memory_usage": 20, "network_io": 300, "disk_io": 5, "error_count": 0}
        )
        _, spike_score = self.m.detect_anomaly(
            {"cpu_usage": 95, "memory_usage": 98, "network_io": 8000, "disk_io": 600, "error_count": 25}
        )
        assert spike_score < normal_score

    def test_custom_threshold_respected(self):
        """With threshold=0 (permissive), borderline readings should be flagged by predict."""
        m_strict = AnomalyMonitor(model_path=_TMP_MODEL,
                                  log_file=os.path.join(self.tmp, "log2.csv"),
                                  anomaly_threshold=0.5)
        # At threshold=0.5 almost everything flags
        is_anom, _ = m_strict.detect_anomaly(
            {"cpu_usage": 15, "memory_usage": 20, "network_io": 300, "disk_io": 5, "error_count": 0}
        )
        assert is_anom

    def test_bad_metrics_returns_false_not_crash(self):
        """If feature extraction fails, should return (False, 0.0) not raise."""
        is_anom, score = self.m.detect_anomaly({})  # missing keys
        assert is_anom is False
        assert score == 0.0


# ---------------------------------------------------------------------------
# Alert triggering
# ---------------------------------------------------------------------------

class TestTriggerAlert:
    def _metrics(self):
        return {"cpu_usage": 95, "memory_usage": 98, "network_io": 9000,
                "disk_io": 600, "error_count": 25, "timestamp": datetime.now()}

    def test_no_crash_without_rcli(self, tmp_path):
        m = _make_monitor(tmp_path, alert_cooldown=0)
        m.rcli_available = False
        m.trigger_alert(self._metrics(), -0.08)  # must not raise

    def test_cooldown_prevents_repeat_alert(self, tmp_path):
        m = _make_monitor(tmp_path, alert_cooldown=60)
        m.last_alert_time = time.time()  # pretend alert just fired
        called = []
        with patch("os.system", side_effect=lambda cmd: called.append(cmd)):
            m.trigger_alert(self._metrics(), -0.08)
        assert len(called) == 0, "Alert fired during cooldown"

    def test_alert_fires_after_cooldown_expires(self, tmp_path):
        m = _make_monitor(tmp_path, alert_cooldown=1)
        m.rcli_available = True
        m.last_alert_time = time.time() - 2  # cooldown already expired
        with patch("os.system", return_value=0) as mock_sys:
            m.trigger_alert(self._metrics(), -0.08)
        mock_sys.assert_called_once()

    def test_rcli_command_format(self, tmp_path):
        """Verify the command uses os.system(f'rcli ask "..."') format."""
        m = _make_monitor(tmp_path, alert_cooldown=0)
        m.rcli_available = True
        m.last_alert_time = 0
        captured = []
        with patch("os.system", side_effect=lambda cmd: captured.append(cmd) or 0):
            m.trigger_alert(self._metrics(), -0.08)
        assert len(captured) == 1
        cmd = captured[0]
        assert cmd.startswith('rcli ask "'), f"Unexpected command format: {cmd}"
        assert cmd.endswith('"'), f"Command should end with quote: {cmd}"

    def test_last_alert_time_updated_on_success(self, tmp_path):
        m = _make_monitor(tmp_path, alert_cooldown=0)
        m.rcli_available = True
        before = time.time()
        with patch("os.system", return_value=0):
            m.trigger_alert(self._metrics(), -0.08)
        assert m.last_alert_time >= before

    def test_last_alert_time_not_updated_on_rcli_failure(self, tmp_path):
        m = _make_monitor(tmp_path, alert_cooldown=0)
        m.rcli_available = True
        m.last_alert_time = 0
        with patch("os.system", return_value=1):  # non-zero = failure
            m.trigger_alert(self._metrics(), -0.08)
        assert m.last_alert_time == 0


# ---------------------------------------------------------------------------
# Log file writing
# ---------------------------------------------------------------------------

class TestLogResult:
    def _metrics(self):
        return {"cpu_usage": 10, "memory_usage": 20, "network_io": 100,
                "disk_io": 1, "error_count": 0, "timestamp": datetime.now()}

    def test_row_written(self, tmp_path):
        m = _make_monitor(tmp_path)
        m.log_result(self._metrics(), False, 0.05)
        df = pd.read_csv(m.log_file)
        assert len(df) == 1

    def test_multiple_rows_appended(self, tmp_path):
        m = _make_monitor(tmp_path)
        for _ in range(5):
            m.log_result(self._metrics(), False, 0.05)
        df = pd.read_csv(m.log_file)
        assert len(df) == 5

    def test_anomaly_flag_stored_correctly(self, tmp_path):
        m = _make_monitor(tmp_path)
        m.log_result(self._metrics(), True, -0.08)
        df = pd.read_csv(m.log_file)
        assert df.iloc[0]["is_anomaly"] == 1

    def test_normal_flag_stored_correctly(self, tmp_path):
        m = _make_monitor(tmp_path)
        m.log_result(self._metrics(), False, 0.05)
        df = pd.read_csv(m.log_file)
        assert df.iloc[0]["is_anomaly"] == 0

    def test_score_stored(self, tmp_path):
        m = _make_monitor(tmp_path)
        m.log_result(self._metrics(), False, 0.12345)
        df = pd.read_csv(m.log_file)
        assert abs(df.iloc[0]["anomaly_score"] - 0.12345) < 0.0001

    def test_all_metric_columns_present(self, tmp_path):
        m = _make_monitor(tmp_path)
        m.log_result(self._metrics(), False, 0.0)
        df = pd.read_csv(m.log_file)
        for col in ["timestamp", "cpu_usage", "memory_usage", "network_io",
                    "disk_io", "error_count", "anomaly_score", "is_anomaly"]:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Single check cycle
# ---------------------------------------------------------------------------

class TestRunSingleCheck:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        m = _make_monitor(tmp_path)
        result = m.run_single_check()
        assert "metrics" in result
        assert "is_anomaly" in result
        assert "anomaly_score" in result

    def test_writes_to_log(self, tmp_path):
        m = _make_monitor(tmp_path)
        m.run_single_check()
        df = pd.read_csv(m.log_file)
        assert len(df) == 1

    def test_no_false_positive_on_idle_system(self, tmp_path):
        """After warm-up (second sample), an idle system should not trigger anomaly."""
        m = _make_monitor(tmp_path, anomaly_threshold=-0.05)
        m.run_single_check()       # warm up counters
        time.sleep(2)
        result = m.run_single_check()
        assert not result["is_anomaly"], (
            f"Idle system flagged as anomaly: score={result['anomaly_score']:.4f}"
        )
