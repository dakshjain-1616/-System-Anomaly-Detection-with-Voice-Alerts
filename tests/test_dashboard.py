"""Tests for dashboard.py — data loading, RCLI check, alert trigger (all with mocked Streamlit)."""

import os
import sys
import tempfile
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Mock streamlit and plotly before importing dashboard
_st = MagicMock()
_st.set_page_config = MagicMock()
_st.error = MagicMock()

sys.modules.setdefault("streamlit", _st)
sys.modules.setdefault("plotly", MagicMock())
sys.modules.setdefault("plotly.graph_objects", MagicMock())
sys.modules.setdefault("plotly.subplots", MagicMock())

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import dashboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_sample_log(path, n_normal=5, n_anomaly=2):
    rows = []
    base = datetime.now()
    for i in range(n_normal):
        rows.append({
            "timestamp": (base + timedelta(seconds=i * 10)).isoformat(),
            "cpu_usage": 10 + i,
            "memory_usage": 20 + i,
            "network_io": 200 + i * 10,
            "disk_io": 5 + i,
            "error_count": 0,
            "anomaly_score": 0.05,
            "is_anomaly": 0,
        })
    for j in range(n_anomaly):
        rows.append({
            "timestamp": (base + timedelta(seconds=(n_normal + j) * 10)).isoformat(),
            "cpu_usage": 95,
            "memory_usage": 97,
            "network_io": 8000,
            "disk_io": 500,
            "error_count": 20,
            "anomaly_score": -0.08,
            "is_anomaly": 1,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# load_anomaly_data
# ---------------------------------------------------------------------------

class TestLoadAnomalyData:
    def test_returns_empty_df_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dashboard, "LOG_FILE", str(tmp_path / "nonexistent.csv"))
        df = dashboard.load_anomaly_data()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_returns_empty_df_for_empty_file(self, tmp_path, monkeypatch):
        log = str(tmp_path / "empty.csv")
        open(log, "w").close()
        monkeypatch.setattr(dashboard, "LOG_FILE", log)
        df = dashboard.load_anomaly_data()
        assert df.empty

    def test_loads_rows_correctly(self, tmp_path, monkeypatch):
        log = str(tmp_path / "log.csv")
        _write_sample_log(log, n_normal=5, n_anomaly=2)
        monkeypatch.setattr(dashboard, "LOG_FILE", log)
        df = dashboard.load_anomaly_data()
        assert len(df) == 7

    def test_timestamp_is_datetime(self, tmp_path, monkeypatch):
        log = str(tmp_path / "log.csv")
        _write_sample_log(log)
        monkeypatch.setattr(dashboard, "LOG_FILE", log)
        df = dashboard.load_anomaly_data()
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_sorted_by_timestamp(self, tmp_path, monkeypatch):
        log = str(tmp_path / "log.csv")
        _write_sample_log(log)
        monkeypatch.setattr(dashboard, "LOG_FILE", log)
        df = dashboard.load_anomaly_data()
        assert df["timestamp"].is_monotonic_increasing

    def test_anomaly_column_present(self, tmp_path, monkeypatch):
        log = str(tmp_path / "log.csv")
        _write_sample_log(log)
        monkeypatch.setattr(dashboard, "LOG_FILE", log)
        df = dashboard.load_anomaly_data()
        assert "is_anomaly" in df.columns

    def test_returns_df_even_if_monitor_not_running(self, tmp_path, monkeypatch):
        """Dashboard must work with existing log even when monitor is stopped."""
        log = str(tmp_path / "log.csv")
        _write_sample_log(log, n_normal=3, n_anomaly=1)
        monkeypatch.setattr(dashboard, "LOG_FILE", log)
        df = dashboard.load_anomaly_data()
        assert not df.empty


# ---------------------------------------------------------------------------
# check_rcli_available
# ---------------------------------------------------------------------------

class TestCheckRcliAvailable:
    def test_returns_true_when_rcli_found(self):
        with patch("os.system", return_value=0):
            assert dashboard.check_rcli_available() is True

    def test_returns_false_when_rcli_missing(self):
        with patch("os.system", return_value=1):
            assert dashboard.check_rcli_available() is False

    def test_returns_bool(self):
        with patch("os.system", return_value=0):
            result = dashboard.check_rcli_available()
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# trigger_test_alert
# ---------------------------------------------------------------------------

class TestTriggerTestAlert:
    def test_returns_tuple(self):
        with patch("os.system", return_value=1):
            result = dashboard.trigger_test_alert()
            assert isinstance(result, tuple) and len(result) == 2

    def test_success_when_rcli_available_and_succeeds(self):
        with patch("os.system", return_value=0):
            success, msg = dashboard.trigger_test_alert()
            assert success is True
            assert isinstance(msg, str)

    def test_failure_when_rcli_not_installed(self):
        with patch("os.system", return_value=1):
            # check_rcli_available returns False → rcli not found
            with patch.object(dashboard, "check_rcli_available", return_value=False):
                success, msg = dashboard.trigger_test_alert()
                assert success is False
                assert "not available" in msg.lower() or "rcli" in msg.lower()

    def test_failure_when_rcli_command_fails(self):
        with patch.object(dashboard, "check_rcli_available", return_value=True):
            with patch("os.system", return_value=2):
                success, msg = dashboard.trigger_test_alert()
                assert success is False

    def test_command_uses_rcli_ask_format(self):
        captured = []
        with patch.object(dashboard, "check_rcli_available", return_value=True):
            with patch("os.system", side_effect=lambda c: captured.append(c) or 0):
                dashboard.trigger_test_alert()
        assert len(captured) == 1
        assert captured[0].startswith('rcli ask "')

    def test_no_crash_when_rcli_raises_exception(self):
        with patch.object(dashboard, "check_rcli_available", return_value=True):
            with patch("os.system", side_effect=Exception("boom")):
                success, msg = dashboard.trigger_test_alert()
                assert success is False
                assert isinstance(msg, str)
