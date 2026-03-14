# System Anomaly Detection with Voice Alerts

> Built by [NEO — Your Autonomous AI Agent](https://heyneo.so)

A fully local, on-device anomaly detection system that monitors CPU, memory, network, and disk I/O in real time, detects anomalies using a trained ML model, and triggers voice alerts via RCLI — no cloud, no Docker, pure Python.

## How it works

```
train.py  ──►  model.pkl  ──►  monitor.py  ──►  anomaly_log.csv  ──►  dashboard.py
(train)        (model)         (daemon)          (log)                 (web UI)
```

- **`train.py`** — generates 5,200 synthetic system log rows, trains an Isolation Forest, prints precision/recall/F1, saves `model.pkl`
- **`monitor.py`** — polls real system metrics every 10 s, scores them against the model, writes every reading to `anomaly_log.csv`, and calls `rcli ask "..."` on anomalies
- **`dashboard.py`** — Streamlit UI that reads `anomaly_log.csv` and auto-refreshes every 5 s; works even when the daemon is stopped
- **`demo.py`** — seeds the log with sample data and runs a 60-second live demo so you can see detection in action immediately

---

## Quick start

### 1. Prerequisites

- Python 3.8+
- RCLI *(optional — needed only for voice alerts)*

```bash
# macOS
brew install rcli

# Linux — check https://github.com/anthropics/rcli for your distro
```

### 2. Clone and install

```bash
git clone https://github.com/your-username/anomaly-detection.git
cd anomaly-detection

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

Expected output:

```
============================================================
  Starting Anomaly Detection Model Training
============================================================
Generating synthetic system metrics data...
Generated 5000 normal samples
Generated 200 anomalous samples

Training Isolation Forest with contamination=0.04
Training on 4000 normal samples

============================================================
  MODEL EVALUATION RESULTS
============================================================
Precision: 0.4598
Recall:    1.0000
F1 Score:  0.6299
============================================================

              precision    recall  f1-score   support
      Normal       1.00      0.95      0.98      1000
     Anomaly       0.46      1.00      0.63        40

Model saved to model.pkl
Training completed successfully!
```

> **Why is precision ~0.46?** Isolation Forest is unsupervised — it doesn't learn from labels. A recall of 1.0 means it catches every real anomaly; the lower precision means some borderline readings also get flagged. The `--threshold` flag lets you tune this trade-off.

---

## Examples

### Example 1 — Run the interactive demo (recommended first step)

The fastest way to see everything working:

```bash
python demo.py
```

This seeds `anomaly_log.csv` with 40 realistic readings (35 normal + 5 injected spikes) and then runs a live 60-second monitor session. Open the dashboard in a second terminal while it's running:

```bash
# terminal 2
streamlit run dashboard.py
```

Demo output:

```
============================================================
  Anomaly Detection — Interactive Demo
============================================================
✓ model.pkl found

============================================================
  Step 1 — Seeding anomaly_log.csv with sample data
============================================================
  Written 40 rows to anomaly_log.csv
  ● Normal readings : 35
  ● Anomaly spikes  : 5

  Tip: open the dashboard now to see the seeded chart.
    streamlit run dashboard.py

Press Enter to start the 60-second live monitor…

============================================================
  Step 2 — Live monitor (60 s, polling every 10 s)
============================================================
  Collecting REAL metrics from your machine every 10 seconds.
  Watch anomaly scores appear — normal readings show in green,
  anomalies in red.  Press Ctrl+C to stop early.

  #    CPU    MEM   NET KB/s  DISK MB/s    SCORE  STATUS
  -----------------------------------------------------------------
  1    8.3%  12.9%      0.0      0.000   -0.0277  ● normal
  2    7.4%  12.9%    122.6      0.000   -0.0183  ● normal
  3    6.2%  12.9%     91.0      0.110   -0.0315  ● normal
  4    6.6%  12.9%    197.1      0.030   -0.0251  ● normal
  5    1.7%  12.9%    122.7      0.060   -0.0510  ● ANOMALY
  6   31.2%  13.1%    118.3      0.050   -0.0023  ● normal

============================================================
  Demo complete
============================================================
  Ran for 66s  |  6 live checks  |  40 total log entries
  Anomalies in log: 5 / 40

  Next steps:
    Dashboard  →  streamlit run dashboard.py
    Daemon     →  python monitor.py  (runs until Ctrl+C)
```

> Scores close to `-0.05` are right on the boundary — adjust `--threshold` to taste (see Example 4 below).

---

### Example 2 — Run the daemon and watch it detect a real anomaly

Start the monitor in one terminal:

```bash
python monitor.py --interval 5 --cooldown 30
```

Normal cycle output (no anomaly):

```
2024-01-15 14:23:10 - INFO - --- Check #1 ---
2024-01-15 14:23:11 - INFO - Metrics collected - CPU: 12.4%, Memory: 18.2%, Errors: 0
2024-01-15 14:23:11 - INFO - Logged result - Anomaly: False, Score: -0.0122
2024-01-15 14:23:11 - INFO - Sleeping for 5 seconds...
```

When an anomaly is detected:

```
2024-01-15 14:24:30 - INFO - --- Check #8 ---
2024-01-15 14:24:31 - INFO - Metrics collected - CPU: 94.7%, Memory: 97.3%, Errors: 22
2024-01-15 14:24:31 - INFO - Logged result - Anomaly: True, Score: -0.0736
2024-01-15 14:24:31 - WARNING - ALERT: Anomaly detected! CPU: 94.7%, Memory: 97.3%, Errors: 22
2024-01-15 14:24:31 - INFO - Voice alert triggered successfully    # if RCLI installed
```

---

### Example 3 — Trigger a test anomaly manually

To see the alert path fire without waiting for your machine to spike, use the dashboard's **"Trigger Test Alert"** button in the sidebar, or run from Python directly:

```python
from monitor import AnomalyMonitor

m = AnomalyMonitor()

# Inject a fake spike reading
spike = {
    "cpu_usage":    95.0,
    "memory_usage": 98.0,
    "network_io":   8500.0,   # KB/s
    "disk_io":      520.0,    # MB/s
    "error_count":  25,
}

is_anomaly, score = m.detect_anomaly(spike)
print(f"Anomaly: {is_anomaly}  Score: {score:.4f}")
# → Anomaly: True  Score: -0.0736
```

---

### Example 4 — Tune the sensitivity

The `--threshold` flag controls how sensitive detection is. The anomaly score ranges roughly from `-0.08` (clear anomaly) to `+0.15` (clearly normal).

```bash
# Default — catches only clear spikes
python monitor.py --threshold -0.05

# Stricter — only flag very extreme readings
python monitor.py --threshold -0.07

# More sensitive — flag anything even slightly unusual
python monitor.py --threshold -0.02
```

Check current score distribution from your log:

```python
import pandas as pd
df = pd.read_csv("anomaly_log.csv")
print(df["anomaly_score"].describe())
print("Anomaly rate:", df["is_anomaly"].mean())
```

---

### Example 5 — Inspect the log

`anomaly_log.csv` is plain CSV — open it in any tool:

```bash
# Last 10 readings
python -c "
import pandas as pd
df = pd.read_csv('anomaly_log.csv')
print(df.tail(10).to_string(index=False))
"
```

```
                 timestamp  cpu_usage  memory_usage  network_io  disk_io  error_count  anomaly_score  is_anomaly
2024-01-15T14:23:10.412     12.4        18.2       139.9     0.014            0       -0.012200           0
2024-01-15T14:23:15.891      9.6        18.1       112.3     0.008            0       -0.011500           0
2024-01-15T14:23:21.043     94.7        97.3      8500.0   520.000           22       -0.073600           1
2024-01-15T14:23:26.198      8.1        18.0       201.4     0.021            0       -0.008900           0
```

---

### Example 6 — Run with a time limit (good for testing)

```bash
# Run for exactly 2 minutes, poll every 5 seconds
python monitor.py --duration 120 --interval 5

# Run for 30 seconds to verify everything is working
python monitor.py --duration 30 --interval 10
```

---

## File structure

```
anomaly-detection/
├── train.py           # Generates data, trains & evaluates model, saves model.pkl
├── monitor.py         # Live monitoring daemon
├── dashboard.py       # Streamlit real-time dashboard
├── demo.py            # Interactive demo — see detection working in 60 seconds
├── requirements.txt   # Python dependencies
├── tests/
│   ├── test_train.py      # 34 tests for train.py
│   ├── test_monitor.py    # 36 tests for monitor.py
│   └── test_dashboard.py  # 16 tests for dashboard.py
└── README.md
```

*`model.pkl` and `anomaly_log.csv` are created at runtime and excluded from version control.*

---

## CLI options

### `monitor.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--interval, -i` | `10` | Seconds between metric polls |
| `--cooldown, -c` | `60` | Min seconds between voice alerts |
| `--duration, -d` | *(forever)* | Stop after N seconds |
| `--threshold, -t` | `-0.05` | Anomaly score cutoff (more negative = stricter) |

```bash
# Poll every 5 s, 30 s cooldown, run for 10 minutes
python monitor.py --interval 5 --cooldown 30 --duration 600
```

---

## Run the tests

```bash
python -m pytest tests/ -v
```

Expected: **85 passed** in ~25 seconds.

---

## Troubleshooting

**`Model file not found`**
Run `python train.py` first.

**`RCLI not found — voice alerts disabled`**
This is a warning, not an error. Everything else works normally. Install RCLI to enable voice alerts.

**Dashboard shows "No data available"**
Either run `python demo.py` to seed sample data, or start `python monitor.py` and wait one poll cycle (10 s).

**False positives / too many alerts**
Make the threshold more negative: `python monitor.py --threshold -0.07`

**Missing real anomalies**
Raise the threshold closer to zero: `python monitor.py --threshold -0.02`

---

## Notes

- All inference is local — no data leaves your machine
- The model is trained on synthetic data; no sensitive information is used
- Voice alert command: `rcli ask "Anomaly detected — CPU at X%, memory at Y%, ..."`
- RCLI is gracefully skipped if not installed — the daemon and dashboard work without it

---

*Built by [NEO — Your Autonomous AI Agent](https://heyneo.so)*
