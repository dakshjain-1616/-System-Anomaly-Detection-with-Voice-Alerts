"""
Demo script — see the anomaly detector working in under 60 seconds.

What it does:
  1. Seeds anomaly_log.csv with 40 realistic readings (normal + anomaly spikes)
     so the dashboard has data to display immediately.
  2. Runs the live monitor for 60 seconds (6 real checks) so you can watch
     your own machine being scored in real time.

Usage:
    python demo.py
    # then in another terminal: streamlit run dashboard.py
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

LOG_FILE = "anomaly_log.csv"
MODEL_FILE = "model.pkl"

HEADER = ["timestamp", "cpu_usage", "memory_usage", "network_io",
          "disk_io", "error_count", "anomaly_score", "is_anomaly"]

# ── Colours for terminal output ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner(text):
    width = 60
    print(f"\n{BOLD}{'=' * width}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'=' * width}{RESET}")


def check_model():
    if not os.path.exists(MODEL_FILE):
        print(f"{RED}✗ model.pkl not found.{RESET}")
        print(f"  Run {BOLD}python train.py{RESET} first, then try again.")
        sys.exit(1)
    print(f"{GREEN}✓ model.pkl found{RESET}")


def seed_log():
    """Write 40 synthetic log entries (35 normal + 5 anomaly spikes) into the log."""
    banner("Step 1 — Seeding anomaly_log.csv with sample data")

    model_data = joblib.load(MODEL_FILE)
    model = model_data["model"]

    rng = np.random.default_rng(seed=7)
    rows = []
    base_time = datetime.now() - timedelta(minutes=40)

    # 35 normal readings
    for i in range(35):
        ts = base_time + timedelta(minutes=i)
        cpu  = float(rng.uniform(5, 55))
        mem  = float(rng.uniform(10, 65))
        net  = float(rng.uniform(50, 1500))
        disk = float(rng.uniform(0.5, 80))
        err  = int(rng.integers(0, 3))
        score = float(model.decision_function([[cpu, mem, net, disk, err]])[0])
        is_anom = int(score < -0.05)
        rows.append([ts.isoformat(), round(cpu,1), round(mem,1),
                     round(net,1), round(disk,2), err, round(score,6), is_anom])

    # 5 anomaly spikes at minutes 36-40
    spikes = [
        (92, 97, 8200, 520, 22),
        (88, 99, 9100, 610, 31),
        (95, 96, 7800, 490, 28),
        (91, 98, 8500, 580, 35),
        (94, 97, 9500, 640, 42),
    ]
    for j, (cpu, mem, net, disk, err) in enumerate(spikes):
        ts = base_time + timedelta(minutes=35 + j)
        score = float(model.decision_function([[cpu, mem, net, disk, err]])[0])
        is_anom = int(score < -0.05)
        rows.append([ts.isoformat(), cpu, mem, net, disk, err, round(score,6), is_anom])

    df = pd.DataFrame(rows, columns=HEADER)
    df.to_csv(LOG_FILE, index=False)

    normal_count  = df["is_anomaly"].eq(0).sum()
    anomaly_count = df["is_anomaly"].eq(1).sum()
    print(f"  Written {len(df)} rows to {LOG_FILE}")
    print(f"  {GREEN}● Normal readings : {normal_count}{RESET}")
    print(f"  {RED}● Anomaly spikes  : {anomaly_count}{RESET}")
    print(f"\n  {YELLOW}Tip: open the dashboard now to see the seeded chart.{RESET}")
    print(f"  {BOLD}  streamlit run dashboard.py{RESET}")


def run_live_monitor():
    """Run the live monitor for 60 seconds and print results in real time."""
    banner("Step 2 — Live monitor (60 s, polling every 10 s)")

    # Import here so train.py output doesn't clutter demo startup
    from monitor import AnomalyMonitor

    print(f"  Collecting REAL metrics from your machine every 10 seconds.")
    print(f"  Watch anomaly scores appear — normal readings show in green,")
    print(f"  anomalies in red.  Press {BOLD}Ctrl+C{RESET} to stop early.\n")

    monitor = AnomalyMonitor(
        log_file=LOG_FILE,
        poll_interval=10,
        alert_cooldown=30,
        anomaly_threshold=-0.05,
    )

    # Suppress the monitor's own logger so we control the output
    import logging
    logging.getLogger().setLevel(logging.ERROR)

    start = time.time()
    check = 0

    print(f"  {'#':<4} {'CPU':>6} {'MEM':>6} {'NET KB/s':>10} {'DISK MB/s':>10} {'SCORE':>8}  STATUS")
    print(f"  {'-'*65}")

    try:
        while time.time() - start < 60:
            check += 1

            metrics = monitor.collect_metrics()
            if metrics is None:
                time.sleep(10)
                continue

            is_anom, score = monitor.detect_anomaly(metrics)
            monitor.log_result(metrics, is_anom, score)

            if is_anom:
                monitor.trigger_alert(metrics, score)
                status = f"{RED}● ANOMALY{RESET}"
            else:
                status = f"{GREEN}● normal {RESET}"

            cpu  = metrics["cpu_usage"]
            mem  = metrics["memory_usage"]
            net  = metrics["network_io"]
            disk = metrics["disk_io"]

            print(
                f"  {check:<4} {cpu:>5.1f}% {mem:>5.1f}%"
                f" {net:>10.1f} {disk:>10.3f}"
                f" {score:>+8.4f}  {status}"
            )

            if time.time() - start < 60:
                time.sleep(10)

    except KeyboardInterrupt:
        print(f"\n  {YELLOW}Stopped by user.{RESET}")

    elapsed = int(time.time() - start)
    df = pd.read_csv(LOG_FILE)
    total   = len(df)
    anomalies = df["is_anomaly"].sum()

    banner("Demo complete")
    print(f"  Ran for {elapsed}s  |  {check} live checks  |  {total} total log entries")
    print(f"  Anomalies in log: {RED}{anomalies}{RESET} / {total}")
    print()
    print(f"  {BOLD}Next steps:{RESET}")
    print(f"    Dashboard  →  {BOLD}streamlit run dashboard.py{RESET}")
    print(f"    Daemon     →  {BOLD}python monitor.py{RESET}  (runs until Ctrl+C)")
    print()


if __name__ == "__main__":
    banner("Anomaly Detection — Interactive Demo")
    check_model()
    seed_log()
    print()
    input(f"  Press {BOLD}Enter{RESET} to start the 60-second live monitor… "
          f"(or Ctrl+C to skip)\n")
    run_live_monitor()
