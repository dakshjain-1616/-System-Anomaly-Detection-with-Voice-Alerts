"""
System Monitoring Daemon with Anomaly Detection

Polls system metrics every 10 seconds, performs anomaly detection,
and triggers voice alerts when anomalies are detected.
"""

import os
import sys
import time
import joblib
import logging
import psutil
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('monitor.log')
    ]
)
logger = logging.getLogger(__name__)


class AnomalyMonitor:
    """
    System metrics monitor with anomaly detection and voice alerting.
    """
    
    def __init__(self, model_path='model.pkl', log_file='anomaly_log.csv',
                 poll_interval=10, alert_cooldown=60, anomaly_threshold=-0.05):
        """
        Initialize the monitor.
        
        Args:
            model_path: Path to the saved Isolation Forest model
            log_file: Path to the anomaly log CSV file
            poll_interval: Seconds between metric polls
            alert_cooldown: Seconds between voice alerts (spam prevention)
        """
        self.model_path = model_path
        self.log_file = log_file
        self.poll_interval = poll_interval
        self.alert_cooldown = alert_cooldown
        self.anomaly_threshold = anomaly_threshold

        self.model = None
        self.feature_cols = None
        self.last_alert_time = 0
        self.rcli_available = False

        # Track previous counter values to compute deltas
        self._prev_net_io = None
        self._prev_disk_io = None
        self._prev_sample_time = None

        # Initialize
        self._load_model()
        self._check_rcli()
        self._init_log_file()
    
    def _load_model(self):
        """Load the trained Isolation Forest model."""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_cols = model_data['feature_cols']
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Feature columns: {self.feature_cols}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            logger.error("Please run train.py first to generate the model.")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _check_rcli(self):
        """Check if RCLI is available for voice alerts."""
        try:
            result = os.system('which rcli > /dev/null 2>&1')
            if result == 0:
                self.rcli_available = True
                logger.info("RCLI is available for voice alerts")
            else:
                self.rcli_available = False
                logger.warning("RCLI not found - voice alerts will be disabled")
        except Exception as e:
            self.rcli_available = False
            logger.warning(f"Error checking RCLI: {e} - voice alerts disabled")
    
    def _init_log_file(self):
        """Initialize the anomaly log CSV file if it doesn't exist."""
        if not os.path.exists(self.log_file):
            columns = ['timestamp', 'cpu_usage', 'memory_usage', 'network_io', 
                      'disk_io', 'error_count', 'anomaly_score', 'is_anomaly']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.log_file, index=False)
            logger.info(f"Created new anomaly log file: {self.log_file}")
        else:
            logger.info(f"Using existing anomaly log file: {self.log_file}")
    
    def collect_metrics(self):
        """
        Collect current system metrics using psutil.
        
        Returns:
            dict: Dictionary containing current system metrics
        """
        try:
            # CPU usage percentage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage percentage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Network I/O — compute delta KB/s since last sample
            net_io = psutil.net_io_counters()
            disk_io_counters = psutil.disk_io_counters()
            now = time.time()

            if self._prev_net_io is not None and self._prev_sample_time is not None:
                elapsed = max(now - self._prev_sample_time, 0.001)
                net_delta = (
                    (net_io.bytes_sent - self._prev_net_io.bytes_sent) +
                    (net_io.bytes_recv - self._prev_net_io.bytes_recv)
                )
                network_io = max(net_delta / elapsed / 1024, 0)  # KB/s

                disk_delta = (
                    (disk_io_counters.read_bytes - self._prev_disk_io.read_bytes) +
                    (disk_io_counters.write_bytes - self._prev_disk_io.write_bytes)
                )
                disk_io = max(disk_delta / elapsed / (1024 * 1024), 0)  # MB/s
            else:
                # First sample — no delta available; use 0 to avoid false positives
                network_io = 0.0
                disk_io = 0.0

            self._prev_net_io = net_io
            self._prev_disk_io = disk_io_counters
            self._prev_sample_time = now
            
            # Error count (simulated based on system load anomalies)
            # In a real scenario, this would come from system logs
            error_count = 0
            if cpu_usage > 80 or memory_usage > 90:
                error_count = np.random.randint(1, 5)
            
            metrics = {
                'timestamp': datetime.now(),
                'cpu_usage': round(cpu_usage, 2),
                'memory_usage': round(memory_usage, 2),
                'network_io': round(network_io, 2),
                'disk_io': round(disk_io, 2),
                'error_count': error_count
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return None
    
    def detect_anomaly(self, metrics):
        """
        Perform anomaly detection on the collected metrics.
        
        Args:
            metrics: Dictionary of system metrics
            
        Returns:
            tuple: (is_anomaly, anomaly_score)
        """
        try:
            # Extract features in the correct order
            features = np.array([[metrics[col] for col in self.feature_cols]])
            
            # Get anomaly score (negative = more anomalous, 0 = boundary)
            anomaly_score = self.model.decision_function(features)[0]

            # Use a custom threshold stricter than the model's contamination boundary
            # so borderline normal readings don't trigger false alerts
            is_anomaly = anomaly_score < self.anomaly_threshold
            
            return is_anomaly, anomaly_score
            
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return False, 0.0
    
    def trigger_alert(self, metrics, anomaly_score):
        """
        Trigger a voice alert using RCLI if cooldown period has passed.
        
        Args:
            metrics: Dictionary of system metrics
            anomaly_score: The anomaly score from detection
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_alert_time < self.alert_cooldown:
            logger.debug("Alert cooldown active, skipping voice alert")
            return
        
        # Prepare alert message
        alert_message = (
            f"Anomaly detected! CPU: {metrics['cpu_usage']}%, "
            f"Memory: {metrics['memory_usage']}%, "
            f"Errors: {metrics['error_count']}"
        )
        
        logger.warning(f"ALERT: {alert_message}")
        
        # Trigger voice alert via RCLI if available
        if self.rcli_available:
            try:
                command = f'rcli ask "{alert_message}"'
                result = os.system(command)
                if result == 0:
                    logger.info("Voice alert triggered successfully")
                    self.last_alert_time = current_time
                else:
                    logger.warning(f"RCLI command failed with exit code: {result}")
            except Exception as e:
                logger.error(f"Error triggering voice alert: {e}")
        else:
            logger.warning("RCLI not available - voice alert skipped (logged only)")
    
    def log_result(self, metrics, is_anomaly, anomaly_score):
        """
        Log the monitoring result to the CSV file.
        
        Args:
            metrics: Dictionary of system metrics
            is_anomaly: Boolean indicating if anomaly was detected
            anomaly_score: The anomaly score
        """
        try:
            log_entry = {
                'timestamp': metrics['timestamp'].isoformat(),
                'cpu_usage': metrics['cpu_usage'],
                'memory_usage': metrics['memory_usage'],
                'network_io': metrics['network_io'],
                'disk_io': metrics['disk_io'],
                'error_count': metrics['error_count'],
                'anomaly_score': round(anomaly_score, 6),
                'is_anomaly': int(is_anomaly)
            }
            
            # Append to CSV
            df = pd.DataFrame([log_entry])
            
            # Check if file exists to determine if header is needed
            file_exists = os.path.exists(self.log_file)
            df.to_csv(self.log_file, mode='a', header=not file_exists, index=False)
            
            logger.info(f"Logged result - Anomaly: {is_anomaly}, Score: {anomaly_score:.4f}")
            
        except Exception as e:
            logger.error(f"Error logging result: {e}")
    
    def run_single_check(self):
        """
        Run a single monitoring check cycle.
        
        Returns:
            dict: The monitoring result or None if failed
        """
        # Collect metrics
        metrics = self.collect_metrics()
        if metrics is None:
            logger.error("Failed to collect metrics")
            return None
        
        logger.info(f"Metrics collected - CPU: {metrics['cpu_usage']}%, "
                   f"Memory: {metrics['memory_usage']}%, "
                   f"Errors: {metrics['error_count']}")
        
        # Detect anomaly
        is_anomaly, anomaly_score = self.detect_anomaly(metrics)
        
        # Log result
        self.log_result(metrics, is_anomaly, anomaly_score)
        
        # Trigger alert if anomaly detected
        if is_anomaly:
            self.trigger_alert(metrics, anomaly_score)
        
        return {
            'metrics': metrics,
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score
        }
    
    def run_daemon(self, duration_seconds=None):
        """
        Run the monitoring daemon continuously.
        
        Args:
            duration_seconds: If specified, run for this many seconds then stop.
                            If None, run indefinitely until interrupted.
        """
        logger.info("=" * 50)
        logger.info("Starting Anomaly Detection Monitor Daemon")
        logger.info(f"Poll interval: {self.poll_interval} seconds")
        logger.info(f"Alert cooldown: {self.alert_cooldown} seconds")
        logger.info(f"RCLI available: {self.rcli_available}")
        logger.info("=" * 50)
        
        start_time = time.time()
        check_count = 0
        
        try:
            while True:
                check_count += 1
                logger.info(f"\n--- Check #{check_count} ---")
                
                self.run_single_check()
                
                # Check if duration limit reached
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info(f"Duration limit ({duration_seconds}s) reached. Stopping.")
                    break
                
                # Sleep until next poll
                logger.info(f"Sleeping for {self.poll_interval} seconds...")
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            logger.info("\nMonitor stopped by user (KeyboardInterrupt)")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            raise
        finally:
            logger.info("=" * 50)
            logger.info(f"Monitor daemon stopped. Total checks: {check_count}")
            logger.info("=" * 50)


def main():
    """
    Main entry point for the monitoring daemon.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='System Anomaly Detection Monitor Daemon'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=None,
        help='Run for specified seconds (default: run indefinitely)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=10,
        help='Polling interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--cooldown', '-c',
        type=int,
        default=60,
        help='Alert cooldown in seconds (default: 60)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=-0.05,
        help='Anomaly score threshold (default: -0.05; more negative = stricter)'
    )

    args = parser.parse_args()

    # Create and run monitor
    monitor = AnomalyMonitor(
        poll_interval=args.interval,
        alert_cooldown=args.cooldown,
        anomaly_threshold=args.threshold
    )
    
    monitor.run_daemon(duration_seconds=args.duration)


if __name__ == "__main__":
    main()