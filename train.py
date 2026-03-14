"""
Anomaly Detection Model Training Script

Generates synthetic system metrics data and trains an Isolation Forest model
to detect anomalies in system behavior.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def generate_normal_data(n_samples=5000, random_state=42):
    """
    Generate synthetic normal system metrics data.
    
    Features:
    - cpu_usage: 1-70% (normal range)
    - memory_usage: 5-80% (normal range)
    - network_io: 10-2000 KB/s
    - disk_io: 0.1-150 MB/s
    - error_count: 0-2 errors
    """
    np.random.seed(random_state)
    
    data = {
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_samples)],
        'cpu_usage': np.random.uniform(1, 70, n_samples),
        'memory_usage': np.random.uniform(5, 80, n_samples),
        'network_io': np.random.uniform(10, 2000, n_samples),
        'disk_io': np.random.uniform(0.1, 150, n_samples),
        'error_count': np.random.randint(0, 3, n_samples)
    }
    
    return pd.DataFrame(data)


def generate_anomalous_data(n_samples=200, random_state=42):
    """
    Generate synthetic anomalous system metrics data.
    
    Anomalies include:
    - Very high CPU usage (>90%)
    - Very high memory usage (>95%)
    - Extreme network activity (>5000 KB/s or <10 KB/s)
    - High disk I/O spikes (>500 MB/s)
    - High error counts (>10 errors)
    """
    np.random.seed(random_state + 1)
    
    data = {
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_samples)],
        'cpu_usage': np.random.uniform(90, 100, n_samples),
        'memory_usage': np.random.uniform(95, 100, n_samples),
        'network_io': np.concatenate([
            np.random.uniform(5000, 10000, n_samples // 2),
            np.random.uniform(0, 10, n_samples // 2)
        ]),
        'disk_io': np.random.uniform(500, 1000, n_samples),
        'error_count': np.random.randint(10, 50, n_samples)
    }
    
    return pd.DataFrame(data)


def prepare_data(normal_df, anomalous_df):
    """
    Combine normal and anomalous data, create labels, and prepare for training.
    
    Labels: 1 = normal (inlier), -1 = anomaly (outlier)
    """
    # Add labels
    normal_df = normal_df.copy()
    anomalous_df = anomalous_df.copy()
    normal_df['label'] = 1  # Normal
    anomalous_df['label'] = -1  # Anomaly
    
    # Combine datasets
    combined_df = pd.concat([normal_df, anomalous_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select feature columns (exclude timestamp and label)
    feature_cols = ['cpu_usage', 'memory_usage', 'network_io', 'disk_io', 'error_count']
    
    X = combined_df[feature_cols].values
    y = combined_df['label'].values
    
    return X, y, feature_cols, combined_df


def train_model(X_train, y_train, contamination=0.04):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Parameters:
    - contamination: Expected proportion of anomalies in the data
    """
    logger.info(f"Training Isolation Forest with contamination={contamination}")
    
    # Isolation Forest returns: 1 for inliers, -1 for outliers
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        max_samples='auto'
    )
    
    # Train only on normal data (where y_train == 1)
    # This is the standard approach for anomaly detection
    normal_mask = y_train == 1
    X_train_normal = X_train[normal_mask]
    
    logger.info(f"Training on {len(X_train_normal)} normal samples")
    model.fit(X_train_normal)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using precision, recall, and F1 score.
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert to binary: 1 = normal, -1 = anomaly
    # For metrics calculation, we'll use the original labels
    
    # Calculate metrics (focus on anomaly detection: -1 class)
    # We need to flip labels for sklearn metrics: 0 = normal, 1 = anomaly
    y_test_binary = (y_test == -1).astype(int)  # 1 if anomaly
    y_pred_binary = (y_pred == -1).astype(int)   # 1 if predicted anomaly
    
    precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("=" * 50)
    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report(y_test_binary, y_pred_binary, 
                                     target_names=['Normal', 'Anomaly'], zero_division=0))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_model(model, feature_cols, filepath='model.pkl'):
    """
    Save the trained model and feature information.
    """
    model_data = {
        'model': model,
        'feature_cols': feature_cols,
        'created_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, filepath)
    logger.info(f"Model saved to {filepath}")


def main():
    """
    Main function to orchestrate data generation, training, and evaluation.
    """
    logger.info("Starting Anomaly Detection Model Training")
    logger.info("=" * 50)
    
    # Step 1: Generate synthetic data
    logger.info("Generating synthetic system metrics data...")
    normal_df = generate_normal_data(n_samples=5000, random_state=42)
    anomalous_df = generate_anomalous_data(n_samples=200, random_state=42)
    
    logger.info(f"Generated {len(normal_df)} normal samples")
    logger.info(f"Generated {len(anomalous_df)} anomalous samples")
    
    # Step 2: Prepare data
    logger.info("Preparing data for training...")
    X, y, feature_cols, combined_df = prepare_data(normal_df, anomalous_df)
    
    # Step 3: Split data
    # Use stratified split to maintain anomaly ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Step 4: Train model
    model = train_model(X_train, y_train, contamination=0.04)
    
    # Step 5: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 6: Save model
    save_model(model, feature_cols, filepath='model.pkl')
    
    logger.info("=" * 50)
    logger.info("Training completed successfully!")
    logger.info(f"Final Metrics - Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return metrics


if __name__ == "__main__":
    main()