"""
Streamlit Dashboard for Anomaly Detection System

Provides real-time visualization of anomaly scores, recent alerts sidebar,
and manual test alert functionality.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="System Anomaly Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
LOG_FILE = 'anomaly_log.csv'
REFRESH_INTERVAL = 5  # seconds


def load_anomaly_data():
    """
    Load anomaly log data from CSV file.
    
    Returns:
        pd.DataFrame: Anomaly log data or empty DataFrame if file not found/empty
    """
    try:
        if not os.path.exists(LOG_FILE):
            return pd.DataFrame()
        
        df = pd.read_csv(LOG_FILE)
        
        if df.empty:
            return df
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
        
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading anomaly log: {e}")
        return pd.DataFrame()


def check_rcli_available():
    """Check if RCLI is available for voice alerts."""
    try:
        result = os.system('which rcli > /dev/null 2>&1')
        return result == 0
    except:
        return False


def trigger_test_alert():
    """Trigger a test voice alert using RCLI."""
    if check_rcli_available():
        try:
            message = "Test alert from Anomaly Detection Dashboard"
            command = f'rcli ask "{message}"'
            result = os.system(command)
            if result == 0:
                return True, "Voice alert triggered successfully!"
            else:
                return False, f"RCLI command failed with exit code: {result}"
        except Exception as e:
            return False, f"Error triggering alert: {e}"
    else:
        return False, "RCLI is not available. Voice alerts are disabled."


def render_header():
    """Render the dashboard header."""
    st.title("🚨 System Anomaly Detection Dashboard")
    st.markdown("---")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "🟢 Active")
    
    with col2:
        rcli_status = "🟢 Available" if check_rcli_available() else "🔴 Not Available"
        st.metric("Voice Alerts", rcli_status)
    
    with col3:
        log_exists = os.path.exists(LOG_FILE)
        log_status = "🟢 Found" if log_exists else "🟡 Not Found"
        st.metric("Log File", log_status)
    
    with col4:
        st.metric("Refresh Rate", f"{REFRESH_INTERVAL}s")
    
    st.markdown("---")


def render_main_chart(df):
    """Render the main anomaly score chart."""
    st.subheader("📊 Anomaly Score Timeline")
    
    if df.empty:
        st.info("📭 No data available yet. The monitor daemon may not be running or hasn't collected any data.")
        st.info("Start the monitor with: `python monitor.py`")
        return
    
    # Create anomaly threshold line
    threshold = 0  # Isolation Forest decision function threshold
    
    # Use plotly for interactive chart
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Anomaly Score Over Time', 'System Metrics'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Anomaly score chart
    colors = ['red' if is_anom else 'green' for is_anom in df['is_anomaly']]
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['anomaly_score'],
            mode='lines+markers',
            name='Anomaly Score',
            marker=dict(color=colors, size=8),
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text="Threshold", row=1, col=1)
    
    # System metrics chart
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['cpu_usage'], 
                  mode='lines', name='CPU %', line=dict(color='orange')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_usage'], 
                  mode='lines', name='Memory %', line=dict(color='purple')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="System Anomaly Detection Dashboard",
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
    fig.update_yaxes(title_text="Usage %", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_records = len(df)
        st.metric("Total Records", total_records)
    
    with col2:
        anomaly_count = df['is_anomaly'].sum()
        st.metric("Anomalies Detected", anomaly_count)
    
    with col3:
        anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    with col4:
        avg_score = df['anomaly_score'].mean()
        st.metric("Avg Anomaly Score", f"{avg_score:.4f}")


def render_sidebar(df):
    """Render the sidebar with recent alerts and controls."""
    st.sidebar.title("🎛️ Control Panel")
    
    # Test Alert Button
    st.sidebar.subheader("🔔 Test Alert")
    if st.sidebar.button("Trigger Test Alert", type="primary"):
        with st.spinner("Triggering test alert..."):
            success, message = trigger_test_alert()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
    
    st.sidebar.markdown("---")
    
    # Recent Alerts Section
    st.sidebar.subheader("🚨 Recent Alerts")
    
    if df.empty:
        st.sidebar.info("No alerts yet. Start the monitor daemon to begin detection.")
    else:
        # Filter for anomalies only
        anomalies = df[df['is_anomaly'] == 1].copy()
        
        if anomalies.empty:
            st.sidebar.success("✅ No anomalies detected recently")
        else:
            # Show last 10 anomalies
            recent_anomalies = anomalies.tail(10).sort_values('timestamp', ascending=False)
            
            for _, row in recent_anomalies.iterrows():
                with st.sidebar.container():
                    st.markdown(f"**🚨 {row['timestamp'].strftime('%H:%M:%S')}**")
                    st.markdown(f"- CPU: {row['cpu_usage']:.1f}%")
                    st.markdown(f"- Memory: {row['memory_usage']:.1f}%")
                    st.markdown(f"- Score: {row['anomaly_score']:.4f}")
                    st.markdown("---")
    
    # System Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    rcli_status = "✅ Available" if check_rcli_available() else "❌ Not Available"
    st.sidebar.markdown(f"**Voice Alerts:** {rcli_status}")
    
    log_exists = os.path.exists(LOG_FILE)
    log_status = "✅ Found" if log_exists else "❌ Not Found"
    st.sidebar.markdown(f"**Log File:** {log_status}")
    
    if not log_exists:
        st.sidebar.warning("⚠️ Start monitor.py to begin data collection")


def main():
    """Main function to run the Streamlit dashboard."""
    # Render header
    render_header()
    
    # Load data
    df = load_anomaly_data()
    
    # Render main chart
    render_main_chart(df)
    
    # Render sidebar
    render_sidebar(df)
    
    # Auto-refresh
    st.markdown("---")
    st.caption(f"⏱️ Dashboard refreshes every {REFRESH_INTERVAL} seconds")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()