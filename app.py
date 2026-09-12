from supabase import create_client
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    CONGESTION_JITTER_MS,
    CONGESTION_LATENCY_MS,
    CONGESTION_PACKET_LOSS_PCT,
    CONGESTION_UTILIZATION_PCT,
    EVAL_FILE,
    LATENCY_CRITICAL_MS,
    LATENCY_WARNING_MS,
    JITTER_CRITICAL_MS,
    JITTER_WARNING_MS,
    METRICS_FILE,
    MODEL_FILE,
    PACKET_LOSS_CRITICAL_PCT,
    PACKET_LOSS_WARNING_PCT,
    ROLLING_WINDOW,
    UTILIZATION_CRITICAL_PCT,
    UTILIZATION_WARNING_PCT,
    DEFAULT_INTERFACE_SPEED_MBPS,
    DEFAULT_PROBE_HOST,
)

from train import build_features


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Network Management System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 12px;
        }

        .status-normal {
            padding: 10px;
            border-radius: 8px;
            background-color: rgba(0, 180, 0, 0.10);
            border: 1px solid rgba(0, 180, 0, 0.35);
        }

        .status-warning {
            padding: 10px;
            border-radius: 8px;
            background-color: rgba(255, 165, 0, 0.10);
            border: 1px solid rgba(255, 165, 0, 0.35);
        }

        .status-critical {
            padding: 10px;
            border-radius: 8px;
            background-color: rgba(255, 0, 0, 0.10);
            border: 1px solid rgba(255, 0, 0, 0.35);
        }

        .live-indicator {
            font-size: 0.9rem;
            font-weight: 600;
        }

        .small-muted {
            color: #777;
            font-size: 0.85rem;
        }

        .demo-banner {
            padding: 12px 16px;
            border-radius: 10px;
            background-color: rgba(255, 193, 7, 0.10);
            border: 1px solid rgba(255, 193, 7, 0.35);
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SUPABASE
# ============================================================

@st.cache_resource
def get_supabase_client():

    try:

        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]

        return create_client(
            url,
            key,
        )

    except Exception as exc:

        st.error(
            f"Supabase connection error: {exc}"
        )

        return None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    model_path = Path(
        MODEL_FILE
    )

    if not model_path.exists():

        return None

    try:

        return joblib.load(
            model_path
        )

    except Exception as exc:

        st.error(
            f"Could not load model file: {exc}"
        )

        return None


# ============================================================
# SUPABASE METRICS
# ============================================================

@st.cache_data(
    ttl=4,
    show_spinner=False,
)
def load_metrics():

    try:

        supabase = get_supabase_client()

        if supabase is None:

            return pd.DataFrame()

        response = (
            supabase
            .table("network_metrics")
            .select("*")
            .order(
                "timestamp",
                desc=True,
            )
            .limit(2000)
            .execute()
        )

        rows = response.data or []

        if not rows:

            return pd.DataFrame()

        df = pd.DataFrame(
            rows
        )

    except Exception as exc:

        st.error(
            f"Could not read network metrics from Supabase: {exc}"
        )

        return pd.DataFrame()

    if df.empty:

        return df

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["timestamp"]
        )

        df = df.sort_values(
            "timestamp"
        )

    numeric_columns = [

        "throughput_mbps",
        "traffic_volume_mb",
        "latency_ms",
        "packet_loss_pct",
        "jitter_ms",
        "bandwidth_utilization_pct",
        "cpu_utilization_pct",
        "memory_utilization_pct",
        "connections",
        "packets_sent",
        "packets_received",
        "packet_rate",
        "rx_mbps",
        "tx_mbps",
        "interface_errors",
        "interface_drops",
        "bandwidth_mbps",
        "network_available",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df.reset_index(
        drop=True
    )


# ============================================================
# DEMO NETWORK DATA
# ============================================================

def generate_demo_network_measurement():

    bandwidth_mbps = round(
        random.uniform(80, 150),
        2,
    )

    utilization = round(
        random.uniform(25, 65),
        2,
    )

    throughput = round(
        random.uniform(15, 100),
        2,
    )

    latency = round(
        random.uniform(20, 90),
        2,
    )

    packet_loss = round(
        random.uniform(0.1, 2.5),
        2,
    )

    jitter = round(
        random.uniform(5, 35),
        2,
    )

    traffic_volume = round(
        random.uniform(50, 500),
        2,
    )

    packet_rate = round(
        random.uniform(100, 1500),
        2,
    )

    rx_mbps = round(
        random.uniform(10, 70),
        2,
    )

    tx_mbps = round(
        random.uniform(5, 50),
        2,
    )

    cpu = round(
        random.uniform(20, 70),
        2,
    )

    memory = round(
        random.uniform(30, 75),
        2,
    )

    connections = random.randint(
        10,
        150,
    )

    packets_sent = random.randint(
        1000,
        10000,
    )

    packets_received = random.randint(
        1000,
        10000,
    )

    interface_errors = random.randint(
        0,
        3,
    )

    interface_drops = random.randint(
        0,
        5,
    )

    return {
        "timestamp": pd.Timestamp.now(
            tz="UTC"
        ),
        "interface": "Live Traffice-interface",
        "probe_host": "Live-network",
        "bandwidth_mbps": bandwidth_mbps,
        "rx_mbps": rx_mbps,
        "tx_mbps": tx_mbps,
        "throughput_mbps": throughput,
        "traffic_volume_mb": traffic_volume,
        "packet_rate": packet_rate,
        "latency_ms": latency,
        "packet_loss_pct": packet_loss,
        "jitter_ms": jitter,
        "bandwidth_utilization_pct": utilization,
        "cpu_utilization_pct": cpu,
        "memory_utilization_pct": memory,
        "interface_errors": interface_errors,
        "interface_drops": interface_drops,
        "network_available": 1,
        "connections": connections,
        "packets_sent": packets_sent,
        "packets_received": packets_received,
    }


# ============================================================
# GENERATE CHANGING LIVE DATA
# ============================================================

def add_demo_measurements(
    df,
    number_of_rows=20,
):

    demo_rows = []

    now = pd.Timestamp.now(
        tz="UTC"
    )

    for i in range(
        number_of_rows
    ):

        row = generate_demo_network_measurement()

        row["timestamp"] = (
            now
            - pd.Timedelta(
                seconds=(
                    number_of_rows - i
                ) * 5
            )
        )

        demo_rows.append(
            row
        )

    demo_df = pd.DataFrame(
        demo_rows
    )

    if df.empty:

        return demo_df

    df = df.copy()

    for column in df.columns:

        if column not in demo_df.columns:

            demo_df[column] = np.nan

    for column in demo_df.columns:

        if column not in df.columns:

            df[column] = np.nan

    historical_count = max(
        len(df) - number_of_rows,
        0,
    )

    historical_df = df.head(
        historical_count
    ).copy()

    combined = pd.concat(
        [
            historical_df,
            demo_df[
                df.columns
            ],
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    return combined


# ============================================================
# STATUS
# ============================================================

def status_for(
    value,
    warning_threshold,
    critical_threshold,
):

    if value is None:

        return "NORMAL"

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return "NORMAL"

    if not np.isfinite(
        value
    ):

        return "NORMAL"

    if value >= critical_threshold:

        return "CRITICAL"

    if value >= warning_threshold:

        return "WARNING"

    return "NORMAL"


def status_emoji(
    status
):

    if status == "CRITICAL":

        return "🔴"

    if status == "WARNING":

        return "🟠"

    return "🟢"


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value
):

    try:

        value = float(
            value
        )

        if np.isfinite(
            value
        ):

            return value

    except (
        TypeError,
        ValueError,
    ):

        pass

    return np.nan


# ============================================================
# RECOMMENDATIONS
# ============================================================

def recommendations(
    utilization,
    latency,
    packet_loss,
    jitter,
):

    recommendations_list = []

    if utilization >= CONGESTION_UTILIZATION_PCT:

        recommendations_list.append(
            "Reduce bandwidth pressure by limiting non-critical "
            "traffic or increasing available link capacity."
        )

    elif utilization >= UTILIZATION_WARNING_PCT:

        recommendations_list.append(
            "Bandwidth utilization is elevated. Monitor traffic "
            "growth and consider load balancing or capacity planning."
        )

    if latency >= CONGESTION_LATENCY_MS:

        recommendations_list.append(
            "Latency is critically high. Investigate congestion, "
            "routing paths, WAN links, and overloaded network devices."
        )

    elif latency >= LATENCY_WARNING_MS:

        recommendations_list.append(
            "Latency is elevated. Check network congestion and "
            "the quality of the current route."
        )

    if packet_loss >= CONGESTION_PACKET_LOSS_PCT:

        recommendations_list.append(
            "Packet loss is critical. Inspect link errors, "
            "interface health, congestion, and physical connectivity."
        )

    elif packet_loss >= PACKET_LOSS_WARNING_PCT:

        recommendations_list.append(
            "Packet loss is elevated. Monitor the affected interface "
            "and investigate retransmissions or unstable connectivity."
        )

    if jitter >= CONGESTION_JITTER_MS:

        recommendations_list.append(
            "Jitter is critically high. Prioritize real-time traffic "
            "and investigate congestion or unstable routing."
        )

    elif jitter >= JITTER_WARNING_MS:

        recommendations_list.append(
            "Jitter is elevated. Monitor real-time traffic quality "
            "and investigate congestion."
        )

    if not recommendations_list:

        recommendations_list.append(
            "Network conditions are currently healthy. "
            "Continue monitoring for future degradation."
        )

    return recommendations_list


# ============================================================
# PLOT METRIC
# ============================================================

def plot_metric(
    data,
    x_column,
    y_column,
    title,
    y_axis_title,
):

    if data.empty:

        st.info(
            "No historical data available."
        )

        return

    if x_column not in data.columns:

        st.info(
            "Timestamp data is not available."
        )

        return

    if y_column not in data.columns:

        st.info(
            f"{y_column} is not available in the collected data."
        )

        return

    chart_data = data[
        [
            x_column,
            y_column,
        ]
    ].dropna()

    if chart_data.empty:

        st.info(
            f"No valid data available for {y_column}."
        )

        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_data[
                x_column
            ],
            y=chart_data[
                y_column
            ],
            mode="lines+markers",
            name=y_column,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_axis_title,
        height=350,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# RUN ML PREDICTIONS
# ============================================================

def run_predictions(
    df,
    models,
):

    results = {
        "available": False,
        "error": None,
    }

    if models is None:

        results["error"] = (
            "No trained model was found. "
            "Run train.py first."
        )

        return results

    if df.empty:

        results["error"] = (
            "Not enough network data."
        )

        return results

    feature_columns = models.get(
        "feature_columns"
    )

    if not feature_columns:

        results["error"] = (
            "The trained model does not contain feature_columns."
        )

        return results

    required_rows = max(
        int(ROLLING_WINDOW),
        2,
    )

    prediction_df = df.tail(
        required_rows
    ).copy()

    try:

        feature_df = build_features(
            prediction_df
        )

    except Exception as exc:

        results["error"] = (
            f"Could not build prediction features: {exc}"
        )

        return results

    if feature_df.empty:

        results["error"] = (
            "Feature dataset is empty."
        )

        return results

    missing_columns = [
        column
        for column in feature_columns
        if column not in feature_df.columns
    ]

    if missing_columns:

        results["error"] = (
            "Missing model features: "
            + ", ".join(
                missing_columns
            )
        )

        return results

    X = feature_df[
        feature_columns
    ].copy()

    X = X.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    X = X.ffill().bfill()

    if X.empty:

        results["error"] = (
            "Prediction features contain no usable data."
        )

        return results

    X_latest = X.tail(
        1
    )

    classifier = (
        models.get(
            "classifier"
        )
        or models.get(
            "classification_model"
        )
        or models.get(
            "model"
        )
    )

    if classifier is not None:

        try:

            prediction = classifier.predict(
                X_latest
            )[0]

            results[
                "classification"
            ] = prediction

        except Exception as exc:

            results[
                "classification_error"
            ] = str(
                exc
            )

    regression_models = models.get(
        "regression_models",
        {},
    )

    if not isinstance(
        regression_models,
        dict,
    ):

        regression_models = {}

    regression_predictions = {}

    for (
        target,
        model,
    ) in regression_models.items():

        try:

            prediction = model.predict(
                X_latest
            )[0]

            regression_predictions[
                target
            ] = float(
                prediction
            )

        except Exception:

            continue

    results[
        "regression_predictions"
    ] = regression_predictions

    target_model_names = {

        "bandwidth_utilization_pct": [
            "bandwidth_utilization_model",
            "utilization_model",
        ],

        "latency_ms": [
            "latency_model",
        ],

        "packet_loss_pct": [
            "packet_loss_model",
        ],

        "jitter_ms": [
            "jitter_model",
        ],
    }

    for (
        target,
        names,
    ) in target_model_names.items():

        if target in regression_predictions:

            continue

        for name in names:

            model = models.get(
                name
            )

            if model is None:

                continue

            try:

                prediction = model.predict(
                    X_latest
                )[0]

                regression_predictions[
                    target
                ] = float(
                    prediction
                )

                break

            except Exception:

                continue

    anomaly_model = (
        models.get(
            "anomaly_model"
        )
        or models.get(
            "anomaly_detector"
        )
        or models.get(
            "isolation_forest"
        )
    )

    if anomaly_model is not None:

        try:

            anomaly_prediction = (
                anomaly_model.predict(
                    X_latest
                )[0]
            )

            results[
                "anomaly_prediction"
            ] = int(
                anomaly_prediction
            )

        except Exception as exc:

            results[
                "anomaly_error"
            ] = str(
                exc
            )

    results[
        "available"
    ] = True

    return results


# ============================================================
# PREDICTION CACHE
# ============================================================

@st.cache_data(
    ttl=0,
    show_spinner=False,
)
def cached_predictions(
    row_count,
    df_for_prediction,
    model_signature,
):

    del row_count
    del model_signature

    models = load_models()

    return run_predictions(
        df_for_prediction,
        models,
    )


def get_predictions(
    df
):

    model_path = Path(
        MODEL_FILE
    )

    if df.empty:

        return {
            "available": False,
            "error": (
                "No network measurements are available."
            ),
        }

    row_count = len(
        df
    )

    try:

        model_stat = model_path.stat()

        model_signature = (
            model_stat.st_mtime_ns,
            model_stat.st_size,
        )

    except OSError:

        model_signature = (
            0,
            0,
        )

    recent_rows = df.tail(
        max(
            int(ROLLING_WINDOW),
            2,
        )
    ).copy()

    return cached_predictions(
        row_count,
        recent_rows,
        model_signature,
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System"
    )

    refresh = st.slider(
        "Refresh interval (seconds)",
        min_value=2,
        max_value=30,
        value=5,
        step=1,
    )

    history_minutes = st.slider(
        "History window (minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
    )

    st.divider()

    if st.button(
        "🔄 Refresh now",
        use_container_width=True,
    ):

        st.rerun()

    st.divider()

    st.subheader(
        "Thresholds"
    )

    st.caption(
        f"Utilization warning: "
        f"{UTILIZATION_WARNING_PCT:.0f}%"
    )

    st.caption(
        f"Utilization critical: "
        f"{UTILIZATION_CRITICAL_PCT:.0f}%"
    )

    st.caption(
        f"Latency warning: "
        f"{LATENCY_WARNING_MS:.0f} ms"
    )

    st.caption(
        f"Latency critical: "
        f"{LATENCY_CRITICAL_MS:.0f} ms"
    )

    st.caption(
        f"Packet loss warning: "
        f"{PACKET_LOSS_WARNING_PCT:.1f}%"
    )

    st.caption(
        f"Packet loss critical: "
        f"{PACKET_LOSS_CRITICAL_PCT:.1f}%"
    )

    st.caption(
        f"Jitter warning: "
        f"{JITTER_WARNING_MS:.0f} ms"
    )

    st.caption(
        f"Jitter critical: "
        f"{JITTER_CRITICAL_MS:.0f} ms"
    )

    st.divider()

    st.info(
        "📊 Demo monitoring values are enabled "
        "for live dashboard demonstration."
    )


# ============================================================
# LIVE DASHBOARD
# ============================================================

@st.fragment(
    run_every=refresh
)
def live_dashboard():

    refresh_count = (
        st.session_state.get(
            "dashboard_refresh_count",
            0,
        )
        + 1
    )

    st.session_state[
        "dashboard_refresh_count"
    ] = refresh_count

    st.title(
        "📡 Predictive Network Management System"
    )

    st.markdown(
        "Real-time network monitoring, predictive analytics, "
        "anomaly detection and proactive network management."
    )

  
    models = load_models()

    load_metrics.clear()

    df = load_metrics()

    # ========================================================
    # GENERATE 20 NEW CHANGING LIVE RECORDS
    # ========================================================

    df = add_demo_measurements(
        df,
        number_of_rows=20,
    )

    st.caption(
        f"📡 Live dashboard • "
        f"Refresh #{refresh_count} • "
        f"Samples: {len(df):,}"
    )

    # ========================================================
    # LATEST RECORD
    # ========================================================

    if df.empty:

        st.error(
            "No network measurements are available."
        )

        return

    latest = df.iloc[
        -1
    ]

    latest_timestamp = latest.get(
        "timestamp"
    )

    # ========================================================
    # HISTORY
    # ========================================================

    if (
        "timestamp" in df.columns
        and pd.notna(
            latest_timestamp
        )
    ):

        cutoff = (
            latest_timestamp
            - pd.Timedelta(
                minutes=history_minutes
            )
        )

        history = df[
            df["timestamp"] >= cutoff
        ].copy()

    else:

        history = df.tail(
            max(
                int(ROLLING_WINDOW),
                1,
            )
        ).copy()

    if history.empty:

        history = df.copy()

    # ========================================================
    # CURRENT METRICS
    # ========================================================

    utilization = safe_float(
        latest.get(
            "bandwidth_utilization_pct",
            np.nan,
        )
    )

    throughput = safe_float(
        latest.get(
            "throughput_mbps",
            np.nan,
        )
    )

    latency = safe_float(
        latest.get(
            "latency_ms",
            np.nan,
        )
    )

    packet_loss = safe_float(
        latest.get(
            "packet_loss_pct",
            np.nan,
        )
    )

    jitter = safe_float(
        latest.get(
            "jitter_ms",
            np.nan,
        )
    )

    # ========================================================
    # STATUS
    # ========================================================

    utilization_status = status_for(
        utilization,
        UTILIZATION_WARNING_PCT,
        UTILIZATION_CRITICAL_PCT,
    )

    latency_status = status_for(
        latency,
        LATENCY_WARNING_MS,
        LATENCY_CRITICAL_MS,
    )

    packet_loss_status = status_for(
        packet_loss,
        PACKET_LOSS_WARNING_PCT,
        PACKET_LOSS_CRITICAL_PCT,
    )

    jitter_status = status_for(
        jitter,
        JITTER_WARNING_MS,
        JITTER_CRITICAL_MS,
    )

    statuses = [
        utilization_status,
        latency_status,
        packet_loss_status,
        jitter_status,
    ]

    if "CRITICAL" in statuses:

        overall_status = "CRITICAL"

    elif "WARNING" in statuses:

        overall_status = "WARNING"

    else:

        overall_status = "NORMAL"

    # ========================================================
    # NETWORK STATUS
    # ========================================================

    if overall_status == "CRITICAL":

        st.error(
            "🔴 NETWORK STATUS: CRITICAL"
        )

    elif overall_status == "WARNING":

        st.warning(
            "🟠 NETWORK STATUS: WARNING"
        )

    else:

        st.success(
            "🟢 NETWORK STATUS: NORMAL"
        )

    # ========================================================
    # CURRENT METRICS
    # ========================================================

    st.subheader(
        "📊 Current Network Metrics"
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    with col1:

        st.metric(
            "Bandwidth Utilization",
            f"{utilization:.1f}%",
            delta=status_emoji(
                utilization_status
            ),
        )

    with col2:

        st.metric(
            "Throughput",
            f"{throughput:.2f} Mbps",
        )

    with col3:

        st.metric(
            "Latency",
            f"{latency:.2f} ms",
            delta=status_emoji(
                latency_status
            ),
        )

    with col4:

        st.metric(
            "Packet Loss",
            f"{packet_loss:.2f}%",
            delta=status_emoji(
                packet_loss_status
            ),
        )

    with col5:

        st.metric(
            "Jitter",
            f"{jitter:.2f} ms",
            delta=status_emoji(
                jitter_status
            ),
        )

    if pd.notna(
        latest_timestamp
    ):

        st.caption(
            f"Latest network sample: "
            f"{latest_timestamp}"
        )

    # ========================================================
    # NETWORK AVAILABILITY
    # ========================================================

    st.subheader(
        "🌐 Network Availability"
    )

    availability_col1, availability_col2 = (
        st.columns(2)
    )

    with availability_col1:

        st.success(
            "🟢 Network Available"
        )

    with availability_col2:

        st.metric(
            "Availability",
            "100%",
        )

    # ========================================================
    # PREDICTIVE ANALYTICS
    # ========================================================

    st.subheader(
        "🤖 Predictive Analytics"
    )

    prediction_results = get_predictions(
        df
    )

    if not prediction_results.get(
        "available",
        False,
    ):

        st.info(
            prediction_results.get(
                "error",
                "Prediction is not available.",
            )
        )

    else:

        prediction_col1, prediction_col2 = (
            st.columns(2)
        )

        with prediction_col1:

            st.markdown(
                "### 🔮 Network Prediction"
            )

            classification = (
                prediction_results.get(
                    "classification"
                )
            )

            if classification is not None:

                classification_text = str(
                    classification
                )

                if classification_text in {
                    "1",
                    "1.0",
                    "True",
                    "true",
                }:

                    st.metric(
                        "Predicted Condition",
                        "⚠️ Degraded",
                    )

                else:

                    st.metric(
                        "Predicted Condition",
                        "✅ Normal",
                    )

            else:

                st.caption(
                    "No classification prediction "
                    "was returned by the model."
                )

        with prediction_col2:

            st.markdown(
                "### 📈 Forecast"
            )

            regression_predictions = (
                prediction_results.get(
                    "regression_predictions",
                    {},
                )
            )

            if regression_predictions:

                forecast_rows = []

                for (
                    target,
                    value,
                ) in regression_predictions.items():

                    forecast_rows.append(
                        {
                            "Metric": target,
                            "Predicted Value": (
                                f"{value:.2f}"
                            ),
                        }
                    )

                st.dataframe(
                    pd.DataFrame(
                        forecast_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.caption(
                    "No regression forecasts "
                    "were returned."
                )

        anomaly_prediction = (
            prediction_results.get(
                "anomaly_prediction"
            )
        )

        if anomaly_prediction is not None:

            if anomaly_prediction == -1:

                st.error(
                    "🚨 ML anomaly detector: "
                    "ANOMALOUS NETWORK BEHAVIOUR"
                )

            else:

                st.success(
                    "✅ ML anomaly detector: "
                    "Normal network behaviour"
                )

    # ========================================================
    # ACTIVE ALERTS
    # ========================================================

    st.subheader(
        "🚨 Active Alerts"
    )

    alerts = []

    if utilization_status != "NORMAL":

        alerts.append(
            f"Bandwidth utilization is "
            f"{utilization_status.lower()}."
        )

    if latency_status != "NORMAL":

        alerts.append(
            f"Latency is "
            f"{latency_status.lower()}."
        )

    if packet_loss_status != "NORMAL":

        alerts.append(
            f"Packet loss is "
            f"{packet_loss_status.lower()}."
        )

    if jitter_status != "NORMAL":

        alerts.append(
            f"Jitter is "
            f"{jitter_status.lower()}."
        )

    if alerts:

        for alert in alerts:

            if overall_status == "CRITICAL":

                st.error(
                    f"🔴 {alert}"
                )

            else:

                st.warning(
                    f"🟠 {alert}"
                )

    else:

        st.success(
            "No active network alerts."
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "💡 Proactive Recommendations"
    )

    recommendation_list = recommendations(
        utilization,
        latency,
        packet_loss,
        jitter,
    )

    for recommendation in recommendation_list:

        st.info(
            f"💡 {recommendation}"
        )

    # ========================================================
    # HISTORICAL PERFORMANCE
    # ========================================================

    st.subheader(
        f"📈 Historical Network Performance "
        f"({history_minutes} minutes)"
    )

    plot_metric(
        history,
        "timestamp",
        "bandwidth_utilization_pct",
        "Bandwidth Utilization",
        "Utilization (%)",
    )

    plot_metric(
        history,
        "timestamp",
        "throughput_mbps",
        "Network Throughput",
        "Throughput (Mbps)",
    )

    plot_metric(
        history,
        "timestamp",
        "latency_ms",
        "Network Latency",
        "Latency (ms)",
    )

    plot_metric(
        history,
        "timestamp",
        "packet_loss_pct",
        "Packet Loss",
        "Packet Loss (%)",
    )

    plot_metric(
        history,
        "timestamp",
        "jitter_ms",
        "Network Jitter",
        "Jitter (ms)",
    )

    # ========================================================
    # ROLLING STATISTICS
    # ========================================================

    st.subheader(
        "📐 Rolling Network Statistics"
    )

    rolling_window = min(
        int(ROLLING_WINDOW),
        len(history),
    )

    if rolling_window > 0:

        rolling_data = history.copy()

        rolling_columns = [
            "bandwidth_utilization_pct",
            "throughput_mbps",
            "latency_ms",
            "packet_loss_pct",
            "jitter_ms",
        ]

        available_rolling_columns = [
            column
            for column in rolling_columns
            if column in rolling_data.columns
        ]

        if available_rolling_columns:

            rolling_means = (
                rolling_data[
                    available_rolling_columns
                ]
                .tail(
                    rolling_window
                )
                .mean()
            )

            (
                stat_col1,
                stat_col2,
                stat_col3,
                stat_col4,
                stat_col5,
            ) = st.columns(5)

            with stat_col1:

                if (
                    "bandwidth_utilization_pct"
                    in rolling_means.index
                ):

                    value = rolling_means[
                        "bandwidth_utilization_pct"
                    ]

                    st.metric(
                        "Avg Utilization",
                        f"{value:.1f}%",
                    )

            with stat_col2:

                if (
                    "throughput_mbps"
                    in rolling_means.index
                ):

                    value = rolling_means[
                        "throughput_mbps"
                    ]

                    st.metric(
                        "Avg Throughput",
                        f"{value:.2f} Mbps",
                    )

            with stat_col3:

                if (
                    "latency_ms"
                    in rolling_means.index
                ):

                    value = rolling_means[
                        "latency_ms"
                    ]

                    st.metric(
                        "Avg Latency",
                        f"{value:.2f} ms",
                    )

            with stat_col4:

                if (
                    "packet_loss_pct"
                    in rolling_means.index
                ):

                    value = rolling_means[
                        "packet_loss_pct"
                    ]

                    st.metric(
                        "Avg Packet Loss",
                        f"{value:.2f}%",
                    )

            with stat_col5:

                if (
                    "jitter_ms"
                    in rolling_means.index
                ):

                    value = rolling_means[
                        "jitter_ms"
                    ]

                    st.metric(
                        "Avg Jitter",
                        f"{value:.2f} ms",
                    )

    # ========================================================
    # RESOURCE UTILIZATION
    # ========================================================

    resource_columns = []

    for column in [
        "cpu_utilization_pct",
        "memory_utilization_pct",
    ]:

        if column in history.columns:

            resource_columns.append(
                column
            )

    if resource_columns:

        st.subheader(
            "💻 System Resource Utilization"
        )

        resource_col1, resource_col2 = (
            st.columns(2)
        )

        if (
            "cpu_utilization_pct"
            in history.columns
        ):

            with resource_col1:

                plot_metric(
                    history,
                    "timestamp",
                    "cpu_utilization_pct",
                    "CPU Utilization",
                    "CPU (%)",
                )

        if (
            "memory_utilization_pct"
            in history.columns
        ):

            with resource_col2:

                plot_metric(
                    history,
                    "timestamp",
                    "memory_utilization_pct",
                    "Memory Utilization",
                    "Memory (%)",
                )

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    st.subheader(
        "🧪 Model Evaluation"
    )

    evaluation_path = Path(
        EVAL_FILE
    )

    if evaluation_path.exists():

        try:

            with evaluation_path.open(
                "r",
                encoding="utf-8",
            ) as f:

                evaluation = json.load(
                    f
                )

            if isinstance(
                evaluation,
                dict,
            ):

                st.json(
                    evaluation
                )

            else:

                st.write(
                    evaluation
                )

        except Exception as exc:

            st.warning(
                f"Could not read model evaluation file: "
                f"{exc}"
            )

    else:

        st.info(
            "Model evaluation results are not available yet."
        )

    # ========================================================
    # LATEST MEASUREMENTS
    # ========================================================

    st.subheader(
        "📋 Latest Measurements"
    )

    table_rows = min(
        20,
        len(df),
    )

    latest_table = df.tail(
        table_rows
    ).copy()

    if "timestamp" in latest_table.columns:

        latest_table[
            "timestamp"
        ] = latest_table[
            "timestamp"
        ].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    for column in latest_table.columns:

        if pd.api.types.is_numeric_dtype(
            latest_table[
                column
            ]
        ):

            latest_table[
                column
            ] = latest_table[
                column
            ].round(
                3
            )

    st.dataframe(
        latest_table,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        f"🔄 Live monitoring active • "
        f"Dashboard refreshes every {refresh} seconds • "
        f"Refresh count: {refresh_count}"
    )


# ============================================================
# RUN DASHBOARD
# ============================================================

live_dashboard()