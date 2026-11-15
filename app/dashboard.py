import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import json
import random
import sqlite3
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics import accuracy_score

from src.simulator import generate_transaction_events
from src.preprocessor import load_and_preprocess_logs
from src.feature_extractor import extract_transaction_features
from src.rca_engine import AdaptiveRCAEngine, DEFAULT_EMA_BASELINES
from src.severity_remediation import assess_severity_and_remediation

st.set_page_config(page_title="Adaptive Payment RCA Dashboard", layout="wide")

DB_PATH = "data/rca_history.db"
SCENARIOS = [
    "Normal",
    "Gateway_Timeout",
    "Authentication_Failure",
    "Core_Banking_Failure",
    "Database_Failure",
    "Network_Failure",
    "External_Service_Failure"
]

def init_and_seed_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            timestamp TEXT,
            hop_order INTEGER,
            service_name TEXT,
            dependency TEXT,
            latency_ms REAL,
            status TEXT,
            error_code TEXT,
            retry_count INTEGER,
            log_message TEXT,
            ground_truth_category TEXT,
            ground_truth_root_cause TEXT,
            ema_processed INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS service_ema_baselines (
            service_name TEXT PRIMARY KEY,
            baseline_ms REAL,
            updated_at TEXT
        )
    """)

    ema_count = conn.execute("SELECT COUNT(*) FROM service_ema_baselines").fetchone()[0]
    if ema_count == 0:
        now_str = datetime.now().isoformat()
        for svc, val in DEFAULT_EMA_BASELINES.items():
            conn.execute(
                "INSERT INTO service_ema_baselines (service_name, baseline_ms, updated_at) VALUES (?, ?, ?)",
                (svc, val, now_str)
            )

    count = conn.execute("SELECT COUNT(*) FROM telemetry_events").fetchone()[0]
    if count == 0 and os.path.exists("data/synthetic_transactions.json"):
        with open("data/synthetic_transactions.json", "r") as f:
            records = json.load(f)
        df_seed = pd.DataFrame(records)
        df_seed["ema_processed"] = 0
        df_seed.to_sql("telemetry_events", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

def load_persisted_ema(conn) -> dict:
    rows = conn.execute("SELECT service_name, baseline_ms FROM service_ema_baselines").fetchall()
    return {svc: val for svc, val in rows}

def save_persisted_ema(conn, baselines: dict):
    now_str = datetime.now().isoformat()
    for svc, val in baselines.items():
        conn.execute(
            "UPDATE service_ema_baselines SET baseline_ms = ?, updated_at = ? WHERE service_name = ?",
            (round(val, 4), now_str, svc)
        )

def ingest_new_transaction(scenario_choice: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    last_tx = conn.execute(
        "SELECT transaction_id FROM telemetry_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    next_num = int(last_tx[0].split("_")[1]) + 1 if last_tx else 1
    new_tx_id = f"TXN_{next_num:04d}"

    chosen_scenario = random.choice(SCENARIOS) if scenario_choice == "Random (Live Traffic Mix)" else scenario_choice
    new_events = generate_transaction_events(new_tx_id, datetime.now(), chosen_scenario)
    df_new = pd.DataFrame(new_events)
    df_new["ema_processed"] = 0
    df_new.to_sql("telemetry_events", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    return new_tx_id

@st.cache_resource
def get_trained_engine():
    df, _ = load_and_preprocess_logs("data/synthetic_transactions.json")
    features_df = extract_transaction_features(df)
    engine = AdaptiveRCAEngine(alpha=0.10)
    X_test, y_test = engine.train_classifier(features_df)
    heldout_acc = round(accuracy_score(y_test, engine.classifier.predict(X_test)) * 100, 2)
    return engine, heldout_acc

def load_all_from_db(engine: AdaptiveRCAEngine):
    conn = sqlite3.connect(DB_PATH)
    engine.load_ema_baselines(load_persisted_ema(conn))

    df = pd.read_sql_query("SELECT * FROM telemetry_events ORDER BY id ASC", conn)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["is_error"] = (df["status"] == "FAILED").astype(int)
    df["norm_latency"] = df.groupby("service_name")["latency_ms"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )
    df = df.sort_values(by=["transaction_id", "hop_order"]).reset_index(drop=True)
    traces = {tx_id: group.to_dict(orient="records") for tx_id, group in df.groupby("transaction_id")}

    features_df = extract_transaction_features(df)
    features_df["predicted_category"] = engine.predict_failure_category(features_df)

    unprocessed_txs = set(df[df["ema_processed"] == 0]["transaction_id"].unique())
    rca_results = {}
    for tx_id, events in traces.items():
        should_update = tx_id in unprocessed_txs
        rca_results[tx_id] = engine.identify_root_cause(events, update_ema=should_update)

    if unprocessed_txs:
        save_persisted_ema(conn, engine.get_ema_baselines())
        conn.execute("UPDATE telemetry_events SET ema_processed = 1 WHERE ema_processed = 0")
        conn.commit()
    conn.close()

    features_df["root_cause_service"] = features_df["transaction_id"].map(
        lambda x: rca_results[x]["root_cause_service"]
    )
    features_df["root_cause_score"] = features_df["transaction_id"].map(
        lambda x: rca_results[x]["root_cause_score"]
    )
    features_df["confidence"] = features_df["transaction_id"].map(
        lambda x: rca_results[x]["confidence"]
    )
    features_df["is_uncertain"] = features_df["transaction_id"].map(
        lambda x: rca_results[x]["is_uncertain"]
    )
    features_df["explanation"] = features_df["transaction_id"].map(
        lambda x: rca_results[x]["explanation"]
    )

    features_df["category_match"] = features_df["ground_truth_category"] == features_df["predicted_category"]
    features_df["root_cause_match"] = features_df["ground_truth_root_cause"] == features_df["root_cause_service"]

    category_counts = features_df["predicted_category"].value_counts().to_dict()
    total_failed_txns = int((features_df["predicted_category"] != "Normal").sum())

    severities, levels, remediations = [], [], []
    for _, row in features_df.iterrows():
        res = assess_severity_and_remediation(
            category=row["predicted_category"],
            root_cause_service=row["root_cause_service"],
            total_latency_ms=row["total_latency_ms"],
            recurrence_count=category_counts.get(row["predicted_category"], 1),
            total_failed_txns=total_failed_txns
        )
        severities.append(res["severity_score"])
        levels.append(res["severity_level"])
        remediations.append(res["remediation"])

    features_df["severity_score"] = severities
    features_df["severity_level"] = levels
    features_df["remediation"] = remediations

    return df, traces, features_df, rca_results

init_and_seed_db()
engine, heldout_test_acc = get_trained_engine()

if "selected_tx" not in st.session_state:
    st.session_state["selected_tx"] = "TXN_0002"

st.sidebar.header("Live Telemetry Ingestion")
st.sidebar.caption("Simulate incoming payment transactions & persist to SQLite DB")

selected_scenario = st.sidebar.selectbox(
    "Select Scenario to Simulate:",
    ["Random (Live Traffic Mix)"] + SCENARIOS
)

if st.sidebar.button("Simulate & Analyze New Transaction", type="primary", use_container_width=True):
    new_id = ingest_new_transaction(selected_scenario)
    st.session_state["selected_tx"] = new_id
    st.sidebar.success(f"Ingested & diagnosed {new_id}!")

auto_stream = st.sidebar.toggle("Auto-Stream Live Events (Every 3s)", value=False)
if auto_stream:
    new_id = ingest_new_transaction("Random (Live Traffic Mix)")
    st.session_state["selected_tx"] = new_id
    st.sidebar.info(f"Streaming active — latest: {new_id}")

df, traces, features_df, rca_results = load_all_from_db(engine)

st.sidebar.divider()
st.sidebar.subheader("Persisted Adaptive EMA Baselines")
ema_df = pd.DataFrame(
    [{"Service": k, "Baseline (ms)": round(v, 2)} for k, v in engine.get_ema_baselines().items()]
)
st.sidebar.dataframe(ema_df, hide_index=True, use_container_width=True)

st.title("Adaptive Root Cause Analysis (RCA) — Distributed Payment Systems")
st.caption("Continuous Telemetry Ingestion, SQLite-Persisted EMA Baselines, Multi-Signal RCA & Automated Remediation")

total_tx = len(features_df)
live_df = features_df[features_df["transaction_id"].str.slice(4).astype(int) > 900]
actual_failed_df = features_df[features_df["ground_truth_category"] != "Normal"]
rc_acc = round(actual_failed_df["root_cause_match"].mean() * 100, 1) if len(actual_failed_df) else 100.0
live_cat_acc = f"{round(live_df['category_match'].mean() * 100, 1)}%" if len(live_df) else "No live txns yet"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Stored Transactions", total_tx)
c2.metric("Held-Out Test Category Acc", f"{heldout_test_acc}%")
c3.metric("Live Simulated Category Acc", live_cat_acc)
c4.metric("Root-Cause Service Accuracy", f"{rc_acc}%")

st.divider()

failed_df = features_df[features_df["predicted_category"] != "Normal"]
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Failure Category Distribution")
    cat_fig = px.bar(
        failed_df["predicted_category"].value_counts().reset_index(),
        x="predicted_category", y="count",
        color="predicted_category",
        labels={"predicted_category": "Failure Category", "count": "Incidents"}
    )
    st.plotly_chart(cat_fig, use_container_width=True)

with col_right:
    st.subheader("Identified Root Cause Services")
    rc_fig = px.pie(failed_df, names="root_cause_service", hole=0.4)
    st.plotly_chart(rc_fig, use_container_width=True)

st.divider()

st.subheader("Incident & Transaction Trace Inspector")
tx_options = features_df["transaction_id"].tolist()
current_tx = st.session_state.get("selected_tx", "TXN_0002")
default_idx = tx_options.index(current_tx) if current_tx in tx_options else 1

selected_tx = st.selectbox(
    "Select a Transaction ID to inspect (latest simulated transactions appear at the bottom):",
    options=tx_options,
    index=default_idx
)
st.session_state["selected_tx"] = selected_tx

tx_row = features_df[features_df["transaction_id"] == selected_tx].iloc[0]
tx_events = traces[selected_tx]
tx_rca = rca_results[selected_tx]

v1, v2, v3, v4 = st.columns(4)
v1.metric("Actual Ground Truth", f"{tx_row['ground_truth_category']} ({tx_row['ground_truth_root_cause']})")
v2.metric("Predicted Outcome", f"{tx_row['predicted_category']} ({tx_row['root_cause_service']})")
v3.metric("Category Check", "PASS" if tx_row["category_match"] else "FAIL")
v4.metric("Root-Cause Service Check", "PASS" if tx_row["root_cause_match"] else "FAIL")

m1, m2, m3, m4 = st.columns(4)
m1.info(f"**Predicted Category:**\n{tx_row['predicted_category']}")
m2.error(f"**Probable Root Cause:**\n{tx_row['root_cause_service']} (Score: {tx_row['root_cause_score']})")
m3.warning(f"**Severity & Confidence:**\n{tx_row['severity_level']} ({tx_row['severity_score']}) | Conf: {tx_row['confidence']}%")
m4.success(f"**Total Latency:**\n{round(tx_row['total_latency_ms'], 2)} ms")

if tx_row["is_uncertain"]:
    st.warning(f"**Diagnostic Explanation:** {tx_row['explanation']}")
else:
    st.info(f"**Diagnostic Explanation:** {tx_row['explanation']}")

st.markdown(f"**Recommended Remediation:** {tx_row['remediation']}")

st.markdown("#### Service Dependency Topology & Fault Origin")
G = nx.DiGraph()
edges = [
    ("Client_App", "API_Gateway"),
    ("API_Gateway", "Auth_Service"),
    ("Auth_Service", "Payment_Service"),
    ("Payment_Service", "Database"),
    ("Payment_Service", "Core_Banking"),
    ("Core_Banking", "External_APIs")
]
G.add_edges_from(edges)

pos = {
    "Client_App": (0, 1),
    "API_Gateway": (1, 1),
    "Auth_Service": (2, 1),
    "Payment_Service": (3, 1),
    "Database": (4, 1.6),
    "Core_Banking": (4, 0.4),
    "External_APIs": (5, 0.4)
}

failed_services = {e["service_name"] for e in tx_events if e["status"] == "FAILED"}
root_service = tx_row["root_cause_service"]

node_colors = []
for node in G.nodes():
    if node == root_service:
        node_colors.append("#E63946")
    elif node in failed_services:
        node_colors.append("#F4A261")
    else:
        node_colors.append("#2A9D8F")

edge_x, edge_y = [], []
for u, v in G.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig_net = go.Figure()
fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2, color="#888"), hoverinfo="none"))
fig_net.add_trace(go.Scatter(
    x=[pos[n][0] for n in G.nodes()],
    y=[pos[n][1] for n in G.nodes()],
    mode="markers+text",
    text=list(G.nodes()),
    textposition="top center",
    marker=dict(size=28, color=node_colors, line=dict(width=2, color="white"))
))
fig_net.update_layout(
    showlegend=False, height=300,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)
st.plotly_chart(fig_net, use_container_width=True)

if tx_rca["evidence"]:
    st.markdown("#### Supporting Diagnostic Evidence $R(s) = w_e E(s) + w_l L(s) + w_d D(s) + w_t T(s)$")
    ev_df = pd.DataFrame(tx_rca["evidence"]).T.reset_index().rename(columns={"index": "Service"})
    st.dataframe(ev_df, use_container_width=True)

st.markdown("#### End-to-End Correlated Transaction Logs")
log_df = pd.DataFrame(tx_events)[["hop_order", "service_name", "dependency", "status", "error_code", "latency_ms", "retry_count", "log_message"]]
st.dataframe(log_df, use_container_width=True)

if auto_stream:
    time.sleep(3)
    st.rerun()