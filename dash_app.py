import streamlit as st
import pandas as pd
import plotly.express as px

from evaluation_and_plot import run_controller_with_metrics

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Optimization Dashboard",
    layout="wide",
)

st.title("AI-Powered Urban Traffic Optimization – Evaluation Dashboard")

st.markdown(
    """
This dashboard compares **Fixed-time** vs **AI (GNN-A2C)** traffic signal control
on your SUMO network.  
All metrics are collected live from the simulation via TraCI.
"""
)

# -------------------------------------------------------------------
# Helper: run simulations and cache results
# -------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def run_episode_cached(mode: str, max_steps: int):
    global_df, tls_df = run_controller_with_metrics(mode=mode, max_steps=max_steps)
    return global_df, tls_df


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Simulation Settings")

max_steps = st.sidebar.slider(
    "Simulation steps", min_value=600, max_value=3600, step=300, value=1800
)

run_fixed = st.sidebar.checkbox("Run Fixed-time baseline", value=True)
run_ai = st.sidebar.checkbox("Run AI (GNN-A2C)", value=True)

run_button = st.sidebar.button("Run simulations")

st.sidebar.markdown("---")
st.sidebar.markdown("Longer episodes give smoother trends but take more time.")

# -------------------------------------------------------------------
# Run simulations when requested
# -------------------------------------------------------------------
if run_button:
    results = {}

    if run_fixed:
        with st.spinner("Running Fixed-time controller..."):
            fixed_global, fixed_tls = run_episode_cached("fixed", max_steps)
            results["Fixed"] = {"global": fixed_global, "tls": fixed_tls}

    if run_ai:
        with st.spinner("Running AI controller..."):
            ai_global, ai_tls = run_episode_cached("ai", max_steps)
            results["AI"] = {"global": ai_global, "tls": ai_tls}

    if not results:
        st.warning("Select at least one controller in the sidebar.")
    else:
        st.success("Simulation(s) completed.")
        st.session_state["results"] = results
else:
    results = st.session_state.get("results", None)

# -------------------------------------------------------------------
# If we have results, build the dashboard
# -------------------------------------------------------------------
if results is None:
    st.info("Use the sidebar to run one or both simulations.")
    st.stop()

# === 1. Global metrics overview =================================================
st.subheader("Global Episode Metrics")

global_frames = []
for label, data in results.items():
    df = data["global"].copy()
    df["mode_label"] = "Fixed-time" if label.lower() == "fixed" else "AI (GNN-A2C)"
    global_frames.append(df)

global_all = pd.concat(global_frames, ignore_index=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average waiting time over time**")
    fig_wait = px.line(
        global_all,
        x="step",
        y="avg_wait_time",
        color="mode_label",
        labels={"step": "Simulation step", "avg_wait_time": "Average waiting time (s)"},
    )
    st.plotly_chart(fig_wait, use_container_width=True)

with col2:
    st.markdown("**Total queue length over time (vehicles in queue)**")
    fig_queue = px.line(
        global_all,
        x="step",
        y="total_queue",
        color="mode_label",
        labels={"step": "Simulation step", "total_queue": "Vehicles in queue"},
    )
    st.plotly_chart(fig_queue, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Vehicles processed over time**")
    fig_proc = px.line(
        global_all,
        x="step",
        y="vehicles_processed_cum",
        color="mode_label",
        labels={
            "step": "Simulation step",
            "vehicles_processed_cum": "Vehicles processed (cumulative)",
        },
    )
    st.plotly_chart(fig_proc, use_container_width=True)

with col4:
    st.markdown("**Reward trend over time**")
    fig_reward = px.line(
        global_all,
        x="step",
        y="reward",
        color="mode_label",
        labels={
            "step": "Simulation step",
            "reward": "Reward (− total wait / 1000)",
        },
    )
    st.plotly_chart(fig_reward, use_container_width=True)

# === 2. Queue vs processed ======================================================
st.subheader("Queue vs Processed Vehicles")

fig_q_vs_p = px.scatter(
    global_all,
    x="vehicles_processed_cum",
    y="total_queue",
    color="mode_label",
    trendline="lowess",
    labels={
        "vehicles_processed_cum": "Vehicles processed (cumulative)",
        "total_queue": "Vehicles in queue",
    },
)
st.plotly_chart(fig_q_vs_p, use_container_width=True)

# === 3. Per-intersection (TLS) performance =====================================
st.subheader("Intersection-level Performance (per TLS)")

tls_frames = []
for label, data in results.items():
    df_tls = data["tls"].copy()
    df_tls["mode_label"] = "Fixed-time" if label.lower() == "fixed" else "AI (GNN-A2C)"
    tls_frames.append(df_tls)

tls_all = pd.concat(tls_frames, ignore_index=True)

available_tls = sorted(tls_all["tls_id"].unique().tolist())
selected_tls = st.selectbox("Select traffic light signal (TLS):", available_tls)

tls_filtered = tls_all[tls_all["tls_id"] == selected_tls]

col_tls1, col_tls2 = st.columns(2)

with col_tls1:
    st.markdown(f"**Queue length over time – {selected_tls}**")
    fig_tls_q = px.line(
        tls_filtered,
        x="step",
        y="queue",
        color="mode_label",
        labels={"step": "Simulation step", "queue": "Vehicles in queue"},
    )
    st.plotly_chart(fig_tls_q, use_container_width=True)

with col_tls2:
    st.markdown(f"**Waiting time over time – {selected_tls}**")
    fig_tls_w = px.line(
        tls_filtered,
        x="step",
        y="wait_time",
        color="mode_label",
        labels={"step": "Simulation step", "wait_time": "Total waiting time (s)"},
    )
    st.plotly_chart(fig_tls_w, use_container_width=True)

# === 4. Fixed vs AI comparison ==================================================
st.subheader("Fixed vs AI – Episode-level Comparison")

summary_rows = []
for label, data in results.items():
    gdf = data["global"]
    mode_label = "Fixed-time" if label.lower() == "fixed" else "AI (GNN-A2C)"

    total_reward = gdf["reward"].sum()
    total_wait = gdf["total_wait_time"].sum()
    mean_avg_wait = gdf["avg_wait_time"].mean()
    max_processed = gdf["vehicles_processed_cum"].max()

    summary_rows.append(
        {
            "mode_label": mode_label,
            "total_reward": total_reward,
            "total_wait_time": total_wait,
            "mean_avg_wait_time": mean_avg_wait,
            "vehicles_processed_total": max_processed,
        }
    )

summary_df = pd.DataFrame(summary_rows)

col_bar1, col_bar2 = st.columns(2)

with col_bar1:
    st.markdown("**Total waiting time (lower is better)**")
    fig_wait_bar = px.bar(
        summary_df,
        x="mode_label",
        y="total_wait_time",
        labels={"mode_label": "Controller", "total_wait_time": "Total waiting time (s)"},
    )
    st.plotly_chart(fig_wait_bar, use_container_width=True)

with col_bar2:
    st.markdown("**Total vehicles processed (higher is better)**")
    fig_proc_bar = px.bar(
        summary_df,
        x="mode_label",
        y="vehicles_processed_total",
        labels={
            "mode_label": "Controller",
            "vehicles_processed_total": "Vehicles processed (episode)",
        },
    )
    st.plotly_chart(fig_proc_bar, use_container_width=True)

if {"Fixed-time", "AI (GNN-A2C)"}.issubset(set(summary_df["mode_label"])):
    fixed_row = summary_df[summary_df["mode_label"] == "Fixed-time"].iloc[0]
    ai_row = summary_df[summary_df["mode_label"] == "AI (GNN-A2C)"].iloc[0]

    improvement_wait = (
        (fixed_row["total_wait_time"] - ai_row["total_wait_time"])
        / fixed_row["total_wait_time"]
        * 100.0
    )

    st.markdown("### Overall Improvement of AI over Fixed-time")

    fig_improve = px.bar(
        x=["Waiting time reduction (%)"],
        y=[improvement_wait],
        labels={"x": "", "y": "Improvement (%)"},
        text=[f"{improvement_wait:.2f}%"],
    )
    fig_improve.update_traces(textposition="outside")
    st.plotly_chart(fig_improve, use_container_width=True)

    st.write(
        f"Estimated **waiting time reduction** of AI vs Fixed-time: "
        f"**{improvement_wait:.2f}%**"
    )

# === 5. CSV export ==============================================================
st.subheader("Download Metrics as CSV")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.markdown("**Global metrics (all controllers)**")
    csv_global = global_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download global metrics CSV",
        data=csv_global,
        file_name="global_metrics_all_controllers.csv",
        mime="text/csv",
    )

with col_dl2:
    st.markdown("**TLS-level metrics (all controllers)**")
    csv_tls = tls_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download per-TLS metrics CSV",
        data=csv_tls,
        file_name="tls_metrics_all_controllers.csv",
        mime="text/csv",
    )
