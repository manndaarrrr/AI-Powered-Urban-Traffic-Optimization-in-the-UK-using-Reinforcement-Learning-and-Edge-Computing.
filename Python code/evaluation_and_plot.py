# -*- coding: utf-8 -*-
"""evaluation_and_plot.py

Evaluation + metrics + helpers for Streamlit dashboard.
"""

from pathlib import Path
import os
import sys
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import traci

# ----------------------------------------------------------------------
# Basic project paths
# ----------------------------------------------------------------------
PROJECT_DIR = Path.cwd()

NETWORK_FILE = PROJECT_DIR / "my_network.net.xml"
ROUTE_FILE   = PROJECT_DIR / "my_routes.rou.xml"
CONFIG_FILE  = PROJECT_DIR / "my_config.sumocfg"
WEIGHTS_FILE = PROJECT_DIR / "gnn_a2c_best.weights.h5"

print("Working directory:", PROJECT_DIR)
print("Network exists? ", NETWORK_FILE.exists())
print("Routes exist?  ", ROUTE_FILE.exists())
print("Config exists? ", CONFIG_FILE.exists())
print("Weights exist? ", WEIGHTS_FILE.exists())

SUMO_HOME = os.environ.get("SUMO_HOME")
if SUMO_HOME is None:
    raise EnvironmentError("SUMO_HOME is not set.")

print("SUMO_HOME:", SUMO_HOME)


def get_sumo_binary(gui: bool = False) -> str:
    """Locate sumo or sumo-gui binary."""
    base_name = "sumo-gui" if gui else "sumo"
    cmd = shutil.which(base_name)
    if cmd is not None:
        return cmd

    bin_dir = Path(SUMO_HOME) / "bin"
    candidate = bin_dir / (base_name + ".exe" if sys.platform.startswith("win") else base_name)

    if not candidate.exists():
        raise FileNotFoundError(f"{base_name} not found at {candidate}")

    return str(candidate)


# ----------------------------------------------------------------------
# Discover TLS and lane mapping
# ----------------------------------------------------------------------
if traci.isLoaded():
    traci.close()

sumo_bin = get_sumo_binary(gui=False)
cmd = [sumo_bin, "-c", str(CONFIG_FILE), "--step-length", "1"]
traci.start(cmd)

tls_ids = traci.trafficlight.getIDList()
print("Total TLS:", len(tls_ids))
print("TLS IDs:", tls_ids)

traci.close()

# --- Build tls_lane_map ---
if traci.isLoaded():
    traci.close()

traci.start([sumo_bin, "-c", str(CONFIG_FILE), "--step-length", "1"])

tls_lane_map = {}
for tls in tls_ids:
    lanes = traci.trafficlight.getControlledLanes(tls)
    lanes = list(dict.fromkeys(lanes))  # remove duplicates
    tls_lane_map[tls] = lanes

print("\nTLS → lanes:")
for tls, lanes in tls_lane_map.items():
    print(tls, ":", lanes)

traci.close()


# ----------------------------------------------------------------------
# State function used by GNN
# ----------------------------------------------------------------------
def get_tls_state(tls_id: str, lane_map: dict) -> list:
    """
    For a single TLS:
      - queue lengths for its lanes
      - waiting times for its lanes
      - current phase index
    """
    lane_ids = lane_map[tls_id]
    queue_lengths = []
    waiting_times = []

    for lane in lane_ids:
        q = traci.lane.getLastStepHaltingNumber(lane)
        w = traci.lane.getWaitingTime(lane)
        queue_lengths.append(q)
        waiting_times.append(w)

    current_phase = traci.trafficlight.getPhase(tls_id)
    return queue_lengths + waiting_times + [current_phase]


# ----------------------------------------------------------------------
# Feature size and graph adjacency
# ----------------------------------------------------------------------
if traci.isLoaded():
    traci.close()

traci.start([sumo_bin, "-c", str(CONFIG_FILE), "--step-length", "1"])

# let simulation advance a few steps so there is traffic
for _ in range(5):
    traci.simulationStep()

lengths = []
for tls in tls_ids:
    s = get_tls_state(tls, tls_lane_map)
    lengths.append(len(s))

feature_size = max(lengths)
num_nodes = len(tls_ids)

print("State lengths per TLS:", lengths)
print("feature_size:", feature_size)
print("num_nodes:", num_nodes)

traci.close()

# --- Build TLS adjacency ---
from collections import defaultdict

if traci.isLoaded():
    traci.close()

traci.start([sumo_bin, "-c", str(CONFIG_FILE), "--step-length", "1"])

tls_adj = {tls: set() for tls in tls_ids}

for tls in tls_ids:
    controlled_links = traci.trafficlight.getControlledLinks(tls)
    # controlled_links: list of lists of (inLane, outLane, via) tuples
    for link_group in controlled_links:
        for (incoming, outgoing, _) in link_group:
            for other_tls in tls_ids:
                if other_tls == tls:
                    continue
                if outgoing in tls_lane_map.get(other_tls, []):
                    tls_adj[tls].add(other_tls)
                    tls_adj[other_tls].add(tls)

traci.close()

# fallback if we found no edges at all
edge_count = sum(len(neigh) for neigh in tls_adj.values())
if edge_count == 0 and len(tls_ids) > 1:
    print("No adjacency found; using simple chain.")
    ordered = list(tls_ids)
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        tls_adj[a].add(b)
        tls_adj[b].add(a)

print("\nAdjacency list:")
for tls, neigh in tls_adj.items():
    print(tls, ":", sorted(list(neigh)))

tls_index = {tls_id: i for i, tls_id in enumerate(tls_ids)}
adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

for tls, neigh in tls_adj.items():
    i = tls_index[tls]
    for nb in neigh:
        j = tls_index[nb]
        adj_matrix[i, j] = 1.0
        adj_matrix[j, i] = 1.0

print("\nadj_matrix shape:", adj_matrix.shape)
print("adj_matrix[0]:", adj_matrix[0])


# ----------------------------------------------------------------------
# Reward function
# ----------------------------------------------------------------------
def compute_global_reward(tls_lane_map: dict) -> float:
    """
    Global reward = - (total waiting time across all controlled lanes) / 1000
    (negative reward; less negative is better).
    """
    total_wait = 0.0
    for tls, lanes in tls_lane_map.items():
        for lane in lanes:
            total_wait += traci.lane.getWaitingTime(lane)
    return -total_wait / 1000.0


# ----------------------------------------------------------------------
# GNN Actor-Critic model (same as training)
# ----------------------------------------------------------------------
class GNNActorCritic(tf.keras.Model):
    def __init__(self, hidden_dim: int, num_actions: int):
        super().__init__()
        self.state_embed = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.post_gnn = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.policy_head = tf.keras.layers.Dense(num_actions)
        self.value_head = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x, adj = inputs  # x: (B, N, F), adj: (B, N, N)

        h = self.state_embed(x)  # (B, N, H)
        h_neigh = tf.matmul(adj, h)  # (B, N, H)

        h_cat = tf.concat([h, h_neigh], axis=-1)  # (B, N, 2H)
        h_out = self.post_gnn(h_cat)  # (B, N, H)

        policy_logits = self.policy_head(h_out)  # (B, N, A)
        graph_embed = tf.reduce_mean(h_out, axis=1)
        value = self.value_head(graph_embed)

        return policy_logits, value


hidden_dim = 64
num_actions = 2  # 0 = keep phase, 1 = switch
gnn_model = GNNActorCritic(hidden_dim, num_actions)

# Build model once with dummy input and load weights
adj_batch_tf = tf.convert_to_tensor(adj_matrix[None, ...], dtype=tf.float32)
dummy_states = tf.random.uniform((1, num_nodes, feature_size), dtype=tf.float32)
gnn_model((dummy_states, adj_batch_tf))

if WEIGHTS_FILE.exists():
    gnn_model.load_weights(WEIGHTS_FILE)
    print("Loaded trained weights from:", WEIGHTS_FILE)
else:
    print("WARNING: weights file not found. Using untrained model.")


def select_actions_from_logits(policy_logits: tf.Tensor) -> np.ndarray:
    """Greedy argmax per node."""
    if isinstance(policy_logits, tf.Tensor):
        policy_logits = policy_logits.numpy()
    return np.argmax(policy_logits, axis=-1)


def apply_actions_to_sumo(actions: np.ndarray, tls_ids_list):
    """
    actions[i] in {0,1} for TLS tls_ids_list[i]
    0 = keep phase, 1 = switch to next phase
    """
    for idx, tls in enumerate(tls_ids_list):
        a = int(actions[idx])
        if a == 0:
            continue
        curr_phase = traci.trafficlight.getPhase(tls)
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)[0]
        num_phases = len(logic.phases)
        next_phase = (curr_phase + 1) % num_phases
        traci.trafficlight.setPhase(tls, next_phase)


# ----------------------------------------------------------------------
# Simple episode runners (no metrics) – kept for compatibility
# ----------------------------------------------------------------------
def run_fixed_time_episode(max_steps: int = 3600):
    """Baseline: SUMO fixed-time controller only; returns return + reward trace."""
    if traci.isLoaded():
        traci.close()

    traci.start([get_sumo_binary(False), "-c", str(CONFIG_FILE), "--step-length", "1"])

    episode_return = 0.0
    rewards_over_time = []

    for t in range(max_steps):
        traci.simulationStep()

        r = compute_global_reward(tls_lane_map)
        episode_return += r
        rewards_over_time.append(r)

        if (t + 1) % 600 == 0:
            print(f"[Fixed] Step {t+1}/{max_steps}, reward: {r:.4f}")

    traci.close()
    print("[Fixed] Episode finished. Total return:", episode_return)
    return episode_return, rewards_over_time


def run_ai_episode_greedy(max_steps: int = 3600):
    """AI controller using trained GNNActorCritic (greedy argmax)."""
    if traci.isLoaded():
        traci.close()

    traci.start([get_sumo_binary(False), "-c", str(CONFIG_FILE), "--step-length", "1"])

    episode_return = 0.0
    rewards_over_time = []

    for t in range(max_steps):
        # build state batch
        all_states = []
        for tls in tls_ids:
            s = get_tls_state(tls, tls_lane_map)
            s_padded = s + [0] * (feature_size - len(s))
            all_states.append(s_padded)

        states_np = np.array(all_states, dtype=np.float32)[None, ...]
        states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)

        policy_logits_tf, value_tf = gnn_model((states_tf, adj_batch_tf), training=False)
        policy_logits = policy_logits_tf[0]  # (N, 2)

        actions = select_actions_from_logits(policy_logits)
        apply_actions_to_sumo(actions, tls_ids)

        traci.simulationStep()

        r = compute_global_reward(tls_lane_map)
        episode_return += r
        rewards_over_time.append(r)

        if (t + 1) % 600 == 0:
            print(f"[AI] Step {t+1}/{max_steps}, reward: {r:.4f}")

    traci.close()
    print("[AI] Episode finished. Total return:", episode_return)
    return episode_return, rewards_over_time


def evaluate_policy(run_fn, label: str, num_episodes: int = 3, max_steps: int = 3600):
    returns = []
    all_traces = []

    for ep in range(num_episodes):
        print(f"\n=== {label} EPISODE {ep+1}/{num_episodes} ===")
        ep_ret, trace = run_fn(max_steps=max_steps)
        returns.append(ep_ret)
        all_traces.append(trace)
        print(f"{label} episode {ep+1} return: {ep_ret:.4f}")

    returns = np.array(returns, dtype=np.float64)
    print(f"\n=== {label} SUMMARY ===")
    print("Returns:", returns)
    print("Mean return:", returns.mean())
    print("Std return:", returns.std())

    return returns, all_traces


# ----------------------------------------------------------------------
# NEW: Detailed per-step metrics for dashboard
# ----------------------------------------------------------------------
def _collect_step_metrics(mode: str, step: int, reward: float,
                          cumulative_processed: int) -> tuple[dict, list[dict], int]:
    """
    Collects global metrics and per-TLS metrics at a single SUMO step.

    Returns:
      global_row (dict),
      tls_rows (list[dict]),
      updated cumulative_processed
    """
    total_wait_time = 0.0
    total_queue = 0
    total_veh = 0

    tls_rows = []

    # vehicles that left the network in THIS step
    processed_step = traci.simulation.getArrivedNumber()
    cumulative_processed += processed_step

    for tls in tls_ids:
        lane_ids = tls_lane_map[tls]
        tls_queue = 0
        tls_wait = 0.0
        tls_veh = 0

        for lane in lane_ids:
            q = traci.lane.getLastStepHaltingNumber(lane)
            w = traci.lane.getWaitingTime(lane)
            n = traci.lane.getLastStepVehicleNumber(lane)

            tls_queue += q
            tls_wait += w
            tls_veh += n

        total_wait_time += tls_wait
        total_queue += tls_queue
        total_veh += tls_veh

        tls_rows.append(
            {
                "mode": mode,
                "step": step,
                "tls_id": tls,
                "queue": tls_queue,
                "wait_time": tls_wait,
                "vehicles_on_lanes": tls_veh,
            }
        )

    avg_wait_time = total_wait_time / max(total_veh, 1)

    global_row = {
        "mode": mode,
        "step": step,
        "reward": reward,
        "total_wait_time": total_wait_time,
        "avg_wait_time": avg_wait_time,
        "total_queue": total_queue,
        "vehicles_processed_step": processed_step,
        "vehicles_processed_cum": cumulative_processed,
    }

    return global_row, tls_rows, cumulative_processed


def run_controller_with_metrics(
    mode: str,
    max_steps: int = 3600,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one episode with either 'fixed' or 'ai' controller and
    return:
      - global_metrics_df
      - tls_metrics_df

    This is the main entry point used by the Streamlit dashboard.
    """
    mode = mode.lower()
    if mode not in {"fixed", "ai"}:
        raise ValueError("mode must be 'fixed' or 'ai'")

    if traci.isLoaded():
        traci.close()

    traci.start([get_sumo_binary(False), "-c", str(CONFIG_FILE), "--step-length", "1"])

    global_rows = []
    tls_rows_all = []

    cumulative_processed = 0
    episode_return = 0.0

    for t in range(max_steps):
        # controller logic
        if mode == "fixed":
            # SUMO fixed-time controller: no manual action
            traci.simulationStep()
        else:
            # AI controller
            all_states = []
            for tls in tls_ids:
                s = get_tls_state(tls, tls_lane_map)
                s_padded = s + [0] * (feature_size - len(s))
                all_states.append(s_padded)

            states_np = np.array(all_states, dtype=np.float32)[None, ...]
            states_tf = tf.convert_to_tensor(states_np, dtype=tf.float32)

            policy_logits_tf, value_tf = gnn_model((states_tf, adj_batch_tf), training=False)
            policy_logits = policy_logits_tf[0]  # (N, 2)

            actions = select_actions_from_logits(policy_logits)
            apply_actions_to_sumo(actions, tls_ids)

            traci.simulationStep()

        # reward and metrics
        r = compute_global_reward(tls_lane_map)
        episode_return += r

        global_row, tls_rows, cumulative_processed = _collect_step_metrics(
            mode=mode,
            step=t + 1,
            reward=r,
            cumulative_processed=cumulative_processed,
        )
        global_rows.append(global_row)
        tls_rows_all.extend(tls_rows)

        if (t + 1) % 600 == 0:
            print(f"[{mode.upper()}] Step {t+1}/{max_steps}, reward: {r:.4f}")

    traci.close()
    print(f"[{mode.upper()}] Episode finished. Total return: {episode_return:.4f}")

    global_df = pd.DataFrame(global_rows)
    tls_df = pd.DataFrame(tls_rows_all)

    return global_df, tls_df


# ----------------------------------------------------------------------
# Optional: Console evaluation (only when run directly, NOT when imported)
# ----------------------------------------------------------------------
try:
    # for notebook environments
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x):
        print(x)


def _run_console_evaluation():
    """Recreates your existing fixed vs AI comparison and summary output."""
    fixed_returns, fixed_traces = evaluate_policy(
        run_fixed_time_episode, "Fixed", num_episodes=3, max_steps=3600
    )
    ai_returns, ai_traces = evaluate_policy(
        run_ai_episode_greedy, "AI", num_episodes=3, max_steps=3600
    )

    # ... (plots and summaries, same as before, optional for Jupyter)
    # You can keep or remove this body if you want.
    print("Console evaluation finished.")


if __name__ == "__main__":
    _run_console_evaluation()
