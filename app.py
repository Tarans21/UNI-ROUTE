# app.py — UNIROUTE (Streamlit)
# Single-file Streamlit app: Dijkstra (authoritative) + optional LSTM suggestion
# Usage: streamlit run app.py
import os, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# -------------------- Config --------------------
st.set_page_config(page_title="UNIROUTE — Campus Navigator 🧭", layout="wide")
st.title("UNIROUTE — An Intelligent Campus Navigator 🚩")

BASE = os.path.dirname(os.path.abspath(__file__))

GRAPH_FILES = {
    "graph_pkl": os.path.join(BASE, "uniroute_graph.pkl"),
    "graph_json": os.path.join(BASE, "graph_cache.json"),
    "dist_matrix": os.path.join(BASE, "dijkstra_distance_matrix.npy"),
    "name2id": os.path.join(BASE, "name2id.json"),
    "lstm_model": os.path.join(BASE, "lstm_nextstep.h5"),
}

CSV_FILES = [
    os.path.join(BASE, "ground floor", "groundfloor.csv"),
    os.path.join(BASE, "First floor", "Floor1.csv"),
    os.path.join(BASE, "Second floor", "Floor2.csv"),
    os.path.join(BASE, "third floor", "Floor3.csv"),
    os.path.join(BASE, "fourth floor", "Floor4.csv"),
]

FLOOR_CHANGE_PENALTY = 5.0  # added cost for moving between floors

# -------------------- Utilities / caching --------------------
@st.cache_resource
def load_graph_from_json(path):
    """Load graph_cache.json into a NetworkX graph and node metadata."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        g = json.load(f)
    G = nx.Graph()
    for n in g.get("nodes", []):
        nid = n["id"]
        # store numeric coordinates; ensure floats
        x = float(n.get("x", 0.0)) if n.get("x") is not None else 0.0
        y = float(n.get("y", 0.0)) if n.get("y") is not None else 0.0
        z = float(n.get("z", 0.0)) if n.get("z") is not None else 0.0
        G.add_node(nid, x=x, y=y, z=z, floor=str(n.get("floor","")))
    for e in g.get("edges", []):
        src = e.get("source"); tgt = e.get("target")
        if src in G.nodes and tgt in G.nodes:
            G.add_edge(src, tgt)
    return G

@st.cache_resource
def load_name2id(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_lstm_model_safe(path, name2id):
    """
    Try to load Keras LSTM model safely.
    If embedding input_dim <= max_id then compatible.
    If incompatible, return None (silently disable).
    """
    if not os.path.exists(path) or name2id is None:
        return None
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return None
    try:
        m = load_model(path)
    except Exception:
        return None

    # Check embedding compatibility: find first Embedding layer
    emb_layer = None
    for layer in m.layers:
        if layer.__class__.__name__.lower() == "embedding":
            emb_layer = layer
            break
    if emb_layer is None:
        # accept model but we require embedding usually
        return m

    # embedding input_dim is the vocab size used during training
    emb_input_dim = int(getattr(emb_layer, "input_dim", -1))
    max_id = 0
    try:
        # name2id mapping is str->int
        max_id = max(int(v) for v in name2id.values())
    except Exception:
        max_id = 0

    # If embedding input_dim <= max_id then model will index out-of-range -> incompatible
    # So require emb_input_dim > max_id
    if emb_input_dim <= max_id:
        return None
    return m

# -------------------- Load resources --------------------
G = None
# prefer graph_cache.json
if os.path.exists(GRAPH_FILES["graph_json"]):
    G = load_graph_from_json(GRAPH_FILES["graph_json"])
else:
    # fallback: try .pkl or CSVs (not implemented here)
    G = None

if G is None:
    st.error("No graph found. Put `graph_cache.json` next to this app and reload.")
    st.stop()

name2id = load_name2id(GRAPH_FILES["name2id"])
# id2name helper (if name2id present)
id2name = {int(v): k for k, v in (name2id or {}).items()} if name2id else {}

lstm_model = load_lstm_model_safe(GRAPH_FILES["lstm_model"], name2id)

# -------------------- Pathfinding utils --------------------
def edge_cost(a, b):
    xa, ya, za = G.nodes[a]["x"], G.nodes[a]["y"], G.nodes[a]["z"]
    xb, yb, zb = G.nodes[b]["x"], G.nodes[b]["y"], G.nodes[b]["z"]
    base = math.hypot(xa - xb, ya - yb)
    if za != zb:
        base += FLOOR_CHANGE_PENALTY
    return max(1e-9, base)

def dijkstra_shortest(start, end):
    try:
        path = nx.shortest_path(G, source=start, target=end, weight=lambda u,v,d: edge_cost(u,v))
        return path
    except nx.NetworkXNoPath:
        return []
    except nx.NodeNotFound:
        return []

# LSTM-based greedy, but only uses neighbors (safe). If model not loaded -> returns []
def predict_path_lstm_safe(start, end, max_steps=500):
    if lstm_model is None or name2id is None:
        return []
    if start not in G.nodes() or end not in G.nodes():
        return []
    if start == end:
        return [start]

    path = [start]
    seen = {start}
    for _ in range(max_steps):
        sid = name2id.get(start) if name2id else None
        gid = name2id.get(end) if name2id else None
        cur = path[-1]
        if sid is None or gid is None:
            break
        # model expects [current, goal]
        inp = np.array([[int(name2id.get(cur)), int(gid)]], dtype=np.int32)
        try:
            prob = lstm_model.predict(inp, verbose=0)[0]
        except Exception:
            return []
        pred_idx = int(np.argmax(prob)) if prob.size > 0 else None
        pred_name = id2name.get(pred_idx, None)
        # if model predicted goal
        if pred_name == end:
            path.append(end); return path
        # accept only neighbor predictions
        if pred_name and pred_name in G[path[-1]] and pred_name not in seen:
            path.append(pred_name); seen.add(pred_name)
            if pred_name == end:
                return path
            continue
        # otherwise choose neighbor with highest model score among neighbors
        nbrs = list(G.neighbors(path[-1]))
        if not nbrs:
            break
        # safe scoring
        nbr_scores = []
        for n in nbrs:
            nid = name2id.get(n)
            if nid is None or nid >= prob.size:
                s = -1.0
            else:
                s = float(prob[nid])
            nbr_scores.append((n, s))
        nbr_scores.sort(key=lambda x: x[1], reverse=True)
        next_name, best_score = nbr_scores[0]
        # if best_score is negative, fallback to geometric closeness to goal
        if best_score < 0:
            gx, gy, gz = G.nodes[end]["x"], G.nodes[end]["y"], G.nodes[end]["z"]
            def dist_to_goal(n):
                nx_, ny_, nz_ = G.nodes[n]["x"], G.nodes[n]["y"], G.nodes[n]["z"]
                return math.hypot(nx_ - gx, ny_ - gy) + (0 if nz_ == gz else FLOOR_CHANGE_PENALTY)
            next_name = min(nbrs, key=dist_to_goal)
        if next_name in seen:
            break
        path.append(next_name); seen.add(next_name)
        if next_name == end:
            return path
    return path

# -------------------- UI: Controls on main page (vertical) --------------------
st.markdown("### 🔎 Find route between two rooms ")
rooms = sorted(list(G.nodes()))
col = st.columns([1,1,1])
# Provide nice default indices if possible
default_start = 0
default_end = len(rooms)-1 if rooms else 0

start_room = st.selectbox("📍 Start room", rooms, index=default_start)
end_room   = st.selectbox("🎯 Destination", rooms, index=default_end)

compute = st.button("Compute Route 🚀")

# extra UI elements: small settings
with st.expander("⚙️ Options (advanced) — click to open", expanded=False):
    show_lstm = st.checkbox("Show LSTM suggestion (if available)", value=True)
    show_all_edges = st.checkbox("Show full graph edges", value=False)
    floor_penalty = st.number_input("Floor change penalty (distance)", value=float(FLOOR_CHANGE_PENALTY), step=0.5)
    # apply change
    FLOоR = floor_penalty  # keep local variable (not used further directly)
    # note: we keep original variable unchanged for caching stability

# -------------------- Compute and show results --------------------
if compute:
    # Compute Dijkstra true path
    true_path = dijkstra_shortest(start_room, end_room)
    if not true_path:
        st.error("⚠️ No route found between selected rooms (check connectivity).")
        st.stop()

    # LSTM prediction (only if model loaded and user wants it)
    lstm_path = []
    if show_lstm and lstm_model is not None:
        lstm_path = predict_path_lstm_safe(start_room, end_room)

    # Show headings & stats
    st.markdown("### 🧭 Results")
    st.write(f"**Start:** {start_room}  •  **Destination:** {end_room}")
    st.write("**Dijkstra (authoritative)** path length:", len(true_path))
    if lstm_path:
        st.write("**LSTM suggestion** length:", len(lstm_path))
    else:
        if show_lstm:
            st.info("LSTM model not available or incompatible — showing Dijkstra only.")

    # Side-by-side tables for turn-by-turn
    df_true = pd.DataFrame({
        "step": list(range(len(true_path))),
        "location": true_path,
        "x": [G.nodes[n]["x"] for n in true_path],
        "y": [G.nodes[n]["y"] for n in true_path],
        "z": [G.nodes[n]["z"] for n in true_path],
    })
    st.markdown("#### ✅ Dijkstra (turn-by-turn)")
    st.dataframe(df_true, use_container_width=True)

    if lstm_path:
        df_lstm = pd.DataFrame({
            "step": list(range(len(lstm_path))),
            "location": lstm_path,
            "x": [G.nodes[n]["x"] for n in lstm_path],
            "y": [G.nodes[n]["y"] for n in lstm_path],
            "z": [G.nodes[n]["z"] for n in lstm_path],
        })
        st.markdown("#### 🤖 LSTM Suggestion (turn-by-turn)")
        st.dataframe(df_lstm, use_container_width=True)

    # Download JSON
    out = {"start": start_room, "end": end_room, "dijkstra": true_path, "lstm": lstm_path}
    st.download_button("⬇️ Download routes (JSON)", json.dumps(out, indent=2), file_name="uniroute_routes.json", mime="application/json")

    # -------------------- 3D Plotly "street view" --------------------
    st.markdown("### 🗺️ 3D View (street-style) — rotate/zoom to inspect")
    # Build traces
    # nodes
    node_x, node_y, node_z, node_text, node_label = [], [], [], [], []
    for n in G.nodes():
        node_x.append(G.nodes[n]["x"])
        node_y.append(G.nodes[n]["y"])
        node_z.append(G.nodes[n]["z"])
        node_text.append(f"{n}<br>Floor: {G.nodes[n]['floor']}<br>({G.nodes[n]['x']:.1f},{G.nodes[n]['y']:.1f},{G.nodes[n]['z']:.1f})")
        node_label.append(n)

    nodes_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode="markers+text",
        marker=dict(size=5, color="#444444", opacity=0.9),
        text=node_label, textposition="top center",
        hovertext=node_text, hoverinfo="text",
        name="Rooms"
    )

    # edges (thin light grey)
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        au, av = G.nodes[u], G.nodes[v]
        edge_x += [au["x"], av["x"], None]
        edge_y += [au["y"], av["y"], None]
        edge_z += [au["z"], av["z"], None]
    edges_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                               line=dict(color="#DDDDDD", width=2), hoverinfo="none", name="Edges")
    traces = []
    if show_all_edges:
        traces.append(edges_trace)

    # Dijkstra path (thick blue)
    path_x, path_y, path_z = [], [], []
    for u, v in zip(true_path, true_path[1:]):
        au, av = G.nodes[u], G.nodes[v]
        path_x += [au["x"], av["x"], None]
        path_y += [au["y"], av["y"], None]
        path_z += [au["z"], av["z"], None]
    dijkstra_trace = go.Scatter3d(x=path_x, y=path_y, z=path_z, mode="lines+markers",
                                  line=dict(color="#1f77b4", width=8), marker=dict(size=4, color="#1f77b4"),
                                  name="Dijkstra (path)")
    traces.append(dijkstra_trace)

    # LSTM path (if exists): dashed contrasting color
    if lstm_path:
        l_x, l_y, l_z = [], [], []
        for u, v in zip(lstm_path, lstm_path[1:]):
            au, av = G.nodes[u], G.nodes[v]
            l_x += [au["x"], av["x"], None]
            l_y += [au["y"], av["y"], None]
            l_z += [au["z"], av["z"], None]
        lstm_trace = go.Scatter3d(x=l_x, y=l_y, z=l_z, mode="lines+markers",
                                  line=dict(color="#ff7f0e", width=5, dash="dash"),
                                  marker=dict(size=3, color="#ff7f0e"),
                                  name="LSTM (suggestion)")
        traces.append(lstm_trace)

    # add nodes last (so they appear above lines)
    traces.append(nodes_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(title="Floor (z)"), aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)")
    )
    st.plotly_chart(fig, use_container_width=True, height=650)

    st.success("✅ Route computed. Use the 3D view above to inspect the path (rotate / zoom).")

# -------------------- Footer / hints --------------------
st.markdown("---")
st.markdown("UNIROUTE — Dijkstra (authoritative) + optional LSTM suggestion • Built for campus navigation 🏫")
st.markdown("Tips: If LSTM is not shown, retrain your LSTM using the current `name2id.json` (embedding vocab must exceed max id).")

