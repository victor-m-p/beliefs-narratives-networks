"""
VMP 2026-02-06 (refactored):

Basic reliability across waves:
1) Panel of 3 scatterplots:
   - interview words (W1 vs W2)
   - LLM-extracted nodes (W1 vs W2; no limit)
   - human canvas edges (W1 vs W2)

2) Reliability table for network metrics (human networks; embeddings pipeline output).

Assumes these exist:
- public/interviews_w*.csv (NOTE: not yet generated - need Phase 0 script)
- public/edges_canvas_w*.csv
- public/llm_extractions/node_extraction_w*/<model>/*.json
- Embeddings data (currently gitignored)

VMP 2026-02-07: tested and run.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy.stats import pearsonr
import networkx as nx

from utilities import wave_1, wave_2, get_public_path, get_llm_extraction_path

# -------------------------
# config
# -------------------------
MODEL = "gpt-4.1-2025-04-14"

outdir = "../fig/consistency"
os.makedirs(outdir, exist_ok=True)

# -------------------------
# small helpers
# -------------------------
def wide_from_agg(df, key="key", wave="wave", value="value"):
    w = (
        df.pivot(index=key, columns=wave, values=value)
          .rename(columns={"1": "wave1", "2": "wave2"})
          .dropna(subset=["wave1", "wave2"])
          .reset_index()
    )
    w.columns.name = None
    return w[[key, "wave1", "wave2"]]

def collect_llm_json_dir(directory):
    rows = []
    for fn in os.listdir(directory):
        if not fn.endswith(".json"):
            continue
        key = os.path.splitext(fn)[0]
        with open(os.path.join(directory, fn), "r", encoding="utf-8") as f:
            data = json.load(f)  # list of dicts
        rows.append({"key": key, "n": len(data)})
    return pd.DataFrame(rows)

def scatter_ax(
    ax, df, xlabel, ylabel, title,
    jitter=0.0, alpha=0.5,
    tick_step=None,      # e.g. 1 or 5; if None uses integer locator
    include_zero=True,   # force limits to include 0
):
    # data (optionally jitter for visibility only)
    x0 = df["wave1"].to_numpy()
    y0 = df["wave2"].to_numpy()
    if jitter and jitter > 0:
        x = x0 + np.random.normal(0, jitter, size=len(x0))
        y = y0 + np.random.normal(0, jitter, size=len(y0))
    else:
        x, y = x0, y0

    ax.scatter(x, y, s=18, alpha=alpha)

    # --- square limits: same span + (optional) include zero ---
    lo = min(df["wave1"].min(), df["wave2"].min())
    hi = max(df["wave1"].max(), df["wave2"].max())

    if include_zero:
        lo = min(lo, 0)
        hi = max(hi, 0)

    # make them clean integers (important for nodes/edges)
    lo = int(np.floor(lo))
    hi = int(np.ceil(hi))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # identity line
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")

    # --- integer ticks ---
    if tick_step is None:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ticks = np.arange(lo, hi + 1, tick_step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    # correlation annotation (on raw, not jittered)
    r, _ = pearsonr(df["wave1"], df["wave2"])
    ax.text(0.05, 0.95, f"$r$ = {r:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

# -------------------------
# 1) panel data
# -------------------------

# (A) interview words (from interviews.csv)
# NOTE: interviews_w*.csv needs to be generated in Phase 0 from private data
try:
    iw1 = pd.read_csv(get_public_path(f"interviews_w{wave_1}.csv"))
    iw2 = pd.read_csv(get_public_path(f"interviews_w{wave_2}.csv"))
    iw1["wave"] = "1"
    iw2["wave"] = "2"
    interview = pd.concat([iw1, iw2], ignore_index=True)
    interview_agg = interview.groupby(["key", "wave"])["words_a"].sum().reset_index(name="words")
    words_wide = wide_from_agg(interview_agg, value="words")
except FileNotFoundError:
    print("Warning: interviews_w*.csv not found (needs Phase 0 generation from private data)")
    words_wide = pd.DataFrame(columns=["key", "Wave 1", "Wave 2"])

# (B) LLM nodes (count JSON length per participant)
llm_node_dir_w1 = get_llm_extraction_path(wave_1, "node_extraction", MODEL)
llm_node_dir_w2 = get_llm_extraction_path(wave_2, "node_extraction", MODEL)

llm_nodes_w1 = collect_llm_json_dir(llm_node_dir_w1).rename(columns={"n": "nodes"})
llm_nodes_w2 = collect_llm_json_dir(llm_node_dir_w2).rename(columns={"n": "nodes"})
llm_nodes_w1["wave"] = "1"
llm_nodes_w2["wave"] = "2"
llm_nodes = pd.concat([llm_nodes_w1, llm_nodes_w2], ignore_index=True)

nodes_wide = wide_from_agg(llm_nodes, value="nodes")

# (C) human edges (from saved edges_canvas.csv)
e1 = pd.read_csv(get_public_path(f"edges_canvas_w{wave_1}.csv"))
e2 = pd.read_csv(get_public_path(f"edges_canvas_w{wave_2}.csv"))

num_e1 = e1.groupby("key").size().reset_index(name="edges")
num_e2 = e2.groupby("key").size().reset_index(name="edges")
num_e1["wave"] = "1"
num_e2["wave"] = "2"
num_e = pd.concat([num_e1, num_e2], ignore_index=True)

edges_wide = wide_from_agg(num_e, value="edges")

# -------------------------
# 2) 1x3 panel
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

scatter_ax(axes[0], words_wide,
           xlabel="Words (W1)", ylabel="Words (W2)",
           title="Interview length",
           jitter=0.0,
           tick_step=500)   # adjust to taste (100/200/250)

scatter_ax(axes[1], nodes_wide,
           xlabel="Nodes (W1)", ylabel="Nodes (W2)",
           title="Extracted nodes",
           jitter=0.05,
           tick_step=5)     # force integer grid

scatter_ax(axes[2], edges_wide,
           xlabel="Edges (W1)", ylabel="Edges (W2)",
           title="Canvas edges",
           jitter=0.05,
           tick_step=5)     # force integer grid

plt.tight_layout()
plt.savefig(os.path.join(outdir, "reliability.pdf"))

''' CONSIDER WHETHER WE WANT THE BELOW:

# -------------------------
# 3) network metric reliability (human; embeddings pipeline)
# -------------------------
nodes_all = pd.read_csv("../data/public/embeddings/nodes.csv")
edges_hum = pd.read_csv("../data/public/embeddings/edge_hum.csv")

# id = key_wave to compare across waves later
nodes_all["id"] = nodes_all["key"].astype(str) + "_" + nodes_all["wave"].astype(str)
edges_hum["id"] = edges_hum["key"].astype(str) + "_" + edges_hum["wave"].astype(str)

# only canvas nodes (include isolates)
nodes_canvas = nodes_all[nodes_all["canvas"] == True].copy()

def build_graph(edges_df, nodes_df, id_value):
    G = nx.Graph()
    nsub = nodes_df[nodes_df["id"] == id_value]
    esub = edges_df[edges_df["id"] == id_value]

    for _, r in nsub.iterrows():
        G.add_node(r["stance"])

    for _, r in esub.iterrows():
        s1, s2 = r.get("stance_1"), r.get("stance_2")
        if pd.notna(s1) and pd.notna(s2) and s1 != s2:
            G.add_edge(s1, s2)
    return G

def metrics_from_graph(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    n_iso = nx.number_of_isolates(G)
    n_comp = nx.number_connected_components(G) if n else 0
    dens = nx.density(G) if n > 1 else 0.0

    # GCC
    if n and m:
        gcc_nodes = max(nx.connected_components(G), key=len)
        H = G.subgraph(gcc_nodes).copy()
    else:
        H = nx.Graph()

    gn = H.number_of_nodes()
    gm = H.number_of_edges()
    gdens = nx.density(H) if gn > 1 else 0.0

    if gn >= 2:
        apl = nx.average_shortest_path_length(H)
        diam = nx.diameter(H)
    else:
        apl = np.nan
        diam = np.nan

    trans = nx.transitivity(H) if gn >= 3 else np.nan
    clust = nx.average_clustering(H) if gn >= 2 else np.nan

    return dict(
        n_nodes=n,
        n_edges=m,
        n_isolates=n_iso,
        n_components=n_comp,
        density=dens,
        gcc_nodes=gn,
        gcc_edges=gm,
        gcc_density=gdens,
        gcc_avg_path_length=apl,
        gcc_diameter=diam,
        gcc_transitivity=trans,
        gcc_avg_clustering=clust,
    )

all_ids = pd.Index(nodes_canvas["id"]).union(edges_hum["id"]).dropna().unique()
rows = []
for idv in all_ids:
    G = build_graph(edges_hum, nodes_canvas, idv)
    d = metrics_from_graph(G)
    d["id"] = idv
    rows.append(d)

net = pd.DataFrame(rows)

# split id -> pid + wave
net["pid"] = net["id"].str.rsplit("_", n=1).str[0]
net["wave"] = net["id"].str.rsplit("_", n=1).str[1].astype(int)

metrics = [c for c in net.columns if c not in ["id", "pid", "wave"]]

out_rows = []
for metric in metrics:
    wide = net.pivot(index="pid", columns="wave", values=metric)
    if 1 in wide.columns and 2 in wide.columns:
        x = wide[1]
        y = wide[2]
        corr = x.corr(y)
    else:
        corr = np.nan

    out_rows.append(dict(
        metric=metric,
        correlation=corr,
        r_squared=(corr ** 2) if pd.notna(corr) else np.nan,
        overall_mean=net[metric].mean(skipna=True),
        overall_sd=net[metric].std(skipna=True),
    ))

reliability = pd.DataFrame(out_rows).sort_values("metric")
reliability.to_csv(os.path.join(outdir, "network_metric_reliability.csv"), index=False)

latex = reliability.to_latex(
    index=False,
    float_format="%.3f",
    na_rep="",
    caption="Reliability and descriptives of network measures (human networks).",
    label="tab:reliability_network_metrics",
    escape=True,
)

with open(os.path.join(outdir, "network_metric_reliability.tex"), "w") as f:
    f.write(latex)

'''