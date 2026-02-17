"""
VMP 2026-02-17: not in preprint.

Population-level topic network 

Aggregates per-participant canvas edges into a single topic-level network
where:
- Nodes = BERTopic topics, sized by how many stances map to them (across all participants)
- Edges = weighted by how many participants drew a canvas connection between
  stances belonging to those two topics, colored by dominant polarity

Layout: ring (circular) arrangement of topics.

VMP 2026-02-11
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from utilities import wave_1, wave_2, get_public_path

# -------------------------
# Config
# -------------------------
WAVES = [1, 2]  # [1, 2] for both, [1] or [2] for single wave

# setup outdir
OUTDIR = Path("../fig/population_topic_network")
OUTDIR.mkdir(parents=True, exist_ok=True)

# bertopic loads
bertopic_dir = Path("../data/public/bertopic/selection")
selection_dir = bertopic_dir / "statement_topics"
best_model = sorted(selection_dir.glob("01__*__statement_topics.csv"))[0]

# Statement-to-topic mapping
df_statements_topics = pd.read_csv(best_model)[["key", "wave", "stance", "topic"]]
#df_statements_topics = df_statements_topics[df_statements_topics["topic"] != -1]

# Topic overview (labels)
overview_dir = bertopic_dir / "overview"
overview_file = sorted(overview_dir.glob("01__*__topic_info.csv"))[0]
df_topic_overview = pd.read_csv(overview_file)[["Topic", "Representation"]]

# -------------------------
# Helper functions
# -------------------------
def load_canvas_data(waves):
    """Load distractors JSON for each wave. Returns {wave: dict_of_participants}."""
    data_by_wave = {}
    for w in waves:
        wave_id = wave_1 if w == 1 else wave_2
        path = get_public_path("distractors_w{wave}.json", wave=wave_id)
        with open(path, encoding="utf-8") as f:
            data_by_wave[w] = json.load(f)
    return data_by_wave

def extract_canvas_stances(data_by_wave, waves):
    """
    From distractors JSON, extract all stances placed on canvas.
    Returns DataFrame with columns [key, wave, stance].
    """
    rows = []
    for w in waves:
        for key, d in data_by_wave[w].items():
            for node in d["positions"]["pos_3"]:
                rows.append({"key": key, "wave": w, "stance": str(node["label"]).strip()})
    return pd.DataFrame(rows)

def extract_canvas_edges(data_by_wave, waves):
    """
    From distractors JSON, extract all canvas edges with polarity.
    Returns DataFrame with columns [key, wave, stance_1, stance_2, polarity].
    """
    rows = []
    for w in waves:
        for key, d in data_by_wave[w].items():
            for e in d["edges"]["edges_3"]:
                pol = e.get("polarity")
                if pol not in ("positive", "negative"):
                    continue
                rows.append({
                    "key": key,
                    "wave": w,
                    "stance_1": str(e["stance_1"]).strip(),
                    "stance_2": str(e["stance_2"]).strip(),
                    "polarity": pol,
                })
    return pd.DataFrame(rows)

def assign_topics(df, df_topics, stance_col="stance"):
    """
    Merge a stance-level DataFrame with the topic assignments.
    Joins on [key, wave, stance].
    """
    return df.merge(
        df_topics[["key", "wave", "stance", "topic"]],
        left_on=["key", "wave", stance_col],
        right_on=["key", "wave", "stance"],
        how="left",
    )


# load data
data_by_wave = load_canvas_data(WAVES)

df_canvas_stances = extract_canvas_stances(data_by_wave, WAVES)
df_canvas_edges = extract_canvas_edges(data_by_wave, WAVES)

# assign topics
df_canvas_stances = assign_topics(df_canvas_stances, df_statements_topics, stance_col="stance")

# For edges: assign topic to each side separately
df_canvas_edges = df_canvas_edges.merge(
    df_statements_topics[["key", "wave", "stance", "topic"]].rename(
        columns={"stance": "stance_1", "topic": "topic_1"}
    ),
    on=["key", "wave", "stance_1"],
    how="left",
)
df_canvas_edges = df_canvas_edges.merge(
    df_statements_topics[["key", "wave", "stance", "topic"]].rename(
        columns={"stance": "stance_2", "topic": "topic_2"}
    ),
    on=["key", "wave", "stance_2"],
    how="left",
)

# here we could filter out just one wave.
# df_canvas_stances = df_canvas_stances[df_canvas_stances["wave"] == 1]
# df_canvas_edges = df_canvas_edges[df_canvas_edges["wave"] == 1]

# here we could also filter out -1 topic
# this removes a huge part of the connections
remove_outlier = True
if remove_outlier:
    df_canvas_stances = df_canvas_stances[df_canvas_stances["topic"] != -1]
    df_canvas_edges = df_canvas_edges[
        (df_canvas_edges["topic_1"] != -1) &
        (df_canvas_edges["topic_2"] != -1)
    ]

### aggregation ###

# cast topics to int (left joins introduce NaN → float; safe after filtering)
df_canvas_stances = df_canvas_stances.dropna(subset=["topic"])
df_canvas_stances["topic"] = df_canvas_stances["topic"].astype(int)
df_canvas_edges = df_canvas_edges.dropna(subset=["topic_1", "topic_2"])
df_canvas_edges["topic_1"] = df_canvas_edges["topic_1"].astype(int)
df_canvas_edges["topic_2"] = df_canvas_edges["topic_2"].astype(int)

# Node sizes: count canvas stances per topic
topic_counts = df_canvas_stances.groupby("topic").size().to_dict()

# For canvas edges, sort the two columns to get unique pairs
df_canvas_edges[["topic_1", "topic_2"]] = np.sort(df_canvas_edges[["topic_1", "topic_2"]], axis=1)

# Drop the stances for now (out of order with topics and we do not need them for this)
df_canvas_edges = df_canvas_edges[["key", "wave", "topic_1", "topic_2", "polarity"]]

# Now we just aggregate (like for previous plots.)
df_canvas_edges["pol_num"] = df_canvas_edges["polarity"].map({"positive": 1, "negative": -1})

df_topic_edges = (
    df_canvas_edges
    .groupby(["topic_1", "topic_2"])["pol_num"]
    .agg(weight="count", polarity_mean="mean")
    .reset_index()
)

### build graph ###
G = nx.Graph()
for _, row in df_topic_edges.iterrows():
    t1, t2 = int(row["topic_1"]), int(row["topic_2"])
    weight = row["weight"]
    pol_mean = row["polarity_mean"]
    if pol_mean > 0:
        polarity = "positive"
    elif pol_mean < 0:
        polarity = "negative"
    else:
        polarity = "both"
    G.add_edge(t1, t2, weight=weight, polarity=polarity, polarity_mean=pol_mean)

G.add_nodes_from(topic_counts.keys())
nx.set_node_attributes(G, topic_counts, "count")

### filter and sort graph ###
# filter graph 
MIN_EDGE_WEIGHT = 12
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < MIN_EDGE_WEIGHT]
G.remove_edges_from(edges_to_remove)

# --- layout ---
# pos = nx.kamada_kawai_layout(G, weight="weight")

# spectral ring: order nodes by Fiedler vector, place on circle
ordered_topics = nx.spectral_ordering(G, normalized=True)
n = len(ordered_topics)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos = {ordered_topics[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

# plot parameters 
EDGE_COLORS = {"positive": "#998ec3", "negative": "#f1a340", "both": "#1b9e77"}

# scale node size 
counts = np.array([G.nodes[t]["count"] for t in G.nodes])
node_sizes = counts / (counts.max() / 3000)  # scale to max size of 800

# scale edge width
weights = np.array([G.edges[u, v]["weight"] for u, v in G.edges], dtype=float)
edge_widths = weights / (weights.max() / 20)

# edge color by polarity  
edge_colors = [EDGE_COLORS[G.edges[u, v]["polarity"]] for u, v in G.edges]

# node color by topic
cmap = plt.get_cmap("tab20")
node_color_map = {ordered_topics[i]: cmap(i % 20) for i in range(n)}
node_colors = [node_color_map[t] for t in G.nodes]

# node labels (just topic ID)
labels = {t: f"T{t}" for t in G.nodes}

# build legend text from df_topic_overview
# Representation is stored as a string repr of a list, so we eval it
topic_legend = {}
for _, row in df_topic_overview.iterrows():
    t = int(row["Topic"])
    rep = row["Representation"]
    if isinstance(rep, str):
        words = eval(rep)[:3]
    else:
        words = list(rep)[:3]
    topic_legend[t] = ", ".join(words)

fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(26, 12),
                                  gridspec_kw={"width_ratios": [3, 1]})
ax.set_aspect("equal")
ax.set_axis_off()
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       edgecolors="black", linewidths=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                       alpha=0.7, ax=ax)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, ax=ax)
ax.set_axis_off()

# legend panel
ax_leg.set_axis_off()
legend_lines = []
for t in sorted(G.nodes()):
    kw = topic_legend.get(t, "")
    cnt = G.nodes[t].get("count", 0)
    legend_lines.append(f"T{t} ({cnt}): {kw}")
ax_leg.text(0, 1, "\n".join(legend_lines), transform=ax_leg.transAxes,
            fontsize=24, verticalalignment="top", fontfamily="monospace")

fig.tight_layout()

'''
Things that would be cool to do: 
1. Get a better topic model (now changed a bit.)
2. Get "significant" edges instead of naive filtering.
3. Try to do it based on co-occurence of topics.
4. Do it with LLM edges.
5. Try different layouts. 

What we are learning: 
- people do not talk about beliefs / morals / ethics a lot.
- most of what we have is behavior, family, taste, ...
'''

### overlay one individual on the population network ###
# pick one random person from wave 1
sample_key = list(data_by_wave[1].keys())[0]
sample_wave = 1

# filter per-participant edges (df_canvas_edges still has key, wave, topic_1, topic_2, polarity)
ind_edges = df_canvas_edges[
    (df_canvas_edges["key"] == sample_key) &
    (df_canvas_edges["wave"] == sample_wave)
].copy()

# aggregate this individual's topic edges
ind_topic_edges = (
    ind_edges
    .groupby(["topic_1", "topic_2"])["pol_num"]
    .agg(weight="count", polarity_mean="mean")
    .reset_index()
)

# which topics does this person touch?
ind_stances = df_canvas_stances[
    (df_canvas_stances["key"] == sample_key) &
    (df_canvas_stances["wave"] == sample_wave)
]
ind_topics = set(ind_stances["topic"].dropna().astype(int))

# --- new figure: population faded + individual highlighted ---
fig2, (ax2, ax2_leg) = plt.subplots(1, 2, figsize=(26, 12),
                                     gridspec_kw={"width_ratios": [3, 1]})
ax2.set_aspect("equal")
ax2.set_axis_off()

# background: population nodes (light gray)
pop_node_colors = ["#e0e0e0"] * G.number_of_nodes()
nx.draw_networkx_nodes(G, pos, node_color=pop_node_colors, node_size=node_sizes,
                       edgecolors="#cccccc", linewidths=0.5, ax=ax2)

# background: population edges (very faint)
nx.draw_networkx_edges(G, pos, edge_color="#e0e0e0", width=edge_widths,
                       alpha=0.3, ax=ax2)

# highlight: individual's topics (colored, with border)
ind_node_list = [t for t in G.nodes if t in ind_topics]
ind_node_sizes = [node_sizes[list(G.nodes).index(t)] for t in ind_node_list]
ind_node_cols = [node_color_map[t] for t in ind_node_list]
if ind_node_list:
    nx.draw_networkx_nodes(G, pos, nodelist=ind_node_list,
                           node_color=ind_node_cols, node_size=ind_node_sizes,
                           edgecolors="black", linewidths=1.5, ax=ax2)

# highlight: individual's edges (colored by polarity)
ind_edge_list = []
ind_edge_colors = []
ind_edge_widths = []
for _, row in ind_topic_edges.iterrows():
    t1, t2 = int(row["topic_1"]), int(row["topic_2"])
    if not G.has_node(t1) or not G.has_node(t2):
        continue
    pol_mean = row["polarity_mean"]
    pol = "positive" if pol_mean > 0 else ("negative" if pol_mean < 0 else "both")
    ind_edge_list.append((t1, t2))
    ind_edge_colors.append(EDGE_COLORS[pol])
    ind_edge_widths.append(max(row["weight"] * 2, 3))

if ind_edge_list:
    nx.draw_networkx_edges(G, pos, edgelist=ind_edge_list,
                           edge_color=ind_edge_colors, width=ind_edge_widths,
                           alpha=0.9, ax=ax2)

# labels (all topics, but bold the individual's)
nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, ax=ax2)

ax2.set_title(f"Individual: {sample_key} (wave {sample_wave})", fontsize=16)

# legend panel (same as population)
ax2_leg.set_axis_off()
ax2_leg.text(0, 1, "\n".join(legend_lines), transform=ax2_leg.transAxes,
             fontsize=24, verticalalignment="top", fontfamily="monospace")

fig2.tight_layout()

# =========================================================================
# LLM edges: same pipeline as above but using d["LLM"]["edge_results"]
# =========================================================================

def extract_llm_edges(data_by_wave, waves):
    """
    From distractors JSON, extract LLM-created edges with polarity.
    Returns DataFrame with columns [key, wave, stance_1, stance_2, polarity].
    """
    rows = []
    for w in waves:
        for key, d in data_by_wave[w].items():
            for e in (d.get("LLM", {}).get("edge_results") or []):
                pol = e.get("polarity")
                if pol not in ("positive", "negative"):
                    continue
                rows.append({
                    "key": key,
                    "wave": w,
                    "stance_1": str(e["stance_1"]).strip(),
                    "stance_2": str(e["stance_2"]).strip(),
                    "polarity": pol,
                })
    return pd.DataFrame(rows)

# extract LLM edges (reuse data_by_wave already loaded above)
df_llm_edges = extract_llm_edges(data_by_wave, WAVES)

# assign topics to each side
df_llm_edges = df_llm_edges.merge(
    df_statements_topics[["key", "wave", "stance", "topic"]].rename(
        columns={"stance": "stance_1", "topic": "topic_1"}
    ),
    on=["key", "wave", "stance_1"],
    how="left",
)
df_llm_edges = df_llm_edges.merge(
    df_statements_topics[["key", "wave", "stance", "topic"]].rename(
        columns={"stance": "stance_2", "topic": "topic_2"}
    ),
    on=["key", "wave", "stance_2"],
    how="left",
)

# filter outlier topic (same as human edges)
if remove_outlier:
    df_llm_edges = df_llm_edges[
        (df_llm_edges["topic_1"] != -1) &
        (df_llm_edges["topic_2"] != -1)
    ]

# cast to int
df_llm_edges = df_llm_edges.dropna(subset=["topic_1", "topic_2"])
df_llm_edges["topic_1"] = df_llm_edges["topic_1"].astype(int)
df_llm_edges["topic_2"] = df_llm_edges["topic_2"].astype(int)

# sort topic pairs
df_llm_edges[["topic_1", "topic_2"]] = np.sort(df_llm_edges[["topic_1", "topic_2"]], axis=1)
df_llm_edges = df_llm_edges[["key", "wave", "topic_1", "topic_2", "polarity"]]

# aggregate
df_llm_edges["pol_num"] = df_llm_edges["polarity"].map({"positive": 1, "negative": -1})

df_llm_topic_edges = (
    df_llm_edges
    .groupby(["topic_1", "topic_2"])["pol_num"]
    .agg(weight="count", polarity_mean="mean")
    .reset_index()
)

# build LLM graph
G_llm = nx.Graph()
for _, row in df_llm_topic_edges.iterrows():
    t1, t2 = int(row["topic_1"]), int(row["topic_2"])
    weight = row["weight"]
    pol_mean = row["polarity_mean"]
    if pol_mean > 0:
        polarity = "positive"
    elif pol_mean < 0:
        polarity = "negative"
    else:
        polarity = "both"
    G_llm.add_edge(t1, t2, weight=weight, polarity=polarity, polarity_mean=pol_mean)

# add all topic nodes (same set as human graph)
G_llm.add_nodes_from(topic_counts.keys())
nx.set_node_attributes(G_llm, topic_counts, "count")

# filter edges (same threshold)
llm_edges_to_remove = [(u, v) for u, v, d in G_llm.edges(data=True) if d["weight"] < MIN_EDGE_WEIGHT]
G_llm.remove_edges_from(llm_edges_to_remove)

# reuse same layout (pos) from human graph for direct comparison
# scale LLM edge widths
llm_weights = np.array([G_llm.edges[u, v]["weight"] for u, v in G_llm.edges], dtype=float)
llm_edge_widths = llm_weights / (llm_weights.max() / 20) if llm_weights.size > 0 else []
llm_edge_colors = [EDGE_COLORS[G_llm.edges[u, v]["polarity"]] for u, v in G_llm.edges]

# plot LLM topic network
fig3, (ax3, ax3_leg) = plt.subplots(1, 2, figsize=(26, 12),
                                     gridspec_kw={"width_ratios": [3, 1]})
ax3.set_aspect("equal")
ax3.set_axis_off()
nx.draw_networkx_nodes(G_llm, pos, node_color=node_colors, node_size=node_sizes,
                       edgecolors="black", linewidths=0.8, ax=ax3)
nx.draw_networkx_edges(G_llm, pos, edge_color=llm_edge_colors, width=llm_edge_widths,
                       alpha=0.7, ax=ax3)
nx.draw_networkx_labels(G_llm, pos, labels=labels, font_size=20, ax=ax3)
ax3.set_title("Population topic network (LLM edges)", fontsize=16)

# legend panel
ax3_leg.set_axis_off()
ax3_leg.text(0, 1, "\n".join(legend_lines), transform=ax3_leg.transAxes,
             fontsize=24, verticalalignment="top", fontfamily="monospace")

fig3.tight_layout()
