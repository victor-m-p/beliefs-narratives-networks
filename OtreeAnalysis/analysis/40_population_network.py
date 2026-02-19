"""
VMP 2026-02-19

Population-level topic network (human canvas edges)

Adds opportunity-based normalization + simple configurable filtering.

Key outputs:
- df_scored: all topic-pairs with m_ij, k_ij, p_obs (=k_ij/m_ij), delta, pval
- df_keep: filtered edges
- G: topic graph with node attr "count" and edge attrs incl. p_obs/delta/pval/polarity_mean

Reproducibility:
- spectral ring ordering stabilized (sorted nodes + anchored rotation + optional flip fix)
- NOTE: spectral ordering is deterministic given node order; SEED is mainly for future layouts

"""

from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import binom
from matplotlib.lines import Line2D

from utilities import wave_1, wave_2, get_public_path


# -------------------------
# Config (single place)
# -------------------------
CFG = dict(
    WAVES=[1, 2],
    REMOVE_OUTLIER=True,     # drop topic == -1
    KEEP_ISOLATES=True,      # keep topics with no kept edges (draw them on ring)
    SEED=123,                # for reproducibility / future random choices

    # filtering
    FILTER_MODE="delta",     # "alpha_delta" | "delta"
    ALPHA=0.1,
    MIN_DELTA=0.15,          # e.g. 0.0 for "p_obs > p_hat", or 0.15 like your example
    MIN_M=10,                # minimum opportunities

    # edge thickness mode
    EDGE_WIDTH="k_ij",       # "p_obs" | "k_ij" | "delta"

    # proportional edge-width scaling (NO floor, no normalization)
    WIDTH_SCALE=2.0,         # linewidth = WIDTH_SCALE * (k_ij or p_obs or delta)

    # layout
    ENFORCE_FLIP=True,       # remove possible mirror flip in spectral ordering (deterministic)

    # plotting
    SHOW_EDGE_LEGEND=False,   # show legend for edge polarity colors

    OUTDIR=Path("../fig/population_topic_network"),
)

''' REASONABLE SETTINGS:
- FILTER_MODE="delta", ALPHA ignored, MIN_DELTA=0.15, MIN_M=10, EDGE_WIDTH="k_ij", WIDTH_SCALE=2, KEEP_ISOLATES=True
- FILTER_MODE="delta", ALPHA ignored, MIN_DELTA=0.15, MIN_M=10, EDGE_WIDTH="p_obs", WIDTH_SCALE=50, KEEP_ISOLATES=True
'''


# -------------------------
# Setup
# -------------------------
np.random.seed(CFG["SEED"])  # mostly future-proof; spectral ring itself is deterministic

CFG["OUTDIR"].mkdir(parents=True, exist_ok=True)

bertopic_dir = Path("../data/public/bertopic/selection")
selection_dir = bertopic_dir / "statement_topics"
best_model = sorted(selection_dir.glob("01__*__statement_topics.csv"))[0]

df_statements_topics = pd.read_csv(best_model)[["key", "wave", "stance", "topic"]]

overview_dir = bertopic_dir / "overview"
overview_file = sorted(overview_dir.glob("01__*__topic_info.csv"))[0]
df_topic_overview = pd.read_csv(overview_file)[["Topic", "Representation"]]


# -------------------------
# Helpers: load + extract + topic assignment
# -------------------------
def load_canvas_data(waves):
    data_by_wave = {}
    for w in waves:
        wave_id = wave_1 if w == 1 else wave_2
        path = get_public_path("distractors_w{wave}.json", wave=wave_id)
        with open(path, encoding="utf-8") as f:
            data_by_wave[w] = json.load(f)
    return data_by_wave


def extract_canvas_stances(data_by_wave, waves):
    rows = []
    for w in waves:
        for key, d in data_by_wave[w].items():
            for node in d["positions"]["pos_3"]:
                rows.append({"key": key, "wave": w, "stance": str(node["label"]).strip()})
    return pd.DataFrame(rows)


def extract_canvas_edges(data_by_wave, waves):
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
    return df.merge(
        df_topics[["key", "wave", "stance", "topic"]],
        left_on=["key", "wave", stance_col],
        right_on=["key", "wave", "stance"],
        how="left",
    )


# -------------------------
# Core counts: m_ij, k_ij
# -------------------------
def compute_m(
    df_topics: pd.DataFrame,
    *,
    by: list[str] = ["key", "wave"],
    topic_col: str = "topic",
) -> tuple[pd.DataFrame, int]:
    pres = df_topics[by + [topic_col]].dropna(subset=[topic_col]).drop_duplicates()
    pres[topic_col] = pres[topic_col].astype(int)

    topics_by_net = pres.groupby(by)[topic_col].apply(lambda s: sorted(set(s.tolist())))

    m_counter = Counter()
    total_m = 0

    for _, topics in topics_by_net.items():
        n = len(topics)
        if n < 2:
            continue
        total_m += n * (n - 1) // 2
        for a, b in combinations(topics, 2):  # sorted -> a < b
            m_counter[(a, b)] += 1

    df_m = pd.DataFrame(
        [(a, b, m) for (a, b), m in m_counter.items()],
        columns=["topic_1", "topic_2", "m_ij"],
    ).sort_values(["topic_1", "topic_2"], ignore_index=True)

    return df_m, total_m


def compute_k(
    df_edges: pd.DataFrame,
    *,
    by: list[str] = ["key", "wave"],
    t1: str = "topic_1",
    t2: str = "topic_2",
    polarity_col: str = "polarity",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    df_e = df_edges[by + [t1, t2, polarity_col]].dropna(subset=[t1, t2]).copy()
    df_e[t1] = df_e[t1].astype(int)
    df_e[t2] = df_e[t2].astype(int)

    # enforce unordered pairs and drop self-loops
    df_e["topic_1"] = df_e[[t1, t2]].min(axis=1)
    df_e["topic_2"] = df_e[[t1, t2]].max(axis=1)
    df_e = df_e[df_e["topic_1"] != df_e["topic_2"]]

    df_e["pol_num"] = df_e[polarity_col].map({"positive": 1, "negative": -1})

    # collapse within-net duplicates
    df_e = (
        df_e.groupby(by + ["topic_1", "topic_2"])["pol_num"]
        .mean()
        .reset_index(name="pol_mean_net")
    )

    df_k = (
        df_e.groupby(["topic_1", "topic_2"])
        .size()
        .reset_index(name="k_ij")
        .sort_values(["topic_1", "topic_2"], ignore_index=True)
    )

    df_pol = (
        df_e.groupby(["topic_1", "topic_2"])["pol_mean_net"]
        .mean()
        .reset_index(name="polarity_mean")
        .sort_values(["topic_1", "topic_2"], ignore_index=True)
    )

    total_k = int(df_e.shape[0])
    return df_k, df_pol, total_k


def merge_mk(df_m: pd.DataFrame, df_k: pd.DataFrame) -> pd.DataFrame:
    out = df_m.merge(df_k, on=["topic_1", "topic_2"], how="left")
    out["k_ij"] = out["k_ij"].fillna(0).astype(int)
    return out


def score_binomial(df_pairs: pd.DataFrame, total_m: int, total_k: int) -> pd.DataFrame:
    p_hat = total_k / total_m if total_m > 0 else np.nan
    out = df_pairs.copy()
    out["p_hat"] = p_hat
    out["p_obs"] = out["k_ij"] / out["m_ij"]        # k_ij / m_ij
    out["delta"] = out["p_obs"] - p_hat
    out["pval"] = binom.sf(out["k_ij"] - 1, out["m_ij"], p_hat)
    return out


# -------------------------
# Filtering + polarity + graph
# -------------------------
def filter_edges(
    df_scored: pd.DataFrame,
    *,
    mode: str,
    alpha: float,
    min_delta: float,
    min_m: int,
) -> pd.DataFrame:
    df = df_scored[df_scored["m_ij"] >= min_m].copy()

    if mode == "alpha_delta":
        return df[(df["pval"] < alpha) & (df["delta"] > min_delta)].copy()

    if mode == "delta":
        return df[df["delta"] > min_delta].copy()

    raise ValueError(f"Unknown filter mode: {mode}")


def add_polarity(df_edges: pd.DataFrame, df_pol: pd.DataFrame) -> pd.DataFrame:
    out = df_edges.merge(df_pol, on=["topic_1", "topic_2"], how="left")

    def pol_label(x):
        if pd.isna(x): return "both"
        if x > 0: return "positive"
        if x < 0: return "negative"
        return "both"

    out["polarity"] = out["polarity_mean"].map(pol_label)
    return out


def build_graph(
    df_edges: pd.DataFrame,
    topic_counts: dict[int, int],
    *,
    keep_isolates: bool = True,
    edge_width_mode: str = "p_obs",   # "p_obs" | "k_ij" | "delta"
) -> nx.Graph:
    G = nx.Graph()

    G.add_nodes_from(sorted(topic_counts))
    nx.set_node_attributes(G, topic_counts, "count")

    for r in df_edges.itertuples(index=False):
        if edge_width_mode == "k_ij":
            width_value = float(r.k_ij)
        elif edge_width_mode == "p_obs":
            width_value = float(r.p_obs)
        elif edge_width_mode == "delta":
            width_value = float(r.delta)
        else:
            raise ValueError(f"Unknown edge_width_mode: {edge_width_mode}")

        G.add_edge(
            int(r.topic_1), int(r.topic_2),
            width_value=width_value,
            k_ij=int(r.k_ij),
            m_ij=int(r.m_ij),
            p_obs=float(r.p_obs),
            delta=float(r.delta),
            pval=float(r.pval),
            polarity=str(r.polarity),
            polarity_mean=float(r.polarity_mean) if pd.notna(r.polarity_mean) else np.nan,
        )

    if not keep_isolates:
        G.remove_nodes_from([n for n in G.nodes if G.degree(n) == 0])

    return G


# -------------------------
# Layout (ring) with stable ordering
# -------------------------
def ring_layout(G: nx.Graph, *, enforce_flip: bool = True) -> tuple[list[int], dict[int, tuple[float, float]]]:
    nodes = list(G.nodes())
    isolates = [n for n in nodes if G.degree(n) == 0]
    connected = [n for n in nodes if G.degree(n) > 0]

    if len(connected) >= 2:
        H = G.subgraph(sorted(connected))  # deterministic adjacency order
        ordered = list(nx.spectral_ordering(H, normalized=True))

        # anchor rotation (removes arbitrary rotation)
        i0 = ordered.index(min(ordered))
        ordered = ordered[i0:] + ordered[:i0]

        # optional deterministic "flip" fix (removes mirror ambiguity)
        if enforce_flip and len(ordered) >= 3:
            if ordered[1] > ordered[-1]:
                ordered = [ordered[0]] + list(reversed(ordered[1:]))
    else:
        ordered = sorted(connected)

    ordered_all = ordered + sorted(isolates)

    n = len(ordered_all)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) if n else np.array([])
    pos = {ordered_all[i]: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}
    return ordered_all, pos


# -------------------------
# Plot helper (proportional widths, optional edge legend)
# -------------------------
def plot_topic_network(
    G: nx.Graph,
    pos: dict,
    *,
    width_scale: float = 1.0,      # linewidth = width_scale * width_value
    show_edge_legend: bool = True,
):
    EDGE_COLORS = {"positive": "#998ec3", "negative": "#f1a340", "both": "#1b9e77"}

    cmap = plt.get_cmap("tab20")
    node_colors = [cmap(int(t) % 20) for t in G.nodes]

    counts = np.array([G.nodes[t]["count"] for t in G.nodes], dtype=float)
    node_sizes = counts / (counts.max() / 3000) if counts.size else []

    edges = list(G.edges())
    if edges:
        w = np.array([G.edges[u, v]["width_value"] for u, v in edges], dtype=float)
        edge_widths = width_scale * w
        edge_colors = [EDGE_COLORS[G.edges[u, v]["polarity"]] for u, v in edges]
    else:
        edge_widths, edge_colors = [], []

    labels = {t: f"T{t}" for t in G.nodes}

    # legend strings (topic panel)
    topic_legend = {}
    for _, row in df_topic_overview.iterrows():
        t = int(row["Topic"])
        rep = row["Representation"]
        words = eval(rep)[:3] if isinstance(rep, str) else list(rep)[:3]
        topic_legend[t] = ", ".join(words)

    legend_lines = []
    for t in sorted(G.nodes()):
        kw = topic_legend.get(t, "")
        cnt = G.nodes[t].get("count", 0)
        legend_lines.append(f"T{t} ({cnt}): {kw}")

    fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(26, 12),
                                     gridspec_kw={"width_ratios": [3, 1]})
    ax.set_aspect("equal")
    ax.set_axis_off()

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="black", linewidths=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=20, ax=ax)

    # Optional edge-color legend on the network axis
    if show_edge_legend:
        handles = [
            Line2D([0], [0], color=EDGE_COLORS["positive"], lw=4, label="supporting"),
            Line2D([0], [0], color=EDGE_COLORS["negative"], lw=4, label="conflicting"),
            Line2D([0], [0], color=EDGE_COLORS["both"], lw=4, label="equally"),
        ]
        ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=14)

    ax_leg.set_axis_off()
    ax_leg.text(0, 1, "\n".join(legend_lines), transform=ax_leg.transAxes,
                fontsize=24, verticalalignment="top", fontfamily="monospace")

    fig.tight_layout()
    return fig


# -------------------------
# Run
# -------------------------
data_by_wave = load_canvas_data(CFG["WAVES"])

df_canvas_stances = extract_canvas_stances(data_by_wave, CFG["WAVES"])
df_canvas_edges = extract_canvas_edges(data_by_wave, CFG["WAVES"])

df_canvas_stances = assign_topics(df_canvas_stances, df_statements_topics, stance_col="stance")

df_canvas_edges = df_canvas_edges.merge(
    df_statements_topics[["key", "wave", "stance", "topic"]].rename(columns={"stance": "stance_1", "topic": "topic_1"}),
    on=["key", "wave", "stance_1"],
    how="left",
)
df_canvas_edges = df_canvas_edges.merge(
    df_statements_topics[["key", "wave", "stance", "topic"]].rename(columns={"stance": "stance_2", "topic": "topic_2"}),
    on=["key", "wave", "stance_2"],
    how="left",
)

if CFG["REMOVE_OUTLIER"]:
    df_canvas_stances = df_canvas_stances[df_canvas_stances["topic"] != -1]
    df_canvas_edges = df_canvas_edges[(df_canvas_edges["topic_1"] != -1) & (df_canvas_edges["topic_2"] != -1)]

df_canvas_stances = df_canvas_stances.dropna(subset=["topic"])
df_canvas_stances["topic"] = df_canvas_stances["topic"].astype(int)

df_canvas_edges = df_canvas_edges.dropna(subset=["topic_1", "topic_2"])
df_canvas_edges["topic_1"] = df_canvas_edges["topic_1"].astype(int)
df_canvas_edges["topic_2"] = df_canvas_edges["topic_2"].astype(int)

topic_counts = df_canvas_stances.groupby("topic").size().to_dict()

df_m, total_m = compute_m(df_canvas_stances[["key", "wave", "topic"]])
df_k, df_pol, total_k = compute_k(df_canvas_edges[["key", "wave", "topic_1", "topic_2", "polarity"]])

df_pairs = merge_mk(df_m, df_k)
df_scored = score_binomial(df_pairs, total_m=total_m, total_k=total_k)

df_keep = filter_edges(
    df_scored,
    mode=CFG["FILTER_MODE"],
    alpha=CFG["ALPHA"],
    min_delta=CFG["MIN_DELTA"],
    min_m=CFG["MIN_M"],
)
df_keep = add_polarity(df_keep, df_pol)

print(f"total_m={total_m} total_k={total_k} p_hat={df_scored['p_hat'].iloc[0]:.4f}")
print(f"pairs={len(df_scored)} kept_edges={len(df_keep)} mode={CFG['FILTER_MODE']}")

G = build_graph(
    df_keep,
    topic_counts,
    keep_isolates=CFG["KEEP_ISOLATES"],
    edge_width_mode=CFG["EDGE_WIDTH"],   # "k_ij" | "p_obs" | "delta"
)

ordered_nodes, pos = ring_layout(G, enforce_flip=CFG["ENFORCE_FLIP"])

fig = plot_topic_network(
    G, pos,
    width_scale=CFG["WIDTH_SCALE"],
    show_edge_legend=CFG["SHOW_EDGE_LEGEND"],
)

fig.savefig(CFG["OUTDIR"] / "population_topic_network.png", dpi=200, bbox_inches="tight")
fig.savefig(CFG["OUTDIR"] / "population_topic_network.pdf", bbox_inches="tight")


### basic information ###

sum_m = df_m["m_ij"].sum()
print(f"total_m = {total_m}")

weighted_mean_p = (df_scored["m_ij"] * df_scored["p_obs"]).sum() / df_scored["m_ij"].sum()
p_hat = df_scored["p_hat"].iloc[0]
print(f"p_hat = {p_hat:.6f}")

print(f"Number of networks (unique key,wave): {df_canvas_stances[['key','wave']].drop_duplicates().shape[0]}")
print(f"Number of topic pairs tested: {len(df_scored)}")
print(f"Number of kept edges: {len(df_keep)}")
print(f"Average topics per network: {df_canvas_stances.groupby(['key','wave'])['topic'].nunique().mean():.2f}")


# =========================================================================
# Individual overlay
# =========================================================================

import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

# -------------------------
# Config (all style knobs in one place)
# -------------------------
IND_CFG = dict(
    # background (population layer)
    BG_NODE_FADE   = 0.25,    # 0 = white, 1 = full topic colour; try 0.15–0.40
    BG_EDGE_FADE   = 0.20,    # same for polarity-coloured pop edges
    BG_EDGE_ALPHA  = 0.5,     # additional alpha on top of fading

    # individual layer
    IND_NODE_LW    = 1.5,     # border linewidth on individual's nodes
    IND_EDGE_SCALE = 5,       # linewidth = weight * scale
    IND_EDGE_MIN   = 4,       # minimum individual edge linewidth
    IND_EDGE_ALPHA = 0.9,

    # shared colours
    EDGE_COLORS    = {"positive": "#998ec3", "negative": "#f1a340", "both": "#1b9e77"},

    # label font size
    LABEL_FONTSIZE = 16,
)

# -------------------------
# Pre-compute population visuals (shared across all individuals)
# -------------------------
_cmap = plt.get_cmap("tab20")
_counts_arr     = np.array([G.nodes[t]["count"] for t in G.nodes], dtype=float)
_node_sizes     = _counts_arr / (_counts_arr.max() / 3000) if _counts_arr.size else np.array([])
_node_color_map = {t: _cmap(int(t) % 20) for t in G.nodes}
_node_order     = list(G.nodes)

_pop_edges = list(G.edges())
_pop_edge_widths = (
    CFG["WIDTH_SCALE"] * np.array([G.edges[u, v]["width_value"] for u, v in _pop_edges], dtype=float)
    if _pop_edges else []
)

def _fade(rgba, fade):
    """Blend an RGBA colour toward white: fade=0 → white, fade=1 → original."""
    r = np.array(rgba, dtype=float)
    r[:3] = fade * r[:3] + (1 - fade)
    return tuple(r)

_pop_node_faded = [_fade(_cmap(int(t) % 20), IND_CFG["BG_NODE_FADE"]) for t in G.nodes]
_pop_edge_faded = [
    _fade(np.array(mcolors.to_rgba(IND_CFG["EDGE_COLORS"][G.edges[u, v]["polarity"]])),
          IND_CFG["BG_EDGE_FADE"])
    for u, v in _pop_edges
]

# topic legend text (built once, used on PDF legend page)
_topic_words = {}
for _, row in df_topic_overview.iterrows():
    t   = int(row["Topic"])
    rep = row["Representation"]
    words = eval(rep)[:3] if isinstance(rep, str) else list(rep)[:3]
    _topic_words[t] = ", ".join(words)

_legend_lines = [
    f"T{t} ({G.nodes[t].get('count', 0)}): {_topic_words.get(t, '')}"
    for t in sorted(G.nodes())
]

# pol_num needed for individual edge aggregation
df_canvas_edges["pol_num"] = df_canvas_edges["polarity"].map({"positive": 1, "negative": -1})

# -------------------------
# Draw helper
# -------------------------
def draw_individual_overlay(ax, key, waves):
    """
    Draw one individual overlaid on the faded population network.
    waves: int or list of ints — which wave(s) to include.
    Edge weight = total count across all included waves; polarity = signed mean.
    No title, no legend — just the ring plot. Caller owns the axes.
    """
    if isinstance(waves, int):
        waves = [waves]

    EC    = IND_CFG["EDGE_COLORS"]
    scale = IND_CFG["IND_EDGE_SCALE"]
    emin  = IND_CFG["IND_EDGE_MIN"]

    # individual's raw edges (no threshold), across specified waves
    ind_edges = df_canvas_edges[
        (df_canvas_edges["key"] == key) & (df_canvas_edges["wave"].isin(waves))
    ]
    if ind_edges.empty:
        ind_topic_edges = pd.DataFrame(columns=["topic_1", "topic_2", "weight", "polarity_mean"])
    else:
        ind_topic_edges = (
            ind_edges
            .groupby(["topic_1", "topic_2"])["pol_num"]
            .agg(weight="count", polarity_mean="mean")
            .reset_index()
        )

    # individual's topics across specified waves
    ind_topics = set(
        df_canvas_stances[
            (df_canvas_stances["key"] == key) & (df_canvas_stances["wave"].isin(waves))
        ]["topic"].dropna().astype(int)
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    # --- background: faded population ---
    nx.draw_networkx_nodes(G, pos, node_color=_pop_node_faded, node_size=_node_sizes,
                           edgecolors="#cccccc", linewidths=0.5, ax=ax)
    if _pop_edges:
        nx.draw_networkx_edges(G, pos, edge_color=_pop_edge_faded, width=_pop_edge_widths,
                               alpha=IND_CFG["BG_EDGE_ALPHA"], ax=ax)

    # --- foreground: individual nodes ---
    ind_node_list  = [t for t in _node_order if t in ind_topics]
    ind_node_sizes = [_node_sizes[_node_order.index(t)] for t in ind_node_list]
    ind_node_cols  = [_node_color_map[t] for t in ind_node_list]
    if ind_node_list:
        nx.draw_networkx_nodes(G, pos, nodelist=ind_node_list,
                               node_color=ind_node_cols, node_size=ind_node_sizes,
                               edgecolors="black", linewidths=IND_CFG["IND_NODE_LW"], ax=ax)

    # --- foreground: individual edges (weight = total count across waves) ---
    ind_edge_list, ind_edge_colors, ind_edge_widths = [], [], []
    for _, row in ind_topic_edges.iterrows():
        t1, t2 = int(row["topic_1"]), int(row["topic_2"])
        if not G.has_node(t1) or not G.has_node(t2):
            continue
        pm  = row["polarity_mean"]
        pol = "positive" if pm > 0 else ("negative" if pm < 0 else "both")
        ind_edge_list.append((t1, t2))
        ind_edge_colors.append(EC[pol])
        ind_edge_widths.append(max(row["weight"] * scale, emin))
    if ind_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=ind_edge_list,
                               edge_color=ind_edge_colors, width=ind_edge_widths,
                               alpha=IND_CFG["IND_EDGE_ALPHA"], ax=ax)

    nx.draw_networkx_labels(G, pos, labels={t: f"T{t}" for t in G.nodes},
                            font_size=IND_CFG["LABEL_FONTSIZE"], ax=ax)


# -------------------------
# Quick preview: first participant, all 3 panels
# -------------------------
_preview_key = list(data_by_wave[1].keys())[0]

fig_preview, axes_preview = plt.subplots(1, 3, figsize=(36, 12))
fig_preview.suptitle(_preview_key, fontsize=13)
for ax_p, (waves, title) in zip(axes_preview, [([1], "Wave 1"), ([2], "Wave 2"), ([1, 2], "Wave 1 + 2")]):
    draw_individual_overlay(ax_p, _preview_key, waves)
    ax_p.set_title(title, fontsize=11)
fig_preview.tight_layout()


# -------------------------
# Compiled PDF: all_individuals.pdf
# -------------------------
_all_keys = sorted(df_canvas_stances["key"].unique())

with PdfPages(CFG["OUTDIR"] / "all_individuals.pdf") as pdf:

    # Page 1: topic legend (reference, not repeated per participant)
    fig_leg, ax_leg = plt.subplots(figsize=(10, 14))
    ax_leg.set_axis_off()
    ax_leg.text(0.05, 0.97, "\n".join(_legend_lines),
                transform=ax_leg.transAxes,
                fontsize=14, verticalalignment="top", fontfamily="monospace")
    fig_leg.tight_layout()
    pdf.savefig(fig_leg, bbox_inches="tight")
    plt.close(fig_leg)

    # Pages 2–N: one page per participant, 3 panels: wave 1 | wave 2 | wave 1+2
    for key in _all_keys:
        fig_trio, axes = plt.subplots(1, 3, figsize=(36, 12))
        fig_trio.suptitle(key, fontsize=13)

        for ax, (waves, title) in zip(axes, [([1], "Wave 1"), ([2], "Wave 2"), ([1, 2], "Wave 1 + 2")]):
            has_data = not df_canvas_stances[
                (df_canvas_stances["key"] == key) & (df_canvas_stances["wave"].isin(waves))
            ].empty
            if has_data:
                draw_individual_overlay(ax, key, waves)
                ax.set_title(title, fontsize=11)
            else:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"no data for {title}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=11, color="gray")

        fig_trio.tight_layout()
        pdf.savefig(fig_trio, bbox_inches="tight")
        plt.close(fig_trio)