"""
VMP 2026-02-06 (refactored):

Basic reliability across waves:
1) Panel of 3 scatterplots:
   - interview words (W1 vs W2)
   - LLM-extracted nodes (W1 vs W2; no limit)
   - human canvas edges (W1 vs W2)

2) Reliability table: descriptives + Pearson r for network measures.

Assumes these exist:
- public/interviews_w*.csv
- public/edges_canvas_w*.csv
- public/llm_extractions/node_extraction_w*/<model>/*.json

OUTPUT:
Figure 3 (main text) — fig/reliability/reliability.svg
Table S4 fig/reliability/reliability_descriptives.tex

VMP 2026-02-07: tested and run.

VMP 2026-09-15: network structure metrics (reviewer request; previously
40_network_metrics.py) merged in. Canvas edges only (edges_3).

Edge COUNTS in Table S4 are edges AS DRAWN: a node-pair marked both supporting
and conflicting counts twice. The structure metrics (mean degree, density,
degree centralization, transitivity) are computed on DE-DUPLICATED pairs, where
such a pair counts once, since the pair is either connected or not.
- Density is mechanically size-dependent (denominator grows with n^2); mean
  degree is the size-robust counterpart.
- Freeman (1979) degree centralization. Isolates are kept as nodes (degree 0).
  Not independent of density: a complete graph scores 0, a star scores 1.
- Transitivity C = 3 * triangles / connected triples. With no connected triples
  C = 0/0; networkx returns 0.0 there, which is wrong, so those are set to NaN.

Additional outputs (all in fig/reliability):
- Section 5, Figure 2 example networks: figure2_examples.csv / .tex
- Section 6, exploratory checks: network_metrics_per_network.csv,
  density_mean_degree.svg, degree_centralization.svg, transitivity.svg
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from scipy.stats import pearsonr

from utilities import wave_1, wave_2, get_public_path, get_llm_extraction_path

# -------------------------
# config
# -------------------------
MODEL = "gpt-4.1-2025-04-14"

outdir = "../fig/reliability"
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

# derive valid key set (210 participants in both waves of distractors)
def load_distractors(wave):
    path = get_public_path(f"distractors_w{wave}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

distractors_w1 = load_distractors(wave_1)
distractors_w2 = load_distractors(wave_2)
valid_keys = set(distractors_w1.keys()) & set(distractors_w2.keys())

words_wide  = words_wide[words_wide["key"].isin(valid_keys)].reset_index(drop=True)
nodes_wide  = nodes_wide[nodes_wide["key"].isin(valid_keys)].reset_index(drop=True)

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
plt.savefig(os.path.join(outdir, "reliability.svg"))

# -------------------------
# 3) unified descriptives + reliability table
# -------------------------

def node_counts_from(data):
    rows = []
    for key, rec in data.items():
        nodes = rec["nodes"]
        n_proposed = len(nodes["generated"])
        n_accepted = sum(1 for x in nodes["final"] if not x.get("is_distractor", False))
        rows.append({"key": key, "proposed": n_proposed, "accepted": n_accepted})
    return pd.DataFrame(rows)

def edge_counts_from(data):
    rows = []
    for key, rec in data.items():
        edges = rec["edges"].get("edges_3", [])
        rows.append({
            "key":      key,
            "total":    len(edges),
            "support":  sum(1 for e in edges if e.get("polarity") == "positive"),
            "conflict": sum(1 for e in edges if e.get("polarity") == "negative"),
        })
    return pd.DataFrame(rows)

nc1 = node_counts_from(distractors_w1)
nc2 = node_counts_from(distractors_w2)
ec1 = edge_counts_from(distractors_w1)
ec2 = edge_counts_from(distractors_w2)

dist_wide = (
    nc1.merge(nc2, on="key", suffixes=("_w1", "_w2"))
       .merge(ec1.merge(ec2, on="key", suffixes=("_w1", "_w2")), on="key")
)

# network structure metrics on de-duplicated pairs (see docstring)
def degree_centralization(G):
    """Freeman (1979) degree centralization, simple undirected graph."""
    n = G.number_of_nodes()
    if n < 3:
        return np.nan                                  # denominator is 0 at n=2
    d = np.array([deg for _, deg in G.degree()])       # isolates contribute 0
    return (d.max() - d).sum() / ((n - 1) * (n - 2))

def transitivity(G):
    """C = 3 * triangles / connected triples. NaN when there are no triples."""
    n_triples = sum(d * (d - 1) // 2 for _, d in G.degree())
    if n_triples == 0:
        return np.nan                                  # 0/0; nx would return 0.0
    return nx.transitivity(G)

metric_rows = []
for wave, data in [(wave_1, distractors_w1), (wave_2, distractors_w2)]:
    for key, rec in data.items():
        accepted = [x["belief"] for x in rec["nodes"]["final"]
                    if not x.get("is_distractor", False)]
        edges = rec["edges"].get("edges_3", [])

        # polarities per unordered pair: (a, b) and (b, a) collapse
        pair_pols = {}
        for e in edges:
            pair = tuple(sorted((e["stance_1"], e["stance_2"])))
            pair_pols.setdefault(pair, set()).add(e["polarity"])

        G = nx.Graph()
        G.add_nodes_from(accepted)                     # isolates first
        G.add_edges_from(pair_pols)
        assert G.number_of_nodes() == len(accepted)    # edges only between accepted nodes

        n, m = len(accepted), G.number_of_edges()
        metric_rows.append({
            "key": key, "wave": wave,
            "n_nodes": n,
            "n_edges": len(edges),                     # as drawn
            "n_pairs": m,                              # de-duplicated
            "pairs_support":  sum(p == {"positive"} for p in pair_pols.values()),
            "pairs_conflict": sum(p == {"negative"} for p in pair_pols.values()),
            "pairs_both":     sum(len(p) > 1 for p in pair_pols.values()),
            "density": m / (n * (n - 1) / 2),
            "mean_degree": 2 * m / n,
            "centralization": degree_centralization(G),
            "transitivity": transitivity(G),
        })

metrics = pd.DataFrame(metric_rows)
metrics_wide = metrics.pivot(index="key", columns="wave")

def summary_row(label, w1, w2):
    paired = pd.DataFrame({"w1": w1, "w2": w2}).dropna()
    r, _ = pearsonr(paired["w1"], paired["w2"])
    return {
        "Measure":    label,
        "Mean W1":    paired["w1"].mean(),
        "SD W1":      paired["w1"].std(),
        "Mean W2":    paired["w2"].mean(),
        "SD W2":      paired["w2"].std(),
        "r":          r,
        "N":          len(paired),
    }

table_rows = [
    summary_row("Interview words",
                words_wide["wave1"], words_wide["wave2"]),
    summary_row("Proposed nodes (LLM, no limit)",
                nodes_wide["wave1"], nodes_wide["wave2"]),
    summary_row("Proposed nodes (LLM, live)",
                dist_wide["proposed_w1"], dist_wide["proposed_w2"]),
    summary_row("Accepted nodes (on canvas)",
                dist_wide["accepted_w1"], dist_wide["accepted_w2"]),
    summary_row("Canvas edges",
                dist_wide["total_w1"], dist_wide["total_w2"]),
    summary_row("Supporting edges",
                dist_wide["support_w1"], dist_wide["support_w2"]),
    summary_row("Conflicting edges",
                dist_wide["conflict_w1"], dist_wide["conflict_w2"]),
    summary_row("Mean degree",
                metrics_wide["mean_degree"][wave_1], metrics_wide["mean_degree"][wave_2]),
    summary_row("Density",
                metrics_wide["density"][wave_1], metrics_wide["density"][wave_2]),
    summary_row("Degree centralization",
                metrics_wide["centralization"][wave_1], metrics_wide["centralization"][wave_2]),
    summary_row("Transitivity",
                metrics_wide["transitivity"][wave_1], metrics_wide["transitivity"][wave_2]),
]

table = pd.DataFrame(table_rows)
print("\n=== Descriptives and reliability ===")
print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# save LaTeX — Table S4
latex = table.to_latex(
    index=False,
    float_format="%.2f",
    na_rep="--",
    caption=("Descriptive statistics and wave-1/wave-2 reliability for interview and network measures. "
             "Edge counts are edges as drawn, so a node-pair marked both supporting and conflicting "
             "counts twice; mean degree, density, degree centralization and transitivity are computed "
             "on de-duplicated node-pairs, where such a pair counts once. Transitivity is undefined for "
             "networks without connected triples; N is the number of participants with the measure "
             "defined in both waves."),
    label="tab:reliability_descriptives",
    escape=True,
)
with open(os.path.join(outdir, "reliability_descriptives.tex"), "w", encoding="utf-8") as f:
    f.write(latex)
print("Saved table to", outdir)

# -------------------------
# 4) pooled (across both waves) quick stats
# -------------------------

# stack wave 1 and wave 2 into one long frame
long = pd.concat([
    dist_wide[["key", "accepted_w1", "total_w1", "support_w1", "conflict_w1"]]
        .rename(columns={"accepted_w1": "accepted", "total_w1": "total",
                         "support_w1": "support", "conflict_w1": "conflict"}),
    dist_wide[["key", "accepted_w2", "total_w2", "support_w2", "conflict_w2"]]
        .rename(columns={"accepted_w2": "accepted", "total_w2": "total",
                         "support_w2": "support", "conflict_w2": "conflict"}),
], ignore_index=True)

n_nets = len(long)

print(f"\n=== Pooled stats across both waves (N = {n_nets} networks) ===")
for col, label in [("accepted", "Accepted nodes"), ("total", "Canvas edges"),
                   ("support", "Supporting edges"), ("conflict", "Conflicting edges")]:
    s = long[col]
    print(f"{label:25s}  mean={s.mean():.2f}  SD={s.std():.2f}  min={s.min():.0f}  max={s.max():.0f}")

no_conflict  = (long["conflict"] == 0).mean() * 100
no_support   = (long["support"]  == 0).mean() * 100
has_both     = ((long["conflict"] > 0) & (long["support"] > 0)).mean() * 100
print(f"\nNetworks with no conflicting edges  : {no_conflict:.1f}%")
print(f"Networks with no supporting edges   : {no_support:.1f}%")
print(f"Networks with both types            : {has_both:.1f}%")

# =========================================================================
# 5) Figure 2 example networks: metrics table
# =========================================================================
# Identified by exact-matching the node labels legible in Figure 2 against the
# accepted-node sets of all 420 participant-waves. Each panel matched exactly
# one network on all of its labels; the runner-up matched at most 40%.
FIGURE_2 = {
    "A": ("24e9f4a8dba444ba", 2),
    "B": ("5fee709bfe55faf2", 1),
    "C": ("9306e20a3c8202a6", 2),
    "D": ("3a4931c85b6bb77d", 2),
    "E": ("d626dfd972cd0a7d", 2),
    "F": ("ea0c4e9dce422f7c", 2),
}

examples = pd.DataFrame(
    [{"Panel": p, "key": k, "wave": w} for p, (k, w) in FIGURE_2.items()]
).merge(metrics, on=["key", "wave"], how="left")

examples = examples.rename(columns={
    "n_nodes": "Nodes", "n_edges": "Edges drawn", "n_pairs": "Edges",
    "pairs_support": "Supporting", "pairs_conflict": "Conflicting", "pairs_both": "Both",
    "mean_degree": "Mean degree", "density": "Density",
    "centralization": "Centralization", "transitivity": "Transitivity",
})

# full view, for checking against the figure colours
# (purple = supporting, orange = conflicting, green = both)
print("\n=== Figure 2 example networks (full, for checking) ===")
print(examples[["Panel", "key", "wave", "Nodes", "Edges drawn", "Edges",
                "Supporting", "Conflicting", "Both", "Mean degree", "Density",
                "Centralization", "Transitivity"]]
      .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# the table itself: "Edges" = distinct pairs, matching both the figure (a pair
# marked both ways is drawn as one green line) and the de-duplicated metrics
examples_table = examples[["Panel", "Nodes", "Edges", "Mean degree", "Density",
                           "Centralization", "Transitivity"]]

examples_table.round(2).to_csv(os.path.join(outdir, "figure2_examples.csv"), index=False)
with open(os.path.join(outdir, "figure2_examples.tex"), "w", encoding="utf-8") as f:
    f.write(examples_table.to_latex(
        index=False, float_format="%.2f", na_rep="--",
        caption=("Network metrics for the six example belief networks in Figure 2. "
                 "All metrics are computed on de-duplicated node-pairs, so a pair "
                 "marked both supporting and conflicting counts once."),
        label="tab:figure2_examples", escape=True))
print(f"Saved {outdir}/figure2_examples.csv and .tex")

# =========================================================================
# 6) exploratory: network metrics vs network size, per-network csv, plots
# =========================================================================
METRIC_COLS = ["density", "mean_degree", "centralization", "transitivity"]

print(f"\n{metrics['n_edges'].sum()} drawn edges -> {metrics['n_pairs'].sum()} distinct pairs "
      f"({metrics['pairs_both'].sum()} pairs drawn with both polarities)")
n_undefined = metrics["transitivity"].isna().sum()
print(f"transitivity undefined (no connected triples): {n_undefined} networks")

print("\n=== Dependence on number of nodes ===")
print(metrics.groupby("n_nodes")[METRIC_COLS].agg(["mean", "count"]).round(2).to_string())

def corr(col, other="n_nodes"):
    """Pearson r on complete cases (transitivity has NaNs)."""
    sub = metrics[[other, col]].dropna()
    return pearsonr(sub[other], sub[col])

print()
r_by_size = {}
for col in METRIC_COLS:
    r, p = corr(col)
    r_by_size[col] = r
    print(f"{col:14s} vs n_nodes: r={r:+.3f} (p={p:.1e})")

# neither centralization nor transitivity is independent of density a priori
print()
for col in ["centralization", "transitivity"]:
    r, p = corr(col, other="density")
    print(f"{col:14s} vs density: r={r:+.3f} (p={p:.1e})")

metrics.round(2).to_csv(os.path.join(outdir, "network_metrics_per_network.csv"), index=False)
print(f"\nSaved {outdir}/network_metrics_per_network.csv")

TITLE_SIZE = 15
LABEL_SIZE = 14
TICK_SIZE = 12

sizes = sorted(metrics["n_nodes"].unique())

def plot_metric(ax_hist, ax_size, col, label, bins):
    """Histogram + by-network-size boxplot for one metric (both waves pooled)."""
    ax_hist.hist(metrics[col].dropna(), bins=bins,
                 color="lightsteelblue", edgecolor="black", linewidth=0.6)
    ax_hist.axvline(metrics[col].mean(), color="black", linestyle="--", linewidth=1)
    ax_hist.set_xlabel(label, fontsize=LABEL_SIZE)
    ax_hist.set_ylabel("Networks", fontsize=LABEL_SIZE)
    ax_hist.set_title(f"{label}: M = {metrics[col].mean():.2f} "
                      f"(SD = {metrics[col].std():.2f})", fontsize=TITLE_SIZE)

    ax_size.boxplot([metrics.loc[metrics["n_nodes"] == s, col].dropna() for s in sizes],
                    positions=sizes, widths=0.6, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor="lightsteelblue", edgecolor="black"),
                    medianprops=dict(color="black"))
    ax_size.axhline(metrics[col].mean(), color="black", linestyle="--", linewidth=1)
    ax_size.set_xticks(sizes)
    ax_size.set_xlabel("Nodes on canvas", fontsize=LABEL_SIZE)
    ax_size.set_ylabel(label, fontsize=LABEL_SIZE)
    ax_size.set_title(f"{label} by size: r = {r_by_size[col]:+.2f}", fontsize=TITLE_SIZE)

    for ax in (ax_hist, ax_size):
        ax.tick_params(axis="both", labelsize=TICK_SIZE)

# plot 1: density and mean degree
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
plot_metric(axes[0][0], axes[0][1], "density", "Density", np.arange(0, 1.05, 0.05))
plot_metric(axes[1][0], axes[1][1], "mean_degree", "Mean degree",
            np.arange(0, metrics["mean_degree"].max() + 0.3, 0.25))
plt.tight_layout()
plt.savefig(os.path.join(outdir, "density_mean_degree.svg"), bbox_inches="tight")

# plot 2: degree centralization
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_metric(axes[0], axes[1], "centralization", "Degree centralization",
            np.arange(0, 1.05, 0.05))
plt.tight_layout()
plt.savefig(os.path.join(outdir, "degree_centralization.svg"), bbox_inches="tight")

# plot 3: global clustering (transitivity)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_metric(axes[0], axes[1], "transitivity", "Transitivity", np.arange(0, 1.05, 0.05))
plt.tight_layout()
plt.savefig(os.path.join(outdir, "transitivity.svg"), bbox_inches="tight")
print(f"Saved density_mean_degree.svg, degree_centralization.svg, transitivity.svg to {outdir}")
