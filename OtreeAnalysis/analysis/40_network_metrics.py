"""
40_network_metrics.py

VMP 2026-09-10

Standard network metrics for the individual belief networks (reviewer request).
Canvas edges only (edges_3); LLM and pairwise edges are excluded.

Part 1 reproduces the edge counts already reported in the manuscript (Section
"Belief networks"), using edges AS DRAWN: a node-pair that a participant marked
both supporting and conflicting counts twice. That is intended, because those
numbers describe supporting and conflicting connections separately.

Part 2 computes density and mean degree on DE-DUPLICATED pairs (such a pair
counts once, since the pair is either connected or not), averaged per
participant. Density is mechanically size-dependent (the denominator grows with
n^2); mean degree is the size-robust counterpart. Both are reported so we can
decide which to put in the manuscript.

Part 3 computes Freeman (1979) degree centralization on the same de-duplicated
graphs. Isolates are kept as nodes (they contribute degree 0). Note that
centralization is not independent of density: a complete graph scores 0 by
construction and a star scores 1, so the correlation with density is reported.

Part 4 computes global clustering (transitivity), C = 3 * triangles / connected
triples, again on the de-duplicated graphs. A graph with no connected triples
at all has C = 0/0; networkx returns 0.0 there, which is wrong, so those
networks are set to NaN and excluded rather than counted as zero clustering.

Reads:
  ../data/public/distractors_w1.json
  ../data/public/distractors_w2.json

Writes:
  ../fig/network_metrics/network_metrics_per_network.csv
  ../fig/network_metrics/density_mean_degree.svg
  ../fig/network_metrics/degree_centralization.svg
  ../fig/network_metrics/transitivity.svg
"""

import os
import json

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

from utilities import wave_1, wave_2, get_public_path

outdir = "../fig/network_metrics"
os.makedirs(outdir, exist_ok=True)

# -------------------------
# load: one row per node-set, one row per drawn edge
# -------------------------
node_rows = []
edge_rows = []
node_lists = {}          # (key, wave) -> accepted node labels, needed for isolates

for wave in [wave_1, wave_2]:
    with open(get_public_path(f"distractors_w{wave}.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, rec in data.items():
        # accepted nodes = everything the participant kept on the canvas,
        # minus distractors, INCLUDING nodes they never connected
        accepted = [n["belief"] for n in rec["nodes"]["final"]
                    if not n.get("is_distractor", False)]
        node_rows.append({"key": key, "wave": wave, "n_nodes": len(accepted)})
        node_lists[(key, wave)] = accepted

        for e in rec["edges"]["edges_3"]:
            edge_rows.append({"key": key, "wave": wave,
                              "stance_1": e["stance_1"],
                              "stance_2": e["stance_2"],
                              "polarity": e["polarity"]})

nodes = pd.DataFrame(node_rows)
edges = pd.DataFrame(edge_rows)

print(f"{len(nodes)} networks ({nodes['key'].nunique()} participants x 2 waves)")
print(f"{len(edges)} drawn edges")

# -------------------------
# Part 1: edge counts as drawn (NOT de-duplicated)
# -------------------------
edges["is_support"] = edges["polarity"] == "positive"
edges["is_conflict"] = edges["polarity"] == "negative"

counts = (
    edges.groupby(["key", "wave"])
         .agg(n_edges=("polarity", "size"),
              n_support=("is_support", "sum"),
              n_conflict=("is_conflict", "sum"))
         .reset_index()
)

counts = nodes.merge(counts, on=["key", "wave"], how="left")

print("\n=== Part 1: counts as drawn, per network ===")
print(counts[["n_nodes", "n_edges", "n_support", "n_conflict"]]
      .agg(["mean", "std", "min", "max"]).round(2).to_string())

print(f"\nNo conflicting connections : {(counts['n_conflict'] == 0).mean() * 100:.0f}%")
print(f"No supporting connections  : {(counts['n_support'] == 0).mean() * 100:.0f}%")
print(f"Both types                 : "
      f"{((counts['n_conflict'] > 0) & (counts['n_support'] > 0)).mean() * 100:.0f}%")

# -------------------------
# Part 2: de-duplicate pairs, then density and mean degree per participant
# -------------------------
# unordered pair, so (a, b) and (b, a) collapse
edges["pair"] = [tuple(sorted(p)) for p in zip(edges["stance_1"], edges["stance_2"])]

pairs = edges.drop_duplicates(subset=["key", "wave", "pair"])
print(f"\n=== Part 2: de-duplicated pairs ===")
print(f"{len(edges)} drawn edges -> {len(pairs)} distinct pairs "
      f"({len(edges) - len(pairs)} pairs drawn with both polarities)")

n_pairs = pairs.groupby(["key", "wave"]).size().reset_index(name="n_pairs")

metrics = counts.merge(n_pairs, on=["key", "wave"], how="left")
metrics["n_possible"] = metrics["n_nodes"] * (metrics["n_nodes"] - 1) // 2

# density: share of possible pairs connected
metrics["density"] = metrics["n_pairs"] / metrics["n_possible"]

# mean degree: connections per belief (each pair contributes to two nodes)
metrics["mean_degree"] = 2 * metrics["n_pairs"] / metrics["n_nodes"]

# -------------------------
# Part 3: Freeman degree centralization
# -------------------------
def degree_centralization(G):
    """Freeman (1979) degree centralization, simple undirected graph."""
    n = G.number_of_nodes()
    if n < 3:
        return np.nan                                  # denominator is 0 at n=2
    d = np.array([deg for _, deg in G.degree()])       # isolates contribute 0
    return (d.max() - d).sum() / ((n - 1) * (n - 2))


graphs = {}
for (key, wave), grp in pairs.groupby(["key", "wave"]):
    G = nx.Graph()
    G.add_nodes_from(node_lists[(key, wave)])          # isolates first
    G.add_edges_from(grp["pair"])
    graphs[(key, wave)] = G

metrics["centralization"] = [degree_centralization(graphs[(k, w)])
                             for k, w in zip(metrics["key"], metrics["wave"])]

# sanity check: graph node/edge counts match the pandas counts
assert all(graphs[(k, w)].number_of_nodes() == n
           for k, w, n in zip(metrics["key"], metrics["wave"], metrics["n_nodes"]))
assert all(graphs[(k, w)].number_of_edges() == m
           for k, w, m in zip(metrics["key"], metrics["wave"], metrics["n_pairs"]))

# -------------------------
# Part 4: global clustering (transitivity)
# -------------------------
def transitivity(G):
    """C = 3 * triangles / connected triples. NaN when there are no triples."""
    n_triples = sum(d * (d - 1) // 2 for _, d in G.degree())
    if n_triples == 0:
        return np.nan                                  # 0/0; nx would return 0.0
    return nx.transitivity(G)


metrics["transitivity"] = [transitivity(graphs[(k, w)])
                           for k, w in zip(metrics["key"], metrics["wave"])]

n_undefined = metrics["transitivity"].isna().sum()
print(f"\ntransitivity undefined (no connected triples): {n_undefined} networks")

for col in ["density", "mean_degree", "centralization", "transitivity"]:
    print(f"\n{col}: M={metrics[col].mean():.3f} (SD={metrics[col].std():.3f}, "
          f"min={metrics[col].min():.2f}, max={metrics[col].max():.2f}, "
          f"N={metrics[col].notna().sum()})")
    print(metrics.groupby("wave")[col].agg(["mean", "std"]).round(3).to_string())

# -------------------------
# size dependence: density is an artifact of n, mean degree is not
# -------------------------
METRIC_COLS = ["density", "mean_degree", "centralization", "transitivity"]

print("\n=== Dependence on number of nodes ===")
print(metrics.groupby("n_nodes")[METRIC_COLS].agg(["mean", "count"]).round(2).to_string())


def corr(col, other="n_nodes"):
    """Pearson r on complete cases (transitivity has NaNs)."""
    sub = metrics[[other, col]].dropna()
    return pearsonr(sub[other], sub[col])


print()
r_density, p_density = corr("density")
r_degree, p_degree = corr("mean_degree")
r_central, p_central = corr("centralization")
r_trans, p_trans = corr("transitivity")
for name, r, p in [("density", r_density, p_density),
                   ("mean_degree", r_degree, p_degree),
                   ("centralization", r_central, p_central),
                   ("transitivity", r_trans, p_trans)]:
    print(f"{name:14s} vs n_nodes: r={r:+.3f} (p={p:.1e})")

# neither centralization nor transitivity is independent of density a priori
print()
for name in ["centralization", "transitivity"]:
    r, p = corr(name, other="density")
    print(f"{name:14s} vs density: r={r:+.3f} (p={p:.1e})")

metrics.round(2).to_csv(os.path.join(outdir, "network_metrics_per_network.csv"), index=False)
print(f"\nSaved {outdir}/network_metrics_per_network.csv")

# -------------------------
# Part 5: the six example networks shown in Figure 2
# -------------------------
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

# polarity composition per distinct pair, to cross-check against the figure colours
# (purple = supporting, orange = conflicting, green = both)
pair_polarities = (
    edges.groupby(["key", "wave", "pair"])["polarity"]
         .agg(lambda s: "both" if s.nunique() > 1 else s.iloc[0])
         .reset_index(name="pair_type")
)
pair_mix = (
    pair_polarities.assign(v=1)
    .pivot_table(index=["key", "wave"], columns="pair_type", values="v", aggfunc="sum")
    .fillna(0).astype(int).reset_index()
)

examples = pd.DataFrame(
    [{"Panel": p, "key": k, "wave": w} for p, (k, w) in FIGURE_2.items()]
).merge(metrics, on=["key", "wave"], how="left").merge(pair_mix, on=["key", "wave"], how="left")

examples = examples.rename(columns={
    "n_nodes": "Nodes", "n_edges": "Edges drawn", "n_pairs": "Distinct pairs",
    "mean_degree": "Mean degree", "density": "Density",
    "centralization": "Centralization", "transitivity": "Transitivity",
    "positive": "Supporting", "negative": "Conflicting", "both": "Both",
})

# full view, printed for checking against the figure (colours, dual-polarity pairs)
check_cols = ["Panel", "key", "wave", "Nodes", "Edges drawn", "Distinct pairs",
              "Supporting", "Conflicting", "Both",
              "Mean degree", "Density", "Centralization", "Transitivity"]
print("\n=== Figure 2 example networks (full, for checking) ===")
print(examples[check_cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# the table itself: "Edges" = distinct pairs, matching both the figure (a pair
# marked both ways is drawn as one green line) and the de-duplicated metrics
table = (examples.rename(columns={"Distinct pairs": "Edges"})
         [["Panel", "Nodes", "Edges", "Mean degree", "Density",
           "Centralization", "Transitivity"]])

print("\n=== Figure 2 example networks (table) ===")
print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

table.round(2).to_csv(os.path.join(outdir, "figure2_examples.csv"), index=False)
with open(os.path.join(outdir, "figure2_examples.tex"), "w", encoding="utf-8") as f:
    f.write(table.to_latex(
        index=False, float_format="%.2f", na_rep="--",
        caption=("Network metrics for the six example belief networks in Figure 2. "
                 "All metrics are computed on de-duplicated node-pairs, so a pair "
                 "marked both supporting and conflicting counts once."),
        label="tab:figure2_examples", escape=True))
print(f"Saved {outdir}/figure2_examples.csv and .tex")

# -------------------------
# Part 6: summary table across all networks
# -------------------------
# Same columns as the Figure 2 table, so the two can be read side by side.
# "Edges" is distinct pairs (de-duplicated), NOT edges as drawn -- see Part 1
# for the as-drawn counts that the manuscript text reports.
SUMMARY_COLS = {
    "Nodes": "n_nodes",
    "Edges": "n_pairs",
    "Mean degree": "mean_degree",
    "Density": "density",
    "Centralization": "centralization",
    "Transitivity": "transitivity",
}

summary = pd.DataFrame(
    {label: [metrics[col].mean(), metrics[col].std()]
     for label, col in SUMMARY_COLS.items()},
    index=["Mean", "SD"],
)

n_min = int(min(metrics[col].notna().sum() for col in SUMMARY_COLS.values()))
n_max = int(max(metrics[col].notna().sum() for col in SUMMARY_COLS.values()))
print("\n=== All networks: mean and SD ===")
print(summary.to_string(float_format=lambda v: f"{v:.2f}"))
print(f"N = {n_max} networks ({n_min} for transitivity: "
      f"{n_undefined} have no connected triples)")

summary.round(2).to_csv(os.path.join(outdir, "all_networks_summary.csv"))
with open(os.path.join(outdir, "all_networks_summary.tex"), "w", encoding="utf-8") as f:
    f.write(summary.to_latex(
        float_format="%.2f", na_rep="--",
        caption=(f"Network metrics across all {n_max} belief networks "
                 f"({nodes['key'].nunique()} participants $\\times$ 2 waves). "
                 "All metrics are computed on de-duplicated node-pairs, so a pair "
                 "marked both supporting and conflicting counts once. "
                 f"Transitivity is undefined for {n_undefined} networks with no "
                 f"connected triples, which are excluded ($N = {n_min}$)."),
        label="tab:all_networks_summary", escape=False))
print(f"Saved {outdir}/all_networks_summary.csv and .tex")

# -------------------------
# plot 1: density and mean degree
# -------------------------
TITLE_SIZE = 15
LABEL_SIZE = 14
TICK_SIZE = 12

sizes = sorted(metrics["n_nodes"].unique())


def plot_metric(ax_hist, ax_size, col, label, bins, r):
    """Histogram + by-network-size boxplot for one metric."""
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
    ax_size.set_title(f"{label} by size: r = {r:+.2f}", fontsize=TITLE_SIZE)

    for ax in (ax_hist, ax_size):
        ax.tick_params(axis="both", labelsize=TICK_SIZE)


fig, axes = plt.subplots(2, 2, figsize=(11, 8))
plot_metric(axes[0][0], axes[0][1], "density", "Density",
            np.arange(0, 1.05, 0.05), r_density)
plot_metric(axes[1][0], axes[1][1], "mean_degree", "Mean degree",
            np.arange(0, metrics["mean_degree"].max() + 0.3, 0.25), r_degree)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "density_mean_degree.svg"), bbox_inches="tight")
print(f"Saved {outdir}/density_mean_degree.svg")

# -------------------------
# plot 2: degree centralization
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_metric(axes[0], axes[1], "centralization", "Degree centralization",
            np.arange(0, 1.05, 0.05), r_central)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "degree_centralization.svg"), bbox_inches="tight")
print(f"Saved {outdir}/degree_centralization.svg")

# -------------------------
# plot 3: global clustering (transitivity)
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_metric(axes[0], axes[1], "transitivity", "Transitivity",
            np.arange(0, 1.05, 0.05), r_trans)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "transitivity.svg"), bbox_inches="tight")
print(f"Saved {outdir}/transitivity.svg")

plt.show()
