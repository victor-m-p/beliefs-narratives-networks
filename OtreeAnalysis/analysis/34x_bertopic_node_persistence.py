"""
VMP 2026-03-15

Transparent raw-data plots for A1 topic persistence.
Companion to 34_bertopic_node_persistence.py — no SE bars, just raw participant
dots + mean line (selected model) or mean lines only (all 10 models).

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/public/bertopic_mapping/edge_mapping__<label>.csv
  ../data/public/bertopic_mapping_llm/edge_mapping_llm__<label>.csv

Writes ../fig/topic_persistence/:
  persistence_a1_canvas_selected.svg  — rank-1, canvas, raw dots + mean
  persistence_a1_llm_selected.svg     — rank-1, LLM,    raw dots + mean
  persistence_a1_canvas_all10.svg     — all 10 models, canvas, mean lines only
  persistence_a1_llm_all10.svg        — all 10 models, LLM,    mean lines only
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helpers import normalize_ab

# -----------------------------
# Config
# -----------------------------
SEL_TOPICS  = Path("../data/public/bertopic/selection")
SEL_MAP     = Path("../data/public/bertopic_mapping")
SEL_MAP_LLM = Path("../data/public/bertopic_mapping_llm")

TOP10_PATH = SEL_TOPICS / "overview_top10.csv"
STMT_DIR   = SEL_TOPICS / "statement_topics"

OUTDIR = Path("../fig/topic_persistence")
OUTDIR.mkdir(parents=True, exist_ok=True)

BINS   = [-0.5, 1.5, 3.5, 5.5, np.inf]
LABELS = ["0-1", "2-3", "4-5", "6+"]

FIGSIZE   = (5, 3.5)
JITTER_H  = 0.12
JITTER_V  = 0.015   # small vertical spread for 0/1 pileups


# -----------------------------
# Data helpers (mirrors 34_)
# -----------------------------
def compute_degree_w1(edge_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(edge_csv)[["wave", "key", "topic_1", "topic_2"]]
    df = df[df["wave"] == 1].copy()
    df = normalize_ab(df, "topic_1", "topic_2")
    df = df.groupby(["key", "topic_1", "topic_2"], as_index=False).size().rename(columns={"size": "n_edges"})
    deg1 = df.groupby(["key", "topic_1"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_1": "topic"})
    cross = df[df["topic_1"] != df["topic_2"]]
    deg2 = cross.groupby(["key", "topic_2"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_2": "topic"})
    return pd.concat([deg1, deg2], ignore_index=True).groupby(["key", "topic"], as_index=False)["degree_wt"].sum()


def compute_degree_unweighted(edge_csv: Path) -> pd.DataFrame:
    """
    Unweighted topic degree: number of distinct topic-level connections
    incident on each topic (each topic pair counts once, regardless of how
    many statement connections underlie it).
    """
    df = pd.read_csv(edge_csv)[["wave", "key", "topic_1", "topic_2"]]
    df = df[df["wave"] == 1].copy()
    df = normalize_ab(df, "topic_1", "topic_2")
    # collapse to distinct topic pairs
    df = df[["key", "topic_1", "topic_2"]].drop_duplicates()
    deg1 = df.groupby(["key", "topic_1"]).size().reset_index(name="degree_wt").rename(columns={"topic_1": "topic"})
    cross = df[df["topic_1"] != df["topic_2"]]
    deg2 = cross.groupby(["key", "topic_2"]).size().reset_index(name="degree_wt").rename(columns={"topic_2": "topic"})
    return pd.concat([deg1, deg2], ignore_index=True).groupby(["key", "topic"], as_index=False)["degree_wt"].sum()


def load_nodes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)[["key", "wave", "topic"]]
    return df[df["wave"].isin([1, 2])].copy()


def build_base_df(df_nodes: pd.DataFrame, deg_w1: pd.DataFrame) -> pd.DataFrame:
    w1 = df_nodes[df_nodes["wave"] == 1][["key", "topic"]].drop_duplicates()
    w2 = df_nodes[df_nodes["wave"] == 2][["key", "topic"]].drop_duplicates().assign(present_w2=1)
    df = w1.merge(deg_w1, on=["key", "topic"], how="left").merge(w2, on=["key", "topic"], how="left")
    df["degree_wt"]  = df["degree_wt"].fillna(0).astype(int)
    df["present_w2"] = df["present_w2"].fillna(0).astype(int)
    df["deg_bin"] = pd.Categorical(
        pd.cut(df["degree_wt"], bins=BINS, labels=LABELS),
        categories=LABELS, ordered=True,
    )
    return df


def get_pp_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Participant-level mean persistence per degree bin."""
    return (
        df.groupby(["key", "deg_bin"], observed=True)["present_w2"]
        .mean()
        .reset_index(name="p_pp")
    )


def get_baseline(df: pd.DataFrame) -> float:
    return float(df.groupby("key")["present_w2"].mean().mean())


def bin_means(pp_bin: pd.DataFrame) -> list[float]:
    return [
        pp_bin.loc[pp_bin["deg_bin"] == b, "p_pp"].mean()
        for b in LABELS
    ]


# -----------------------------
# Plot helpers
# -----------------------------
def plot_selected(pp_bin: pd.DataFrame, baseline: float, outpath: Path) -> None:
    """Raw participant dots + mean line + baseline for one model."""
    rng    = np.random.default_rng(42)
    x_base = np.arange(len(LABELS), dtype=float)

    data_by_bin = [
        pp_bin.loc[pp_bin["deg_bin"] == b, "p_pp"].dropna().to_numpy()
        for b in LABELS
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # raw dots
    for i, yvals in enumerate(data_by_bin):
        if len(yvals) == 0:
            continue
        xj = x_base[i] + rng.uniform(-JITTER_H, JITTER_H, size=len(yvals))
        yj = yvals + rng.uniform(-JITTER_V, JITTER_V, size=len(yvals))
        ax.scatter(xj, yj, s=14, alpha=0.25, color="lightsteelblue", linewidths=0, zorder=2)

    # mean line
    means = [yvals.mean() if len(yvals) > 0 else np.nan for yvals in data_by_bin]
    ax.plot(x_base, means, "o-", color="black", markersize=7, linewidth=1.4, zorder=4)

    # baseline
    ax.axhline(baseline, linestyle="--", linewidth=1.2, color="gray", zorder=3)

    ax.set_xticks(x_base)
    ax.set_xticklabels(LABELS, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Topic degree in wave 1", fontsize=13)
    ax.set_ylabel("Topic persistence rate", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(outpath), bbox_inches="tight")
    plt.close(fig)


def plot_all10(all_means: list[list[float]], outpath: Path) -> None:
    """One mean line per model, all 10 overlaid."""
    x_base = np.arange(len(LABELS), dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for means in all_means:
        ax.plot(x_base, means, "o-", color="lightsteelblue",
                markersize=4, linewidth=1.0, alpha=0.7, zorder=2)

    ax.set_xticks(x_base)
    ax.set_xticklabels(LABELS, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Topic degree in wave 1", fontsize=13)
    ax.set_ylabel("Topic persistence rate", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(outpath), bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Run
# -----------------------------
top10 = pd.read_csv(TOP10_PATH)

canvas_means_all, llm_means_all = [], []
pp_bin_c1 = pp_bin_l1 = pd.DataFrame()
baseline_c1 = baseline_l1 = float("nan")

for rank, r in enumerate(top10.itertuples(index=False), 1):
    label    = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    stmt_csv = STMT_DIR / f"{label}__statement_topics.csv"

    # canvas
    df_c     = build_base_df(load_nodes(stmt_csv), compute_degree_w1(SEL_MAP / f"edge_mapping__{label}.csv"))
    pp_bin_c = get_pp_bin(df_c)
    canvas_means_all.append(bin_means(pp_bin_c))

    # LLM
    df_l     = build_base_df(load_nodes(stmt_csv), compute_degree_w1(SEL_MAP_LLM / f"edge_mapping_llm__{label}.csv"))
    pp_bin_l = get_pp_bin(df_l)
    llm_means_all.append(bin_means(pp_bin_l))

    if rank == 1:
        pp_bin_c1, baseline_c1 = pp_bin_c, get_baseline(df_c)
        pp_bin_l1, baseline_l1 = pp_bin_l, get_baseline(df_l)

    print(f"[ok] rank {rank}")

# Plot 1 & 2: selected model raw dots
plot_selected(pp_bin_c1, baseline_c1, OUTDIR / "persistence_a1_canvas_selected.svg")
plot_selected(pp_bin_l1, baseline_l1, OUTDIR / "persistence_a1_llm_selected.svg")

# Plot 3 & 4: all 10 models, mean lines only
plot_all10(canvas_means_all, OUTDIR / "persistence_a1_canvas_all10.svg")
plot_all10(llm_means_all,    OUTDIR / "persistence_a1_llm_all10.svg")

print("\nSaved to:", OUTDIR)


# -----------------------------
# Topic-level persistence table (rank-1 / canvas)
# -----------------------------
# For each topic: P(present in wave 2 | present in wave 1), averaged over
# participants who have the topic in wave 1.
r1       = top10.iloc[0]
label_r1 = f"01__{r1.embed_model_outname}__run_{r1.run_id}"
df_r1    = build_base_df(
    load_nodes(STMT_DIR / f"{label_r1}__statement_topics.csv"),
    compute_degree_w1(SEL_MAP / f"edge_mapping__{label_r1}.csv"),
)

mean_degree = df_r1.groupby("topic")["degree_wt"].mean().rename("mean_degree")

topic_tbl = (
    df_r1
    .groupby("topic")["present_w2"]
    .agg(n_participants="count", p_persist="mean")
    .join(mean_degree)
    .reset_index()
    .sort_values("topic")
    .round({"p_persist": 2, "mean_degree": 2})
)
topic_tbl.to_latex(OUTDIR / "topic_persistence_table.tex", index=False, float_format="%.2f")
print(topic_tbl.to_string(index=False))
print("\nSaved topic table to:", OUTDIR / "topic_persistence_table.tex")

# -----------------------------
# Participant-level CSV for mixed-effects analysis
# -----------------------------
deg_unwt = compute_degree_unweighted(SEL_MAP / f"edge_mapping__{label_r1}.csv")

export_df = (
    df_r1[["key", "topic", "degree_wt", "present_w2"]]
    .merge(deg_unwt.rename(columns={"degree_wt": "degree_unwt"}), on=["key", "topic"], how="left")
    .fillna({"degree_unwt": 0})
    .assign(degree_unwt=lambda d: d["degree_unwt"].astype(int))
    [["key", "topic", "degree_unwt", "degree_wt", "present_w2"]]
    .rename(columns={
        "key":          "participant_id",
        "topic":        "topic_id",
        "degree_unwt":  "degree_unweighted",
        "degree_wt":    "degree_weighted",
        "present_w2":   "present_wave2",
    })
    .sort_values(["participant_id", "topic_id"])
)

export_df.to_csv(OUTDIR / "topic_persistence.csv", index=False)
print(f"\nSaved mixed-effects CSV: {len(export_df)} rows, {export_df['participant_id'].nunique()} participants")
print("Saved to:", OUTDIR / "topic_persistence_mixed.csv")

# -----------------------------
# Extra: no-outlier and unweighted variants (canvas, selected + all 10)
# -----------------------------

def compute_degree_no_outlier(edge_csv: Path) -> pd.DataFrame:
    """Weighted degree, excluding all edges that touch topic -1."""
    df = pd.read_csv(edge_csv)[["wave", "key", "topic_1", "topic_2"]]
    df = df[df["wave"] == 1].copy()
    df = normalize_ab(df, "topic_1", "topic_2")
    df = df[(df["topic_1"] != -1) & (df["topic_2"] != -1)]
    df = df.groupby(["key", "topic_1", "topic_2"], as_index=False).size().rename(columns={"size": "n_edges"})
    deg1 = df.groupby(["key", "topic_1"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_1": "topic"})
    cross = df[df["topic_1"] != df["topic_2"]]
    deg2 = cross.groupby(["key", "topic_2"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_2": "topic"})
    return pd.concat([deg1, deg2], ignore_index=True).groupby(["key", "topic"], as_index=False)["degree_wt"].sum()


no_out_means_all, unwt_means_all = [], []
pp_bin_no_out = pp_bin_unwt = pd.DataFrame()
baseline_no_out = baseline_unwt = float("nan")

for rank, r in enumerate(top10.itertuples(index=False), 1):
    label    = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    stmt_csv = STMT_DIR / f"{label}__statement_topics.csv"
    edge_csv = SEL_MAP / f"edge_mapping__{label}.csv"

    # no-outlier: filter topic -1 from nodes too
    nodes_no_out = load_nodes(stmt_csv)
    nodes_no_out = nodes_no_out[nodes_no_out["topic"] != -1]
    df_no_out = build_base_df(nodes_no_out, compute_degree_no_outlier(edge_csv))
    pp_no_out = get_pp_bin(df_no_out)
    no_out_means_all.append(bin_means(pp_no_out))

    # unweighted
    df_unwt = build_base_df(load_nodes(stmt_csv), compute_degree_unweighted(edge_csv))
    pp_unwt = get_pp_bin(df_unwt)
    unwt_means_all.append(bin_means(pp_unwt))

    if rank == 1:
        pp_bin_no_out   = pp_no_out
        baseline_no_out = get_baseline(df_no_out)
        pp_bin_unwt     = pp_unwt
        baseline_unwt   = get_baseline(df_unwt)

    print(f"[ok-extra] rank {rank}")

# Selected model plots
plot_selected(pp_bin_no_out, baseline_no_out, OUTDIR / "persistence_a1_canvas_no_outlier.svg")
plot_selected(pp_bin_unwt,   baseline_unwt,   OUTDIR / "persistence_a1_canvas_unweighted.svg")

# All-10 spaghetti plots
plot_all10(no_out_means_all, OUTDIR / "persistence_a1_canvas_no_outlier_all10.svg")
plot_all10(unwt_means_all,   OUTDIR / "persistence_a1_canvas_unweighted_all10.svg")

print("\nSaved extra plots to:", OUTDIR)
