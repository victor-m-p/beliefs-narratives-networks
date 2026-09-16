"""
38_criterion.py

VMP 2026-03-15
Criterion validity: topic persistence as a function of wave-1 topic degree.
Raw participant dots + mean line (selected model) or mean lines only (all 10 models).

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/public/bertopic_mapping/edge_mapping__<label>.csv
  ../data/public/bertopic_mapping_llm/edge_mapping_llm__<label>.csv

Writes ../fig/criterion/:
  persistence_a1_canvas_selected.svg  → Figure 7 (V2, selected model, canvas)
  persistence_a1_llm_selected.svg     → Figure 7 (V2, selected model, LLM)
  persistence_a1_canvas_all10.svg     → Figure S12 (V2, all 10 models, canvas)
  persistence_a1_llm_all10.svg        → Figure S12 (V2, all 10 models, LLM)
  topic_persistence_table.tex / .csv  → Table S9
  topic_persistence.csv               (participant-level, for mixed-effects analysis)

VMP 2026-09-16: Table S9 rebuilt on the full participant x topic grid, so it
reports both retention and the rate at which a topic appears anew in wave 2
(previously 42_criterion2.py, now removed). Participant-level summary of new
topics added at the end, printed only.
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

OUTDIR = Path("../fig/criterion")
OUTDIR.mkdir(parents=True, exist_ok=True)

BINS   = [-0.5, 1.5, 3.5, np.inf]
LABELS = ["0-1", "2-3", "4+"]

FIGSIZE   = (5, 3.5)
JITTER_H  = 0.12
JITTER_V  = 0.015   # small vertical spread for 0/1 pileups


# -----------------------------
# Data helpers
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

    ax.set_xticks(x_base)
    ax.set_xticklabels(LABELS, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Topic degree in wave 1", fontsize=13)
    ax.set_ylabel("Topic persistence rate", fontsize=13)
    fig.tight_layout()
    fig.savefig(str(outpath), bbox_inches="tight")
    plt.close(fig)


def plot_all10(all_means: list[list[float]], outpath: Path,
               ylim: tuple[float, float] | None = None) -> None:
    """One mean line per model, all 10 overlaid."""
    x_base = np.arange(len(LABELS), dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for means in all_means:
        ax.plot(x_base, means, "o-", color="lightsteelblue",
                markersize=4, linewidth=1.0, alpha=0.7, zorder=2)

    if ylim is not None:
        ax.set_ylim(ylim)

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

# Plot 3 & 4: all 10 models, mean lines only — shared y-axis
_all10_flat = [v for means in canvas_means_all + llm_means_all for v in means if not np.isnan(v)]
_pad = 0.05 * (max(_all10_flat) - min(_all10_flat))
_shared_ylim = (min(_all10_flat) - _pad, max(_all10_flat) + _pad)

plot_all10(canvas_means_all, OUTDIR / "persistence_a1_canvas_all10.svg", ylim=_shared_ylim)
plot_all10(llm_means_all,    OUTDIR / "persistence_a1_llm_all10.svg",    ylim=_shared_ylim)

print("\nSaved to:", OUTDIR)


# -----------------------------
# Topic-level persistence table (rank-1 / canvas) — Table S9
# -----------------------------
# Built on the FULL participant x topic grid, not only the pairs a participant
# had in wave 1, so both conditionals can be read symmetrically:
#
#   Retained in W2   P(topic in wave 2 | topic in wave 1)      - stickiness
#   New in W2        P(topic in wave 2 | topic NOT in wave 1)  - base rate
#
# Their difference is the persistence effect. A topic can look sticky simply
# because it is common: if it appears anew at nearly the rate it is retained,
# wave-1 presence carries little information. Both denominators are reported,
# because they are complementary and sum to the participants: the topics with
# the highest retention are exactly those with the fewest participants
# available to pick them up, so those rates rest on small samples.
r1_idx   = 8                      # 0-based index for model 09
r1       = top10.iloc[r1_idx]
label_r1 = f"{r1_idx+1:02d}__{r1.embed_model_outname}__run_{r1.run_id}"
nodes_r1 = load_nodes(STMT_DIR / f"{label_r1}__statement_topics.csv")
deg_r1   = compute_degree_w1(SEL_MAP / f"edge_mapping__{label_r1}.csv")
df_r1    = build_base_df(nodes_r1, deg_r1)

w1_r1 = nodes_r1[nodes_r1["wave"] == 1][["key", "topic"]].drop_duplicates()
w2_r1 = nodes_r1[nodes_r1["wave"] == 2][["key", "topic"]].drop_duplicates()

participants = sorted(nodes_r1["key"].unique())
topics       = sorted(nodes_r1["topic"].unique())

grid = pd.MultiIndex.from_product([participants, topics],
                                  names=["key", "topic"]).to_frame(index=False)
grid = grid.merge(w1_r1.assign(in_w1=1), on=["key", "topic"], how="left")
grid = grid.merge(w2_r1.assign(in_w2=1), on=["key", "topic"], how="left")
grid = grid.merge(deg_r1, on=["key", "topic"], how="left")
grid[["in_w1", "in_w2", "degree_wt"]] = (
    grid[["in_w1", "in_w2", "degree_wt"]].fillna(0).astype(int))

had_it  = grid[grid["in_w1"] == 1]
did_not = grid[grid["in_w1"] == 0]

topic_tbl = pd.DataFrame({
    "N present (W1)":    had_it.groupby("topic").size(),
    "Retained in W2":    had_it.groupby("topic")["in_w2"].mean(),
    "N absent (W1)":     did_not.groupby("topic").size(),
    "New in W2":         did_not.groupby("topic")["in_w2"].mean(),
    "Topic degree (W1)": had_it.groupby("topic")["degree_wt"].mean(),
}).reset_index().rename(columns={"topic": "Topic"})

topic_tbl["Difference"] = topic_tbl["Retained in W2"] - topic_tbl["New in W2"]
topic_tbl = topic_tbl[["Topic", "N present (W1)", "Retained in W2",
                       "N absent (W1)", "New in W2", "Difference",
                       "Topic degree (W1)"]].sort_values("Topic")

# the two groups partition the participants
assert (topic_tbl["N present (W1)"] + topic_tbl["N absent (W1)"]
        == len(participants)).all()

# the wave-1 columns must reproduce the frame the figures above are built on
_check = df_r1.groupby("topic").agg(n=("present_w2", "size"),
                                    p=("present_w2", "mean"),
                                    d=("degree_wt", "mean")).reset_index()
_check = topic_tbl.merge(_check, left_on="Topic", right_on="topic")
assert (_check["N present (W1)"] == _check["n"]).all()
assert (_check["Retained in W2"] - _check["p"]).abs().max() < 1e-9
assert (_check["Topic degree (W1)"] - _check["d"]).abs().max() < 1e-9

print("\n=== Topic persistence: stickiness vs base rate ===")
print(topic_tbl.round(2).to_string(index=False))

pooled_retained = had_it["in_w2"].mean()
pooled_new      = did_not["in_w2"].mean()
print(f"\nPooled over all participant-topic pairs:")
print(f"  Retained in W2 : {pooled_retained:.3f}  (n={len(had_it)})")
print(f"  New in W2      : {pooled_new:.3f}  (n={len(did_not)})")
print(f"  Difference     : {pooled_retained - pooled_new:+.3f}")

topic_tbl.round(2).to_csv(OUTDIR / "topic_persistence_table.csv", index=False)
topic_tbl.to_latex(OUTDIR / "topic_persistence_table.tex", index=False, float_format="%.2f")
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
print("Saved to:", OUTDIR / "topic_persistence.csv")


# -----------------------------
# Participant-level: how many topics are new in wave 2?
# -----------------------------
# The topic-level table asks, per topic, how often it is retained or appears
# anew. A reviewer could equally mean the participant-level reading: how much of
# an individual's wave-2 belief set is new. Printed only, not saved.
w1_topics = w1_r1.groupby("key")["topic"].apply(set).reindex(participants)
w2_topics = w2_r1.groupby("key")["topic"].apply(set).reindex(participants)

assert w1_topics.notna().all() and w2_topics.notna().all()

per = pd.DataFrame({"key": participants})
per["n_w1"]  = [len(s) for s in w1_topics]
per["n_w2"]  = [len(s) for s in w2_topics]
per["n_new"] = [len(b - a) for a, b in zip(w1_topics, w2_topics)]

# fraction of the wave-2 topic set that was not present in wave 1
per["frac_of_w2"] = per["n_new"] / per["n_w2"]
per["any_new"]    = (per["n_new"] > 0).astype(int)

summary = (per[["n_w1", "n_w2", "n_new", "frac_of_w2", "any_new"]]
           .agg(["mean", "std", "min", "max"]).T
           .rename(index={
               "n_w1":       "Topics in W1",
               "n_w2":       "Topics in W2",
               "n_new":      "New topics in W2 (count)",
               "frac_of_w2": "New / topics in W2 (proportion)",
               "any_new":    "Has any new topic (0/1)",
           }))

print(f"\n=== Participant-level (N={len(per)}) ===")
print(summary.round(2).to_string())

n_any = int(per["any_new"].sum())
print(f"\nParticipants with at least one new topic in W2: "
      f"{n_any} of {len(per)} ({n_any / len(per) * 100:.1f}%)")
print(f"Participants with no new topics:                 {len(per) - n_any}")

# does the count of new topics just track how much a participant mentions?
print(f"\ncorr(new topics, topics in W2) = {per['n_new'].corr(per['n_w2']):+.2f}")
