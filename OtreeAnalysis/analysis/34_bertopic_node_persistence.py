"""
VMP 2026-02-02 (refactored 2026-03-09)

Topic persistence vs wave-1 topic degree, for TOP10 selected runs.
Two analyses per model:
  A1 — person-weighted persistence curve + unconditional baseline
  A2 — topic-adjusted (other-participant baseline) persistence curve

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/public/bertopic_mapping/edge_mapping__<label>.csv
  ../data/public/bertopic_mapping_llm/edge_mapping_llm__<label>.csv

Writes:
  ../fig/BERTopic/topic_persist/
    <label>__persistence_a1.{pdf,svg}
    <label>__persistence_a2.{pdf,svg}
    summary__persistence_a1.{pdf,svg}
    summary__persistence_a2.{pdf,svg}
    overview_top10__persistence_degree.csv

  ../fig/BERTopic/topic_persist_llm/
    <label>__persistence_llm_a1.{pdf,svg}
    <label>__persistence_llm_a2.{pdf,svg}
    summary__persistence_llm_a1.{pdf,svg}
    summary__persistence_llm_a2.{pdf,svg}
    overview_top10__persistence_degree_llm.csv
"""

from __future__ import annotations

import shutil
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

OUTDIR     = Path("../fig/BERTopic/topic_persist")
LLM_OUTDIR = Path("../fig/BERTopic/topic_persist_llm")
WIPE_OUTDIR = True

BINS   = [-0.5, 1.5, 3.5, 5.5, np.inf]
LABELS = ["0-1", "2-3", "4-5", "6+"]

FIGSIZE = (3.5, 2.5)


# -----------------------------
# Helpers
# -----------------------------
def compute_degree_w1(edge_csv: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [key, topic, degree_wt] giving the
    weighted degree of each topic in each participant's wave-1 belief network.

    The edge CSV has one row per statement-level canvas connection. Each
    connection is mapped to a (topic_1, topic_2) pair. The weighted degree of
    a topic is the total number of canvas connections incident on it:
      - cross-topic edge (T1, T2): contributes 1 to T1 and 1 to T2.
      - self-loop (T, T): contributes 1 to T (counted once, not twice).
    """
    df = pd.read_csv(edge_csv)[["wave", "key", "topic_1", "topic_2"]]
    df = df[df["wave"] == 1].copy()

    # consistent ordering for T1, T2 for aggregation
    df = normalize_ab(df, "topic_1", "topic_2")

    # Count statement connections per (key, topic_1, topic_2) triple
    df = df.groupby(["key", "topic_1", "topic_2"], as_index=False).size().rename(columns={"size": "n_edges"})

    # Degree contribution from topic_1 side (includes self-loops once)
    deg1 = df.groupby(["key", "topic_1"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_1": "topic"})

    # Degree contribution from topic_2 side (cross-topic edges only — self-loops already counted above)
    cross = df[df["topic_1"] != df["topic_2"]]
    deg2 = cross.groupby(["key", "topic_2"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_2": "topic"})

    out = pd.concat([deg1, deg2], ignore_index=True).groupby(["key", "topic"], as_index=False)["degree_wt"].sum()
    return out


def load_nodes(stmt_csv: Path) -> pd.DataFrame:
    """Returns [key, wave, topic] for waves 1 and 2."""
    df = pd.read_csv(stmt_csv)[["key", "wave", "topic"]]
    return df[df["wave"].isin([1, 2])].copy()


def build_base_df(df_nodes: pd.DataFrame, deg_w1: pd.DataFrame) -> pd.DataFrame:
    """One row per (participant × topic) present in Wave 1."""
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


def compute_a1(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    # per-participant weighted degree bins.
    pp_bin = df.groupby(["key", "deg_bin"], observed=True)["present_w2"].mean().reset_index(name="p_pp")
    agg = pp_bin.groupby("deg_bin", observed=True)["p_pp"].agg(p="mean", n="size", sd="std").reset_index()
    
    # 95% CI.
    agg["se"] = agg["sd"] / np.sqrt(agg["n"])
    agg["ci_lo"] = agg["p"] - 1.96 * agg["se"]
    agg["ci_hi"] = agg["p"] + 1.96 * agg["se"]
    
    # baseline: P(topic in wave 2 | topic in wave 1)
    baseline = df.groupby("key")["present_w2"].mean().mean()
    return agg, baseline


def compute_a2(df: pd.DataFrame) -> pd.DataFrame:
    """Topic-adjusted persistence curve, person-weighted (mirrors A1)."""
    topic_sum = df.groupby("topic")["present_w2"].transform("sum")
    topic_n   = df.groupby("topic")["present_w2"].transform("count")
    resid_y   = df["present_w2"] - (topic_sum - df["present_w2"]) / (topic_n - 1)
    pp_bin = df.assign(resid_y=resid_y).groupby(["key", "deg_bin"], observed=True)["resid_y"].mean().reset_index(name="r_pp")
    agg = pp_bin.groupby("deg_bin", observed=True)["r_pp"].agg(p="mean", n="size", sd="std").reset_index()
    agg = agg.set_index("deg_bin").reindex(LABELS).reset_index()

    # 95% CI.
    agg["se"]    = agg["sd"] / np.sqrt(agg["n"])
    agg["ci_lo"] = agg["p"] - 1.96 * agg["se"]
    agg["ci_hi"] = agg["p"] + 1.96 * agg["se"]
    return agg


def save_fig(fig, stem: Path) -> None:
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    plt.close(fig)


def plot_a1(agg: pd.DataFrame, baseline: float, stem: Path) -> None:
    x, y = agg["deg_bin"].astype(str), agg["p"].to_numpy(float)
    lo, hi = agg["ci_lo"].to_numpy(float), agg["ci_hi"].to_numpy(float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3)
    ax.axhline(baseline, linestyle="--", linewidth=1.2, color="gray",
               label=f"Baseline = {baseline:.2f}")
    ax.set_xlabel("Topic degree in wave 1")
    ax.set_ylabel("P(topic in wave 2)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_fig(fig, stem)


def plot_a2(agg: pd.DataFrame, stem: Path) -> None:
    x, y = agg["deg_bin"].astype(str), agg["p"].to_numpy(float)
    lo, hi = agg["ci_lo"].to_numpy(float), agg["ci_hi"].to_numpy(float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3)
    ax.axhline(0, linestyle="--", linewidth=1.0, color="gray", label="0 = no excess")
    ax.set_xlabel("Topic degree in wave 1")
    ax.set_ylabel("Excess topic persistence")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_fig(fig, stem)


def p_map(agg: pd.DataFrame) -> dict:
    return dict(zip(agg["deg_bin"].astype(str), agg["p"]))


def plot_summary_a1(records: list[dict], stem: Path) -> None:
    """
    Spaghetti + main-model plot for A1.
    records: list of dicts with keys {rank, agg, baseline}.
    Rank-1 model is highlighted; all others drawn as thin grey lines.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = LABELS

    first_grey = True
    for rec in records:
        agg = rec["agg"]
        y = agg.set_index("deg_bin").reindex(LABELS)["p"].to_numpy(float)
        if rec["rank"] == 1:
            main_rec = rec
        else:
            lbl = "Candidate models" if first_grey else "_nolegend_"
            ax.plot(x, y, color="0.75", linewidth=0.8, zorder=1, label=lbl)
            first_grey = False

    agg1 = main_rec["agg"].set_index("deg_bin").reindex(LABELS).reset_index()
    y  = agg1["p"].to_numpy(float)
    lo = agg1["ci_lo"].to_numpy(float)
    hi = agg1["ci_hi"].to_numpy(float)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3, zorder=3,
                label="Selected model")

    ax.set_xlabel("Topic degree in wave 1")
    ax.set_ylabel("P(topic in wave 2)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_fig(fig, stem)


def plot_summary_a2(records: list[dict], stem: Path) -> None:
    """
    Spaghetti + main-model plot for A2.
    records: list of dicts with keys {rank, agg}.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = LABELS

    first_grey = True
    for rec in records:
        agg = rec["agg"]
        y = agg.set_index("deg_bin").reindex(LABELS)["p"].to_numpy(float)
        if rec["rank"] == 1:
            main_rec = rec
        else:
            lbl = "Candidate models" if first_grey else "_nolegend_"
            ax.plot(x, y, color="0.75", linewidth=0.8, zorder=1, label=lbl)
            first_grey = False

    agg2 = main_rec["agg"].set_index("deg_bin").reindex(LABELS).reset_index()
    y  = agg2["p"].to_numpy(float)
    lo = agg2["ci_lo"].to_numpy(float)
    hi = agg2["ci_hi"].to_numpy(float)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3, zorder=3,
                label="Selected model")
    ax.axhline(0, linestyle="--", linewidth=1.0, color="gray", zorder=2)

    ax.set_xlabel("Topic degree in wave 1")
    ax.set_ylabel("Excess topic persistence")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_fig(fig, stem)


# -----------------------------
# Canvas edges
# -----------------------------
if WIPE_OUTDIR:
    shutil.rmtree(OUTDIR, ignore_errors=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

top10 = pd.read_csv(TOP10_PATH)

rows = []
summary_a1, summary_a2 = [], []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    label = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    edge_csv = SEL_MAP  / f"edge_mapping__{label}.csv"
    stmt_csv = STMT_DIR / f"{label}__statement_topics.csv"

    df = build_base_df(load_nodes(stmt_csv), compute_degree_w1(edge_csv))

    agg1, baseline = compute_a1(df)
    agg2 = compute_a2(df)

    plot_a1(agg1, baseline, OUTDIR / f"{label}__persistence_a1")
    plot_a2(agg2, OUTDIR / f"{label}__persistence_a2")

    summary_a1.append(dict(rank=rank, agg=agg1, baseline=baseline))
    summary_a2.append(dict(rank=rank, agg=agg2))

    pm1, pm2 = p_map(agg1), p_map(agg2)
    rows.append(dict(
        label=label,
        baseline=baseline,
        a1_p01=pm1.get("0-1"), a1_p23=pm1.get("2-3"), a1_p45=pm1.get("4-5"), a1_p6p=pm1.get("6+"),
        a1_delta=pm1.get("6+", np.nan) - baseline,
        a2_p01=pm2.get("0-1"), a2_p23=pm2.get("2-3"), a2_p45=pm2.get("4-5"), a2_p6p=pm2.get("6+"),
    ))
    print("[ok]", label)

plot_summary_a1(summary_a1, OUTDIR / "summary__persistence_a1")
plot_summary_a2(summary_a2, OUTDIR / "summary__persistence_a2")

pd.DataFrame(rows).to_csv(OUTDIR / "overview_top10__persistence_degree.csv", index=False)
print("\nSaved:", OUTDIR)

# -----------------------------
# LLM edges
# -----------------------------
if WIPE_OUTDIR:
    shutil.rmtree(LLM_OUTDIR, ignore_errors=True)
LLM_OUTDIR.mkdir(parents=True, exist_ok=True)

rows_llm = []
summary_llm_a1, summary_llm_a2 = [], []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    label    = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    edge_csv = SEL_MAP_LLM / f"edge_mapping_llm__{label}.csv"
    stmt_csv = STMT_DIR     / f"{label}__statement_topics.csv"

    df = build_base_df(load_nodes(stmt_csv), compute_degree_w1(edge_csv))

    agg1, baseline = compute_a1(df)
    agg2           = compute_a2(df)

    plot_a1(agg1, baseline, LLM_OUTDIR / f"{label}__persistence_llm_a1")
    plot_a2(agg2,           LLM_OUTDIR / f"{label}__persistence_llm_a2")

    summary_llm_a1.append(dict(rank=rank, agg=agg1, baseline=baseline))
    summary_llm_a2.append(dict(rank=rank, agg=agg2))

    pm1, pm2 = p_map(agg1), p_map(agg2)
    rows_llm.append(dict(
        label=label,
        baseline=baseline,
        a1_p01=pm1.get("0-1"), a1_p23=pm1.get("2-3"), a1_p45=pm1.get("4-5"), a1_p6p=pm1.get("6+"),
        a1_delta=pm1.get("6+", np.nan) - baseline,
        a2_p01=pm2.get("0-1"), a2_p23=pm2.get("2-3"), a2_p45=pm2.get("4-5"), a2_p6p=pm2.get("6+"),
    ))
    print("[ok-llm]", label)

plot_summary_a1(summary_llm_a1, LLM_OUTDIR / "summary__persistence_llm_a1")
plot_summary_a2(summary_llm_a2, LLM_OUTDIR / "summary__persistence_llm_a2")

pd.DataFrame(rows_llm).to_csv(LLM_OUTDIR / "overview_top10__persistence_degree_llm.csv", index=False)
print("\nSaved LLM:", LLM_OUTDIR)
