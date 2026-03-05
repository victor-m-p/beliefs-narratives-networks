"""
VMP 2026-02-02 (refactored 2026-03-05)

Topic persistence vs wave-1 topic degree, for TOP10 selected runs.
Two analyses per model:
  A1 — person-weighted persistence curve + unconditional baseline
  A2 — topic-adjusted (other-participant baseline) persistence curve

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/public/bertopic_mapping/edge_mapping__<label>.csv
  ../data/public/bertopic_mapping_llm/edge_mapping_llm__<label>.csv

Writes (run twice; toggle REMOVE_OUTLIER_TOPIC):
  ../fig/BERTopic/topic_persist/outlier_{remove|include}/
    <label>__persistence_a1.{pdf,svg}
    <label>__persistence_a2.{pdf,svg}
    overview_top10__persistence_degree.csv

  ../fig/BERTopic/topic_persist_llm/outlier_{remove|include}/
    <label>__persistence_llm_a1.{pdf,svg}
    <label>__persistence_llm_a2.{pdf,svg}
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
REMOVE_OUTLIER_TOPIC = False  # toggle manually

SEL_TOPICS = Path("../data/public/bertopic/selection")
SEL_MAP    = Path("../data/public/bertopic_mapping")
SEL_MAP_LLM = Path("../data/public/bertopic_mapping_llm")

TOP10_PATH = SEL_TOPICS / "overview_top10.csv"
STMT_DIR   = SEL_TOPICS / "statement_topics"

OUT_BASE     = Path("../fig/BERTopic/topic_persist")
LLM_OUT_BASE = Path("../fig/BERTopic/topic_persist_llm")
WIPE_OUTDIR  = True

BINS   = [-0.5, 1.5, 3.5, 5.5, np.inf]
LABELS = ["0-1", "2-3", "4-5", "6+"]
ORDER  = ["0-1", "2-3", "4-5", "6+"]

FIGSIZE = (3.5, 2.5)


# -----------------------------
# Helpers
# -----------------------------
def compute_degree_w1(edge_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(edge_csv)[["wave", "key", "topic_1", "topic_2"]].copy()
    if REMOVE_OUTLIER_TOPIC:
        df = df[(df["topic_1"] != -1) & (df["topic_2"] != -1)].copy()
    df = df[df["wave"] == 1].copy()
    df["key"]     = df["key"].astype(str)
    df["topic_1"] = pd.to_numeric(df["topic_1"], errors="coerce").astype("Int64")
    df["topic_2"] = pd.to_numeric(df["topic_2"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["topic_1", "topic_2"]).copy()
    df["topic_1"] = df["topic_1"].astype(int)
    df["topic_2"] = df["topic_2"].astype(int)
    df = normalize_ab(df, "topic_1", "topic_2")
    df = df.groupby(["key", "topic_1", "topic_2"], as_index=False).size().rename(columns={"size": "n_edges"})
    deg1 = df.groupby(["key", "topic_1"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_1": "topic"})
    deg2 = df.groupby(["key", "topic_2"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_2": "topic"})
    out  = pd.concat([deg1, deg2], ignore_index=True).groupby(["key", "topic"], as_index=False)["degree_wt"].sum()
    out["degree_wt"] = out["degree_wt"].astype(int)
    out["topic"]     = out["topic"].astype(int)
    out["key"]       = out["key"].astype(str)
    return out


def load_nodes(stmt_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(stmt_csv)[["key", "wave", "topic"]].copy()
    if REMOVE_OUTLIER_TOPIC:
        df = df[df["topic"] != -1].copy()
    df["key"]   = df["key"].astype(str)
    df["wave"]  = pd.to_numeric(df["wave"],  errors="raise").astype(int)
    df["topic"] = pd.to_numeric(df["topic"], errors="raise").astype(int)
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
        categories=ORDER, ordered=True,
    )
    return df


def compute_a1(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Person-weighted persistence curve. Returns (agg, baseline)."""
    pp_bin = df.groupby(["key", "deg_bin"], observed=True)["present_w2"].mean().reset_index(name="p_pp")
    agg = pp_bin.groupby("deg_bin", observed=True)["p_pp"].agg(p="mean", n="size", sd="std").reset_index()
    agg = agg.set_index("deg_bin").reindex(ORDER).reset_index()
    agg["se"]    = agg["sd"] / np.sqrt(agg["n"])
    agg["ci_lo"] = agg["p"] - 1.96 * agg["se"]
    agg["ci_hi"] = agg["p"] + 1.96 * agg["se"]
    baseline = df.groupby("key")["present_w2"].mean().mean()
    return agg, baseline


def compute_a2(df: pd.DataFrame) -> pd.DataFrame:
    """Topic-adjusted (other-participant baseline) persistence curve."""
    topic_sum = df.groupby("topic")["present_w2"].transform("sum")
    topic_n   = df.groupby("topic")["present_w2"].transform("count")
    resid_y   = df["present_w2"] - (topic_sum - df["present_w2"]) / (topic_n - 1)
    agg = df.assign(resid_y=resid_y).groupby("deg_bin", observed=True)["resid_y"].agg(
        p="mean", n="size", sd="std").reset_index()
    agg = agg.set_index("deg_bin").reindex(ORDER).reset_index()
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
    ax.legend(fontsize=9)
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
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, stem)


def p_map(agg: pd.DataFrame) -> dict:
    return dict(zip(agg["deg_bin"].astype(str), agg["p"]))


# -----------------------------
# Canvas edges
# -----------------------------
_suffix = "outlier_remove" if REMOVE_OUTLIER_TOPIC else "outlier_include"
OUTDIR = OUT_BASE / _suffix
if WIPE_OUTDIR:
    shutil.rmtree(OUTDIR, ignore_errors=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

top10 = pd.read_csv(TOP10_PATH)

rows = []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    label    = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    edge_csv = SEL_MAP  / f"edge_mapping__{label}.csv"
    stmt_csv = STMT_DIR / f"{label}__statement_topics.csv"

    if not edge_csv.exists() or not stmt_csv.exists():
        print("[skip]", label)
        continue

    df = build_base_df(load_nodes(stmt_csv), compute_degree_w1(edge_csv))

    agg1, baseline = compute_a1(df)
    agg2           = compute_a2(df)

    plot_a1(agg1, baseline, OUTDIR / f"{label}__persistence_a1")
    plot_a2(agg2,           OUTDIR / f"{label}__persistence_a2")

    pm1, pm2 = p_map(agg1), p_map(agg2)
    rows.append(dict(
        label=label,
        remove_outlier_topic=REMOVE_OUTLIER_TOPIC,
        baseline=baseline,
        a1_p01=pm1.get("0-1"), a1_p23=pm1.get("2-3"), a1_p45=pm1.get("4-5"), a1_p6p=pm1.get("6+"),
        a1_delta=pm1.get("6+", np.nan) - baseline,
        a2_p01=pm2.get("0-1"), a2_p23=pm2.get("2-3"), a2_p45=pm2.get("4-5"), a2_p6p=pm2.get("6+"),
    ))
    print("[ok]", label)

pd.DataFrame(rows).to_csv(OUTDIR / "overview_top10__persistence_degree.csv", index=False)
print("\nSaved:", OUTDIR)


# -----------------------------
# LLM edges
# -----------------------------
LLM_OUTDIR = LLM_OUT_BASE / _suffix
if WIPE_OUTDIR:
    shutil.rmtree(LLM_OUTDIR, ignore_errors=True)
LLM_OUTDIR.mkdir(parents=True, exist_ok=True)

rows_llm = []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    label    = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    edge_csv = SEL_MAP_LLM / f"edge_mapping_llm__{label}.csv"
    stmt_csv = STMT_DIR     / f"{label}__statement_topics.csv"

    if not edge_csv.exists() or not stmt_csv.exists():
        print("[skip-llm]", label)
        continue

    df = build_base_df(load_nodes(stmt_csv), compute_degree_w1(edge_csv))

    agg1, baseline = compute_a1(df)
    agg2           = compute_a2(df)

    plot_a1(agg1, baseline, LLM_OUTDIR / f"{label}__persistence_llm_a1")
    plot_a2(agg2,           LLM_OUTDIR / f"{label}__persistence_llm_a2")

    pm1, pm2 = p_map(agg1), p_map(agg2)
    rows_llm.append(dict(
        label=label,
        remove_outlier_topic=REMOVE_OUTLIER_TOPIC,
        baseline=baseline,
        a1_p01=pm1.get("0-1"), a1_p23=pm1.get("2-3"), a1_p45=pm1.get("4-5"), a1_p6p=pm1.get("6+"),
        a1_delta=pm1.get("6+", np.nan) - baseline,
        a2_p01=pm2.get("0-1"), a2_p23=pm2.get("2-3"), a2_p45=pm2.get("4-5"), a2_p6p=pm2.get("6+"),
    ))
    print("[ok-llm]", label)

pd.DataFrame(rows_llm).to_csv(LLM_OUTDIR / "overview_top10__persistence_degree_llm.csv", index=False)
print("\nSaved LLM:", LLM_OUTDIR)
