"""
VMP 2026-02-02

Topic persistence vs wave-1 topic degree (0,1,2,3,4+),
ONLY for TOP10 selected runs.

Reads:
  ../data/data-<date_w2>/topics_bertopic/selection/overview_top10.csv
  ../data/data-<date_w2>/topics_bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/data-<date_w2>/mapping_bertopic/selection/edge_mapping__<label>.csv

Writes (run twice; toggle REMOVE_OUTLIER_TOPIC):
  ../fig/BERTopic/topic_persist/outlier_remove/
    <label>__persistence.png
    overview_top10__persistence_degree.csv

  ../fig/BERTopic/topic_persist/outlier_include/
    <label>__persistence.png
    overview_top10__persistence_degree.csv

VMP 2026-02-08: tested and run.
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
# toggle manually
REMOVE_OUTLIER_TOPIC = False  # True/False

SEL_TOPICS = Path("../data/public/bertopic/selection")
SEL_MAP    = Path("../data/public/bertopic_mapping")

TOP10_PATH = SEL_TOPICS / "overview_top10.csv"
STMT_DIR   = SEL_TOPICS / "statement_topics"

OUT_BASE = Path("../fig/BERTopic/topic_persist")
OUTDIR = OUT_BASE / ("outlier_remove" if REMOVE_OUTLIER_TOPIC else "outlier_include")
WIPE_OUTDIR = True

# bins: 0,1,2,3,4+
BINS   = [-0.5, 0.5, 1.5, 2.5, 3.5, np.inf]
LABELS = ["0", "1", "2", "3", "4+"]
ORDER  = ["0", "1", "2", "3", "4+"]

FIGSIZE = (6.5, 3.2)

# -----------------------------
# Helpers
# -----------------------------
def compute_degree_w1(edge_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(edge_csv)[["wave", "key", "topic_1", "topic_2"]].copy()

    if REMOVE_OUTLIER_TOPIC:
        df = df[(df["topic_1"] != -1) & (df["topic_2"] != -1)].copy()

    df = df[df["wave"] == 1].copy()
    df["key"] = df["key"].astype(str)
    df["topic_1"] = pd.to_numeric(df["topic_1"], errors="coerce").astype("Int64")
    df["topic_2"] = pd.to_numeric(df["topic_2"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["topic_1", "topic_2"]).copy()
    df["topic_1"] = df["topic_1"].astype(int)
    df["topic_2"] = df["topic_2"].astype(int)

    # undirected normalization
    df = normalize_ab(df, "topic_1", "topic_2")

    # weighted degree = sum of edge multiplicities incident on topic node
    df = df.groupby(["key", "topic_1", "topic_2"], as_index=False).size().rename(columns={"size": "n_edges"})

    deg1 = df.groupby(["key", "topic_1"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_1": "topic"})
    deg2 = df.groupby(["key", "topic_2"])["n_edges"].sum().reset_index(name="degree_wt").rename(columns={"topic_2": "topic"})
    out = pd.concat([deg1, deg2], ignore_index=True).groupby(["key", "topic"], as_index=False)["degree_wt"].sum()

    out["degree_wt"] = out["degree_wt"].astype(int)
    out["topic"] = out["topic"].astype(int)
    out["key"] = out["key"].astype(str)
    return out


def load_nodes(stmt_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(stmt_csv)[["key", "wave", "topic"]].copy()
    if REMOVE_OUTLIER_TOPIC:
        df = df[df["topic"] != -1].copy()
    df["key"] = df["key"].astype(str)
    df["wave"] = pd.to_numeric(df["wave"], errors="raise").astype(int)
    df["topic"] = pd.to_numeric(df["topic"], errors="raise").astype(int)
    return df[df["wave"].isin([1, 2])].copy()


def baseline_from_wave2(df_nodes: pd.DataFrame) -> float:
    # p_t = prevalence across participants in wave 2
    df_w2 = df_nodes[df_nodes["wave"] == 2][["key", "topic"]].drop_duplicates()
    if df_w2.empty:
        return np.nan

    keys = df_w2["key"].drop_duplicates()
    topics = df_w2["topic"].drop_duplicates()

    grid = pd.MultiIndex.from_product([keys, topics], names=["key", "topic"]).to_frame(index=False)
    present = df_w2.assign(present=1)

    g = grid.merge(present, on=["key", "topic"], how="left")
    g["present"] = g["present"].fillna(0).astype(int)

    p_t = g.groupby("topic")["present"].mean().astype(float)
    if p_t.sum() == 0:
        return np.nan
    q_t = p_t / p_t.sum()

    return float((q_t * p_t).sum())


def persistence_curve(df_nodes: pd.DataFrame, deg_w1: pd.DataFrame) -> pd.DataFrame:
    w1 = df_nodes[df_nodes["wave"] == 1][["key", "topic"]].drop_duplicates().assign(present_w1=1)
    w2 = df_nodes[df_nodes["wave"] == 2][["key", "topic"]].drop_duplicates().assign(present_w2=1)

    df = (
        w1.merge(deg_w1, on=["key", "topic"], how="left")
          .merge(w2, on=["key", "topic"], how="left")
    )
    df["degree_wt"] = df["degree_wt"].fillna(0).astype(int)
    df["present_w2"] = df["present_w2"].fillna(0).astype(int)

    df["deg_bin"] = pd.cut(df["degree_wt"], bins=BINS, labels=LABELS)
    df["deg_bin"] = pd.Categorical(df["deg_bin"], categories=ORDER, ordered=True)

    agg = df.groupby("deg_bin", observed=True)["present_w2"].agg(p="mean", n="size").reset_index()
    agg = agg.set_index("deg_bin").reindex(ORDER).reset_index()

    agg["se"] = np.sqrt(agg["p"] * (1 - agg["p"]) / agg["n"])
    agg["ci_lo"] = agg["p"] - 1.96 * agg["se"]
    agg["ci_hi"] = agg["p"] + 1.96 * agg["se"]
    return agg


def plot_curve(agg: pd.DataFrame, baseline: float, outpath: Path) -> None:
    x = agg["deg_bin"].astype(str)
    y = agg["p"].to_numpy(float)
    lo = agg["ci_lo"].to_numpy(float)
    hi = agg["ci_hi"].to_numpy(float)

    plt.figure(figsize=FIGSIZE)
    plt.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=3, label="Conditional Probability")
    plt.axhline(baseline, linestyle="--", linewidth=1.2, label="Baseline Probability")
    plt.xlabel("Topic degree in wave 1")
    plt.ylabel("Topic probability in wave 2")
    plt.title("")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------------
# Main
# -----------------------------
if WIPE_OUTDIR:
    shutil.rmtree(OUTDIR, ignore_errors=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

top10 = pd.read_csv(TOP10_PATH)

rows = []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    label = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"

    edge_csv = SEL_MAP / f"edge_mapping__{label}.csv"
    stmt_csv = STMT_DIR / f"{label}__statement_topics.csv"

    if not edge_csv.exists() or not stmt_csv.exists():
        print("[skip]", label, "missing files")
        continue

    deg_w1 = compute_degree_w1(edge_csv)
    nodes = load_nodes(stmt_csv)

    baseline = baseline_from_wave2(nodes)
    agg = persistence_curve(nodes, deg_w1)

    p_map = dict(zip(agg["deg_bin"].astype(str), agg["p"]))
    p_deg0 = float(p_map.get("0", np.nan))
    p_deg1 = float(p_map.get("1", np.nan))
    p_deg2 = float(p_map.get("2", np.nan))
    p_deg3 = float(p_map.get("3", np.nan))
    p_deg4p = float(p_map.get("4+", np.nan))
    delta = p_deg4p - baseline if np.isfinite(p_deg4p) and np.isfinite(baseline) else np.nan

    plot_curve(agg, baseline, OUTDIR / f"{label}__persistence.png")

    rows.append(dict(
        label=label,
        remove_outlier_topic=REMOVE_OUTLIER_TOPIC,
        baseline=baseline,
        p_deg0=p_deg0, p_deg1=p_deg1, p_deg2=p_deg2, p_deg3=p_deg3, p_deg4p=p_deg4p,
        delta_deg4p_baseline=delta,
    ))

    print("[ok]", label, "delta_deg4p_baseline=", delta)

df = pd.DataFrame(rows).sort_values("delta_deg4p_baseline", ascending=False).reset_index(drop=True)
df.to_csv(OUTDIR / "overview_top10__persistence_degree.csv", index=False)

print("\nSaved:", OUTDIR)