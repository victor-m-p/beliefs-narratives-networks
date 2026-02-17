"""
VMP 2026-02-17

Edge reliability across waves for BERTopic topic networks.

Question: if edge (t1, t2) exists in W1 for a participant,
how likely is it to also exist in W2?

For each W1 topic pair we track two things:
  - both topics present in W2?
  - edge present in W2?

Reported separately for connected pairs (edge in W1)
vs unconnected pairs (no edge in W1, baseline).

Output at participant level and aggregated across runs.

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/public/bertopic_mapping/edge_mapping__<label>.csv

Writes:
  ../fig/BERTopic/edge_reliability/
    <label>__participant_level.csv
    edge_reliability_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers import normalize_ab

# ---- config ----
REMOVE_OUTLIER_TOPIC = False

SEL_TOPICS = Path("../data/public/bertopic/selection")
SEL_MAP = Path("../data/public/bertopic_mapping")
TOP10_PATH = SEL_TOPICS / "overview_top10.csv"
STMT_DIR = SEL_TOPICS / "statement_topics"

OUTDIR = Path("../fig/BERTopic/edge_reliability")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---- helpers ----
def load_topic_presence(stmt_csv: Path) -> pd.DataFrame:
    """Per-participant topic presence: key, wave, topic (deduplicated)."""
    df = pd.read_csv(stmt_csv)[["key", "wave", "topic"]].copy()
    if REMOVE_OUTLIER_TOPIC:
        df = df[df["topic"] != -1]
    df["key"] = df["key"].astype(str)
    df = df[df["wave"].isin([1, 2])].copy()
    return df[["key", "wave", "topic"]].drop_duplicates()


def load_edges(edge_csv: Path) -> pd.DataFrame:
    """Per-participant topic-level edges: key, wave, topic_1, topic_2 (deduplicated)."""
    df = pd.read_csv(edge_csv)[["key", "wave", "topic_1", "topic_2"]].copy()
    if REMOVE_OUTLIER_TOPIC:
        df = df[(df["topic_1"] != -1) & (df["topic_2"] != -1)]
    df["key"] = df["key"].astype(str)
    df["topic_1"] = pd.to_numeric(df["topic_1"], errors="coerce").astype("Int64")
    df["topic_2"] = pd.to_numeric(df["topic_2"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["topic_1", "topic_2"]).copy()
    df["topic_1"] = df["topic_1"].astype(int)
    df["topic_2"] = df["topic_2"].astype(int)
    df = df[df["wave"].isin([1, 2])].copy()
    df = normalize_ab(df, "topic_1", "topic_2")
    return df[["key", "wave", "topic_1", "topic_2"]].drop_duplicates()


def compute_participant_level(presence: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Per-participant counts for connected and unconnected W1 pairs."""
    pres = presence.groupby(["key", "wave"])["topic"].apply(set).to_dict()
    edge_sets = (
        edges.groupby(["key", "wave"])
        .apply(lambda g: set(zip(g["topic_1"], g["topic_2"])), include_groups=False)
        .to_dict()
    )

    keys = sorted(set(k for k, w in pres if w == 1) & set(k for k, w in pres if w == 2))

    rows = []
    for key in keys:
        topics_w1 = pres.get((key, 1), set())
        topics_w2 = pres.get((key, 2), set())
        edges_w1 = edge_sets.get((key, 1), set())
        edges_w2 = edge_sets.get((key, 2), set())

        # all W1 topic pairs (normalized: t1 < t2)
        topics_w1_sorted = sorted(topics_w1)
        all_pairs = set()
        for i, t1 in enumerate(topics_w1_sorted):
            for t2 in topics_w1_sorted[i + 1:]:
                all_pairs.add((t1, t2))

        # connected (edge in W1)
        conn_n = 0
        conn_both_w2 = 0
        conn_edge_w2 = 0
        # unconnected (no edge in W1)
        unconn_n = 0
        unconn_both_w2 = 0
        unconn_edge_w2 = 0

        for t1, t2 in all_pairs:
            both_in_w2 = (t1 in topics_w2) and (t2 in topics_w2)
            edge_in_w2 = (t1, t2) in edges_w2

            if (t1, t2) in edges_w1:
                conn_n += 1
                conn_both_w2 += both_in_w2
                conn_edge_w2 += edge_in_w2
            else:
                unconn_n += 1
                unconn_both_w2 += both_in_w2
                unconn_edge_w2 += edge_in_w2

        rows.append(dict(
            key=key,
            conn_n=conn_n,
            conn_both_w2=conn_both_w2,
            conn_edge_w2=conn_edge_w2,
            unconn_n=unconn_n,
            unconn_both_w2=unconn_both_w2,
            unconn_edge_w2=unconn_edge_w2,
        ))

    return pd.DataFrame(rows)


def safe_div(a, b):
    return a / b if b > 0 else np.nan


def aggregate(df_part: pd.DataFrame) -> dict:
    """Average participant-level rates (each participant weighted equally)."""
    df = df_part.copy()

    # per-participant rates (NaN if denominator is 0)
    df["conn_p_both_w2"] = df.apply(lambda r: safe_div(r["conn_both_w2"], r["conn_n"]), axis=1)
    df["conn_p_edge_w2"] = df.apply(lambda r: safe_div(r["conn_edge_w2"], r["conn_n"]), axis=1)
    df["unconn_p_both_w2"] = df.apply(lambda r: safe_div(r["unconn_both_w2"], r["unconn_n"]), axis=1)
    df["unconn_p_edge_w2"] = df.apply(lambda r: safe_div(r["unconn_edge_w2"], r["unconn_n"]), axis=1)

    return dict(
        n_participants=len(df),
        conn_n=int(df["conn_n"].sum()),
        conn_p_both_w2=df["conn_p_both_w2"].mean(),
        conn_p_edge_w2=df["conn_p_edge_w2"].mean(),
        unconn_n=int(df["unconn_n"].sum()),
        unconn_p_both_w2=df["unconn_p_both_w2"].mean(),
        unconn_p_edge_w2=df["unconn_p_edge_w2"].mean(),
    )


# ---- main (single run for now) ----
top10 = pd.read_csv(TOP10_PATH)
r = top10.iloc[0]
label = f"01__{r.embed_model_outname}__run_{r.run_id}"

edge_csv = SEL_MAP / f"edge_mapping__{label}.csv"
stmt_csv = STMT_DIR / f"{label}__statement_topics.csv"

presence = load_topic_presence(stmt_csv)
edges = load_edges(edge_csv)

df_part = compute_participant_level(presence, edges)
agg = aggregate(df_part)

print(f"Run: {label}")
print(f"N participants: {agg['n_participants']}")
print()
print(f"{'':20s} {'Connected':>12s} {'Unconnected':>12s}")
print(f"{'N pairs':20s} {agg['conn_n']:12d} {agg['unconn_n']:12d}")
print(f"{'P(both in W2)':20s} {agg['conn_p_both_w2']:12.3f} {agg['unconn_p_both_w2']:12.3f}")
print(f"{'P(edge in W2)':20s} {agg['conn_p_edge_w2']:12.3f} {agg['unconn_p_edge_w2']:12.3f}")
print(f"{'edge / both':20s} {agg['conn_p_edge_w2']/agg['conn_p_both_w2']:12.3f} {agg['unconn_p_edge_w2']/agg['unconn_p_both_w2']:12.3f}")
