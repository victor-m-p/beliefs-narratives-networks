"""
VMP 2026-02-02

Create stance-edge → topic-edge mappings
ONLY for the TOP10 selected BERTopic runs.

Reads:
  ../data/data-<date_w2>/topics_bertopic/selection/
    overview_top10.csv
    statement_topics/*.csv

Writes:
  ../data/data-<date_w2>/mapping_bertopic/selection/
    edge_mapping__<label>.csv

VMP 2026-02-08: tested and run.
"""

from __future__ import annotations

'''
VMP 2026-02-06 (refactored):
Maps BERTopic topics to participant edges.
Uses sanitized public data.
Works with gitignored topics_bertopic and mapping_bertopic folders.
'''

import json
import shutil
from pathlib import Path

import pandas as pd

from utilities import wave_1, wave_2, get_public_path


# -----------------------------
# Config
# -----------------------------
SELECTION_ROOT = Path("../data/public/bertopic/selection")
TOP10_PATH = SELECTION_ROOT / "overview_top10.csv"
STATEMENT_DIR = SELECTION_ROOT / "statement_topics"

OUT_ROOT = Path("../data/public/bertopic_mapping")
WIPE_OUT_ROOT = True

EXCLUDE_OUTLIERS_IN_LOOKUP = False

EDGE_KEY = ("edges", "edges_3")  # v["edges"]["edges_3"]


# -----------------------------
# Reset output
# -----------------------------
if WIPE_OUT_ROOT:
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load participant edges (W1/W2) from public (sanitized)
# -----------------------------
distractors_w1_path = get_public_path("distractors_w{wave}.json", wave=wave_1)
distractors_w2_path = get_public_path("distractors_w{wave}.json", wave=wave_2)

with open(distractors_w1_path, encoding='utf-8') as f:
    data_w1 = json.load(f)
data_w1['242bdf6bc5bb0b94']['edges']['edges_3']

with open(distractors_w2_path, encoding='utf-8') as f:
    data_w2 = json.load(f)

edges_w1 = pd.concat(
    (pd.DataFrame(v[EDGE_KEY[0]][EDGE_KEY[1]]).assign(key=str(k)) for k, v in data_w1.items()),
    ignore_index=True,
)
edges_w2 = pd.concat(
    (pd.DataFrame(v[EDGE_KEY[0]][EDGE_KEY[1]]).assign(key=str(k)) for k, v in data_w2.items()),
    ignore_index=True,
)

edges_w1["wave"] = 1
edges_w2["wave"] = 2

edges_all = pd.concat([edges_w1, edges_w2], ignore_index=True)

edges_all["key"] = edges_all["key"].astype(str)
edges_all["stance_1"] = edges_all["stance_1"].astype(str).str.strip()
edges_all["stance_2"] = edges_all["stance_2"].astype(str).str.strip()

print(f"Loaded edges: n={len(edges_all)} (waves 1+2 pooled)")


# -----------------------------
# Load TOP10 manifest
# -----------------------------
top10 = pd.read_csv(TOP10_PATH)

required = ["embed_model_outname", "run_id"]
for c in required:
    if c not in top10.columns:
        raise ValueError(f"Missing column {c} in {TOP10_PATH}")


# -----------------------------
# Build mappings (selection only)
# -----------------------------
for rank, r in enumerate(top10.itertuples(index=False), 1):
    label = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    st_path = STATEMENT_DIR / f"{label}__statement_topics.csv"

    if not st_path.exists():
        print(f"[skip] missing statement_topics: {st_path.name}")
        continue

    # ---- load statement topics ----
    df_topics = pd.read_csv(st_path)

    df_topics["key"] = df_topics["key"].astype(str)
    df_topics["wave"] = pd.to_numeric(df_topics["wave"], errors="raise").astype(int)
    df_topics["stance"] = df_topics["stance"].astype(str).str.strip()
    df_topics["topic"] = pd.to_numeric(df_topics["topic"], errors="raise").astype(int)

    df_topics = df_topics[df_topics["wave"].isin([1, 2])].copy()

    if EXCLUDE_OUTLIERS_IN_LOOKUP:
        df_topics = df_topics[df_topics["topic"] != -1].copy()

    keep_cols = [c for c in ["key", "wave", "stance", "topic", "topic_conf", "assigned_prob"]
                 if c in df_topics.columns]
    df_topics = df_topics[keep_cols]

    topics_lookup = (
        df_topics
        .drop_duplicates(subset=["key", "wave", "stance"])
        .set_index(["key", "wave", "stance"])
    )

    # ---- attach topics to edges ----
    edges = edges_all.copy()

    edges = edges.join(
        topics_lookup.rename(columns={
            "topic": "topic_1",
            "topic_conf": "topic_conf_1",
            "assigned_prob": "assigned_prob_1",
        }),
        on=["key", "wave", "stance_1"],
        how="left",
    )

    edges = edges.join(
        topics_lookup.rename(columns={
            "topic": "topic_2",
            "topic_conf": "topic_conf_2",
            "assigned_prob": "assigned_prob_2",
        }),
        on=["key", "wave", "stance_2"],
        how="left",
    )

    edges["label"] = label

    base_cols = ["label", "key", "wave", "stance_1", "stance_2"]
    if "polarity" in edges.columns:
        base_cols.append("polarity")
    if "timestamp" in edges.columns:
        base_cols.append("timestamp")

    topic_cols = ["topic_1", "topic_2"]
    if "topic_conf_1" in edges.columns:
        topic_cols += ["topic_conf_1", "topic_conf_2"]
    if "assigned_prob_1" in edges.columns:
        topic_cols += ["assigned_prob_1", "assigned_prob_2"]

    edges_out = edges[base_cols + topic_cols].copy()
    out_csv = OUT_ROOT / f"edge_mapping__{label}.csv"
    edges_out.to_csv(out_csv, index=False)

    n_missing = edges_out["topic_1"].isna().sum() + edges_out["topic_2"].isna().sum()
    print(f"[ok] {label}: edges={len(edges_out)} missing endpoints={n_missing}")

print("\nDone. Wrote human-edge mappings to:", OUT_ROOT.resolve())


# =============================================================
# LLM-extracted edges  (same topic mapping, different edge source)
# =============================================================
# reuse the same topic lookup here since the nodes are shared
# only difference is where the edges live. 
# this is not used in the preprint but testing this for the main submission.

LLM_OUT_ROOT = Path("../data/public/bertopic_mapping_llm")
if WIPE_OUT_ROOT:
    shutil.rmtree(LLM_OUT_ROOT, ignore_errors=True)
LLM_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _extract_llm_edges(data: dict, wave: int) -> pd.DataFrame:
    """Extract LLM edge_results from every participant into a DataFrame."""
    frames = []
    for k, v in data.items():
        recs = v.get("LLM", {}).get("edge_results", [])
        if not recs:
            continue
        df = pd.DataFrame(recs).assign(key=str(k))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stance_1", "stance_2", "polarity", "key", "wave"])
    out = pd.concat(frames, ignore_index=True)
    out["wave"] = wave
    return out


edges_llm_w1 = _extract_llm_edges(data_w1, wave=1)
edges_llm_w2 = _extract_llm_edges(data_w2, wave=2)

edges_llm_all = pd.concat([edges_llm_w1, edges_llm_w2], ignore_index=True)
edges_llm_all["key"] = edges_llm_all["key"].astype(str)
edges_llm_all["stance_1"] = edges_llm_all["stance_1"].astype(str).str.strip()
edges_llm_all["stance_2"] = edges_llm_all["stance_2"].astype(str).str.strip()

print(f"\nLoaded LLM edges: n={len(edges_llm_all)} (waves 1+2 pooled)")

# --- Build LLM-edge mappings (reuse same TOP10 + statement topics) ---
for rank, r in enumerate(top10.itertuples(index=False), 1):
    label = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    st_path = STATEMENT_DIR / f"{label}__statement_topics.csv"

    if not st_path.exists():
        print(f"[skip-llm] missing statement_topics: {st_path.name}")
        continue

    # ---- load statement topics (identical to human block) ----
    df_topics = pd.read_csv(st_path)
    df_topics["key"] = df_topics["key"].astype(str)
    df_topics["wave"] = pd.to_numeric(df_topics["wave"], errors="raise").astype(int)
    df_topics["stance"] = df_topics["stance"].astype(str).str.strip()
    df_topics["topic"] = pd.to_numeric(df_topics["topic"], errors="raise").astype(int)
    df_topics = df_topics[df_topics["wave"].isin([1, 2])].copy()

    if EXCLUDE_OUTLIERS_IN_LOOKUP:
        df_topics = df_topics[df_topics["topic"] != -1].copy()

    keep_cols = [c for c in ["key", "wave", "stance", "topic", "topic_conf", "assigned_prob"]
                 if c in df_topics.columns]
    df_topics = df_topics[keep_cols]

    topics_lookup = (
        df_topics
        .drop_duplicates(subset=["key", "wave", "stance"])
        .set_index(["key", "wave", "stance"])
    )

    # ---- attach topics to LLM edges ----
    edges = edges_llm_all.copy()

    edges = edges.join(
        topics_lookup.rename(columns={
            "topic": "topic_1",
            "topic_conf": "topic_conf_1",
            "assigned_prob": "assigned_prob_1",
        }),
        on=["key", "wave", "stance_1"],
        how="left",
    )

    edges = edges.join(
        topics_lookup.rename(columns={
            "topic": "topic_2",
            "topic_conf": "topic_conf_2",
            "assigned_prob": "assigned_prob_2",
        }),
        on=["key", "wave", "stance_2"],
        how="left",
    )

    edges["label"] = label

    base_cols = ["label", "key", "wave", "stance_1", "stance_2"]
    if "polarity" in edges.columns:
        base_cols.append("polarity")
    if "explicitness" in edges.columns:
        base_cols.append("explicitness")
    if "strength" in edges.columns:
        base_cols.append("strength")

    topic_cols = ["topic_1", "topic_2"]
    if "topic_conf_1" in edges.columns:
        topic_cols += ["topic_conf_1", "topic_conf_2"]
    if "assigned_prob_1" in edges.columns:
        topic_cols += ["assigned_prob_1", "assigned_prob_2"]

    edges_out = edges[base_cols + topic_cols].copy()
    out_csv = LLM_OUT_ROOT / f"edge_mapping_llm__{label}.csv"
    edges_out.to_csv(out_csv, index=False)

    n_missing = edges_out["topic_1"].isna().sum() + edges_out["topic_2"].isna().sum()
    print(f"[ok-llm] {label}: edges={len(edges_out)} missing endpoints={n_missing}")

print("\nDone. Wrote LLM-edge mappings to:", LLM_OUT_ROOT.resolve())
