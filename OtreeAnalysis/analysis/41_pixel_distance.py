"""
VMP 2026-02-17: not in preprint.

Exploratory: 
1. does spatial proximity on the canvas predict edges?
2. does spatial proximity correlate with semantic similarity?

Reads:
  public/distractors_w*.json

Writes:
  ../fig/pixel_distance/
"""

from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utilities import wave_1, wave_2, get_public_path

# -------------------------
# Config
# -------------------------
POS_KEY = "pos_3"
EDGES_KEY = "edges_3"

outdir = "../fig/pixel_distance"
os.makedirs(outdir, exist_ok=True)

# load data
with open(get_public_path("distractors_w{wave}.json", wave=wave_1), encoding="utf-8") as f:
    data_w1 = json.load(f)
with open(get_public_path("distractors_w{wave}.json", wave=wave_2), encoding="utf-8") as f:
    data_w2 = json.load(f)

# construct dataframe
def _get_positions(bundle):
    """Node label → (x, y) from canvas positions."""
    return {n["label"]: (n["x"], n["y"]) for n in bundle["positions"][POS_KEY]}


def _human_edge_lookup(bundle) -> dict[tuple, dict]:
    """(sorted pair) → {polarity, strength} for human-drawn edges."""
    lookup: dict[tuple, dict] = {}
    for e in bundle["edges"][EDGES_KEY]:
        s1 = str(e["stance_1"]).strip()
        s2 = str(e["stance_2"]).strip()
        pair = tuple(sorted([s1, s2]))
        pol = e["polarity"]
        if pair in lookup and lookup[pair]["polarity"] != pol:
            lookup[pair] = {"polarity": "both", "strength": None}
        else:
            lookup[pair] = {"polarity": pol, "strength": None}
    return lookup


def _llm_edge_lookup(bundle) -> dict[tuple, dict]:
    """(sorted pair) → {polarity, strength} for LLM-extracted edges."""
    lookup: dict[tuple, dict] = {}
    for e in bundle.get("LLM", {}).get("edge_results", []):
        s1 = str(e["stance_1"]).strip()
        s2 = str(e["stance_2"]).strip()
        pair = tuple(sorted([s1, s2]))
        lookup[pair] = {"polarity": e["polarity"], "strength": e.get("strength")}
    return lookup


def build_pairwise(data: dict, wave: int, edge_source: str = "human") -> pd.DataFrame:
    edge_fn = _human_edge_lookup if edge_source == "human" else _llm_edge_lookup
    rows = []
    for key, bundle in data.items():
        pos = _get_positions(bundle)
        edge_lookup = edge_fn(bundle)

        stances = sorted(pos.keys())
        for a, b in combinations(stances, 2):
            pair = tuple(sorted([a, b]))
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

            info = edge_lookup.get(pair)
            rows.append({
                "key": key,
                "wave": wave,
                "stance_1": pair[0],
                "stance_2": pair[1],
                "distance": dist,
                "has_edge": info is not None,
                "polarity": info["polarity"] if info else "none",
                "strength": info["strength"] if info else None,
            })

    return pd.DataFrame(rows)

# construct dataframes (manually double checked an example)
df_w1 = build_pairwise(data_w1, wave=1, edge_source="human")
df_w2 = build_pairwise(data_w2, wave=2, edge_source="human")
df = pd.concat([df_w1, df_w2], ignore_index=True)

# basic information
print(f"\nTotal pairs: {len(df)} (W1={len(df_w1)}, W2={len(df_w2)})")
print(f"Pairs with edge: {df['has_edge'].sum()} ({100*df['has_edge'].mean():.1f}%)")
print(df["polarity"].value_counts().to_string())

# -------------------------
# 1a. Distance → any edge (binned)
# -------------------------
N_BINS = 10
df["dist_bin"] = pd.qcut(df["distance"], q=N_BINS, duplicates="drop")

# -------------------------
# Distance → edge type (binned)
# -------------------------
df["is_positive"] = df["polarity"] == "positive"
df["is_negative"] = df["polarity"] == "negative"

agg2 = df.groupby("dist_bin", observed=True).agg(
    mean_dist=("distance", "mean"),
    p_positive=("is_positive", "mean"),
    p_negative=("is_negative", "mean"),
    p_edge=("has_edge", "mean"),
    n=("has_edge", "size"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(agg2["mean_dist"], agg2["p_positive"], "o-", color="#998ec3", label="P(supporting)")
ax.plot(agg2["mean_dist"], agg2["p_negative"], "o-", color="#f1a340", label="P(conflicting)")
ax.set_xlabel("Mean pixel distance (binned)")
ax.set_ylabel("P(edge type)")
ax.set_title("Proximity vs. edge polarity")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(outdir, "distance_vs_polarity.png"), dpi=300)
plt.close(fig)

# -------------------------
# Same as above, but normalize distance.
# -------------------------
# Normalize distance within each (key, wave) to [0,1] so we're comparing
# relative placement rather than absolute pixel values.
df["dist_norm"] = df.groupby(["key", "wave"])["distance"].transform(
    lambda s: (s - s.min()) / (s.max() - s.min())
)

df["dist_norm_bin"] = pd.qcut(df["dist_norm"], q=N_BINS, duplicates="drop")

agg3 = df.groupby("dist_norm_bin", observed=True).agg(
    mean_dist_norm=("dist_norm", "mean"),
    p_edge=("has_edge", "mean"),
    p_positive=("is_positive", "mean"),
    p_negative=("is_negative", "mean"),
    n=("has_edge", "size"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(agg3["mean_dist_norm"], agg3["p_positive"], "o-", color="#998ec3", label="P(supporting)")
ax.plot(agg3["mean_dist_norm"], agg3["p_negative"], "o-", color="#f1a340", label="P(conflicting)")
ax.set_xlabel("Normalized distance (0=closest, 1=farthest)")
ax.set_ylabel("P(edge type)")
ax.set_title("Normalized proximity vs. edge polarity")
ax.legend()

fig.tight_layout()
fig.savefig(os.path.join(outdir, "normdist_vs_polarity.png"), dpi=300)
plt.close(fig)

''' FURTHER THINGS:
1. mixed-effects logistic regression like this:

has_edge ~ distance + (1 | key)

could consider having wave somewhere e.g.
as main effect or also random effect.
but the main driver should be key
(i.e., participants using canvas differently).
'''

# =========================================================
# 1-LLM. Same proximity analysis but for LLM-extracted edges
# one difference is that we have measure of "strength".
# =========================================================
df_llm_w1 = build_pairwise(data_w1, wave=1, edge_source="llm")
df_llm_w2 = build_pairwise(data_w2, wave=2, edge_source="llm")
df_llm = pd.concat([df_llm_w1, df_llm_w2], ignore_index=True)

print(f"\n=== LLM edges ===")
print(f"Total pairs: {len(df_llm)} (W1={len(df_llm_w1)}, W2={len(df_llm_w2)})")
print(f"Pairs with edge: {df_llm['has_edge'].sum()} ({100*df_llm['has_edge'].mean():.1f}%)")
print(df_llm["polarity"].value_counts().to_string())

df_llm["is_positive"] = df_llm["polarity"] == "positive"
df_llm["is_negative"] = df_llm["polarity"] == "negative"

# bin distance
df_llm["dist_bin"] = pd.qcut(df_llm["distance"], q=N_BINS, duplicates="drop")

# reproduce the plot by polarity for the LLM.
agg_llm2 = df_llm.groupby("dist_bin", observed=True).agg(
    mean_dist=("distance", "mean"),
    p_positive=("is_positive", "mean"),
    p_negative=("is_negative", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(agg_llm2["mean_dist"], agg_llm2["p_positive"], "o-", color="#998ec3", label="P(supporting)")
ax.plot(agg_llm2["mean_dist"], agg_llm2["p_negative"], "o-", color="#f1a340", label="P(conflicting)")
ax.set_xlabel("Mean pixel distance (binned)")
ax.set_ylabel("P(edge type)")
ax.set_title("LLM: Proximity vs. edge polarity")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(outdir, "llm_distance_vs_polarity.png"), dpi=300)
plt.close(fig)

# 1c-LLM. Distance → strength (positive edges only) + P(supporting) on twin axis
df_llm_pos = df_llm[df_llm["polarity"] == "positive"].copy()
df_llm_pos["strength"] = pd.to_numeric(df_llm_pos["strength"])
df_llm_pos["dist_bin_str"] = pd.qcut(df_llm_pos["distance"], q=N_BINS, duplicates="drop")

agg_str = df_llm_pos.groupby("dist_bin_str", observed=True).agg(
    mean_dist=("distance", "mean"),
    mean_strength=("strength", "mean"),
    se_strength=("strength", "sem"),
).reset_index()

# P(supporting) uses all pairs (same bins as strength)
agg_prob = df_llm.groupby("dist_bin", observed=True).agg(
    mean_dist=("distance", "mean"),
    p_positive=("is_positive", "mean"),
).reset_index()

fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()

ax1.errorbar(agg_str["mean_dist"], agg_str["mean_strength"],
             yerr=agg_str["se_strength"], fmt="o-", capsize=3,
             color="#998ec3", label="Mean strength (positive)")
ax2.plot(agg_prob["mean_dist"], agg_prob["p_positive"],
         "o-", color="#f1a340", label="P(supporting)")

ax1.set_xlabel("Mean pixel distance (binned)")
ax1.set_ylabel("Mean LLM edge strength", color="#998ec3")
ax2.set_ylabel("P(supporting)", color="#f1a340")
ax1.tick_params(axis="y", labelcolor="#998ec3")
ax2.tick_params(axis="y", labelcolor="#f1a340")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax1.set_title("LLM: Proximity vs. supporting edge strength & probability")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "llm_distance_vs_strength.png"), dpi=300)
plt.close(fig)

# =========================================================
# 2. Semantic distance vs pixel distance
# This is actually not super promising I must say.
# Clearly some relation, but super weak.
# =========================================================
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cosine_dist

all_stances = pd.unique(df[["stance_1", "stance_2"]].values.ravel())
stance_list = all_stances.tolist()

# run two models to verify that patterns are consistent.
# the QWEN model takes a few minutes to run on CPU.
EMBED_MODELS = [
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
    ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B"),
]

# encode with both models
cosine_cols = {}
for short_name, hf_name in EMBED_MODELS:
    print(f"\nEncoding with {short_name} ...")
    device = "cpu" if hf_name.startswith("Qwen/") else None
    mdl = SentenceTransformer(hf_name, device=device)
    emb = mdl.encode(stance_list, batch_size=64, convert_to_numpy=True)
    emb_lookup = dict(zip(all_stances, emb))

    col = f"cosine_{short_name}"
    df[col] = [
        cosine_dist(emb_lookup[s1], emb_lookup[s2])
        for s1, s2 in zip(df["stance_1"], df["stance_2"])
    ]
    cosine_cols[short_name] = col

# 2x2 plot: scatter (left) + binned (right) per model (rows)
df["dist_bin_sem"] = pd.qcut(df["distance"], q=N_BINS, duplicates="drop")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for row, (short_name, col) in enumerate(cosine_cols.items()):
    agg_sem = df.groupby("dist_bin_sem", observed=True).agg(
        mean_dist=("distance", "mean"),
        mean_cosine=(col, "mean"),
        se_cosine=(col, "sem"),
    ).reset_index()

    ax_scatter = axes[row, 0]
    ax_scatter.scatter(df["distance"], df[col], alpha=0.04, s=3, color="steelblue", rasterized=True)
    ax_scatter.set_xlabel("Pixel distance")
    ax_scatter.set_ylabel("Cosine distance")
    ax_scatter.set_title(short_name)

    ax_binned = axes[row, 1]
    ax_binned.errorbar(agg_sem["mean_dist"], agg_sem["mean_cosine"],
                       yerr=agg_sem["se_cosine"], fmt="o-", capsize=3, color="steelblue")
    ax_binned.set_xlabel("Mean pixel distance (binned)")
    ax_binned.set_ylabel("Mean cosine distance")
    ax_binned.set_title(f"{short_name} (binned)")

fig.tight_layout()
fig.savefig(os.path.join(outdir, "pixel_vs_cosine.png"), dpi=300)
plt.close(fig)

''' FURTHER THINGS:
Possible that a few latent dimensions of the embedding are driving distance.
Could be investigated further.
'''