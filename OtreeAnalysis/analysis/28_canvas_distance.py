"""
28_canvas_distance.py

VMP 2026-03-15

Canvas distance vs edge polarity: do participants place connected nodes closer together?

Reads:
  ../data/public/distractors_w*.json

Writes:
  ../fig/canvas_distance/normdist_means.pdf
  ../fig/canvas_distance/normdist_rawdots.pdf    (Figure S11 V2 — means + raw dots)
"""

from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilities import wave_1, wave_2, get_public_path

# -------------------------
# Config
# -------------------------
POS_KEY   = "pos_3"
EDGES_KEY = "edges_3"

NORM_BINS = np.arange(0, 1.01, 0.1)

COL_SUP = "#998ec3"
COL_CON = "#f1a340"

outdir = "../fig/canvas_distance"
os.makedirs(outdir, exist_ok=True)

# -------------------------
# Load data
# -------------------------
with open(get_public_path("distractors_w{wave}.json", wave=wave_1), encoding="utf-8") as f:
    data_w1 = json.load(f)
with open(get_public_path("distractors_w{wave}.json", wave=wave_2), encoding="utf-8") as f:
    data_w2 = json.load(f)


def _get_positions(bundle):
    return {n["label"]: (n["x"], n["y"]) for n in bundle["positions"][POS_KEY]}


def _human_edge_lookup(bundle) -> dict[tuple, str]:
    lookup: dict[tuple, str] = {}
    for e in bundle["edges"][EDGES_KEY]:
        s1, s2 = str(e["stance_1"]).strip(), str(e["stance_2"]).strip()
        pair = tuple(sorted([s1, s2]))
        pol = e["polarity"]
        if pair in lookup and lookup[pair] != pol:
            lookup[pair] = "both"
        else:
            lookup[pair] = pol
    return lookup


def build_pairwise(data: dict, wave: int) -> pd.DataFrame:
    rows = []
    for key, bundle in data.items():
        pos         = _get_positions(bundle)
        edge_lookup = _human_edge_lookup(bundle)
        stances     = sorted(pos.keys())
        for a, b in combinations(stances, 2):
            pair = tuple(sorted([a, b]))
            x1, y1 = pos[a]
            x2, y2 = pos[b]
            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            pol  = edge_lookup.get(pair)
            rows.append({
                "key":         key,
                "wave":        wave,
                "distance":    dist,
                "has_edge":    pol is not None,
                "polarity":    pol if pol else "none",
            })
    return pd.DataFrame(rows)


df = pd.concat(
    [build_pairwise(data_w1, wave=1), build_pairwise(data_w2, wave=2)],
    ignore_index=True,
)

# -------------------------
# Normalize distance within (key, wave)
# -------------------------
df["dist_norm"] = df.groupby(["key", "wave"])["distance"].transform(
    lambda s: (s - s.min()) / (s.max() - s.min())
)

df["is_positive"] = (df["polarity"] == "positive").astype(float)
df["is_negative"] = (df["polarity"] == "negative").astype(float)

df["dist_norm_bin"] = pd.cut(df["dist_norm"], bins=NORM_BINS, include_lowest=True)

# -------------------------
# Bin-level means
# -------------------------
agg = df.groupby("dist_norm_bin", observed=True).agg(
    mid=("dist_norm", "mean"),
    p_positive=("is_positive", "mean"),
    p_negative=("is_negative", "mean"),
    n=("dist_norm", "size"),
).reset_index()

x = agg["mid"].to_numpy()

# -------------------------
# Plot 1: means only
# -------------------------
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(x, agg["p_positive"], "o-", color=COL_SUP, label="Supporting")
ax.plot(x, agg["p_negative"], "o-", color=COL_CON, label="Conflicting")
ax.set_xlabel("Normalized distance", fontsize=13)
ax.set_ylabel("Fraction", fontsize=13)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "normdist_means.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: normdist_means.pdf")

# -------------------------
# Plot 2: raw dots + means
# -------------------------
# Participant-level fraction per bin
pp = df.groupby(["key", "wave", "dist_norm_bin"], observed=True).agg(
    p_positive=("is_positive", "mean"),
    p_negative=("is_negative", "mean"),
).reset_index()

# map bin → x position (bin midpoint index)
bin_order = agg["dist_norm_bin"].tolist()
bin_to_x  = {b: x[i] for i, b in enumerate(bin_order)}

rng       = np.random.default_rng(42)
JITTER_H  = 0.018
JITTER_V  = 0.012

fig, ax = plt.subplots(figsize=(4, 3))

for col, color, label in [
    ("p_positive", COL_SUP, "Supporting"),
    ("p_negative", COL_CON, "Conflicting"),
]:
    # raw dots
    xvals = np.array([bin_to_x[b] for b in pp["dist_norm_bin"]])
    yvals = pp[col].to_numpy()
    xj    = xvals + rng.uniform(-JITTER_H, JITTER_H, size=len(xvals))
    yj    = yvals + rng.uniform(-JITTER_V, JITTER_V, size=len(yvals))
    ax.scatter(xj, yj, s=6, alpha=0.15, color=color, linewidths=0, zorder=2)

    # mean line
    ax.plot(x, agg[col], "o-", color=color, markersize=5,
            linewidth=1.4, zorder=4, label=label)

ax.set_xlabel("Normalized distance", fontsize=13)
ax.set_ylabel("Fraction", fontsize=13)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "normdist_rawdots.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved: normdist_rawdots.pdf")
