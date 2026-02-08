"""
VMP 2026-02-06 (refactored)

Make 2x2 plots of network ratings:
1) raw (all sources)
2) raw (human-only)
3) z (global mean/sd across all sources)
4) z (global mean/sd across human-only)

Input: public/distractors_w*.json (sanitized)
Output: ../fig/rating_networks/

VMP 2026-02-07: tested and run.
"""

import os
import json
import numpy as np
import pandas as pd

from utilities import wave_2, get_public_path
from helpers import mean_se_plot_side

# -------------------------
# config
# -------------------------
wave = wave_2

outdir = "../fig/rating_networks"
os.makedirs(outdir, exist_ok=True)

SOURCE_LABEL = {
    "user": "Self",
    "llm": "LLM",
    "user_random": "Self + Random",
    "llm_random": "LLM + Random",
}
ORDER_ALL = ["Self", "LLM", "Self + Random", "LLM + Random"]
ORDER_HUM = ["Self", "Self + Random"]

ALL_SOURCES = ["user", "llm", "user_random", "llm_random"]
HUMAN_SOURCES = ["user", "user_random"]  # "exclude LLM" => drop llm + llm_random

# -------------------------
# load + tidy
# -------------------------
distractors_path = get_public_path("distractors_w{wave}.json", wave=wave)
with open(distractors_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nc = pd.concat(
    (pd.DataFrame(v["network_compare"]).assign(key=k) for k, v in data.items()),
    ignore_index=True,
)

nc["rating_left"]  = pd.to_numeric(nc["rating_left"], errors="coerce")
nc["rating_right"] = pd.to_numeric(nc["rating_right"], errors="coerce")

left = nc[["key", "left", "rating_left"]].rename(columns={"left": "source", "rating_left": "rating"})
right = nc[["key", "right", "rating_right"]].rename(columns={"right": "source", "rating_right": "rating"})
ratings = pd.concat([left, right], ignore_index=True).dropna(subset=["rating"])

# -------------------------
# helpers
# -------------------------
def participant_means(df_long):
    return df_long.groupby(["key", "source"], as_index=False)["rating"].mean()

# group by key: mean and SD per participant.
def add_within_key_z(df_long, keycol="key", ycol="rating", outcol="rating_z"):
    df_long = df_long.copy()
    g = df_long.groupby(keycol)[ycol]
    mu = g.transform("mean")
    sd = g.transform("std")  # ddof=1 by default

    # protect against sd==0 (or NaN if only 1 obs): return 0 in those cases
    df_long[outcol] = (df_long[ycol] - mu) / sd
    df_long[outcol] = df_long[outcol].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_long

def plot_mean_se(df, ycol, title, ylab, order, outname, figsize=(6, 4)):
    # mean_se_plot_side expects xcol values to match label_map/order,
    # so we pass xcol="source" and provide label_map+order in label space.
    mean_se_plot_side(
        df=df,
        xcol="source",
        ycol=ycol,
        title=title,
        ylab=ylab,
        label_map=SOURCE_LABEL,
        order=order,
        ci_mult=1.0,
        rotate_xticks=0,
        connect_ids=True,
        figsize=figsize,
        outname=os.path.join(outdir, outname),
    )

# -------------------------
# (1) RAW: all sources
# -------------------------
r_all = ratings[ratings["source"].isin(ALL_SOURCES)]
pm_all = participant_means(r_all)
plot_mean_se(
    df=pm_all,
    ycol="rating",
    title="Participant-level ratings",
    ylab="Rating",
    order=ORDER_ALL,
    outname="network_raw__all.png",
)

# -------------------------
# (2) RAW: human-only
# -------------------------
r_hum = ratings[ratings["source"].isin(HUMAN_SOURCES)]
pm_hum = participant_means(r_hum)
plot_mean_se(
    df=pm_hum,
    ycol="rating",
    title="Participant-level ratings",
    ylab="Rating",
    order=ORDER_HUM,
    outname="network_raw__human.png",
)

# (3) Z: all sources (within-participant)
rz_all = add_within_key_z(r_all)
pmz_all = rz_all.groupby(["key", "source"], as_index=False)["rating_z"].mean()
plot_mean_se(
    df=pmz_all,
    ycol="rating_z",
    title="Participant-level ratings",
    ylab="Rating z-scored (within participant)",
    order=ORDER_ALL,
    outname="network_z__all.png",
)

# (4) Z: human-only (within-participant, computed on human-only subset)
# they will all pass through 0 here because that is the mean.
rz_hum = add_within_key_z(r_hum)
pmz_hum = rz_hum.groupby(["key", "source"], as_index=False)["rating_z"].mean()
plot_mean_se(
    df=pmz_hum,
    ycol="rating_z",
    title="Participant-level ratings",
    ylab="Rating z-scored (within participant)",
    order=ORDER_HUM,
    outname="network_z__human.png",
)

print("Saved to:", outdir)

# -------------------------
# (5) COMBINED: human-only, raw vs z (same style as mean_se_plot_side)
# -------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True)

# Left: raw (human-only)
mean_se_plot_side(
    df=pm_hum,
    xcol="source",
    ycol="rating",
    title="",
    ylab="Rating",
    label_map=SOURCE_LABEL,
    order=ORDER_HUM,
    ci_mult=1.0,
    rotate_xticks=0,
    connect_ids=True,
    ax=axes[0],       
    outname=None,
)

# Right: z (human-only, within participant)
mean_se_plot_side(
    df=pmz_hum,
    xcol="source",
    ycol="rating_z",
    title="",
    ylab="Rating z-scored",
    label_map=SOURCE_LABEL,
    order=ORDER_HUM,
    ci_mult=1.0,
    rotate_xticks=0,
    connect_ids=True,
    ax=axes[1],        
    outname=None,
)

fig.suptitle("Participant-level network ratings", y=1)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "network_human__raw_vs_z__side_by_side.png"), dpi=300, bbox_inches="tight")
plt.close(fig)