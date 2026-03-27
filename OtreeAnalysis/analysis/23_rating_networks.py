"""
VMP 2026-02-06 (refactored 2026-03-15)

Two plots of raw network ratings:
  (1) Canvas vs Canvas + Random
  (2) LLM vs LLM + Random

Input: public/distractors_w*.json (sanitized)
Output: ../fig/ratings/

OUTPUT: Figure 5A (canvas_rating.svg) and Figure 5B (llm_rating.svg).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilities import wave_2, get_public_path
from helpers import mean_se_plot_side

# -------------------------
# Config
# -------------------------
wave = wave_2

outdir = "../fig/ratings"
os.makedirs(outdir, exist_ok=True)

SOURCE_LABEL = {
    "user":        "Canvas",
    "llm":         "LLM",
    "user_random": "Canvas + Random",
    "llm_random":  "LLM + Random",
}

ORDER_CANVAS = ["Canvas", "Canvas + Random"]
ORDER_LLM    = ["LLM",    "LLM + Random"]

CANVAS_SOURCES = ["user", "user_random"]
LLM_SOURCES    = ["llm",  "llm_random"]

# -------------------------
# Load + tidy
# -------------------------
distractors_path = get_public_path("distractors_w{wave}.json", wave=wave)
with open(distractors_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nc = pd.concat(
    (pd.DataFrame(v["network_compare"]).assign(key=k) for k, v in data.items()),
    ignore_index=True,
)

nc["rating_left"]  = pd.to_numeric(nc["rating_left"],  errors="coerce")
nc["rating_right"] = pd.to_numeric(nc["rating_right"], errors="coerce")

left  = nc[["key", "left",  "rating_left" ]].rename(columns={"left":  "source", "rating_left":  "rating"})
right = nc[["key", "right", "rating_right"]].rename(columns={"right": "source", "rating_right": "rating"})
ratings = pd.concat([left, right], ignore_index=True).dropna(subset=["rating"])


def participant_means(df_long):
    return df_long.groupby(["key", "source"], as_index=False)["rating"].mean()


def save_rating_plot(pm, order, outpath):
    fig, ax = plt.subplots(figsize=(4, 3))
    mean_se_plot_side(
        df=pm, xcol="source", ycol="rating",
        title="", ylab="Rating",
        label_map=SOURCE_LABEL, order=order,
        rotate_xticks=0, connect_ids=True, show_mean_dot=True,
        ax=ax, outname=None,
    )
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Plot 1: Canvas vs Canvas + Random
# -------------------------
pm_canvas = participant_means(ratings[ratings["source"].isin(CANVAS_SOURCES)])
save_rating_plot(pm_canvas, ORDER_CANVAS, os.path.join(outdir, "canvas_rating.svg"))

# -------------------------
# Plot 2: LLM vs LLM + Random
# -------------------------
pm_llm = participant_means(ratings[ratings["source"].isin(LLM_SOURCES)])
save_rating_plot(pm_llm, ORDER_LLM, os.path.join(outdir, "llm_rating.svg"))

print("Saved to:", outdir)

# -------------------------
# Mean + SE summary (all four sources)
# -------------------------
ALL_SOURCES = ["user", "llm", "user_random", "llm_random"]
pm_all = participant_means(ratings[ratings["source"].isin(ALL_SOURCES)])

se_summary = (
    pm_all
    .groupby("source")["rating"]
    .agg(n="count", mean="mean", sd="std")
    .assign(se=lambda df: df["sd"] / np.sqrt(df["n"]))
    .loc[ALL_SOURCES]
    .rename(index=SOURCE_LABEL)
    [["n", "mean", "sd", "se"]]
)

print("\nMean + SE of raw ratings (participant-level means, all four sources):")
print(se_summary.to_string(float_format="{:.4f}".format))

# -------------------------
# Numbers for results section
# -------------------------
for source, row in se_summary.iterrows():
    print(f"{source}: M={row['mean']:.1f}, SD={row['sd']:.1f}")
