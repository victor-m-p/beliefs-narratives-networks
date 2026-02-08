'''
VMP 2026-02-06 (refactored):

Accuracy + relevance of LLM-extracted nodes.
Real vs distractor summaries.
One figure with Wave 1 and Wave 2 side-by-side (optional shaded exclusion bands).
Uses sanitized public data (node ratings are safe).

VMP 2026-02-07: tested and run.
'''

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from utilities import wave_1, wave_2, get_public_path

# -------------------------
# config
# -------------------------
FIG_DIR = "../fig/ratings"
os.makedirs(FIG_DIR, exist_ok=True)

DISTRACTOR_DROP = "I eat beans to get enough cholesterol"  # drop this one
SUMMARY_ORDER = ["Real summary", "Fake summary"]
HUE_ORDER = ["Accuracy", "Relevance"]

palette_boxes  = ["#a6c4e0", "#f6c59f"]   # light blue / light orange
palette_points = ["#1f77b4", "#ff7f0e"]   # strong blue / strong orange
box_width = 0.6

# -------------------------
# load + reshape
# -------------------------
def load_wave_long(wave):
    curation_path = get_public_path("curation_w{wave}.json", wave=wave)
    with open(curation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    acc = pd.concat(
        (pd.DataFrame(v["nodes"]["accuracy"]).assign(key=k) for k, v in data.items()),
        ignore_index=True,
    ).rename(columns={"rating": "accuracy"})

    rel = pd.concat(
        (pd.DataFrame(v["nodes"]["relevance"]).assign(key=k) for k, v in data.items()),
        ignore_index=True,
    ).rename(columns={"rating": "relevance"})

    # drop this distractor item
    acc = acc[acc["belief"] != DISTRACTOR_DROP]
    rel = rel[rel["belief"] != DISTRACTOR_DROP]

    df = acc.merge(rel, on=["key", "is_distractor", "belief"], how="inner")

    long = df.melt(
        id_vars=["key", "is_distractor"],
        value_vars=["accuracy", "relevance"],
        var_name="metric",
        value_name="rating",
    )

    long["wave"] = f"Wave {wave}"
    long["summary_type"] = long["is_distractor"].map({False: "Real summary", True: "Fake summary"})
    long["metric_label"] = long["metric"].map({"accuracy": "Accuracy", "relevance": "Relevance"})
    return long

df_long = pd.concat(
    [
        load_wave_long(wave_1),
        load_wave_long(wave_2),
    ],
    ignore_index=True,
)

# -------------------------
# plotting (two panels)
# -------------------------
def plot_panel(ax, df, highlight_regions=True, add_legend=False):
    sns.boxplot(
        data=df,
        x="summary_type",
        y="rating",
        hue="metric_label",
        order=SUMMARY_ORDER,
        hue_order=HUE_ORDER,
        width=box_width,
        palette=palette_boxes,
        showfliers=False,
        ax=ax,
        boxprops=dict(edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        medianprops=dict(color="black", linewidth=1),
    )

    handles, labels = ax.get_legend_handles_labels()

    sns.stripplot(
        data=df,
        x="summary_type",
        y="rating",
        hue="metric_label",
        order=SUMMARY_ORDER,
        hue_order=HUE_ORDER,
        dodge=True,
        alpha=0.1,
        size=4,
        jitter=0.1,
        palette=palette_points,
        linewidth=0,
        legend=False,
        ax=ax,
    )

    if highlight_regions:
        # shade Accuracy only: left half of each x-category cluster
        y_min, y_max = ax.get_ylim()
        shade_color = "#f4c2c2"
        for x_center, lab in zip(ax.get_xticks(), SUMMARY_ORDER):
            x_left = x_center - box_width / 2
            acc_width = box_width / 2

            if lab == "Real summary":
                y0, y1 = -2, 60
            else:  # Fake summary
                y0, y1 = 40, y_max + 2

            ax.add_patch(Rectangle((x_left, y0), acc_width, y1 - y0,
                                   facecolor=shade_color, alpha=0.3, zorder=0))
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("")
    ax.set_ylabel("Rating")
    ax.legend_.remove()

    if add_legend:
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
            ncol=2,
            title=None,
        )

fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), dpi=300, sharey=True)

for ax, wlab in zip(axes, [f"Wave {wave_1}", f"Wave {wave_2}"]):
    d = df_long[df_long["wave"] == wlab]
    plot_panel(ax, d, highlight_regions=True, add_legend=False)
    ax.set_title(wlab)

axes[0].set_ylabel("Rating")
axes[1].set_ylabel("")

# grab legend entries once (from either axis)
handles, labels = axes[0].get_legend_handles_labels()

# remove per-axis legends
for ax in axes:
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

# add one centered, horizontal legend for the whole figure
fig.legend(
    handles[:2], labels[:2], 
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,
    frameon=False,
    title=None,
)

# leave space for the legend at the bottom
plt.tight_layout(rect=[0, 0.08, 1, 1])

fig.suptitle("Ratings of LLM-generated summaries", y=1.02)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "nodes.pdf"), bbox_inches="tight")
plt.close(fig)
