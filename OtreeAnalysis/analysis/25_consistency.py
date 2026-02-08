"""
VMP 2026-02-06 (refactored)
Participant-level consistency between:
(1) canvas vs pairwise interview
(2) canvas vs LLM

Requires:
  public/edges_canvas_w*.csv
  public/edges_pairwise_w*.csv (NOTE: may not exist in safe data)
  public/edges_llm_w*.csv
  public/distractors_w*.json

VMP 2026-02-07: tested and run.
"""

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from utilities import wave_2, get_public_path

wave = wave_2

outdir = "../fig/consistency"
os.makedirs(outdir, exist_ok=True)

THEME = {
    "cmap": "Blues",
    "linecolor": "gray",
    "linewidths": 0.5,
    "dpi": 300,
    "figsize_default": (6, 5),
    "annot_percent_fmt": ".1f",
}

ORDER = ("conflict", "no connection", "support")


def plot_heatmap_table(table, annot=None, title=None, xlabel=None, ylabel=None,
                       theme=THEME, outpath=None, figsize=None, cbar=True, cbar_label=None):

    if figsize is None:
        figsize = theme["figsize_default"]

    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        table,
        annot=annot, fmt="",
        cmap=theme["cmap"],
        cbar=cbar,
        linewidths=theme["linewidths"],
        linecolor=theme["linecolor"]
    )
    if cbar and cbar_label:
        ax.collections[0].colorbar.set_label(cbar_label)

    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=theme["dpi"], bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def heatmap_avg_percent_by_key(df, key_col, p1, p2, col_p1, col_p2,
                               order=ORDER, title=None, outpath=None, theme=THEME,
                               figsize=(4, 4)):

    counts = df.groupby([key_col, p1, p2]).size().reset_index(name="n")
    counts["pct"] = 100 * counts["n"] / counts.groupby(key_col)["n"].transform("sum")

    wide = counts.pivot_table(index=key_col, columns=[p1, p2], values="pct", fill_value=0)
    avg = wide.mean(axis=0).unstack().reindex(index=order, columns=order).round(1)

    # avoid DataFrame.applymap (your earlier error)
    annot = avg.apply(lambda col: col.map(lambda x: f"{x:{theme['annot_percent_fmt']}}%"))

    plot_heatmap_table(
        avg, annot=annot, title=title,
        xlabel=col_p2, ylabel=col_p1,
        theme=theme, outpath=outpath,
        figsize=figsize, cbar=True, cbar_label="% of edges"
    )
    return avg


# -------------------------
# Load canonical edges (from public folder)
# -------------------------
df_canvas = pd.read_csv(get_public_path(f"edges_canvas_w{wave}.csv"))
df_llm = pd.read_csv(get_public_path(f"edges_llm_w{wave}.csv"))
# NOTE: pairwise edges may not exist in sanitized data (contained interview Q&A)
try:
    df_pairwise = pd.read_csv(get_public_path(f"edges_pairwise_w{wave}.csv"))
except FileNotFoundError:
    print("Warning: edges_pairwise not found (may be excluded from safe data)")
    df_pairwise = pd.DataFrame(columns=["key", "stance_1", "stance_2", "pairwise"])

# -------------------------
# (1) Canvas vs Pairwise
# -------------------------
df_canvas_pairwise = (
    df_pairwise
    .merge(df_canvas, on=["key", "stance_1", "stance_2"], how="left")
    .fillna({"canvas": "no connection"})
)

heatmap_avg_percent_by_key(
    df_canvas_pairwise, key_col="key",
    p1="pairwise", p2="canvas",
    col_p1="Pairwise edges", col_p2="Canvas edges",
    title="Participant consistency",
    outpath=os.path.join(outdir, "canvas_interview.png"),
)

# -------------------------
# (2) Canvas vs LLM (needs final nodes grid)
# -------------------------
distractors_path = get_public_path("distractors_w{wave}.json", wave=wave)
with open(distractors_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def pairwise_beliefs_per_key(data, list_path=("nodes", "final"), nodename="belief"):
    a, b = list_path
    rows = []
    for key, bundle in data.items():
        items = bundle[a][b]
        for x, y in combinations(items, 2):
            rows.append({"key": key, "stance_1": x[nodename], "stance_2": y[nodename]})
    df = pd.DataFrame(rows)
    df[["stance_1", "stance_2"]] = np.sort(df[["stance_1", "stance_2"]].to_numpy(), axis=1)
    return df

grid = pairwise_beliefs_per_key(data)

df_canvas_llm = df_canvas.merge(df_llm, on=["key", "stance_1", "stance_2"], how="outer")
final = grid.merge(df_canvas_llm, on=["key", "stance_1", "stance_2"], how="left")

final["canvas"] = final["canvas"].fillna("no connection")
final["llm"]    = final["llm"].fillna("no connection")

heatmap_avg_percent_by_key(
    final, key_col="key",
    p1="llm", p2="canvas",
    col_p1="LLM edges", col_p2="Canvas edges",
    title="Canvas-LLM consistency",
    outpath=os.path.join(outdir, "canvas_llm.png"),
)
