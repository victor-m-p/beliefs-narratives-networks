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


# =============================================================
# below not used for the preprint
# =============================================================
# Question: are inconsistent / missing edges between less relevant nodes?
#   - Node relevance: data[key]["nodes"]["relevance"] → {belief, relevance 0-100}
#   - LLM edge strength: data[key]["LLM"]["edge_results"] → strength 0-100

# ---- 1. Build node-relevance lookup (key, stance) → relevance ----
relevance_rows = []
for key, bundle in data.items():
    for item in bundle.get("nodes", {}).get("relevance", []):
        relevance_rows.append({
            "key": key,
            "stance": item["belief"],
            "relevance": item["relevance"],
            "is_distractor": item.get("is_distractor", False),
        })
df_relevance = pd.DataFrame(relevance_rows)
df_relevance = df_relevance[~df_relevance["is_distractor"]].copy()
df_relevance["stance"] = df_relevance["stance"].astype(str).str.strip()

rel_lookup = df_relevance.set_index(["key", "stance"])["relevance"]

def attach_relevance(df, stance_col):
    """Join relevance for one stance endpoint."""
    idx = pd.MultiIndex.from_arrays(
        [df["key"].astype(str), df[stance_col].astype(str).str.strip()],
        names=["key", "stance"],
    )
    return rel_lookup.reindex(idx).values

# ---- 2. Canvas vs Pairwise: mean relevance per 3×3 cell ----
cp = df_canvas_pairwise.copy()
cp["rel_1"] = attach_relevance(cp, "stance_1")
cp["rel_2"] = attach_relevance(cp, "stance_2")
cp["rel_mean"] = (cp["rel_1"] + cp["rel_2"]) / 2
cp["rel_prod"] = cp["rel_1"] * cp["rel_2"] / 100 

for agg_col, label in [("rel_mean", "avg relevance"), ("rel_prod", "avg product relevance")]:
    tbl = cp.groupby(["pairwise", "canvas"])[agg_col].mean().unstack()
    tbl = tbl.reindex(index=ORDER, columns=ORDER)
    print(f"\n{label}:")
    print(tbl.round(1).to_string())

    annot = tbl.apply(lambda col: col.map(lambda x: f"{x:.1f}" if pd.notna(x) else ""))
    plot_heatmap_table(
        tbl, annot=annot,
        title=f"Canvas vs Pairwise — {label}",
        xlabel="Canvas edges", ylabel="Pairwise edges",
        outpath=os.path.join(outdir, f"relevance_canvas_pairwise_{agg_col}.png"),
        figsize=(4, 4), cbar_label="Relevance",
    )

# ---- 3. Canvas vs LLM: mean relevance per 3×3 cell ----
cl = final.copy()
cl["rel_1"] = attach_relevance(cl, "stance_1")
cl["rel_2"] = attach_relevance(cl, "stance_2")
cl["rel_mean"] = (cl["rel_1"] + cl["rel_2"]) / 2
cl["rel_prod"] = cl["rel_1"] * cl["rel_2"] / 100

for agg_col, label in [("rel_mean", "avg relevance"), ("rel_prod", "avg product relevance")]:
    tbl = cl.groupby(["llm", "canvas"])[agg_col].mean().unstack()
    tbl = tbl.reindex(index=ORDER, columns=ORDER)
    print(f"\n{label}:")
    print(tbl.round(1).to_string())

    annot = tbl.apply(lambda col: col.map(lambda x: f"{x:.1f}" if pd.notna(x) else ""))
    plot_heatmap_table(
        tbl, annot=annot,
        title=f"Canvas vs LLM — {label}",
        xlabel="Canvas edges", ylabel="LLM edges",
        outpath=os.path.join(outdir, f"relevance_canvas_llm_{agg_col}.png"),
        figsize=(4, 4), cbar_label="Relevance",
    )

# ---- 4. LLM edge strength per 3×3 cell (canvas vs LLM) ----
# Build a lookup: (key, stance_1, stance_2) → LLM strength
llm_strength_rows = []
for key, bundle in data.items():
    for e in bundle.get("LLM", {}).get("edge_results", []):
        s1, s2 = str(e["stance_1"]).strip(), str(e["stance_2"]).strip()
        pair = tuple(sorted([s1, s2]))
        llm_strength_rows.append({
            "key": key,
            "stance_1": pair[0],
            "stance_2": pair[1],
            "llm_strength": e.get("strength", np.nan),
        })
df_llm_strength = pd.DataFrame(llm_strength_rows)
df_llm_strength = df_llm_strength.drop_duplicates(subset=["key", "stance_1", "stance_2"])

cl_s = cl.merge(df_llm_strength, on=["key", "stance_1", "stance_2"], how="left")

tbl_str = cl_s.groupby(["llm", "canvas"])["llm_strength"].mean().unstack()
tbl_str = tbl_str.reindex(index=ORDER, columns=ORDER)

annot_str = tbl_str.apply(lambda col: col.map(lambda x: f"{x:.1f}" if pd.notna(x) else "—"))
plot_heatmap_table(
    tbl_str, annot=annot_str,
    title="Canvas vs LLM — avg LLM strength",
    xlabel="Canvas edges", ylabel="LLM edges",
    outpath=os.path.join(outdir, "strength_canvas_llm.png"),
    figsize=(4, 4), cbar_label="LLM strength (0-100)",
)

# ---- 5. LLM edge strength per 3×3 cell (pairwise vs canvas) ----
cp_s = cp.merge(df_llm_strength, on=["key", "stance_1", "stance_2"], how="left")

tbl_str_pw = cp_s.groupby(["pairwise", "canvas"])["llm_strength"].mean().unstack()
tbl_str_pw = tbl_str_pw.reindex(index=ORDER, columns=ORDER)

annot_pw = tbl_str_pw.apply(lambda col: col.map(lambda x: f"{x:.1f}" if pd.notna(x) else "—"))
plot_heatmap_table(
    tbl_str_pw, annot=annot_pw,
    title="Canvas vs Pairwise — avg LLM strength",
    xlabel="Canvas edges", ylabel="Pairwise edges",
    outpath=os.path.join(outdir, "strength_canvas_pairwise.png"),
    figsize=(4, 4), cbar_label="LLM strength (0-100)",
)