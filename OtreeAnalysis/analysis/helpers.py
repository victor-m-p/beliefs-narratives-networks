import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import os


# Used by: 01_curation.py
def safe_json_loads(value, fallback="NA"):
    try:
        if pd.isna(value):
            return fallback
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


# Used by: 28_bertopic_prepare.py
def extract_nodes(data_dict, node_type):
    df_nodes = pd.concat(
        (
            pd.DataFrame(v['nodes'][node_type]).assign(key=k)
            for k, v in data_dict.items()
        ),
        ignore_index=True
    )
    return df_nodes


# Used by: 28_bertopic_prepare.py
def extract_LLM_edges(data_dict, mdl):
    df_edges = pd.concat(
        (
            pd.DataFrame(v['LLM'][mdl]).assign(key=k)
            for k, v in data_dict.items()
        ),
        ignore_index=True
    )
    return df_edges


# Used by: 28_bertopic_prepare.py
def extract_hum_edges(data_dict, type):
    df_edges = pd.concat(
        (
            pd.DataFrame(v['edges'][type]).assign(key=k)
            for k, v in data_dict.items()
        ),
        ignore_index=True
    )
    return df_edges


# Used by: 34_bertopic_pred.py
def normalize_ab(df, a='a', b='b'):
    """Normalize a-b columns so that a <= b."""
    df = df.copy()
    df[[a, b]] = np.sort(df[[a, b]].to_numpy(), axis=1)
    return df


# --- Embedding helpers (used by 29_bertopic_fit.py via embed_df) ---

def apply_prefix(texts, prefix=None):
    if prefix is None:
        return texts
    return [prefix + t for t in texts]


# Used by: 29_bertopic_fit.py
def embed_df(df, model, text_col="stance", batch_size=64, encode_normalize=False, prefix=None):
    """
    Encode df[text_col] using SentenceTransformer, with optional prefix and optional
    model-side L2 normalization (normalize_embeddings=True).

    Returns:
      df_reset, E (np.ndarray shape [n, d])
    """
    df = df.reset_index(drop=True).copy()
    texts = apply_prefix(df[text_col].tolist(), prefix)

    E = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=encode_normalize
    )
    return df, E


# --- Plotting helpers ---

from matplotlib.patches import Polygon

# Used by: 23_rating_networks.py, 31_bertopic_retest.py
def mean_se_plot_side(
    df,
    xcol,
    ycol,
    xlab="",
    ylab="",
    title="",
    label_map=None,
    order=None,
    outname=None,
    ci_mult=1.0,              # 1.0 = ±1 SE; 1.96 ≈ 95% CI
    # layout / styling
    box_offset=-0.13,         # left shift
    point_offset=+0.13,       # right shift
    box_width=0.25,           # thinner boxes
    jitter=0.06,              # horizontal jitter for points
    point_size=22,
    point_alpha=0.35,
    show_fliers=False,
    rotate_xticks=30,
    connect_ids=False,
    figsize=(7.8, 4.2),
    ax=None,
):
    df_plot = df.copy()

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # map labels for display (but keep xcol name the same)
    if label_map is not None:
        df_plot[xcol] = df_plot[xcol].astype(object).map(label_map).fillna(df_plot[xcol])

    if order is None:
        order = sorted(df_plot[xcol].dropna().unique())

    # summary stats (mean, sd, n, se)
    g = df_plot.groupby(xcol)[ycol]
    summary = g.agg(mean="mean", sd="std", n="count").reset_index()
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    summary["err"] = ci_mult * summary["se"]

    # enforce order for plotting/summary
    summary = summary.set_index(xcol).loc[order].reset_index()

    # numeric x positions for categories
    x_base = np.arange(len(order), dtype=float)

    # --- 1) Boxplots shifted left ---
    data_by_group = [
        df_plot.loc[df_plot[xcol] == grp, ycol].dropna().to_numpy()
        for grp in order
    ]
    box_positions = x_base + box_offset

    bp = ax.boxplot(
        data_by_group,
        positions=box_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=show_fliers,
        manage_ticks=False,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
        patch.set_alpha(0.7)
    for key in ("whiskers", "caps", "medians"):
        for line in bp[key]:
            line.set_color("black")
            line.set_linewidth(1.0)

    # --- 2) Points shifted right ("dots") ---
    point_positions = x_base + point_offset
    rng = np.random.default_rng(0)

    for i, grp in enumerate(order):
        yvals = data_by_group[i]
        if len(yvals) == 0:
            continue
        xj = point_positions[i] + rng.uniform(-jitter, jitter, size=len(yvals))
        ax.scatter(xj, yvals, s=point_size, alpha=point_alpha, color="lightsteelblue", linewidths=0)

    # --- 3) Mean ± SE as a diamond ("lozenge") ---
    means = summary["mean"].to_numpy()
    errs  = summary["err"].to_numpy()
    diamond_halfwidth = 0.04

    for x0, m, e in zip(point_positions, means, errs):
        verts = [
            (x0, m + e),
            (x0 + diamond_halfwidth, m),
            (x0, m - e),
            (x0 - diamond_halfwidth, m),
        ]
        poly = Polygon(
            verts,
            closed=True,
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
        ax.add_patch(poly)

    # --- 4) optional: connect individual IDs across conditions ---
    if connect_ids:
        wide = (
            df_plot
            .pivot(index="key", columns=xcol, values=ycol)
            .reindex(columns=order)
        )
        for _, row in wide.iterrows():
            yvals = row.values
            if np.any(np.isnan(yvals)):
                continue
            ax.plot(
                point_positions,
                yvals,
                color="lightsteelblue",
                alpha=0.15,
                linewidth=0.6,
                zorder=0
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(order)

    if rotate_xticks:
        plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(-0.6, len(order) - 0.4)

    if outname is not None:
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        fig.savefig(outname, bbox_inches="tight", dpi=300)
        if created_fig:
            plt.close(fig)
    else:
        if created_fig:
            plt.show()

    return summary
