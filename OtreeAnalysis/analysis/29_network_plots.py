"""
29_network_plots.py

VMP 2026-03-05

Individual belief networks (stance level, no topic coloring).
Node style matches the canvas app (MapNodePlacement.html).
One compiled PDF: wave 1 | wave 2 per page, participant key shown.

Reads:
  ../data/public/distractors_w*.json

Writes:
  ../fig/networks/all_networks.pdf            → Figure 2 (main text)
  ../fig/networks/individual/<i>_<key>.svg   (one per participant)
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

from utilities import wave_1, wave_2, get_public_path


# -------------------------
# Config
# -------------------------
POS_KEY   = "pos_3"
EDGES_KEY = "edges_3"
CANVAS        = 585          # canvas pixel size (matches app)
CANVAS_TOP    = CANVAS + 50  # canvas top extended to keep labels inside border
PAD           = 20           # whitespace around canvas border

# node style — from canvasfunc.js DRAW_DEFAULTS:
#   defaultRadius=20 (in 585px canvas), strokeWidth=2, fill=#8CC7CA
NODE_COLOR   = "#8cc7ca"
NODE_STROKE  = "#111111"
NODE_RADIUS  = 20            # data units; matches app defaultRadius
NODE_SIZE    = 800           # pt² ≈ π×(20 × 0.806 pt/unit)²
NODE_STROKE_LW = 1.5         # linewidths; matches app strokeWidth=2

BORDER_COLOR = "#000000"
BORDER_LW    = 2.5           # matches app borderWidth=3

# label: bottom-anchored so offset is constant regardless of line count
LABEL_Y_OFFSET = NODE_RADIUS + 4   # data units above node centre → bottom of text block
LABEL_VA       = "bottom"

# edge style — canvasfunc: minWidth=1, maxWidth=10, default strength=50 → 5.5 px
EDGE_COLORS = {"positive": "#998ec3", "negative": "#f1a340", "both": "#1b9e77"}
EDGE_WIDTH  = 5.5

LABEL_FONTSIZE  = 11   # matches app labelFont: '14px' scaled to figure size
PANEL_TITLE_FS  = 20
FIGSIZE         = (14, 7)     # landscape: two square-ish panels side by side

KEY_FONTSIZE = 9    # participant ID shown at top of each page

OUTDIR    = Path("../fig/networks")
INDIV_DIR = OUTDIR / "individual"


# -------------------------
# Setup — wipe on rerun
# -------------------------
if OUTDIR.exists():
    shutil.rmtree(OUTDIR)
OUTDIR.mkdir(parents=True)
INDIV_DIR.mkdir(parents=True)

distractors_w1_path = get_public_path("distractors_w{wave}.json", wave=wave_1)
distractors_w2_path = get_public_path("distractors_w{wave}.json", wave=wave_2)

with open(distractors_w1_path, encoding="utf-8") as f:
    data_w1 = json.load(f)
with open(distractors_w2_path, encoding="utf-8") as f:
    data_w2 = json.load(f)

# only participants present in both waves
keys = sorted(set(data_w1) & set(data_w2))


# -------------------------
# Helpers
# -------------------------
def invert_y(pos):
    """Flip y so canvas origin is bottom-left (SVG/HTML uses top-left)."""
    return {k: (x, CANVAS - y) for k, (x, y) in pos.items()}


def collapse_polarity(p):
    return "both" if ("positive" in p and "negative" in p) \
        else ("positive" if "positive" in p else "negative")


def build_stance_graph(data_by_key, key):
    d   = data_by_key[key]
    pos = invert_y({n["label"]: (n["x"], n["y"]) for n in d["positions"][POS_KEY]})

    G = nx.Graph()
    G.add_nodes_from(pos)

    for e in d["edges"][EDGES_KEY]:
        u   = str(e["stance_1"]).strip()
        v   = str(e["stance_2"]).strip()
        pol = e.get("polarity")
        if pol not in {"positive", "negative"}:
            continue
        if G.has_edge(u, v):
            G.edges[u, v]["_p"].add(pol)
        else:
            G.add_edge(u, v, _p={pol})

    for u, v, ed in G.edges(data=True):
        ed["polarity"] = collapse_polarity(ed.pop("_p"))

    return pos, G


def draw_stance(ax, G, pos, title=""):
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-PAD, CANVAS + PAD)
    ax.set_ylim(-PAD, CANVAS_TOP + PAD)

    # canvas border (extended upward)
    ax.add_patch(Rectangle(
        (0, 0), CANVAS, CANVAS_TOP,
        fill=False, linewidth=BORDER_LW, edgecolor=BORDER_COLOR,
    ))

    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_FS, pad=4)

    nx.draw_networkx_nodes(
        G, pos,
        node_color=NODE_COLOR,
        edgecolors=NODE_STROKE,
        linewidths=NODE_STROKE_LW,
        node_size=NODE_SIZE,
        ax=ax,
    )

    if G.number_of_edges() > 0:
        ecols = [EDGE_COLORS[G.edges[u, v]["polarity"]] for u, v in G.edges]
        nx.draw_networkx_edges(G, pos, edge_color=ecols, width=EDGE_WIDTH, ax=ax)

    labels = {n: textwrap.fill(str(n), width=18) for n in G.nodes}
    label_pos = {n: (x, y + LABEL_Y_OFFSET) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G, label_pos, labels=labels,
        font_size=LABEL_FONTSIZE,
        verticalalignment=LABEL_VA,
        ax=ax,
    )

# -------------------------
# Compile PDF
# -------------------------
def make_page(key, page_num, total):
    pos_w1, G_w1 = build_stance_graph(data_w1, key)
    pos_w2, G_w2 = build_stance_graph(data_w2, key)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    draw_stance(ax1, G_w1, pos_w1, title="Wave 1")
    draw_stance(ax2, G_w2, pos_w2, title="Wave 2")

    fig.suptitle(
        f"[{page_num:02d}/{total}]  {key}",
        fontsize=KEY_FONTSIZE, x=0.01, ha="left", va="top",
    )
    fig.tight_layout()
    return fig


n = len(keys)

with PdfPages(OUTDIR / "all_networks.pdf") as pdf:
    for i, key in enumerate(keys):
        fig = make_page(key, i + 1, n)
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(str(INDIV_DIR / f"{i + 1:02d}_{key}.svg"), bbox_inches="tight")
        plt.close(fig)

print("Saved to:", OUTDIR)
