"""
VMP 2026-02-06 (refactored)

2x2 per-participant stance/topic network plots for ONE selected BERTopic run.

Input:
  ../data/topics_bertopic/selection/overview_top10.csv
  ../data/topics_bertopic/selection/statement_topics/<label>__statement_topics.csv
  public/distractors_w*.json (sanitized)

Output (PDF only):
  ../fig/BERTopic/stance_topic/<label>__outlierFalse/individual/*.pdf
  ../fig/BERTopic/stance_topic/<label>__outlierFalse/<label>__ALL.pdf

and same for outlierTrue

VMP 2026-02-08: tested and run.
"""

from __future__ import annotations

import re
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pypdf import PdfWriter

from utilities import wave_1, wave_2, get_public_path

# -------------------------
# Config (tweak here)
# -------------------------
EXCLUDE_OUTLIER_TOPIC = False  # toggle manually True/False

POS_KEY, EDGES_KEY = "pos_3", "edges_3"
CANVAS, PAD = 585, 25

APPEND_TOPIC = True
STANCE_NODE_SIZE = 160
TOPIC_NODE_BASE, TOPIC_NODE_CAP = 220, 650
EDGE_W_MIN, EDGE_W_MAX = 2.5, 7.0

EDGE_COLORS = {"positive": "#998ec3", "negative": "#f1a340", "both": "#1b9e77"}
DEFAULT_NODE_COLOR = "#cccccc"
NODE_STROKE = "#000000"

FIGSIZE, DPI = (10, 10), 300
LABEL_FONTSIZE, TITLE_FONTSIZE = 8, 12

SEL = Path("../data/public/bertopic/selection")
TOP10 = pd.read_csv(SEL / "overview_top10.csv")
STMT_DIR = SEL / "statement_topics"

# Choose the run: pick whichever model is ranked 01 (best DBCV)
# otherwise manually input LABEL.
_matches = sorted(STMT_DIR.glob("01__*__statement_topics.csv"))
if not _matches:
    raise FileNotFoundError("No 01__* statement_topics file found in " + str(STMT_DIR))
LABEL = _matches[0].stem.replace("__statement_topics", "")

# Output folders
OUTROOT = Path(f"../fig/BERTopic/stance_topic/{LABEL}__outlier{EXCLUDE_OUTLIER_TOPIC}")
OUTIND  = OUTROOT / "individual"
OUTIND.mkdir(parents=True, exist_ok=True)


# -------------------------
# Load participant data from public (sanitized)
# -------------------------
distractors_w1_path = get_public_path("distractors_w{wave}.json", wave=wave_1)
distractors_w2_path = get_public_path("distractors_w{wave}.json", wave=wave_2)

with open(distractors_w1_path, encoding='utf-8') as f:
    data_w1 = json.load(f)
with open(distractors_w2_path, encoding='utf-8') as f:
    data_w2 = json.load(f)
keys = sorted(set(data_w1) & set(data_w2))


# -------------------------
# Load statement topics (this is the "point to a model" step)
# -------------------------
df_stmt = pd.read_csv(STMT_DIR / f"{LABEL}__statement_topics.csv")[["key", "wave", "stance", "topic"]]
df_stmt["key"] = df_stmt["key"].astype(str)
df_stmt["wave"] = pd.to_numeric(df_stmt["wave"], errors="raise").astype(int)
df_stmt["stance"] = df_stmt["stance"].astype(str).str.strip()
df_stmt["topic"] = pd.to_numeric(df_stmt["topic"], errors="raise").astype(int)
if EXCLUDE_OUTLIER_TOPIC:
    df_stmt = df_stmt[df_stmt["topic"] != -1].copy()


# -------------------------
# Helpers
# -------------------------
def invert_y(pos): return {k: (x, CANVAS - y) for k, (x, y) in pos.items()}

def clean_filename(s):
    s = re.sub(r"[^\w\-_\. ]", "_", str(s)).strip().replace(" ", "_")
    return s[:180]

def collapse_polarity(p):
    return "both" if ("positive" in p and "negative" in p) else ("positive" if "positive" in p else "negative")

def topic_palette(topics):
    cmap = plt.get_cmap("tab20")
    topics = sorted(set(int(t) for t in topics))
    def hx(rgba):
        r, g, b, _ = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
    return {t: hx(cmap(i % 20)) for i, t in enumerate(topics)}

def stance_to_topic(key, wave):
    d = df_stmt[(df_stmt.key == str(key)) & (df_stmt.wave == int(wave))]
    return dict(zip(d.stance, d.topic))

def topics_present(data_by_key, key, s2t):
    stances = [n["label"] for n in data_by_key[key]["positions"][POS_KEY]]
    out = set()
    for s in stances:
        t = s2t.get(str(s).strip())
        if t is None: continue
        t = int(t)
        if EXCLUDE_OUTLIER_TOPIC and t == -1: continue
        out.add(t)
    return out

# -------------------------
# Graphs
# -------------------------
def build_stance_graph(data_by_key, key, s2t, tcolor):
    d = data_by_key[key]
    pos = invert_y({n["label"]: (n["x"], n["y"]) for n in d["positions"][POS_KEY]})
    G = nx.Graph()
    G.add_nodes_from(pos)

    for n in G.nodes:
        s = str(n).strip()
        t = s2t.get(s)
        if t is None:
            G.nodes[n]["hex"] = DEFAULT_NODE_COLOR
            G.nodes[n]["disp"] = s
        else:
            t = int(t)
            G.nodes[n]["hex"] = tcolor.get(t, DEFAULT_NODE_COLOR)
            G.nodes[n]["disp"] = f"{s}\n(T{t})" if APPEND_TOPIC else s

    for e in d["edges"][EDGES_KEY]:
        u = str(e["stance_1"]).strip()
        v = str(e["stance_2"]).strip()
        pol = e.get("polarity")
        if pol not in {"positive", "negative"}: continue
        if G.has_edge(u, v):
            G.edges[u, v]["_p"].add(pol)
        else:
            G.add_edge(u, v, _p={pol})

    for u, v, ed in G.edges(data=True):
        ed["polarity"] = collapse_polarity(ed.pop("_p"))

    return pos, G

def build_topic_graph(data_by_key, key, s2t, tcolor):
    d = data_by_key[key]
    pos_st = invert_y({n["label"]: (n["x"], n["y"]) for n in d["positions"][POS_KEY]})

    members = {}
    for stance, xy in pos_st.items():
        t = s2t.get(str(stance).strip())
        if t is None: continue
        t = int(t)
        if EXCLUDE_OUTLIER_TOPIC and t == -1: continue
        members.setdefault(t, []).append(xy)

    topics = sorted(members)
    pos = {t: tuple(np.asarray(members[t], float).mean(axis=0)) for t in topics}

    G = nx.Graph()
    G.add_nodes_from(topics)
    for t in topics:
        n = len(members[t])
        G.nodes[t]["hex"] = tcolor.get(t, DEFAULT_NODE_COLOR)
        G.nodes[t]["disp"] = f"T{t}"
        G.nodes[t]["node_size"] = min(TOPIC_NODE_BASE * np.sqrt(max(n, 1)), TOPIC_NODE_CAP)

    for e in d["edges"][EDGES_KEY]:
        s1 = str(e["stance_1"]).strip()
        s2 = str(e["stance_2"]).strip()
        t1, t2 = s2t.get(s1), s2t.get(s2)
        if t1 is None or t2 is None: continue
        t1, t2 = int(t1), int(t2)
        if EXCLUDE_OUTLIER_TOPIC and (t1 == -1 or t2 == -1): continue
        pol = e.get("polarity")
        if pol not in {"positive", "negative"}: continue

        if G.has_edge(t1, t2):
            G.edges[t1, t2]["_p"].add(pol)
            G.edges[t1, t2]["count"] += 1
        else:
            G.add_edge(t1, t2, _p={pol}, count=1)

    for u, v, ed in G.edges(data=True):
        ed["polarity"] = collapse_polarity(ed.pop("_p"))

    return pos, G


# -------------------------
# Drawing + 2x2 plot
# -------------------------
def draw(ax, G, pos, title, node_sizes=None, wrap=True, wrap_width=18, edge_width_attr=None):
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(-PAD, CANVAS + PAD)
    ax.set_ylim(-PAD, CANVAS + PAD)
    ax.add_patch(Rectangle((0, 0), CANVAS, CANVAS, fill=False, linewidth=1.0, edgecolor="black"))

    node_colors = [G.nodes[n].get("hex", DEFAULT_NODE_COLOR) for n in G.nodes]
    sizes = [STANCE_NODE_SIZE] * G.number_of_nodes() if node_sizes is None else [node_sizes.get(n, STANCE_NODE_SIZE) for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors=NODE_STROKE, linewidths=0.6, node_size=sizes, ax=ax)

    ecols = [EDGE_COLORS[G.edges[u, v]["polarity"]] for u, v in G.edges]
    if edge_width_attr is None:
        ewidths = [3.0] * G.number_of_edges()
    else:
        raw = np.array([G.edges[u, v].get(edge_width_attr, 1) for u, v in G.edges], float)
        if raw.size == 0:
            ewidths = []
        else:
            rmin, rmax = raw.min(), raw.max()
            ewidths = [EDGE_W_MIN] * len(raw) if rmax <= rmin else list(EDGE_W_MIN + (raw - rmin) * (EDGE_W_MAX - EDGE_W_MIN) / (rmax - rmin))

    nx.draw_networkx_edges(G, pos, edge_color=ecols, width=ewidths, ax=ax)

    labels = {n: str(G.nodes[n].get("disp", n)) for n in G.nodes}
    if wrap:
        labels = {n: textwrap.fill(txt, width=wrap_width) for n, txt in labels.items()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=LABEL_FONTSIZE, ax=ax)


def plot_2x2(key):
    s2t1 = stance_to_topic(key, 1)
    s2t2 = stance_to_topic(key, 2)

    topics_union = topics_present(data_w1, key, s2t1) | topics_present(data_w2, key, s2t2)
    tcolor = topic_palette(topics_union) if topics_union else {}

    pos_s1, G_s1 = build_stance_graph(data_w1, key, s2t1, tcolor)
    pos_s2, G_s2 = build_stance_graph(data_w2, key, s2t2, tcolor)
    pos_t1, G_t1 = build_topic_graph(data_w1, key, s2t1, tcolor)
    pos_t2, G_t2 = build_topic_graph(data_w2, key, s2t2, tcolor)

    s1 = {n: G_t1.nodes[n]["node_size"] for n in G_t1.nodes}
    s2 = {n: G_t2.nodes[n]["node_size"] for n in G_t2.nodes}

    fig, ax = plt.subplots(2, 2, figsize=FIGSIZE, dpi=DPI)
    ax = ax.ravel()

    draw(ax[0], G_s1, pos_s1, "Wave 1 (stances)", wrap=True,  edge_width_attr=None)
    draw(ax[1], G_t1, pos_t1, "Wave 1 (topics)",  node_sizes=s1, wrap=False, edge_width_attr="count")
    draw(ax[2], G_s2, pos_s2, "Wave 2 (stances)", wrap=True,  edge_width_attr=None)
    draw(ax[3], G_t2, pos_t2, "Wave 2 (topics)",  node_sizes=s2, wrap=False, edge_width_attr="count")

    #fig.suptitle(f"Participant ID: {str(key)}", fontsize=TITLE_FONTSIZE)
    #fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.tight_layout()
    return fig

# -------------------------
# Run
# -------------------------
# wipe individual pdfs
for p in OUTIND.glob("*.pdf"):
    p.unlink()

# export individuals
for k in keys:
    fig = plot_2x2(k)
    fig.savefig(OUTIND / f"{clean_filename(k)}.pdf")
    plt.close(fig)

# merge
out_pdf = OUTROOT / f"{LABEL}__ALL.pdf"
writer = PdfWriter()
for p in sorted(OUTIND.glob("*.pdf")):
    writer.append(str(p))
with open(out_pdf, "wb") as f:
    writer.write(f)

print("Saved:", OUTROOT)
print("Merged ->", out_pdf)
