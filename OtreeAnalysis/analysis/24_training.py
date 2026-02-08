'''
VMP 2026-02-06 (refactored):
Analyzes training performance across waves.
Uses sanitized public data.

VMP 2026-02-07: tested and run.
'''

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import wave_1, wave_2, get_public_path

# -------------------------
# config
# -------------------------
outdir = "../fig/consistency"
os.makedirs(outdir, exist_ok=True)

# -------------------------
# -------------------------
STANCE_MAP = {
    # Alex
    'Alex attends a conversational meetup': 1,
    'Alex feels embarrassed speaking out loud': -1,
    'Alex has a busy job': -1,
    'Alex practices daily with an app': 1,
    'Alex sees career benefits from learning Spanish': 1,
    'Alex wants to speak with locals': 1,
    'Colleagues are learning Spanish': 1,
    # Jordan
    'A friend has invited Jordan to join their gym sessions': 1,
    'Jordan feels self-conscious exercising in front of others': -1,
    'Jordan often feels exhausted after work': -1,
    'Jordan plans to go to the gym three times a week': 1,
    'Jordan wants to have more energy during the day': 1,
    'Jordan wants to improve their physical fitness': 1,
    'The gym is located on Jordan’s way home from work': 1,
    # Riley
    'A close friend also volunteers there': 1,
    'Riley registered for a weekly volunteer shift': 1,
    'Riley sometimes needs to take care of a younger relative on short notice': -1,
    'Riley tries to plan ahead using reminders and a shared calendar': 1,
    'Riley wants to contribute to the local community': 1,
    'The community center is close to Riley’s home': 1,
    'Unexpected caregiving needs sometimes interfere with Riley’s plans': -1
}

TASK_ORDER = ["support", "conflict"]
TASK_PALETTE = {"support": "#1f77b4", "conflict": "#ff7f0e"}

# -------------------------
# helpers
# -------------------------
def load_json(wave):
    distractors_path = get_public_path("distractors_w{wave}.json", wave=wave)
    with open(distractors_path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_from_dirs(df, dir1, dir2, task="task"):
    df = df.dropna(subset=[dir1, dir2]).copy()
    df["consistent"] = df[dir1] * df[dir2]
    df["correct_direction"] = (
        ((df[task] == "support")  & (df["consistent"] == 1)) |
        ((df[task] == "conflict") & (df["consistent"] == -1))
    ).astype(int)
    return df

def training_acc(data, pres_offset):
    out = []
    for k, v in data.items():
        tr = v["training"]
        omap = {sk: i for i, sk in enumerate(tr["train_order"])}
        for task, rows in [("support", tr["train_pos"]), ("conflict", tr["train_neg"])]:
            df = pd.DataFrame(rows).assign(key=k, task=task)
            df = df[df["attempt_index"] == 0]
            if df.empty:
                continue
            df = df.explode("edges")
            df = df[df["edges"].apply(lambda x: isinstance(x, dict))].copy()
            df["stance_1"] = df["edges"].apply(lambda d: d.get("stance_1"))
            df["stance_2"] = df["edges"].apply(lambda d: d.get("stance_2"))
            df["presentation_order"] = df["scenario_key"].map(omap) + pres_offset
            out.append(df[["key","task","presentation_order","stance_1","stance_2"]])
    if not out:
        return pd.DataFrame(columns=["key","presentation_order","task","accuracy"])
    e = pd.concat(out, ignore_index=True).drop_duplicates(["key","presentation_order","task","stance_1","stance_2"])
    e["s1"] = e["stance_1"].map(STANCE_MAP)
    e["s2"] = e["stance_2"].map(STANCE_MAP)
    e = score_from_dirs(e, "s1", "s2", task="task")
    return (e.groupby(["key","presentation_order","task"])["correct_direction"]
             .mean().reset_index(name="accuracy"))

def canvas_acc(data, wave, pres_order):
    # edges
    e = pd.read_csv(get_public_path(f"edges_canvas_w{wave}.csv")).rename(columns={"canvas":"task"})
    e = e[e["task"].isin(["support","conflict"])].copy()

    # dirs from LLM effect, restricted to final nodes
    nodes_llm = pd.concat(
        (pd.DataFrame(v["LLM"]["node_results"]["results"]).assign(key=k) for k, v in data.items()),
        ignore_index=True
    )[["key","stance","effect"]]

    nodes_final = pd.concat(
        (pd.DataFrame(v["nodes"]["final"]).assign(key=k) for k, v in data.items()),
        ignore_index=True
    ).rename(columns={"belief":"stance"})[["key","stance"]]

    dirs = (nodes_llm.merge(nodes_final, on=["key","stance"], how="inner")
                    .assign(dir=lambda d: d["effect"].map({"INCREASE": 1, "DECREASE": -1}))
                    .dropna(subset=["dir"]))[["key","stance","dir"]]

    e = (e.merge(dirs.rename(columns={"stance":"stance_1","dir":"s1"}), on=["key","stance_1"], how="inner")
           .merge(dirs.rename(columns={"stance":"stance_2","dir":"s2"}), on=["key","stance_2"], how="inner"))

    e = score_from_dirs(e, "s1", "s2", task="task")
    e["presentation_order"] = pres_order
    return (e.groupby(["key","presentation_order","task"])["correct_direction"]
             .mean().reset_index(name="accuracy"))

# -------------------------
# compute
# -------------------------
data_w1 = load_json(wave_1)
data_w2 = load_json(wave_2)

df_acc_train_w1  = training_acc(data_w1, pres_offset=0)   # 0,1,2
df_acc_canvas_w1 = canvas_acc(data_w1, wave_1, pres_order=3)
df_acc_train_w2  = training_acc(data_w2, pres_offset=4)   # 4
df_acc_canvas_w2 = canvas_acc(data_w2, wave_2, pres_order=5)

df_plot = pd.concat([df_acc_train_w1, df_acc_canvas_w1, df_acc_train_w2, df_acc_canvas_w2], ignore_index=True)

x_label_map = {
    0: "Train 1 (w1)",
    1: "Train 2 (w1)",
    2: "Train 3 (w1)",
    3: "Canvas (w1)",
    4: "Train 4 (w2)",
    5: "Canvas (w2)",
}

# -------------------------
# plot: points (individuals) + mean/CI line
# -------------------------
from matplotlib.lines import Line2D

eps = 0.02  # vertical jitter size 
df_pts = df_plot.copy()
df_pts["accuracy_j"] = np.clip(df_pts["accuracy"] + np.random.uniform(-eps, eps, len(df_pts)), 0, 1)

fig, ax = plt.subplots(figsize=(6.8, 4))

sns.stripplot(
    data=df_pts,
    x="presentation_order", y="accuracy_j",
    hue="task", hue_order=TASK_ORDER, palette=TASK_PALETTE,
    dodge=True, jitter=0.18, alpha=0.18, size=3,
    ax=ax, zorder=2
)

sns.pointplot(
    data=df_plot,
    x="presentation_order", y="accuracy",
    hue="task", hue_order=TASK_ORDER, palette=TASK_PALETTE,
    dodge=True, errorbar=("ci", 95),
    markers="o", linestyles="-",
    ax=ax, zorder=3
)

# x labels etc...
xs = np.sort(df_plot["presentation_order"].unique())
ax.set_xticks(xs)
ax.set_xticklabels([x_label_map.get(x, str(x)) for x in xs], rotation=45, ha="right")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("")
ax.set_ylabel("Fraction consistent", size=14)

# remove seaborn's auto legend(s)
if ax.get_legend() is not None:
    ax.get_legend().remove()

# add clean, opaque legend
handles = [
    Line2D([0], [0], marker="o", linestyle="-", color=TASK_PALETTE["support"],  label="support"),
    Line2D([0], [0], marker="o", linestyle="-", color=TASK_PALETTE["conflict"], label="conflict"),
]
ax.legend(handles=handles, loc="lower right", frameon=True)

fig.tight_layout()
plt.savefig(os.path.join(outdir, "training.pdf"))
