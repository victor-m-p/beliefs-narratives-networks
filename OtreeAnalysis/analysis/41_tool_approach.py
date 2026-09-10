"""
41_tool_approach.py

VMP 2026-09-10

Is between-participant variation in belief-network structure partly just
variation in how people use the canvas, rather than in their beliefs?

We compute mean degree three ways for each participant:

  (1) REAL W1 / W2 - their own belief network in each wave (edges_3,
                     de-duplicated pairs, accepted nodes), as in 40_*.
  (2) TRAINING     - the three training vignettes (Jordan, Riley, Alex),
                     averaged. Training happens in WAVE 1 ONLY. These use fixed
                     statements about a fictional person, so nothing about the
                     participant's own beliefs should show up here.

Training detail: each vignette is drawn in two separate phases, a supporting
phase (train_pos) and a conflicting phase (train_neg), on the same 7 statements.
We take the final passing attempt of each phase and union the pairs, which
mirrors the real canvas where both edge types live on one network. Scenarios are
keyed by scenario_key, not example_index, because presentation order is
randomised per participant.

Training is a supervised task: some edges are required to pass, so there is a
floor (minimum 4 pairs). That floor is the same for everyone, so it shifts the
training measure without biasing the correlation, but it does compress variance.

The three correlations plotted are:
  real W1 vs real W2   - the test-retest benchmark, how well the measure
                         predicts itself across roughly a week
  training vs real W1  - same session
  training vs real W2  - different session, so shared session state (fatigue,
                         engagement, mouse comfort) cannot explain it

If training predicts the real network about as well as the real network predicts
itself, then a substantial part of the variance in these networks is drawing
style rather than belief structure.

Reads:
  ../data/public/distractors_w1.json
  ../data/public/distractors_w2.json

Writes:
  ../fig/tool_approach/tool_approach_per_participant.csv
  ../fig/tool_approach/training_vs_real.svg
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from utilities import wave_1, wave_2, get_public_path

outdir = "../fig/tool_approach"
os.makedirs(outdir, exist_ok=True)

SCENARIOS = {"example1": "Alex", "example2": "Jordan", "example3": "Riley"}

data_by_wave = {}
for wave in [wave_1, wave_2]:
    with open(get_public_path(f"distractors_w{wave}.json"), "r", encoding="utf-8") as f:
        data_by_wave[wave] = json.load(f)
    print(f"wave {wave}: {len(data_by_wave[wave])} participants")

# -------------------------
# real task: mean degree on the participant's own belief network, both waves
# -------------------------
real_rows = []
for wave, data in data_by_wave.items():
    for key, rec in data.items():
        accepted = [n["belief"] for n in rec["nodes"]["final"]
                    if not n.get("is_distractor", False)]
        pairs = {tuple(sorted((e["stance_1"], e["stance_2"])))
                 for e in rec["edges"]["edges_3"]}
        real_rows.append({"key": key, "wave": wave,
                          "n_nodes": len(accepted),
                          "n_pairs": len(pairs),
                          "mean_degree": 2 * len(pairs) / len(accepted)})

real = pd.DataFrame(real_rows)
real_wide = real.pivot(index="key", columns="wave", values="mean_degree")
real_wide.columns = [f"real_w{w}" for w in real_wide.columns]

# -------------------------
# training: mean degree on each of the three vignettes (wave 1 only)
# -------------------------
train_rows = []
for key, rec in data_by_wave[wave_1].items():
    training = rec["training"]

    for scenario_key, scenario_name in SCENARIOS.items():
        pairs = set()
        labels = set()

        for field in ["train_pos", "train_neg"]:
            # final passing attempt for this scenario and phase
            passed = [a for a in training[field]
                      if a["scenario_key"] == scenario_key and a["status"] == "ok"]
            attempt = passed[-1]
            labels.update(p["label"] for p in attempt["positions"])
            for e in attempt["edges"]:
                pairs.add(tuple(sorted((e["stance_1"], e["stance_2"]))))

        train_rows.append({"key": key,
                           "scenario": scenario_name,
                           "n_nodes": len(labels),
                           "n_pairs": len(pairs),
                           "mean_degree": 2 * len(pairs) / len(labels)})

train = pd.DataFrame(train_rows)

print(f"\n{len(train)} training canvases "
      f"({train['key'].nunique()} participants x {train['scenario'].nunique()} scenarios)")
print(f"nodes per training canvas: {sorted(train['n_nodes'].unique())}")
print("\ntraining mean degree by scenario:")
print(train.groupby("scenario")["mean_degree"].agg(["mean", "std", "min", "max"])
      .round(2).to_string())

# -------------------------
# is drawing style stable across the three vignettes?
# -------------------------
train_wide = train.pivot(index="key", columns="scenario", values="mean_degree")

print("\n=== Consistency across the three training vignettes ===")
print(train_wide.corr().round(2).to_string())

k = train_wide.shape[1]
alpha = (k / (k - 1)) * (1 - train_wide.var(ddof=1).sum()
                         / train_wide.sum(axis=1).var(ddof=1))
print(f"\nCronbach's alpha across the {k} vignettes: {alpha:.2f}")

# -------------------------
# assemble and correlate
# -------------------------
df = real_wide.join(train_wide.mean(axis=1).rename("training")).dropna().reset_index()

print(f"\n=== Mean degree, N={len(df)} ===")
print(df[["training", "real_w1", "real_w2"]]
      .agg(["mean", "std", "min", "max"]).round(2).to_string())

COMPARISONS = [
    ("real_w1", "real_w2", f"Wave {wave_1} mean degree", f"Wave {wave_2} mean degree",
     "Test-retest"),
    ("training", "real_w1", "Training mean degree", f"Wave {wave_1} mean degree",
     f"Training vs wave {wave_1}"),
    ("training", "real_w2", "Training mean degree", f"Wave {wave_2} mean degree",
     f"Training vs wave {wave_2}"),
]

print()
for xcol, ycol, _, _, label in COMPARISONS:
    r, p = pearsonr(df[xcol], df[ycol])
    rho, p_rho = spearmanr(df[xcol], df[ycol])
    print(f"{label:22s} r={r:+.3f} (p={p:.2e})  rho={rho:+.3f}  r^2={r ** 2:.3f}")

print("\nPer-vignette correlation with the real network (wave 1):")
for s in train_wide.columns:
    sub = train_wide[[s]].join(real_wide["real_w1"]).dropna()
    r_s, p_s = pearsonr(sub[s], sub["real_w1"])
    print(f"  {s:8s} r={r_s:+.3f} (p={p_s:.2e})")

df.round(3).to_csv(os.path.join(outdir, "tool_approach_per_participant.csv"), index=False)
print(f"\nSaved {outdir}/tool_approach_per_participant.csv")

# -------------------------
# plot: the three correlations, on shared axes so they are comparable
# -------------------------
TITLE_SIZE = 15
LABEL_SIZE = 14
TICK_SIZE = 12

lo = min(df[["training", "real_w1", "real_w2"]].min()) - 0.2
hi = max(df[["training", "real_w1", "real_w2"]].max()) + 0.2

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

for ax, (xcol, ycol, xlabel, ylabel, label) in zip(axes, COMPARISONS):
    x, y = df[xcol], df[ycol]
    ax.scatter(x, y, s=26, alpha=0.45, color="lightsteelblue",
               edgecolors="black", linewidths=0.4)

    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.4)

    r, _ = pearsonr(x, y)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(f"{label}: r = {r:+.2f}", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig(os.path.join(outdir, "training_vs_real.svg"), bbox_inches="tight")
print(f"Saved {outdir}/training_vs_real.svg")
plt.show()
