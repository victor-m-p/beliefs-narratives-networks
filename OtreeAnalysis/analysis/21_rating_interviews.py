'''
VMP 2026-02-06 (refactored):

Analyzes interview ratings across both waves.
Uses sanitized public data (interview_rating field is safe).

VMP 2026-02-07: tested and run.
'''

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import wave_1, wave_2, get_public_path

# outpath
outpath_fig = "../fig/ratings"
os.makedirs(outpath_fig, exist_ok=True)

# load data from public (sanitized)
curation_w1_path = get_public_path("curation_w{wave}.json", wave=wave_1)
curation_w2_path = get_public_path("curation_w{wave}.json", wave=wave_2)

with open(curation_w1_path, "r", encoding="utf-8") as f:
    data_w1 = json.load(f)

with open(curation_w2_path, "r", encoding="utf-8") as f:
    data_w2 = json.load(f)

# extract ratings 
def extract_ratings(data):
    rows_num = []

    for key, v in data.items():
        ratings = v.get("interview_rating", None)
        for field, val in ratings.items():
            if field.endswith("_num"):
                feature = field[:-4]  
                rows_num.append({"key": key, "feature": feature, "value": val})

    return pd.DataFrame(rows_num)

# long-format dataframes
df_w1 = extract_ratings(data_w1)  # n=247
df_w2 = extract_ratings(data_w2)  # n=217

# map short feature names -> long labels
feature_label_map = {
    "overall":      "Overall experience",
    "relevant":     "Questions were relevant",
    "easy_chat":    "Easy to express thoughts",
    "comfort":      "Comfortable being honest",
    "creepy":       "Interacting was creepy",
}

df_w1['feature_label'] = df_w1['feature'].map(feature_label_map)
df_w2['feature_label'] = df_w2['feature'].map(feature_label_map)

# plot
feature_order = [
    "Overall experience",
    "Questions were relevant",
    "Easy to express thoughts",
    "Comfortable being honest",
    "Interacting was creepy",
]

palette_box = "#a6c4e0"     # light blue
palette_points = "#1f77b4"  # strong blue

fig, axes = plt.subplots(1, 2, figsize=(7, 4), dpi=300, sharey=True)

for ax, df_plot, title in zip(
    axes,
    [df_w1, df_w2],
    [f"Wave {wave_1}", f"Wave {wave_2}"]
):
    sns.boxplot(
        data=df_plot,
        x="feature_label",
        y="value",
        order=feature_order,
        ax=ax,
        width=0.6,
        color=palette_box,
        showfliers=False,
        boxprops=dict(edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        medianprops=dict(color="black", linewidth=2),
    )

    sns.stripplot(
        data=df_plot,
        x="feature_label",
        y="value",
        order=feature_order,
        ax=ax,
        color=palette_points,
        size=4,
        jitter=0.1,
        alpha=0.2,
        linewidth=0,
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30, labelrotation=30)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

axes[0].set_ylabel("Rating")
axes[1].set_ylabel("")

fig.suptitle("Ratings of the LLM-guided interview")
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(outpath_fig, "interview.pdf"), bbox_inches='tight')
plt.close(fig)

''' Summary of feedback (qualitative)
1. Was fun.
2. Was not pushy (compared to other studies).
3. Smooth, especially transcripts.
4. Went well.
5. Went fine, a few typos in transcriptions.
6. Smooth and fast replies. 
7. Don't mind doing again.
8. Relevant questions but a bit repetitive (interview could have been shorter).
9. Recording feature worked well
'''