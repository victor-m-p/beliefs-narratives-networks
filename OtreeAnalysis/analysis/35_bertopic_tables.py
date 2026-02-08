'''
VMP 2026-02-03:
Generating tables and checking numbers that we need for the preprint.

VMP 2026-02-08: tested and run.
'''

import numpy as np 
import pandas as pd 
import os 

outpath = "../fig/tables"
os.makedirs(outpath, exist_ok=True)

# test-retest
with_outliers = pd.read_csv("../fig/BERTopic/retest/phi__excludeOutliersFalse/overview_phi.csv")
without_outliers = pd.read_csv("../fig/BERTopic/retest/phi__excludeOutliersTrue/overview_phi.csv")

## phi coefficient
### for the main model (with outliers)
with_outliers['mean_self'].iloc[0] #.36
with_outliers['mean_other'].iloc[0] # .10

### for the main model (without outliers)
without_outliers['mean_self'].iloc[0] # .31 
without_outliers['mean_other'].iloc[0] # .03

### across all 10 models (phi)
with_outliers['mean_self'].mean() # .38
with_outliers['mean_self'].min() # .34
with_outliers['mean_self'].max() # .43

# AUC
with_outliers['auc'].iloc[0] # .80
with_outliers['auc'].mean() # .78
with_outliers['auc'].max() # .82
with_outliers['auc'].min() # .72

'''
Create table including outliers with phi and AUC.
'''

table = with_outliers[["label", "n_topics", "mean_self", "mean_other", "auc"]].copy()
table['m'] = table['mean_self'] - table['mean_other']
table.sort_values('m')

# derive columns
table["Model"] = table["label"].str.replace(r"__run.*$", "", regex=True)
table["Model"] = table["Model"].apply(lambda s: r"\texttt{" + s.replace("_", r"\_") + "}")
table[r"$\Delta\bar{\phi}$"] = table["mean_self"] - table["mean_other"]

# select + final column names (single pass)
table = table.rename(columns={
    "n_topics": "K",
    "mean_self": r"$\bar{\phi}_{\mathrm{self}}$",
    "mean_other": r"$\bar{\phi}_{\mathrm{other}}$",
    "auc": "AUC",
})[[
    "Model",
    "K",
    r"$\bar{\phi}_{\mathrm{self}}$",
    r"$\bar{\phi}_{\mathrm{other}}$",
    r"$\Delta\bar{\phi}$",
    "AUC",
]]

latex = table.to_latex(
    index=False,
    escape=False,
    column_format="lrrrrr",
    float_format=lambda x: f"{x:.2f}",
)

# save
outname = os.path.join(outpath, "phi_auc_table.tex")
with open(outname, "w", encoding="utf-8") as f:
    f.write(latex)

### table of the overview for the selected model ###
bertopic_path = "../data/public/bertopic/selection/overview/"
from pathlib import Path
_matches = sorted(Path(bertopic_path).glob("01__*__topic_overview.csv"))
if not _matches:
    raise FileNotFoundError("No 01__* topic_overview file found in " + bertopic_path)
bertopic_df = pd.read_csv(_matches[0])

# helpers for writing a nice latex table
def _esc(s):
    if pd.isna(s):
        return ""
    s = str(s)
    for a, b in {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        s = s.replace(a, b)
    return s

rep_cols = [c for c in ["RepDoc1", "RepDoc2", "RepDoc3"] if c in bertopic_df.columns]
d = bertopic_df[["Topic", "Count", "Keywords"] + rep_cols].copy()

def _join_reps(row):
    reps = [str(row[c]).strip() for c in rep_cols if pd.notna(row[c]) and str(row[c]).strip()]
    return ", ".join(reps)

d["Example"] = d.apply(_join_reps, axis=1)

# sort (keeps -1)
d["Topic_num"] = pd.to_numeric(d["Topic"], errors="coerce")
d = d.sort_values(["Topic_num", "Topic"], kind="mergesort").drop(columns="Topic_num")

# escape
d["Keywords"] = d["Keywords"].map(_esc)
d["Example"] = d["Example"].map(_esc)

lines = [
    r"\setlength{\LTleft}{0pt}",
    r"\setlength{\LTright}{0pt}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\renewcommand{\arraystretch}{1.05}",
    r"\begingroup\small",
    r"\begin{longtable}{r r >{\raggedright\arraybackslash}p{0.44\textwidth} >{\raggedright\arraybackslash}p{0.44\textwidth}}",
    r"\toprule",
    r"Topic & n & Keywords & Representative Docs \\",
    r"\midrule",
    r"\endfirsthead",
    r"\toprule",
    r"Topic & n & Keywords & Representative Docs \\",
    r"\midrule",
    r"\endhead",
    r"\midrule",
    r"\multicolumn{4}{r}{Continued on next page} \\",
    r"\midrule",
    r"\endfoot",
    r"\bottomrule",
    r"\endlastfoot",
]
lines += [f"{r.Topic} & {r.Count} & {r.Keywords} & {r.Example} \\\\" for r in d.itertuples(index=False)]
lines += [r"\end{longtable}", r"\endgroup"]

outname = os.path.join(outpath, "BERTopic_overview.txt")
with open(outname, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("Wrote:", outname)