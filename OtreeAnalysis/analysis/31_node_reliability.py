"""
31_bertopic_retest.py

VMP 2026-02-02
Node-topic test-retest reliability across waves.

For each participant: binary vector of topic presence (wave 1 vs wave 2).
Phi coefficient: within-person (same participant, W1 vs W2) vs between-person.
Outliers (topic == -1) included as a topic.

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv

Writes:
  ../fig/node_reliability/node_phi_boxdots.svg  (selected model 09__) → Figure 4 (V2)
  ../fig/node_reliability/node_phi_table.tex    (summary across all 10 models) → Table S6 (V2)

VMP 2026-02-08: tested and run.
VMP 2026-03-27: simplified - always include outliers, one plot + table output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers import mean_se_plot_side

# -----------------------------
# 0) Config
# -----------------------------
SELECTION_ROOT = Path("../data/public/bertopic/selection")
TOP10_PATH = SELECTION_ROOT / "overview_top10.csv"
STATEMENT_DIR = SELECTION_ROOT / "statement_topics"

OUTDIR = Path("../fig/node_reliability")
OUTDIR.mkdir(parents=True, exist_ok=True)


PLOT_MAX_OTHER = 500
MIN_KEYS_REQUIRED = 5

PLOT_STYLE = dict(
    ci_mult=1.0,
    box_offset=-0.13,
    point_offset=+0.13,
    box_width=0.25,
    jitter=0.06,
    point_size=18,
    point_alpha=0.25,
    show_fliers=False,
    rotate_xticks=0,
    connect_ids=False,
    figsize=(6.2, 4.0),
    fontsize=14,
)

# -----------------------------
# 1) Helpers
# -----------------------------
def safe_read_statement_topics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["key"] = df["key"].astype(str)
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype("Int64")
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["wave", "topic"]).copy()
    df["wave"] = df["wave"].astype(int)
    df["topic"] = df["topic"].astype(int)
    return df[df["wave"].isin([1, 2])].copy()


def build_binary_presence(df_raw: pd.DataFrame):
    """Binary topic-presence matrices for wave 1 and wave 2 (outliers included as topic -1)."""
    df = df_raw.copy()

    topics_all = sorted(df["topic"].unique().tolist())
    if not topics_all:
        return np.empty((0, 0)), np.empty((0, 0)), []

    counts = (
        df.groupby(["key", "wave", "topic"]).size()
          .unstack(fill_value=0)
          .reindex(columns=topics_all, fill_value=0)
    )
    binary = (counts > 0).astype(np.int8)

    w1 = binary.xs(1, level="wave")
    w2 = binary.xs(2, level="wave")

    keys = w1.index.intersection(w2.index)
    w1, w2 = w1.loc[keys], w2.loc[keys]

    keep = (w1.sum(axis=1) > 0) & (w2.sum(axis=1) > 0)
    keys = list(w1.index[keep])

    return w1.loc[keys].to_numpy(np.int8), w2.loc[keys].to_numpy(np.int8), keys


def phi_matrix(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """N×N phi matrix: entry [i,j] = phi(A[i,:], B[j,:])."""
    a = A @ B.T
    sumA = A.sum(axis=1)[:, None]
    sumB = B.sum(axis=1)[None, :]
    b = sumA - a
    c = sumB - a
    d = A.shape[1] - (a + b + c)
    num = a * d - b * c
    den = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    out = np.full_like(num, np.nan, dtype=float)
    ok = den > eps
    out[ok] = num[ok] / den[ok]
    return out


# -----------------------------
# 2) Plot: selected model (09__)
# -----------------------------
top10 = pd.read_csv(TOP10_PATH)
r = top10.iloc[8]
label = f"09__{r.embed_model_outname}__run_{r.run_id}"
st_path = STATEMENT_DIR / f"{label}__statement_topics.csv"

df_raw = safe_read_statement_topics(st_path)
A, B, keys = build_binary_presence(df_raw)

rng = np.random.default_rng(42)
P = phi_matrix(A, B)
self_phi  = np.diag(P)
other_phi = P[~np.eye(len(keys), dtype=bool)]
self_phi  = self_phi[np.isfinite(self_phi)]
other_phi = other_phi[np.isfinite(other_phi)]
if len(other_phi) > PLOT_MAX_OTHER:
    other_phi = rng.choice(other_phi, size=PLOT_MAX_OTHER, replace=False)

label_self  = r"$\phi_{\mathrm{within}}$"
label_other = r"$\phi_{\mathrm{between}}$"

df_plot = pd.DataFrame({
    "group": [label_self] * len(self_phi) + [label_other] * len(other_phi),
    "value": np.concatenate([self_phi, other_phi]),
})

mean_se_plot_side(
    df_plot,
    xcol="group",
    ycol="value",
    xlab="",
    ylab=r"Phi coefficient ($\phi$)",
    title="",
    order=[label_self, label_other],
    outname=str(OUTDIR / "node_phi_boxdots.svg"),
    **PLOT_STYLE,
)
print(f"Saved: {OUTDIR / 'node_phi_boxdots.svg'}")

# -----------------------------
# 3) Table: all 10 models
# -----------------------------
summary_rows = []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    run_label = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    run_st_path = STATEMENT_DIR / f"{run_label}__statement_topics.csv"

    if not run_st_path.exists():
        print(f"Skipping {run_label} (missing file)")
        continue

    df_r = safe_read_statement_topics(run_st_path)
    A_r, B_r, keys_r = build_binary_presence(df_r)
    if len(keys_r) < MIN_KEYS_REQUIRED:
        continue

    P_r = phi_matrix(A_r, B_r)
    self_r  = np.diag(P_r)[np.isfinite(np.diag(P_r))]
    other_r = P_r[~np.eye(len(keys_r), dtype=bool)]
    other_r = other_r[np.isfinite(other_r)]

    phi_within  = float(np.mean(self_r))
    phi_between = float(np.mean(other_r))
    summary_rows.append(dict(
        label=run_label,
        n_topics=A_r.shape[1],
        n_participants=len(keys_r),
        phi_within=round(phi_within, 3),
        phi_between=round(phi_between, 3),
        phi_delta=round(phi_within - phi_between, 3),
    ))
    print(f"{run_label}: phi_within={phi_within:.3f}  phi_between={phi_between:.3f}")

df_summary = pd.DataFrame(summary_rows)

table = df_summary[["label", "n_topics", "phi_within", "phi_between", "phi_delta"]].copy()
table["Model"] = table["label"].str.replace(r"__run.*$", "", regex=True)
table["Model"] = table["Model"].apply(lambda s: r"\texttt{" + s.replace("_", r"\_") + "}")
table = table.rename(columns={
    "n_topics": r"$n_{\mathrm{topics}}$",
    "phi_within": r"$\bar{\phi}_{\mathrm{within}}$",
    "phi_between": r"$\bar{\phi}_{\mathrm{between}}$",
    "phi_delta": r"$\Delta\bar{\phi}$",
})[["Model", r"$n_{\mathrm{topics}}$", r"$\bar{\phi}_{\mathrm{within}}$",
    r"$\bar{\phi}_{\mathrm{between}}$", r"$\Delta\bar{\phi}$"]]

latex = table.to_latex(
    index=False,
    escape=False,
    column_format="lrrrr",
    float_format=lambda x: f"{x:.2f}",
)
outname = OUTDIR / "node_phi_table.tex"
with open(outname, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"Saved table: {outname}")
