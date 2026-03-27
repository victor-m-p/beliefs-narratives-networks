"""
VMP 2026-02-17

Edge reliability across waves for BERTopic topic networks.

Question: if edge (t1, t2) exists in W1 for a participant,
how likely is it to also exist in W2?

For each W1 topic pair we track two things:
  - both topics present in W2?
  - edge present in W2?

Reported separately for connected pairs (edge in W1)
vs unconnected pairs (no edge in W1, baseline).

Output at participant level and aggregated across runs.

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/public/bertopic_mapping/edge_mapping__<label>.csv

Writes:
  ../fig/BERTopic/edge_reliability/
    <label>__participant_level.csv
    edge_reliability_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from helpers import normalize_ab, mean_se_plot_side

# ---- config ----
SEL_TOPICS = Path("../data/public/bertopic/selection")
SEL_MAP = Path("../data/public/bertopic_mapping")
TOP10_PATH = SEL_TOPICS / "overview_top10.csv"
STMT_DIR = SEL_TOPICS / "statement_topics"

OUTDIR = Path("../fig/BERTopic/edge_reliability")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ---- helpers ----
def load_edges(edge_csv: Path) -> pd.DataFrame:
    """Per-participant topic-level edges: key, wave, topic_1, topic_2 (deduplicated)."""
    df = pd.read_csv(edge_csv)[["key", "wave", "topic_1", "topic_2"]]
    df = normalize_ab(df, "topic_1", "topic_2")
    return df[["key", "wave", "topic_1", "topic_2"]].drop_duplicates()


top10 = pd.read_csv(TOP10_PATH)
r = top10.iloc[8]
label = f"09__{r.embed_model_outname}__run_{r.run_id}"

edge_csv = SEL_MAP / f"edge_mapping__{label}.csv"
stmt_csv = STMT_DIR / f"{label}__statement_topics.csv"

presence = pd.read_csv(stmt_csv)[["key", "wave", "topic"]].drop_duplicates()
edges = load_edges(edge_csv)

PLOT_MAX_OTHER = 500
EDGE_PHI_OUTDIR = OUTDIR / "edge_phi"
EDGE_PHI_OUTDIR.mkdir(parents=True, exist_ok=True)

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


def build_edge_matrices(presence: pd.DataFrame, edges: pd.DataFrame):
    """Build binary edge matrices (N_participants × N_pairs) for both waves."""
    all_topics = sorted(presence["topic"].unique())
    all_pairs = [(t1, t2) for i, t1 in enumerate(all_topics) for t2 in all_topics[i:]]
    pair_to_idx = {p: i for i, p in enumerate(all_pairs)}
    n_pairs = len(all_pairs)

    common_keys = sorted(
        set(presence[presence["wave"] == 1]["key"]) &
        set(presence[presence["wave"] == 2]["key"])
    )
    n_keys = len(common_keys)
    key_to_idx = {k: i for i, k in enumerate(common_keys)}

    A = np.zeros((n_keys, n_pairs), dtype=np.int8)
    B = np.zeros((n_keys, n_pairs), dtype=np.int8)

    for _, row in edges[edges["wave"] == 1].iterrows():
        ki = key_to_idx.get(row["key"])
        pi = pair_to_idx.get((row["topic_1"], row["topic_2"]))
        if ki is not None and pi is not None:
            A[ki, pi] = 1

    for _, row in edges[edges["wave"] == 2].iterrows():
        ki = key_to_idx.get(row["key"])
        pi = pair_to_idx.get((row["topic_1"], row["topic_2"]))
        if ki is not None and pi is not None:
            B[ki, pi] = 1

    return A, B, common_keys, all_topics, all_pairs


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


# Build binary matrices for selected run using shared helper
A_edge, B_edge, common_keys, all_topics, all_pairs = build_edge_matrices(presence, edges)
n_keys = len(common_keys) # n=210
n_pairs = len(all_pairs) # n=136
print(f"\nEdge-phi: {n_pairs} possible topic-pair columns ({len(all_topics)} topics)")
print(f"Edge-phi: {n_keys} participants with data in both waves")

# Phi matrix and extract within/between
P_edge = phi_matrix(A_edge, B_edge)

rng = np.random.default_rng(42)
self_phi  = np.diag(P_edge) # shape 210
other_phi = P_edge[~np.eye(n_keys, dtype=bool)] # shape 43890
self_phi  = self_phi[np.isfinite(self_phi)]
other_phi = other_phi[np.isfinite(other_phi)]
if len(other_phi) > PLOT_MAX_OTHER:
    other_phi = rng.choice(other_phi, size=PLOT_MAX_OTHER, replace=False)

# ---- Plot: within- vs between-person phi ----
label_self  = r"$\phi_{\mathrm{within}}$"
label_other = r"$\phi_{\mathrm{between}}$"

df_edge_phi = pd.DataFrame({
    "group": [label_self] * len(self_phi) + [label_other] * len(other_phi),
    "value": np.concatenate([self_phi, other_phi]),
})

mean_se_plot_side(
    df_edge_phi,
    xcol="group",
    ycol="value",
    xlab="",
    ylab=r"Phi coefficient ($\phi$)",
    title="",
    order=[label_self, label_other],
    outname=str(EDGE_PHI_OUTDIR / "edge_phi_boxdots.svg"),
    **PLOT_STYLE,
)
print(f"Saved: {EDGE_PHI_OUTDIR / 'edge_phi_boxdots.svg'}")

# ---- Within-person 2x2 contingency table ----
# Rows = W1 edge status, Columns = W2 edge status.
# Each cell computed per participant across all n_pairs columns, then averaged.
# Universe: all global topic pairs, including pairs the participant never had.
a_pp = (A_edge * B_edge).sum(axis=1).astype(float)              # W1 present, W2 present
b_pp = (A_edge * (1 - B_edge)).sum(axis=1).astype(float)        # W1 present, W2 absent
c_pp = ((1 - A_edge) * B_edge).sum(axis=1).astype(float)        # W1 absent,  W2 present
d_pp = ((1 - A_edge) * (1 - B_edge)).sum(axis=1).astype(float)  # W1 absent,  W2 absent

# ---- Summary table: edge phi across all 10 candidate runs ----
summary_rows = []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    run_label = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    run_edge_csv = SEL_MAP / f"edge_mapping__{run_label}.csv"
    run_stmt_csv = STMT_DIR / f"{run_label}__statement_topics.csv"

    if not run_edge_csv.exists() or not run_stmt_csv.exists():
        print(f"Skipping {run_label} (missing files)")
        continue

    run_presence = pd.read_csv(run_stmt_csv)[["key", "wave", "topic"]].drop_duplicates()
    run_edges = load_edges(run_edge_csv)

    A_r, B_r, keys_r, topics_r, pairs_r = build_edge_matrices(run_presence, run_edges)
    if len(keys_r) == 0:
        continue

    P_r = phi_matrix(A_r, B_r)
    self_r = np.diag(P_r)
    other_r = P_r[~np.eye(len(keys_r), dtype=bool)]
    self_r = self_r[np.isfinite(self_r)]
    other_r = other_r[np.isfinite(other_r)]

    phi_within = float(np.mean(self_r))
    phi_between = float(np.mean(other_r))
    summary_rows.append(dict(
        label=run_label,
        n_topics=len(topics_r),
        n_pairs=len(pairs_r),
        n_participants=len(keys_r),
        phi_within=round(phi_within, 3),
        phi_between=round(phi_between, 3),
        phi_delta=round(phi_within - phi_between, 3),
    ))
    print(f"{run_label}: phi_within={phi_within:.3f}  phi_between={phi_between:.3f}")

df_summary = pd.DataFrame(summary_rows)

table = df_summary[["label", "n_pairs", "phi_within", "phi_between", "phi_delta"]].copy()
table["Model"] = table["label"].str.replace(r"__run.*$", "", regex=True)
table["Model"] = table["Model"].apply(lambda s: r"\texttt{" + s.replace("_", r"\_") + "}")
table = table.rename(columns={
    "n_pairs": r"$n_{\mathrm{pairs}}$",
    "phi_within": r"$\bar{\phi}_{\mathrm{within}}$",
    "phi_between": r"$\bar{\phi}_{\mathrm{between}}$",
    "phi_delta": r"$\Delta\bar{\phi}$",
})[["Model", r"$n_{\mathrm{pairs}}$", r"$\bar{\phi}_{\mathrm{within}}$",
    r"$\bar{\phi}_{\mathrm{between}}$", r"$\Delta\bar{\phi}$"]]

latex = table.to_latex(
    index=False,
    escape=False,
    column_format="lrrrr",
    float_format=lambda x: f"{x:.2f}",
)
outname = EDGE_PHI_OUTDIR / "edge_phi_table.tex"
with open(outname, "w", encoding="utf-8") as f:
    f.write(latex)
print(f"\nSaved table: {outname}")


# ---- Save 2x2 contingency table (raw counts + row %) ----
TABLES_DIR = Path("../fig/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

a_m = a_pp.mean(); b_m = b_pp.mean()
c_m = c_pp.mean(); d_m = d_pp.mean()
r1 = a_m + b_m
r2 = c_m + d_m

contingency_latex = (
    r"\begin{tabular}{lccc}" + "\n"
    r"\toprule" + "\n"
    r" & W2 Present & W2 Absent & Row total \\" + "\n"
    r"\midrule" + "\n"
    rf"W1 Present & {a_m:.1f} ({a_m/r1*100:.1f}\%) & {b_m:.1f} ({b_m/r1*100:.1f}\%) & {r1:.1f} (100\%) \\" + "\n"
    rf"W1 Absent  & {c_m:.1f} ({c_m/r2*100:.1f}\%) & {d_m:.1f} ({d_m/r2*100:.1f}\%) & {r2:.1f} (100\%) \\" + "\n"
    r"\bottomrule" + "\n"
    r"\end{tabular}"
)

cont_outname = TABLES_DIR / "edge_contingency_table.tex"
with open(cont_outname, "w", encoding="utf-8") as f:
    f.write(contingency_latex)
print(f"Saved: {cont_outname}")