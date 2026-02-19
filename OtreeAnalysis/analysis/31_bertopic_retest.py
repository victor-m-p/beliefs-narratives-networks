"""
VMP 2026-02-02

Test-retest reliability of BERTopic topic presence (binary),
restricted to TOP10 selected runs.

Updated now to only compute Phi coefficient (removed Jaccard etc.)
But now running with both outliers True and False (toggle manually.)

VMP 2026-02-08: tested and run.

NB: Probably we actually should have Jaccard again since $phi$ can 
become dominated by shared absences when topics are sparse.
"""

from __future__ import annotations

import shutil
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

# >>> TOGGLE THIS MANUALLY <<<
EXCLUDE_OUTLIERS = False

SUFFIX = f"excludeOutliers{EXCLUDE_OUTLIERS}"
FIG_ROOT = Path(f"../fig/BERTopic/retest/phi__{SUFFIX}")

WIPE_FIG_ROOT = True

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
)

# -----------------------------
# 1) Utilities (unchanged)
# -----------------------------
def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def safe_read_statement_topics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["key"] = df["key"].astype(str)
    df["wave"] = pd.to_numeric(df["wave"], errors="coerce").astype("Int64")
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["wave", "topic"]).copy()
    df["wave"] = df["wave"].astype(int)
    df["topic"] = df["topic"].astype(int)
    return df[df["wave"].isin([1, 2])].copy()


def count_keywave_all_outlier(df_raw: pd.DataFrame) -> int:
    g = df_raw.groupby(["key", "wave"])["topic"]
    return int(g.apply(lambda s: (s == -1).all()).sum())


def build_binary_presence(df_raw: pd.DataFrame):
    df = df_raw.copy()
    if EXCLUDE_OUTLIERS:
        df = df[df["topic"] != -1].copy()

    topics_all = sorted(df["topic"].unique().tolist())
    if not topics_all:
        return np.empty((0, 0)), np.empty((0, 0)), [], 0

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

    return (
        w1.loc[keys].to_numpy(np.int8),
        w2.loc[keys].to_numpy(np.int8),
        keys,
        len(topics_all),
    )


def phi_matrix(A: np.ndarray, B: np.ndarray, eps=1e-12) -> np.ndarray:
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


def summarize_phi(P: np.ndarray) -> dict:
    n = P.shape[0]
    diag = np.diag(P)
    off = P[~np.eye(n, dtype=bool)]

    diag = diag[np.isfinite(diag)]
    off = off[np.isfinite(off)]

    return dict(
        mean_self=np.mean(diag),
        mean_other=np.mean(off),
        delta_mean=np.mean(diag) - np.mean(off),
        auc=auc_self_vs_other(diag, off),
    )


def auc_self_vs_other(self_vals, other_vals):
    other_sorted = np.sort(other_vals)
    lt = np.searchsorted(other_sorted, self_vals, side="left")
    le = np.searchsorted(other_sorted, self_vals, side="right")
    return float((lt + 0.5 * (le - lt)).sum() / (len(self_vals) * len(other_vals)))


# -----------------------------
# 2) Main
# -----------------------------
if WIPE_FIG_ROOT:
    reset_dir(FIG_ROOT)
else:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

top10 = pd.read_csv(TOP10_PATH)

rows = []

for rank, r in enumerate(top10.itertuples(index=False), 1):
    label = f"{rank:02d}__{r.embed_model_outname}__run_{r.run_id}"
    st_path = STATEMENT_DIR / f"{label}__statement_topics.csv"

    if not st_path.exists():
        continue

    df_raw = safe_read_statement_topics(st_path)
    n_all_out = count_keywave_all_outlier(df_raw)

    A, B, keys, K = build_binary_presence(df_raw)
    if len(keys) < MIN_KEYS_REQUIRED or K == 0:
        continue

    P = phi_matrix(A, B)
    stats = summarize_phi(P)

    out_run = FIG_ROOT / f"run_{label}"
    out_run.mkdir(parents=True, exist_ok=True)

    # plot
    rng = np.random.default_rng(42)
    self_p = np.diag(P)
    other_p = P[~np.eye(P.shape[0], dtype=bool)]

    # subsample self–other for plotting only
    other_p = other_p[np.isfinite(other_p)]
    if len(other_p) > PLOT_MAX_OTHER:
        other_p = rng.choice(other_p, size=PLOT_MAX_OTHER, replace=False)

    self_p = self_p[np.isfinite(self_p)]

    label_self = r"$\phi_{\mathrm{self}}$"
    label_other = r"$\phi_{\mathrm{other}}$"
    
    df_plot = pd.DataFrame({
        "group": [label_self] * len(self_p) + [label_other] * len(other_p),
        "value": np.concatenate([self_p, other_p]),
    })

    mean_se_plot_side(
        df_plot,
        xcol="group",
        ycol="value",
        xlab="",
        ylab=r"Phi coefficient ($\phi$)",
        title="",
        order=[label_self, label_other],
        outname=str(out_run / "topic_presence_phi_boxdots.png"),
        **PLOT_STYLE,
    )
    
    rows.append(dict(
        label=label,
        n_topics=K,
        n_common_keys=len(keys),
        n_keywave_all_outlier=n_all_out,
        **stats,
    ))

df_phi = pd.DataFrame(rows)
df_phi.to_csv(FIG_ROOT / "overview_phi.csv", index=False)

print("Saved:", FIG_ROOT)
