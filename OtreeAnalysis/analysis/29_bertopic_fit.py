"""
bertopic_grid_base.py

VMP 2026-01-26
Phase A: Base BERTopic grid search across embedding models.
- For each embedding model:
  - Embed statements (optionally with prefix), L2-normalize on encode
  - For each (UMAP, HDBSCAN) hyperparam combo:
      * Fit BERTopic
      * Save a "manual evaluation pack":
          - params.json
          - topic_info.csv
          - topic_overview.csv  (top words + representative sentences; UNIQUE texts)
          - statement_topics.csv (statement-level topics + topic_conf + assigned_prob)
      * Save a run summary row:
          n_topics, outlier_rate, outlier_0_ratio, dbcv, distortion, mean_assigned_prob

Outputs:
../data/data-<date>/topics_bertopic/<model_outname>/runs/run_<run_id>/*
../data/data-<date>/topics_bertopic/<model_outname>/base_grid_summary.csv
../data/data-<date>/topics_bertopic/overview_all_models__base_grid_summary.csv  (sorted by DBCV desc)

Notes:
- This script does NOT do topic reduction (nr_topics K). That's the next step.
- mean_assigned_prob: mean (over non-outlier docs) of P(assigned topic | doc), if mapping is unambiguous.
- DBCV uses HDBSCAN's built-in relative_validity_ (can be NaN in degenerate cases).
- outlier_rate: fraction assigned to topic -1 (noise).
- outlier_0_ratio: (topic -1 + topic 0) fraction (paper-compatible, but topic 0 is not "outlier" by default).

VMP 2026-02-08: tested and run.
"""

import os
import json
import time
import hashlib
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv() # HF key, not required

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

import torch
from utilities import EMB_MODELS
from helpers import embed_df

import shutil
from pathlib import Path

# -----------------------------
# 0) Config
# -----------------------------

emb_dir = "../data/public/embeddings"
nodes_path = os.path.join(emb_dir, "nodes.csv")

def reset_dir(path: str, must_contain: str = "bertopic") -> None:
    """Delete + recreate directory. Safety: refuses unless `must_contain` is in resolved path."""
    p = Path(path).resolve()
    if must_contain not in str(p):
        raise ValueError(f"Refusing to delete suspicious path: {p}")
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

out_root = "../data/public/bertopic"

WIPE_OUTPUTS = True
if WIPE_OUTPUTS:
    reset_dir(out_root, must_contain="bertopic")
else:
    os.makedirs(out_root, exist_ok=True)

device = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

RANDOM_STATE = 42
BATCH_SIZE = 64

VECTORIZER = dict(
    stop_words="english",
    ngram_range=(1, 3),
    min_df=1
)

UMAP_GRID = [
    dict(n_neighbors=15, n_components=10, min_dist=0.1),
    dict(n_neighbors=30, n_components=10, min_dist=0.1),
    dict(n_neighbors=45, n_components=10, min_dist=0.1),
    dict(n_neighbors=15, n_components=2,  min_dist=0.1),
    dict(n_neighbors=30, n_components=2,  min_dist=0.1),
    dict(n_neighbors=45, n_components=2,  min_dist=0.1),
]

HDB_GRID = [
    dict(min_cluster_size=25, min_samples=5),
    dict(min_cluster_size=50, min_samples=5),
    dict(min_cluster_size=75, min_samples=5),
]

N_REP_DOCS = 5
N_TOPIC_WORDS = 12


# -----------------------------
# 1) Helpers
# -----------------------------

def safe_filename(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")

def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def run_hash(d: dict) -> str:
    b = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.md5(b).hexdigest()[:8]

def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def mean_cosine_dist_to_centroid(emb_unit: np.ndarray, topics: np.ndarray, eps: float = 1e-12) -> float:
    """Mean cosine distance to assigned-topic centroid in embedding space (excludes -1)."""
    topics = np.asarray(topics)
    mask = topics != -1
    if mask.sum() == 0:
        return np.nan

    X = emb_unit[mask]
    t = topics[mask]

    centroids = {}
    for k in np.unique(t):
        ck = X[t == k].mean(axis=0)
        ck = ck / (np.linalg.norm(ck) + eps)
        centroids[k] = ck

    d = np.empty(len(X), dtype=float)
    for i, (vec, k) in enumerate(zip(X, t)):
        d[i] = 1 - float(np.dot(vec, centroids[k]))
    return float(d.mean())

def compute_dbcv(topic_model) -> float:
    return float(getattr(topic_model.hdbscan_model, "relative_validity_", np.nan))

def summarize_run(topic_model: BERTopic, emb_unit: np.ndarray, assigned_prob: np.ndarray) -> dict:
    topics_arr = np.asarray(topic_model.topics_)
    uniq = set(topics_arr.tolist())
    n_topics = len(uniq) - (1 if -1 in uniq else 0)

    n_total = len(topics_arr)
    n_outliers = int((topics_arr == -1).sum())
    n_topic0 = int((topics_arr == 0).sum())

    outlier_rate = n_outliers / n_total              
    topic0_ratio = n_topic0 / n_total                
    outlier_0_ratio = (n_outliers + n_topic0) / n_total  

    distortion = mean_cosine_dist_to_centroid(emb_unit, topics_arr)
    m = np.nanmean(assigned_prob)
    mean_assigned_prob = float(m) if np.isfinite(m) else np.nan
    dbcv = compute_dbcv(topic_model)                 

    return dict(
        n_total=n_total,                            
        n_topics=n_topics,
        dbcv=dbcv,

        n_outliers=n_outliers,
        n_topic0=n_topic0,
        outlier_rate=outlier_rate,
        topic0_ratio=topic0_ratio,                   
        outlier_0_ratio=outlier_0_ratio,

        distortion=distortion,
        mean_assigned_prob=mean_assigned_prob,
    )

def _clean_for_dedupe(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r"\s+", " ", s)

def representative_docs_table(topic_model, topic_id, docs, probs, n):
    """Up to n representative docs for a topic, UNIQUE by text. Tops up for -1."""
    docs = list(docs)
    topics_arr = np.asarray(topic_model.topics_)
    out, seen = [], set()

    def add_txt(txt: str):
        key = _clean_for_dedupe(txt)
        if key in seen:
            return
        seen.add(key)
        out.append(txt)

    if probs is not None and topic_id != -1:
        idx = np.where(topics_arr == topic_id)[0]
        if len(idx) > 0:
            if probs.ndim == 2 and topic_id < probs.shape[1]:
                scores = probs[idx, topic_id]
            elif probs.ndim == 2:
                scores = probs[idx].max(axis=1)
            else:
                scores = probs[idx]
            order = idx[np.argsort(scores)[::-1]]
            for i in order:
                add_txt(docs[i])
                if len(out) >= n:
                    return out[:n]

    try:
        reps = topic_model.get_representative_docs(topic_id) or []
        for txt in reps:
            add_txt(txt)
            if len(out) >= n:
                return out[:n]
    except Exception:
        pass

    idx = np.where(topics_arr == topic_id)[0]
    for i in idx:
        add_txt(docs[i])
        if len(out) >= n:
            break

    return out[:n]

def build_topic_overview(topic_model: BERTopic, docs: list[str], probs,
                         n_words: int = 12, n_rep_docs: int = 5) -> pd.DataFrame:
    info = topic_model.get_topic_info().copy()
    rows = []
    for _, r in info.iterrows():
        tid = int(r["Topic"])
        cnt = int(r["Count"])
        if tid == -1:
            keywords = "OUTLIER"
        else:
            words = [w for w, _ in (topic_model.get_topic(tid) or [])[:n_words]]
            keywords = ", ".join(words)

        reps = representative_docs_table(topic_model, tid, docs, probs, n_rep_docs)
        row = {"Topic": tid, "Count": cnt, "Keywords": keywords}
        for i in range(n_rep_docs):
            row[f"RepDoc{i+1}"] = reps[i] if i < len(reps) else ""
        rows.append(row)
    return pd.DataFrame(rows)

def compute_assigned_prob(topics: np.ndarray, probs) -> np.ndarray:
    """Per-doc prob of assigned topic (NaN for -1 or if mapping ambiguous)."""
    topics = np.asarray(topics)
    assigned = np.full(len(topics), np.nan, dtype=float)

    if probs is None or getattr(probs, "ndim", 0) != 2:
        return assigned

    non_out = sorted([t for t in set(topics.tolist()) if t != -1])
    K = probs.shape[1]

    if non_out == list(range(K)):
        topic_to_col = {t: t for t in non_out}
    elif len(non_out) == K:
        topic_to_col = {t: j for j, t in enumerate(non_out)}
    else:
        return assigned

    for t, j in topic_to_col.items():
        mask = topics == t
        assigned[mask] = probs[mask, j]

    return assigned


# -----------------------------
# 2) Load data
# -----------------------------

nodes = pd.read_csv(nodes_path).copy()
nodes["stance"] = nodes["stance"].astype(str)
nodes = nodes[nodes["stance"].str.strip().ne("")].reset_index(drop=True)

docs = nodes["stance"].tolist()
id_cols = [c for c in ["key", "wave", "stance", "canvas"] if c in nodes.columns]

print("statements:", len(nodes))
print("unique participants:", nodes["key"].nunique())
print("device:", device)

# -----------------------------
# 3) Loop models + grid
# -----------------------------

all_summary_rows = []

for mspec in EMB_MODELS:
    model_outname = safe_filename(mspec["name"])
    model_name = mspec["hf"]
    prefix = mspec.get("prefix", None)

    model_dir = os.path.join(out_root, model_outname)
    runs_dir = os.path.join(model_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Embedding model:", model_outname)
    print("HF:", model_name)
    print("prefix:", repr(prefix))
    print("=" * 70)

    # patch for the QWEN model
    use_device = device
    if model_name.startswith("Qwen/") and device == "mps":
        use_device = "cpu"

    st_model = SentenceTransformer(model_name, device=use_device)

    df_emb, emb = embed_df(
        df=nodes,
        model=st_model,
        text_col="stance",
        batch_size=BATCH_SIZE,
        encode_normalize=True,
        prefix=prefix
    )

    emb_unit = l2_normalize_rows(emb)

    summary_rows = []

    for ucfg in UMAP_GRID:
        for hcfg in HDB_GRID:
            cfg = {
                "embed_model_outname": model_outname,
                "embed_hf": model_name,
                "prefix": prefix,
                "encode_normalize": True,
                "vectorizer": VECTORIZER,
                "umap": ucfg,
                "hdbscan": hcfg,
                "random_state": RANDOM_STATE,
            }

            rid = f"{now_id()}_{run_hash(cfg)}"
            run_dir = os.path.join(runs_dir, f"run_{rid}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n[RUN] {rid}")
            print(" UMAP:", ucfg)
            print(" HDB :", hcfg)

            umap_model = UMAP(
                n_neighbors=ucfg["n_neighbors"],
                n_components=ucfg["n_components"],
                min_dist=ucfg["min_dist"],
                metric="cosine",
                random_state=RANDOM_STATE,
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=hcfg["min_cluster_size"],
                min_samples=hcfg["min_samples"],
                metric="euclidean",
                cluster_selection_method="eom",
                gen_min_span_tree=True, # for DCBV
                prediction_data=True,
            )

            vectorizer_model = CountVectorizer(**VECTORIZER)

            topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                calculate_probabilities=True,
                verbose=False,
            )

            topics, probs = topic_model.fit_transform(docs, emb_unit)
            topics_arr = np.asarray(topics)

            assigned_prob = compute_assigned_prob(topics_arr, probs)

            with open(os.path.join(run_dir, "params.json"), "w") as f:
                json.dump(cfg, f, indent=2)

            topic_info = topic_model.get_topic_info()
            topic_info.to_csv(os.path.join(run_dir, "topic_info.csv"), index=False)

            overview = build_topic_overview(
                topic_model, docs, probs,
                n_words=N_TOPIC_WORDS,
                n_rep_docs=N_REP_DOCS
            )
            overview.to_csv(os.path.join(run_dir, "topic_overview.csv"), index=False)

            main = df_emb[id_cols].copy()
            main["topic"] = topics_arr
            main["topic_conf"] = probs.max(axis=1) if probs is not None and getattr(probs, "ndim", 0) == 2 else np.nan
            main["assigned_prob"] = assigned_prob
            main.to_csv(os.path.join(run_dir, "statement_topics.csv"), index=False)

            summ = summarize_run(topic_model, emb_unit, assigned_prob)

            summ_row = {
                "embed_model_outname": model_outname,
                "run_id": rid,
                "run_dir": os.path.basename(run_dir),
                **{f"umap_{k}": v for k, v in ucfg.items()},
                **{f"hdb_{k}": v for k, v in hcfg.items()},
                **summ,
            }

            summary_rows.append(summ_row)
            all_summary_rows.append(summ_row)

    df_sum = pd.DataFrame(summary_rows).sort_values(
        ["dbcv", "outlier_rate", "distortion"],
        ascending=[False, True, True],
        na_position="last"
    )
    df_sum.to_csv(os.path.join(model_dir, "base_grid_summary.csv"), index=False)

    print("\nSaved base grid summary to:", os.path.join(model_dir, "base_grid_summary.csv"))
    print(df_sum.head(10).to_string(index=False))

df_all = pd.DataFrame(all_summary_rows).sort_values(
    ["dbcv", "outlier_rate", "distortion"],
    ascending=[False, True, True],
    na_position="last"
).reset_index(drop=True)

out_all = os.path.join(out_root, "overview_all_models__base_grid_summary.csv")
df_all.to_csv(out_all, index=False)

print("\nSaved GLOBAL merged summary to:", out_all)
print("Done. Outputs under:", out_root)