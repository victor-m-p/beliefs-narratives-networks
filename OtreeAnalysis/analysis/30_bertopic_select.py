"""
VMP 2026-02-02

Select TOP10 BERTopic runs and copy key artifacts into:

../data/data-<date_w2>/topics_bertopic/selection/
  statement_topics/
  overview/

Also writes:
  selection/overview_top10.csv
  selection/manifest.csv

VMP 2026-02-08: tested and run.
"""

import os
import shutil
from pathlib import Path

import pandas as pd
# -------------------
# Config
# -------------------
inpath = Path("../data/public/bertopic")

TOP_N = 10
MAX_OUTLIER0 = 0.40
MIN_TOPICS, MAX_TOPICS = 5, 40

selection_folder = inpath / "selection"
statement_folder = selection_folder / "statement_topics"
overview_folder  = selection_folder / "overview"

# -------------------
# Load + filter + pick top10
# -------------------
mdl = pd.read_csv(inpath / "overview_all_models__base_grid_summary.csv")
print(len(mdl))  # 72

mdl = mdl[mdl["outlier_0_ratio"] <= MAX_OUTLIER0]
print(len(mdl))  # 61

mdl = mdl[(mdl["n_topics"] >= MIN_TOPICS) & (mdl["n_topics"] <= MAX_TOPICS)]
print(len(mdl))  # 40

# -------------------
# Reset selection folder
# -------------------
shutil.rmtree(selection_folder, ignore_errors=True)
statement_folder.mkdir(parents=True, exist_ok=True)
overview_folder.mkdir(parents=True, exist_ok=True)

# assumes file is already sorted by dbcv desc (as in your grid script)
top10 = mdl.head(TOP_N).copy()
top10.to_csv(selection_folder / "overview_top10.csv", index=False)

# -------------------
# Copy artifacts
# -------------------
rows = []
for rank, r in enumerate(top10.itertuples(index=False), 1):
    model = r.embed_model_outname
    run_id = r.run_id
    run_dir = inpath / model / "runs" / f"run_{run_id}"

    base = f"{rank:02d}__{model}__run_{run_id}"

    src_stmt = run_dir / "statement_topics.csv"
    src_info = run_dir / "topic_info.csv"
    src_ov   = run_dir / "topic_overview.csv"
    src_par  = run_dir / "params.json"

    dst_stmt = statement_folder / f"{base}__statement_topics.csv"
    dst_info = overview_folder  / f"{base}__topic_info.csv"
    dst_ov   = overview_folder  / f"{base}__topic_overview.csv"
    dst_par  = overview_folder  / f"{base}__params.json"

    ok_stmt = src_stmt.exists()
    ok_info = src_info.exists()
    ok_ov   = src_ov.exists()
    ok_par  = src_par.exists()

    if ok_stmt: shutil.copyfile(src_stmt, dst_stmt)
    if ok_info: shutil.copyfile(src_info, dst_info)
    if ok_ov:   shutil.copyfile(src_ov,   dst_ov)
    if ok_par:  shutil.copyfile(src_par,  dst_par)

    rows.append(dict(
        rank=rank, embed_model_outname=model, run_id=run_id,
        run_dir=str(run_dir),
        statement_topics=str(dst_stmt) if ok_stmt else "",
        topic_info=str(dst_info) if ok_info else "",
        topic_overview=str(dst_ov) if ok_ov else "",
        params=str(dst_par) if ok_par else "",
        ok_stmt=ok_stmt, ok_info=ok_info, ok_overview=ok_ov, ok_params=ok_par
    ))

pd.DataFrame(rows).to_csv(selection_folder / "manifest.csv", index=False)
print("Wrote:", selection_folder)
