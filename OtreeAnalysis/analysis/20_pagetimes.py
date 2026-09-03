'''
VMP 2026-02-06 (refactored):
- Analyzes page times from sanitized curation data
- Saves figures to ../fig/pagetimes
- Uses public data (page_times field is safe, doesn't contain interviews)

NOTE: Figures are diagnostic only — not reported in manuscript.
Used to verify participants spent reasonable time on each survey page.

VMP 2026-02-07: tested and run.

VMP 2026-09-02: added training-duration tables (reviewer question: how long
did the training take?). Two tables are written to ../fig/pagetimes:
  - training_rounds_w{1,2}   coarse: one row per training round
  - training_subtasks_w{1,2} fine:   one row per round x sub-task
Durations are reported as median and IQR across participants, in minutes.
A round is intro + map + edge_pos + edge_neg; see INCLUDE_BRIEF for the
general instructions page that precedes round 1.
Both waves are now processed in a single run (the figures are unchanged).
Switch SOURCE to "distractors" to restrict to attention-check passers.
'''

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import wave_1, wave_2, get_public_path

# setup
WAVES = [wave_1, wave_2]

# "curation" = all participants; "distractors" = attention-check passers only
SOURCE = "curation"

# decimal places for reported durations (minutes)
DECIMALS = 2

# `training_brief` is a general instructions page rather than a training task,
# and it only precedes round 1. Excluded by default so that every round is the
# same four sub-tasks (intro + map + edge_pos + edge_neg) and rounds compare
# like with like. Its duration is separable: page times are backward diffs
# between submits, so dropping it does not shift any round's time.
INCLUDE_BRIEF = False

# create outfolder
outpath_fig = f"../fig/pagetimes"
os.makedirs(outpath_fig, exist_ok=True)

# page times
def participant_page_times(data, participant_id):
    dict_id = data[participant_id]
    events = pd.DataFrame(dict_id['page_times']).sort_values('ts').reset_index(drop=True)

    # Submits only (your old filter kept everything because label is always truthy)
    submits = events[events['label'].str.endswith(':submit')].copy()
    submits['page'] = submits['label'].str.replace(':submit', '', regex=False)

    # Durations: default = time since previous submit
    submits['duration_sec'] = submits['ts'].diff()

    # Special case: first submit uses consent:render -> consent:submit
    cons_render = events.loc[events['label'] == 'consent:render', 'ts']
    if not cons_render.empty and not submits.empty:
        first_idx = submits.index[0]
        submits.loc[first_idx, 'duration_sec'] = submits.loc[first_idx, 'ts'] - cons_render.iloc[0]

    # Optional: ensure positive, drop any remaining NaNs (e.g., if consent:render missing)
    submits = submits.dropna(subset=['duration_sec'])

    # Sequence suffix for repeated pages (interview screens, etc.)
    submits = submits.sort_values('ts').copy()
    occ = submits.groupby('page').cumcount() + 1
    dup = submits['page'].duplicated(keep=False)
    submits['page_seq'] = np.where(dup, submits['page'] + occ.astype(str), submits['page'])

    # Identifier
    submits['participant_id'] = participant_id

    # Keep useful columns
    return submits[['participant_id', 'page', 'page_seq', 'ts', 'duration_sec', 'label']]


def load_page_times(wave):
    """Flat page-time table for one wave."""
    path = get_public_path(f"{SOURCE}_w{{wave}}.json", wave=wave)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.concat(
        [participant_page_times(data, key) for key in data.keys()],
        ignore_index=True,
    )

    # get the submit rank
    df = df.sort_values(['participant_id', 'ts']).copy()
    df['submit_rank'] = df.groupby('participant_id').cumcount()
    return df


def plot_page_times(df, wave):
    labels = (
        df.groupby('page_seq')['submit_rank']
          .median()
          .sort_values()
          .index
          .to_list()
    )

    data_by_page = [df.loc[df['page_seq'] == lab, 'duration_sec'].values for lab in labels]

    plt.figure(figsize=(14, 6))
    plt.boxplot(data_by_page, tick_labels=labels, showfliers=False, widths=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Time spent (seconds)")
    plt.title("Time per page")
    plt.tight_layout()
    plt.savefig(os.path.join(outpath_fig, f"pagetimes_w{wave}.png"), dpi=300)
    plt.close()


# -------------------------
# training durations
# -------------------------
# The training block is `training_brief` followed by one or more rounds of
# intro -> map -> edge_pos -> edge_neg (3 rounds in w1, 1 round in w2).
# `progress_practice` is the transition page out of training and is excluded.
SUBTASK_LABELS = {
    'intro': 'Read scenario',
    'map': 'Place statements',
    'edge_pos': 'Draw supporting links',
    'edge_neg': 'Draw conflicting links',
}
SUBTASK_ORDER = ['intro', 'map', 'edge_pos', 'edge_neg']
TRAINING_RE = re.compile(r'^training_(intro|map|edge_pos|edge_neg)_(\d+)$')


def label_training(df):
    """Tag training rows with round (1-indexed) and sub-task; drop everything else."""
    parsed = df['page'].str.extract(TRAINING_RE)
    parsed.columns = ['subtask', 'round']

    out = df.copy()
    out['subtask'] = parsed['subtask']
    # 0-indexed in the instrument, 1-indexed for reporting
    out['round'] = pd.to_numeric(parsed['round'], errors='coerce') + 1

    # training_brief is a one-off page before round 1, not part of any round
    if INCLUDE_BRIEF:
        brief = out['page'] == 'training_brief'
        out.loc[brief, 'subtask'] = 'brief'
        out.loc[brief, 'round'] = 0

    return out.dropna(subset=['subtask'])


def summarise(series):
    """Median and IQR over participants, in minutes."""
    minutes = series / 60
    q1, q3 = minutes.quantile([0.25, 0.75])
    return pd.Series({
        'n': int(minutes.notna().sum()),
        'median_min': minutes.median(),
        'q1_min': q1,
        'q3_min': q3,
    })


def training_tables(df, wave):
    """Coarse (per round) and fine (per round x sub-task) duration tables."""
    tr = label_training(df)

    # ---- fine: one row per round x sub-task ----
    # each participant sees a given round/sub-task once, so no per-participant
    # aggregation is needed here
    fine = (
        tr.groupby(['round', 'subtask'])['duration_sec']
          .apply(summarise)
          .unstack()
          .reset_index()
    )
    fine['subtask_order'] = fine['subtask'].map(
        {**{s: i for i, s in enumerate(SUBTASK_ORDER)}, 'brief': -1}
    )
    fine = fine.sort_values(['round', 'subtask_order']).drop(columns='subtask_order')
    fine.insert(0, 'wave', wave)
    fine['subtask_label'] = fine['subtask'].map(SUBTASK_LABELS).fillna('Instructions')

    # ---- coarse: total per participant per round, then summarised ----
    per_participant = (
        tr.groupby(['participant_id', 'round'])['duration_sec']
          .sum()
          .reset_index()
    )
    coarse = (
        per_participant.groupby('round')['duration_sec']
                       .apply(summarise)
                       .unstack()
                       .reset_index()
    )

    # whole training block per participant (brief + every round)
    total = (
        tr.groupby('participant_id')['duration_sec']
          .sum()
          .pipe(summarise)
          .to_frame().T
    )
    total.insert(0, 'round', -1)  # sentinel: full block
    coarse = pd.concat([coarse, total], ignore_index=True)
    coarse.insert(0, 'wave', wave)
    coarse['label'] = np.where(
        coarse['round'] == -1, 'Training block (total)',
        np.where(coarse['round'] == 0, 'Instructions', 'Round ' + coarse['round'].astype(int).astype(str))
    )

    out = []
    for tbl in (coarse, fine):
        tbl['n'] = tbl['n'].astype(int)
        tbl['round'] = tbl['round'].astype(int)
        for col in ('median_min', 'q1_min', 'q3_min'):
            tbl[col] = tbl[col].round(DECIMALS)
        out.append(tbl)

    return out[0], out[1]


def fmt(tbl, label_col):
    """Human-readable view: median [IQR] in minutes."""
    return pd.DataFrame({
        label_col: tbl[label_col],
        'n': tbl['n'],
        'median_min': tbl['median_min'],
        'IQR_min': tbl.apply(
            lambda r: f"{r['q1_min']:.{DECIMALS}f}-{r['q3_min']:.{DECIMALS}f}", axis=1
        ),
    })


# -------------------------
# run
# -------------------------
for wave in WAVES:
    df = load_page_times(wave)
    plot_page_times(df, wave)

    coarse, fine = training_tables(df, wave)

    csv_kwargs = dict(index=False, float_format=f"%.{DECIMALS}f")
    coarse[['wave', 'round', 'label', 'n', 'median_min', 'q1_min', 'q3_min']].to_csv(
        os.path.join(outpath_fig, f"training_rounds_w{wave}.csv"), **csv_kwargs)
    fine[['wave', 'round', 'subtask', 'subtask_label', 'n', 'median_min', 'q1_min', 'q3_min']].to_csv(
        os.path.join(outpath_fig, f"training_subtasks_w{wave}.csv"), **csv_kwargs)

    n_part = df['participant_id'].nunique()
    print(f"\n{'=' * 72}")
    print(f"WAVE {wave}  (source: {SOURCE}, n = {n_part} participants)")
    print('=' * 72)

    print("\n[1] Training duration by round")
    print(fmt(coarse, 'label').to_string(index=False))

    print("\n[2] Training duration by round x sub-task")
    fine_view = fmt(fine, 'subtask_label')
    fine_view.insert(0, 'round', np.where(fine['round'] == 0, '-', fine['round'].astype(int).astype(str)))
    print(fine_view.to_string(index=False))

print(f"\nWrote figures and tables to {outpath_fig}")
