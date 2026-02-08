'''
VMP 2026-02-06 (refactored):
- Analyzes page times from sanitized curation data
- Saves figures to ../fig/pagetimes
- Uses public data (page_times field is safe, doesn't contain interviews)

VMP 2026-02-07: tested and run.
'''

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utilities import wave_1, wave_2, get_public_path

# setup
wave = wave_2  # change here for wave 1 vs wave 2

# load data from public (sanitized, but includes page_times)
curation_path = get_public_path("curation_w{wave}.json", wave=wave)
with open(curation_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

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

# extract data for all participants
dict_keys = data.keys()
page_times_list = []
for key in dict_keys:
    df = participant_page_times(data, key)
    page_times_list.append(df)
df = pd.concat(page_times_list, ignore_index=True)

# get the submit rank 
df = df.sort_values(['participant_id', 'ts']).copy()
df['submit_rank'] = df.groupby('participant_id').cumcount()

labels = (
    df.groupby('page_seq')['submit_rank']
      .median()
      .sort_values()
      .index
      .to_list()
)

data_by_page = [df.loc[df['page_seq'] == lab, 'duration_sec'].values for lab in labels]

# plot 
plt.figure(figsize=(14, 6))
plt.boxplot(data_by_page, tick_labels=labels, showfliers=False, widths=1)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Time spent (seconds)")
plt.title("Time per page")
plt.tight_layout()
plt.savefig(os.path.join(outpath_fig, f"pagetimes_w{wave}.png"), dpi=300)