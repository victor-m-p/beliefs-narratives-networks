'''
VMP 2026-02-06 (refactored):
- Prepares nodes.csv for downstream BERTopic analysis (29_bertopic_fit.py).
- Uses sanitized public data.
- Saves to ../data/public.

VMP 2026-02-08: tested and run.
VMP 2026-03-27: removed unused edge preparation code; renamed to 28_prepare_nodes.py.
'''

import os
import json
import pandas as pd
from utilities import wave_1, wave_2, get_public_path
from helpers import extract_nodes

# outpath
outdir = "../data/public"
os.makedirs(outdir, exist_ok=True)

# load data from public (sanitized)
distractors_w1_path = get_public_path("distractors_w{wave}.json", wave=wave_1)
distractors_w2_path = get_public_path("distractors_w{wave}.json", wave=wave_2)

with open(distractors_w1_path, 'r', encoding='utf-8') as f:
    data_w1 = json.load(f)

with open(distractors_w2_path, "r", encoding='utf-8') as f:
    data_w2 = json.load(f)

### prepare nodes ###
# all generated
node_w1 = extract_nodes(data_w1, 'generated')
node_w2 = extract_nodes(data_w2, 'generated')

# only accepted
nodes_a_w1 = extract_nodes(data_w1, 'final')
nodes_a_w2 = extract_nodes(data_w2, 'final')

# curation
nodes_a_w1 = nodes_a_w1.rename(columns={'belief': 'stance'})
nodes_a_w2 = nodes_a_w2.rename(columns={'belief': 'stance'})
nodes_a_w1 = nodes_a_w1[['stance', 'key']]
nodes_a_w2 = nodes_a_w2[['stance', 'key']]

# merge to flag which nodes ended up on the canvas
node_w1 = node_w1.merge(
    nodes_a_w1.drop_duplicates().assign(canvas=True),
    on=['stance', 'key'],
    how='left'
).assign(canvas=lambda d: d['canvas'].fillna(False))

node_w2 = node_w2.merge(
    nodes_a_w2.drop_duplicates().assign(canvas=True),
    on=['stance', 'key'],
    how='left'
).assign(canvas=lambda d: d['canvas'].fillna(False))

# attach wave label and save
node_w1['wave'] = '1'
node_w2['wave'] = '2'
node_final = pd.concat([node_w1, node_w2])
node_final = node_final[['key', 'stance', 'wave', 'canvas']].dropna()

node_final.to_csv(os.path.join(outdir, "nodes.csv"), index=False)
