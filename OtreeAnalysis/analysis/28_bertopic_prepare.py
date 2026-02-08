'''
VMP 2026-02-06 (refactored):
- Currently only curates human edges (not LLM.)
- Saves also a nodes.csv file which is used in other analysis (consider splitting this out.)
- Uses sanitized public data.
- Saves to embeddings folder (which is gitignored).

VMP 2026-02-08: tested and run.
'''

import os
import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from utilities import wave_1, wave_2, get_public_path
from helpers import extract_nodes, extract_hum_edges, extract_LLM_edges

device = "mps" if torch.backends.mps.is_available() else "cpu"

# outpath
outdir = "../data/public/embeddings"
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

# merge this 
node_w1 = node_w1.merge(
    nodes_a_w1.drop_duplicates().assign(canvas=True),
    on=['stance','key'],
    how='left'
).assign(canvas=lambda d: d['canvas'].fillna(False))

node_w2 = node_w2.merge(
    nodes_a_w2.drop_duplicates().assign(canvas=True),
    on=['stance', 'key'],
    how='left'
).assign(canvas=lambda d: d['canvas'].fillna(False))

# attach label and remove columns
node_w1['wave'] = '1'
node_w2['wave'] = '2'
node_final = pd.concat([node_w1, node_w2])
node_final = node_final[['key', 'stance', 'wave', 'canvas']].dropna()

# save the final nodes 
node_final.to_csv(os.path.join(outdir, "nodes.csv"), index=False)

### prepare edges ### 
edge_llm1 = extract_LLM_edges(data_w1, 'edge_results')
edge_llm2 = extract_LLM_edges(data_w2, 'edge_results')

edge_hum1 = extract_hum_edges(data_w1, 'edges_3')
edge_hum2 = extract_hum_edges(data_w2, 'edges_3')

def rename_hum(edge_dataframe): 
    return edge_dataframe.rename(columns={
        'from': 'stance_1',
        'to': 'stance_2'
    })

edge_hum1 = rename_hum(edge_hum1)
edge_hum2 = rename_hum(edge_hum2)

# prepare edge final hum + LLM
edge_hum1['wave'] = 1
edge_hum2['wave'] = 2 
edge_hum_final = pd.concat([edge_hum1, edge_hum2])
edge_hum_final = edge_hum_final.dropna()

edge_llm1['wave'] = 1
edge_llm2['wave'] = 2
edge_llm_final = pd.concat([edge_llm1, edge_llm2])
edge_llm_final = edge_llm_final.dropna()

# save 
edge_hum_final.to_csv(os.path.join(outdir, "edge_hum.csv"), index=False)
edge_llm_final.to_csv(os.path.join(outdir, "edge_llm.csv"), index=False)