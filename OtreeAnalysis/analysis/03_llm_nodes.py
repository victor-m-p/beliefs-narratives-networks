'''
VMP 2026-02-06 (refactored):
Runs node extraction without the limit imposed in the study.
Runs for all participants in curation (before filtering by attention checks).
Output needed for 12_basic_reliability.py.

SENSITIVE: Reads interview transcripts from private/curation_w*.json
OUTPUTS:
  - private/llm_prompts/ (contains interview text)
  - public/llm_extractions/ (safe structured nodes/edges)

VMP: Verified that this saves to the right directories and is broadly consistent (2026-02-06).
I did not rerun all extractions. 
'''
import json
import os
from utilities import wave_1, wave_2, get_private_path, get_public_path
from llm_utilities import (
    call_openai,
    make_node_prompt,
    NodeModelList,
    make_edge_prompt,
    EdgeModelList,
)
import json 

# function to extract nodes 
def extract_transcript(data, id):

    dict_id = data[id]

    # get the Q-A 
    main_interview = dict_id['interviews']['main']
    transcript = {item["question"]: item["answer"] for i, item in enumerate(main_interview)}

    return transcript

# setup
wave = wave_1
model = 'gpt-4.1-2025-04-14' # same as in survey

# load data (from private - contains interviews)
curation_path = get_private_path("curation_w{wave}.json", wave=wave)
with open(curation_path, 'r') as f:
    data = json.load(f)

# Setup output paths
path_prompt_nodes = get_private_path(f"llm_prompts/node_prompts_w{wave}/{model}")
path_llm_nodes = get_public_path(f"llm_extractions/node_extraction_w{wave}/{model}")
os.makedirs(path_prompt_nodes, exist_ok=True)
os.makedirs(path_llm_nodes, exist_ok=True)

for id in data.keys(): 
    # extract transcript
    transcript = extract_transcript(data, id)

    # if no transcript exists then skip
    if transcript is None: 
        continue 
    
    # make node prompt + save
    prompt = make_node_prompt(transcript)
    with open(os.path.join(path_prompt_nodes, f"{id}.txt"), "w") as f:
        f.write(prompt)

    # run node extraction with the specified model
    print(f"node extraction: running {id} with {model}")
    response = call_openai(NodeModelList, prompt, model_name=model)
    results = json.loads(response.model_dump_json(indent=2))['results']
    
    # save results 
    with open(os.path.join(path_llm_nodes, f"{id}.json"), "w") as f:
        json.dump(results, f, indent=2)

### assigning edges to these nodes ###
path_prompt_edges = get_private_path(f"llm_prompts/edge_prompts_w{wave}/{model}")
path_llm_edges = get_public_path(f"llm_extractions/edge_extraction_w{wave}/{model}")
os.makedirs(path_prompt_edges, exist_ok=True)
os.makedirs(path_llm_edges, exist_ok=True)

# load the nodes from before 
for id in data.keys():
    # extract transcript 
    transcript = extract_transcript(data, id)
    
    # if no transcript then continue silently
    if transcript is None: 
        continue
    
    # extract saved nodes
    with open(os.path.join(path_llm_nodes, f"{id}.json"), "r") as f:
        nodes_data = json.load(f)

    # create edge prompt 
    nodes_list = [node['stance'] for node in nodes_data]
    prompt = make_edge_prompt(transcript, nodes_list)

    # save edge prompt
    with open(os.path.join(path_prompt_edges, f"{id}.txt"), "w") as f:
        f.write(prompt)

    # run edge extraction 
    print(f"edge extraction: running {id} with {model}")
    response = call_openai(EdgeModelList, prompt, model_name=model)
    results = json.loads(response.model_dump_json(indent=2))['results']

    # save LLM edges
    with open(os.path.join(path_llm_edges, f"{id}.json"), "w") as f:
        json.dump(results, f, indent=2)