'''
VMP 2026-02-06 (refactored):
Curates Prolific/oTree data and anonymizes IDs using secret key.
Loads from private/raw/, saves to private/.

VMP verified (2026-02-06) that this produces exactly the same file as originally.
'''

import os
import json
import hashlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from utilities import date_w1, date_w2, wave_1, wave_2, get_private_path
from helpers import safe_json_loads

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Choose wave
wave = wave_2  # Change to wave_2 for Wave 2
date = date_w1 if wave == wave_1 else date_w2

# Load cleaned data from private/raw/ (output of 00_pre_cleaning)
csv_path = get_private_path(f"raw/all_apps_wide_{date}_clean.csv")
df = pd.read_csv(csv_path)
print(f"✓ Loaded cleaned CSV: {len(df)} participants from {csv_path}")
pd.set_option('display.max_colwidth', None)

# anonymization
SALT = os.getenv("ANON_SALT")

def anonymize_label(x: str, salt: str = SALT, length: int = 16) -> str:
    s = str(x)
    h = hashlib.sha256((salt + s).encode("utf-8")).hexdigest()
    return h[:length]

df["participant.label"] = df["participant.label"].apply(anonymize_label)

# create outfolder 
os.makedirs(f"../data/data-{date}/clean", exist_ok=True)

# consent
def collect_data(df, base_str, i, wave):
    data = {}
    
    code_i = df['participant.code'].iloc[i]
    label_i = df['participant.label'].iloc[i]
    is_wave2 = (wave == 2)

    # ---- common part: exists in both waves ----
    data[label_i] = {
        # label 
        'code': code_i,
        
        # page times
        'page_times': safe_json_loads(df[f'{base_str}.page_timings_json'].iloc[i]),
        
        # consent 
        'consent': df[f'{base_str}.consent_given'].iloc[i],

        # training
        'training': {
            "train_order": safe_json_loads(df[f"{base_str}.training_order_json"].iloc[i]),
            "train_map": safe_json_loads(df[f"{base_str}.training_map_attempts_json"].iloc[i]),
            "train_pos": safe_json_loads(df[f"{base_str}.training_pos_attempts_json"].iloc[i]),
            "train_neg": safe_json_loads(df[f"{base_str}.training_neg_attempts_json"].iloc[i]),
        },

        # meat scale
        'meat_scale': {
            'present': df[f'{base_str}.meat_consumption_present'].iloc[i],
            'past': df[f'{base_str}.meat_consumption_past'].iloc[i],
            'future': df[f'{base_str}.meat_consumption_future'].iloc[i],
        },

        # dissonance scale
        'dissonance': {
            'personal': df[f'{base_str}.dissonance_personal'].iloc[i],
            'social': df[f'{base_str}.dissonance_social'].iloc[i],
        },
        
        # interviews
        'interviews': {
            'test': df[f'{base_str}.interview_test'].iloc[i],
            'test_audio': df[f'{base_str}.audio_data'].iloc[i],
            'main': safe_json_loads(df[f'{base_str}.conversation_json'].iloc[i]),
            'feedback': df[f'{base_str}.interview_feedback'].iloc[i] 
        },
        
        # interview feedback
        'interview_rating': {
            'overall_num': df[f'{base_str}.conv_overall_0_100'].iloc[i],
            'overall_cat': df[f'{base_str}.conv_overall_cat'].iloc[i],
            'relevant_num': df[f'{base_str}.conv_relevant_0_100'].iloc[i],
            'relevant_cat': df[f'{base_str}.conv_relevant_cat'].iloc[i],
            'easy_chat_num': df[f'{base_str}.conv_easy_chat_0_100'].iloc[i],
            'easy_chat_cat': df[f'{base_str}.conv_easy_chat_cat'].iloc[i],
            'comfort_num': df[f'{base_str}.conv_comfort_0_100'].iloc[i],
            'comfort_cat': df[f'{base_str}.conv_comfort_cat'].iloc[i],
            'creepy_num': df[f'{base_str}.conv_creepy_0_100'].iloc[i],
            'creepy_cat': df[f'{base_str}.conv_creepy_cat'].iloc[i],
        },
        
        # separately for now.
        'interview_feedback': df[f'{base_str}.conv_open_feedback'].iloc[i],
        
        # llm
        'LLM': {
            'node_prompt': df[f'{base_str}.prompt_used'].iloc[i],
            'node_results': safe_json_loads(df[f'{base_str}.llm_result'].iloc[i]),
            'edge_prompt': df[f'{base_str}.llm_edge_prompt'].iloc[i],
            'edge_results': safe_json_loads(df[f'{base_str}.llm_edges'].iloc[i]),
            'edge_random': safe_json_loads(df[f'{base_str}.llm_edges_random'].iloc[i]),
        },
        
        # distractor nodes: 
        'distractors': {
            'failed': df[f'{base_str}.distractor_problem'].iloc[i],
            'ratings': safe_json_loads(df[f'{base_str}.distractor_ratings'].iloc[i]), 
        },
        
        # nodes 
        'nodes': {
            'generated': json.loads(df[f'{base_str}.generated_nodes'].iloc[i]), 
            'final': json.loads(df[f'{base_str}.final_nodes'].iloc[i]),
            'accuracy': json.loads(df[f'{base_str}.generated_nodes_accuracy'].iloc[i]), 
            'relevance': json.loads(df[f'{base_str}.generated_nodes_relevance'].iloc[i])
        },
        
        # positions 
        'positions': {
            'pos_1': json.loads(df[f'{base_str}.positions_1'].iloc[i]),
            'pos_2': json.loads(df[f'{base_str}.positions_2'].iloc[i]),
            'pos_3': json.loads(df[f'{base_str}.positions_3'].iloc[i]),
        },    
        
        # edges
        'edges': {
            'edges_2': json.loads(df[f'{base_str}.edges_2'].iloc[i]),
            'edges_3': json.loads(df[f'{base_str}.edges_3'].iloc[i]),
        },  
        
        # demographics 
        'demographics': {
            'age': df[f'{base_str}.age'].iloc[i],
            'gender': df[f'{base_str}.gender'].iloc[i],
            'education': df[f'{base_str}.education'].iloc[i],
            'politics': df[f'{base_str}.politics'].iloc[i],
            'state': df[f'{base_str}.state'].iloc[i],
            'zipcode': df[f'{base_str}.zipcode'].iloc[i],
        },

        # final feedback
        'final_feedback': df[f'{base_str}.final_feedback'].iloc[i],

        # ---- second-wave-specific stuff: default to None for wave 1 ----

        # network comparison
        'network_compare': (
            safe_json_loads(df[f'{base_str}.network_compare_results'].iloc[i])
            if is_wave2 else None
        ),
        
        # plausibility
        'pairwise_interview': (
            safe_json_loads(df[f'{base_str}.pair_interview_json'].iloc[i])
            if is_wave2 else None
        ),
        
        # questionnaire    
        'questionnaires': (
            {
                'VEMI': safe_json_loads(df[f'{base_str}.vemi_responses'].iloc[i]),
                'MEMI': safe_json_loads(df[f'{base_str}.memi_responses'].iloc[i]),
            } if is_wave2 else None
        ),
    }
    return data

# gather data 
master_data = {}
base_str = 'otreesurvey_app.1.player'
for i in range(len(df)):
    participant_data = collect_data(df, base_str, i, wave)
    master_data.update(participant_data)  # merge into master

# maybe need this 
def convert_to_builtin_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
master_data = convert_to_builtin_type(master_data)

# Save curated data to private/
curation_path = get_private_path(f"curation_w{wave}.json")
with open(curation_path, 'w', encoding='utf-8') as f:
    json.dump(master_data, f, indent=2)

print(f"✓ Curated Wave {wave} data: {len(master_data)} participants")
print(f"✓ Saved to {curation_path}")