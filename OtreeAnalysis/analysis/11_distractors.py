'''
VMP 2026-02-06 (refactored):
Loads sanitized curation data from public/.

Filters out people that rate the "seahorse" distractor too highly.
Creates distractors_w*.json in public/ folder.

VMP (run on: 2026-02-06): 
'''

import pandas as pd
import json
from utilities import wave_1, wave_2, get_public_path

# load data (from public - sanitized)
curation_w1_path = get_public_path("curation_w{wave}.json", wave=wave_1)
curation_w2_path = get_public_path("curation_w{wave}.json", wave=wave_2)

with open(curation_w1_path, 'r') as f:
    data_w1 = json.load(f)

with open(curation_w2_path, 'r') as f:
    data_w2 = json.load(f)

# overview of distractors 
def curate_distractors(data):

    distractor_df = pd.concat(
        (
            pd.DataFrame(v['distractors']['ratings'])
            .assign(key=k)
            .drop(columns='index', errors='ignore')
            for k, v in data.items()
        ),
        ignore_index=True
    )
    
    return distractor_df 

distractors_w1 = curate_distractors(data_w1)
distractors_w2 = curate_distractors(data_w2)

# filter only by the seahorse distractor.
distractor = "My friends often go out to eat seahorse"
distractors_w1 = distractors_w1[distractors_w1['belief']==distractor]
distractors_w2 = distractors_w2[distractors_w2['belief']==distractor]

# now find keys with rating >= 40 
failed_keys_w1_alt = distractors_w1[distractors_w1['rating'] >= 40]['key'].unique().tolist()
failed_keys_w2_alt = distractors_w2[distractors_w2['rating'] >= 40]['key'].unique().tolist()

# check how many fail this attention check
failed_keys_w1_alt # n=1 key
failed_keys_w2_alt # n=5 keys

# now do intersection and remove them
data_w1_clean_alt = {k: v for k, v in data_w1.items() if k not in failed_keys_w1_alt}
data_w2_clean_alt = {k: v for k, v in data_w2.items() if k not in failed_keys_w2_alt}

len(data_w1_clean_alt) # 246
len(data_w2_clean_alt) # 212

# find intersection
w1_keys_alt = set(data_w1_clean_alt.keys())
w2_keys_alt = set(data_w2_clean_alt.keys())
intersection_keys_alt = w1_keys_alt & w2_keys_alt
len(intersection_keys_alt) # 210 --> instead of n=183 above

data_w1_final_alt = {k: v for k, v in data_w1_clean_alt.items() if k in intersection_keys_alt}
data_w2_final_alt = {k: v for k, v in data_w2_clean_alt.items() if k in intersection_keys_alt}

# save to public/ (attention-filtered data)
distractors_w1_path = get_public_path("distractors_w{wave}.json", wave=wave_1)
distractors_w2_path = get_public_path("distractors_w{wave}.json", wave=wave_2)

with open(distractors_w1_path, 'w') as f:
    json.dump(data_w1_final_alt, f, indent=2)

with open(distractors_w2_path, 'w') as f:
    json.dump(data_w2_final_alt, f, indent=2)

print(f"✓ Saved {len(data_w1_final_alt)} participants to {distractors_w1_path}")
print(f"✓ Saved {len(data_w2_final_alt)} participants to {distractors_w2_path}")
