'''
VMP 2026-02-06 (refactored):
- Fixes specific data issues in private curation files
- Overwrites private/curation_w*.json

VMP verified (2026-02-06) that this produces exactly the same file as originally.
'''

import json
import copy
from utilities import wave_1, wave_2, get_private_path

# load data from private (contains interviews)
curation_w1_path = get_private_path("curation_w{wave}.json", wave=wave_1)
curation_w2_path = get_private_path("curation_w{wave}.json", wave=wave_2)

with open(curation_w1_path, 'r') as f:
    data_w1 = json.load(f)

with open(curation_w2_path, 'r') as f:
    data_w2 = json.load(f)
    
''' 
one person is logged as having had 9 questions instead of 8 in wave 1
'''

qa_list = data_w1['b3f7e055bf4647c4']['interviews']['main']

if len(qa_list) == 9: 
    qa_list.pop()

# also remove this from the time keeping
page_times = data_w1['b3f7e055bf4647c4']['page_times']

# iterate backwards and remove the first matching item
for i in range(len(page_times) - 1, -1, -1):
    if page_times[i].get('label') == 'interviewmain:submit':
        page_times.pop(i)
        break

'''
a few cases of wrong names for edges by the LLM.
(1) hallucinates something that is just not there
(2) gives a very slightly different name.

For case (1) remove, for case (2) rename.
'''

def fix_llm_nodes(data, rename_by_key, remove_by_key, llm_edges_key="edge_results"):
    """
    Only touches keys mentioned in rename_by_key/remove_by_key.
    - Renames stance_1/stance_2 using rename_by_key[key]
    - Drops edges if stance_1 or stance_2 is in remove_by_key[key]
    """
    data_fixed = dict(data)
    implicated = set(rename_by_key.keys()) | set(remove_by_key.keys())

    for key in implicated:
        if key not in data:
            continue

        d = copy.deepcopy(data[key])
        rename_map = rename_by_key.get(key, {}) or {}
        remove_set = remove_by_key.get(key, set()) or set()

        new_edges = []
        for e in d["LLM"][llm_edges_key]:
            s1 = rename_map.get(e["stance_1"], e["stance_1"])
            s2 = rename_map.get(e["stance_2"], e["stance_2"])

            if (s1 in remove_set) or (s2 in remove_set):
                continue

            e2 = dict(e)
            e2["stance_1"] = s1
            e2["stance_2"] = s2
            new_edges.append(e2)

        d["LLM"][llm_edges_key] = new_edges
        data_fixed[key] = d

    return data_fixed

def fix_llm_edges(data, llm_edges_key="edge_results", valid=("positive", "negative")):
    """
    Touches ALL keys.
    Drops any LLM edge whose polarity is not in valid.
    """
    valid = set(valid)
    data_fixed = dict(data)

    for key in data.keys():
        d = copy.deepcopy(data[key])
        edges = d["LLM"][llm_edges_key]
        d["LLM"][llm_edges_key] = [e for e in edges if e.get("polarity") in valid]
        data_fixed[key] = d

    return data_fixed

# manual LLM node fixes
RENAME_W1 = {
    "90a26c5862848d7f": {
        "I eat meat when it is on sale": "I buy meat when it is on sale",
    }
}

REMOVE_W1 = {
    "90a26c5862848d7f": {
        "I only eat certain types of meat",
    },
    "78497a8102898b56": {
        "I don't feel bad about eating meat"
    },
    "25d9c5f16c668c52": {
        "My perspective on eating meat is that we need to. It seems to be a very healthy thing for the body, and I think that everyone should eat meat, at least in moderation."
    },
    "847c51b8edf36462": {
        "We have no interest in being vegetarians"
    },
    "9d41a316406ab604": {
        "I might try to reduce meat a bit more in the future, mainly for health and environmental reasons"
    },
}

RENAME_W2 = {
    "24e9f4a8dba444ba": {
        "Eat meat with most meals": "I eat meat with most meals"
    },
    "11e5443a9c9c5d6e": {
        "It's been hard to change my diet": "I find it hard to change my diet"
    },
    "c314920d84ff0215": {
        "Balancd diet is important to me": "Balanced diet is important to me"
    },
    
}

REMOVE_W2 = {
    "b8a1d8c350a12933": {
        "I wish I didn't crave the taste of meat"
    },
    "405ae83ad699f77e": {
        "I love the taste of meat"
    },
    "333a4e955656276f": {
        "I try to consume meat in a healthy way."
    }
}

# apply fixes
# Wave 1
data_w1 = fix_llm_nodes(data_w1, RENAME_W1, REMOVE_W1, llm_edges_key="edge_results")
data_w1 = fix_llm_edges(data_w1, llm_edges_key="edge_results", valid=("positive", "negative"))

data_w2 = fix_llm_nodes(data_w2, RENAME_W2, REMOVE_W2, llm_edges_key="edge_results")
data_w2 = fix_llm_edges(data_w2, llm_edges_key="edge_results", valid=("positive", "negative"))

### save corrected data (overwrite private files) ###
with open(curation_w1_path, 'w', encoding='utf-8') as f:
    json.dump(data_w1, f, indent=2)

with open(curation_w2_path, 'w', encoding='utf-8') as f:
    json.dump(data_w2, f, indent=2)

print(f"✓ Fixed data issues in {curation_w1_path}")
print(f"✓ Fixed data issues in {curation_w2_path}")
