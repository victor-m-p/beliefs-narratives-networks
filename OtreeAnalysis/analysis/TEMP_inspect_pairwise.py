"""
TEMP — inspect pairwise_interview data (private).
Delete when done.

Usage:
  python TEMP_inspect_pairwise.py            # prints first N=5 participants
  python TEMP_inspect_pairwise.py <key>      # prints one specific participant
  python TEMP_inspect_pairwise.py --random   # prints one random participant
"""
import json
import sys
import random

PRIVATE_W2 = "../data/private/curation_w2.json"
N_DEFAULT  = 5   # participants to show when no argument given

with open(PRIVATE_W2, encoding="utf-8") as f:
    data = json.load(f)

# keep only participants who have pairwise data
participants = {k: v for k, v in data.items() if v.get("pairwise_interview")}
print(f"Participants with pairwise_interview: {len(participants)}\n")


def print_participant(key: str, bundle: dict) -> None:
    pw = bundle["pairwise_interview"]
    print("=" * 70)
    print(f"PARTICIPANT: {key}   ({len(pw)} pairs)")
    print("=" * 70)
    for item in pw:
        print(f"\n  Pair {item['pair_index']}:")
        print(f"    Belief A : {item['pair'][0]}")
        print(f"    Belief B : {item['pair'][1]}")
        print(f"    Question : {item['question']}")
        print(f"    Answer   : {item['answer']}")
        print(f"    Coding   : {item['connection_choice']}")
        print(f"    Mode     : {item['input_mode']}")
    print()


# ---- argument handling ----
args = sys.argv[1:]

if not args:
    keys = list(participants.keys())[:N_DEFAULT]
    for k in keys:
        print_participant(k, participants[k])

elif args[0] == "--random":
    k = random.choice(list(participants.keys()))
    print_participant(k, participants[k])

else:
    k = args[0]
    if k in participants:
        print_participant(k, participants[k])
    else:
        print(f"Key '{k}' not found. Available keys (first 20):")
        for key in list(participants.keys())[:20]:
            print(" ", key)


# ---- save all participants to individual .txt files ----
import os
OUTDIR = "../data/private/pairwise"
os.makedirs(OUTDIR, exist_ok=True)

for key, bundle in participants.items():
    pw = bundle["pairwise_interview"]
    lines = []
    lines.append("=" * 70)
    lines.append(f"PARTICIPANT: {key}   ({len(pw)} pairs)")
    lines.append("=" * 70)
    for item in pw:
        lines.append(f"\n  Pair {item['pair_index']}:")
        lines.append(f"    Belief A : {item['pair'][0]}")
        lines.append(f"    Belief B : {item['pair'][1]}")
        lines.append(f"    Question : {item['question']}")
        lines.append(f"    Answer   : {item['answer']}")
        lines.append(f"    Coding   : {item['connection_choice']}")
        lines.append(f"    Mode     : {item['input_mode']}")
    with open(os.path.join(OUTDIR, f"{key}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

print(f"Saved {len(participants)} files to {OUTDIR}")