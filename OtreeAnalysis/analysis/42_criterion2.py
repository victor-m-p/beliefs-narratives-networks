"""
42_criterion2.py

VMP 2026-09-11

Topic-level persistence from wave 1 to wave 2 (reviewer request).

MAY SUPERSEDE Table S9, which comes from 38_criterion.py. Note that
38_criterion.py also produces Figure 7 and Figure S12, which are NOT reproduced
here; do not delete it.

Table S9 is anchored on wave 1: every row is a (participant, topic) pair the
participant had in WAVE 1, so it reports stickiness given presence and nothing
else. A reviewer asked how often beliefs appear in wave 2 that were not in
wave 1, which that table structurally cannot answer.

Here we build the full participant x topic grid instead, which gives both
conditional probabilities symmetrically:

  Retained in W2   P(topic in wave 2 | topic in wave 1)   - stickiness
  Acquired in W2   P(topic in wave 2 | topic NOT in wave 1) - base rate

Their difference is the actual persistence effect. A topic can look sticky
simply because it is common: if it is acquired at nearly the rate it is
retained, wave-1 presence carries little information.

Both denominators are reported, because they are complementary and sum to the
210 participants: the topics with the highest retention are exactly those with
the fewest participants available to acquire them, so those acquisition rates
rest on small samples.

Reads:
  ../data/public/bertopic/selection/overview_top10.csv
  ../data/public/bertopic/selection/statement_topics/<label>__statement_topics.csv
  ../data/public/bertopic_mapping/edge_mapping__<label>.csv

Writes:
  ../fig/criterion2/topic_persistence_table.csv
  ../fig/criterion2/topic_persistence_table.tex
"""

import os

import pandas as pd

from helpers import normalize_ab

SEL_TOPICS = "../data/public/bertopic/selection"
SEL_MAP = "../data/public/bertopic_mapping"

outdir = "../fig/criterion2"
os.makedirs(outdir, exist_ok=True)

# selected BERTopic run: row 8 of the top-10 overview, i.e. model "09"
SELECTED_IDX = 8

top10 = pd.read_csv(f"{SEL_TOPICS}/overview_top10.csv")
row = top10.iloc[SELECTED_IDX]
label = f"{SELECTED_IDX + 1:02d}__{row['embed_model_outname']}__run_{row['run_id']}"
print(f"selected model: {label}")

# -------------------------
# nodes: which topics each participant has, in each wave
# -------------------------
nodes = pd.read_csv(f"{SEL_TOPICS}/statement_topics/{label}__statement_topics.csv")
nodes = nodes[nodes["wave"].isin([1, 2])][["key", "wave", "topic"]]

w1 = nodes[nodes["wave"] == 1][["key", "topic"]].drop_duplicates()
w2 = nodes[nodes["wave"] == 2][["key", "topic"]].drop_duplicates()

participants = sorted(nodes["key"].unique())
topics = sorted(nodes["topic"].unique())

print(f"{len(participants)} participants, {len(topics)} topics")
print(f"wave 1: {len(w1)} participant-topic pairs")
print(f"wave 2: {len(w2)} participant-topic pairs")

# -------------------------
# wave-1 topic degree (weighted: statement connections underlying each topic)
# -------------------------
edges = pd.read_csv(f"{SEL_MAP}/edge_mapping__{label}.csv")[
    ["wave", "key", "topic_1", "topic_2"]]
edges = edges[edges["wave"] == 1]
edges = normalize_ab(edges, "topic_1", "topic_2")
edges = (edges.groupby(["key", "topic_1", "topic_2"], as_index=False)
              .size().rename(columns={"size": "n_edges"}))

# each topic pair contributes to both endpoints; self-pairs contribute once
deg_1 = (edges.groupby(["key", "topic_1"])["n_edges"].sum()
              .reset_index(name="degree").rename(columns={"topic_1": "topic"}))
deg_2 = (edges[edges["topic_1"] != edges["topic_2"]]
         .groupby(["key", "topic_2"])["n_edges"].sum()
         .reset_index(name="degree").rename(columns={"topic_2": "topic"}))
degree_w1 = (pd.concat([deg_1, deg_2], ignore_index=True)
             .groupby(["key", "topic"], as_index=False)["degree"].sum())

# -------------------------
# full grid: every participant x every topic, present or not, in each wave
# -------------------------
grid = pd.MultiIndex.from_product([participants, topics],
                                  names=["key", "topic"]).to_frame(index=False)
grid = grid.merge(w1.assign(in_w1=1), on=["key", "topic"], how="left")
grid = grid.merge(w2.assign(in_w2=1), on=["key", "topic"], how="left")
grid = grid.merge(degree_w1, on=["key", "topic"], how="left")

grid["in_w1"] = grid["in_w1"].fillna(0).astype(int)
grid["in_w2"] = grid["in_w2"].fillna(0).astype(int)
grid["degree"] = grid["degree"].fillna(0).astype(int)

print(f"grid: {len(grid)} rows ({len(participants)} x {len(topics)})")

# -------------------------
# the two conditionals, split on wave-1 presence
# -------------------------
had_it = grid[grid["in_w1"] == 1]
did_not = grid[grid["in_w1"] == 0]

table = pd.DataFrame({
    "N present (W1)":    had_it.groupby("topic").size(),
    "Retained in W2":    had_it.groupby("topic")["in_w2"].mean(),
    "N absent (W1)":     did_not.groupby("topic").size(),
    "Acquired in W2":    did_not.groupby("topic")["in_w2"].mean(),
    "Topic degree (W1)": had_it.groupby("topic")["degree"].mean(),
}).reset_index().rename(columns={"topic": "Topic"})

table["Difference"] = table["Retained in W2"] - table["Acquired in W2"]
table = table[["Topic", "N present (W1)", "Retained in W2",
               "N absent (W1)", "Acquired in W2", "Difference",
               "Topic degree (W1)"]]

# the two groups partition the participants
assert (table["N present (W1)"] + table["N absent (W1)"] == len(participants)).all()

print("\n=== Topic persistence: stickiness vs base rate ===")
print(table.round(2).to_string(index=False))

pooled_retained = had_it["in_w2"].mean()
pooled_acquired = did_not["in_w2"].mean()
print(f"\nPooled over all participant-topic pairs:")
print(f"  Retained in W2 : {pooled_retained:.3f}  (n={len(had_it)})")
print(f"  Acquired in W2 : {pooled_acquired:.3f}  (n={len(did_not)})")
print(f"  Difference     : {pooled_retained - pooled_acquired:+.3f}")

# -------------------------
# check the retention column against the table 38_criterion.py produced
# -------------------------
prev_path = "../fig/criterion/topic_persistence.csv"
if os.path.exists(prev_path):
    prev = (pd.read_csv(prev_path)
            .groupby("topic_id")
            .agg(n_prev=("participant_id", "size"),
                 p_prev=("present_wave2", "mean"),
                 deg_prev=("degree_weighted", "mean"))
            .reset_index().rename(columns={"topic_id": "Topic"}))
    check = table.merge(prev, on="Topic")
    same = (
        (check["N present (W1)"] == check["n_prev"]).all()
        and (check["Retained in W2"] - check["p_prev"]).abs().max() < 1e-9
        and (check["Topic degree (W1)"] - check["deg_prev"]).abs().max() < 1e-9
    )
    print(f"\nRetention columns match 38_criterion.py (Table S9) exactly: {same}")
else:
    print(f"\n{prev_path} not found; skipping cross-check")

table.round(2).to_csv(f"{outdir}/topic_persistence_table.csv", index=False)
with open(f"{outdir}/topic_persistence_table.tex", "w", encoding="utf-8") as f:
    f.write(table.to_latex(
        index=False, float_format="%.2f",
        caption=("Topic-level persistence from wave 1 to wave 2. "
                 "``Retained in W2'' is the fraction of participants who had the "
                 "topic in wave 1 and also had it in wave 2; ``Acquired in W2'' is "
                 "the fraction of participants who did NOT have it in wave 1 but "
                 "had it in wave 2. Their difference isolates persistence from the "
                 "topic's overall prevalence. Topic $-1$ is the BERTopic outlier "
                 "topic."),
        label="tab:topic_persistence2", escape=True))
print(f"\nSaved {outdir}/topic_persistence_table.csv and .tex")

# -------------------------
# Participant-level: how many topics are new in wave 2?
# -------------------------
# The topic-level table above asks, per topic, how often it is retained or
# acquired. A reviewer could equally mean the participant-level reading: how
# much of an individual's wave-2 belief set is new. Printed only, not saved.
w1_topics = w1.groupby("key")["topic"].apply(set).reindex(participants)
w2_topics = w2.groupby("key")["topic"].apply(set).reindex(participants)

assert w1_topics.notna().all() and w2_topics.notna().all()

per = pd.DataFrame({"key": participants})
per["n_w1"] = [len(s) for s in w1_topics]
per["n_w2"] = [len(s) for s in w2_topics]
per["n_new"] = [len(b - a) for a, b in zip(w1_topics, w2_topics)]

# fraction of the wave-2 topic set that was not present in wave 1
per["frac_of_w2"] = per["n_new"] / per["n_w2"]
per["any_new"] = (per["n_new"] > 0).astype(int)

summary = (per[["n_w1", "n_w2", "n_new", "frac_of_w2", "any_new"]]
           .agg(["mean", "std", "min", "max"]).T
           .rename(index={
               "n_w1":       "Topics in W1",
               "n_w2":       "Topics in W2",
               "n_new":      "New topics in W2 (count)",
               "frac_of_w2": "New / topics in W2 (proportion)",
               "any_new":    "Has any new topic (0/1)",
           }))

print(f"\n=== Participant-level (N={len(per)}) ===")
print(summary.round(2).to_string())

n_any = int(per["any_new"].sum())
print(f"\nParticipants with at least one new topic in W2: "
      f"{n_any} of {len(per)} ({n_any / len(per) * 100:.1f}%)")
print(f"Participants with no new topics:                 {len(per) - n_any}")

# does the count of new topics just track how much a participant mentions?
r_size = per["n_new"].corr(per["n_w2"])
print(f"\ncorr(new topics, topics in W2) = {r_size:+.2f}")
