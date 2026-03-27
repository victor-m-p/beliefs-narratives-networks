## Directory structure

```
OtreeAnalysis/
  analysis/          # All scripts, run from this directory
  data/public/       # Sanitized data (shared):
                     #   edges_*.csv, interviews_*.csv, nodes.csv
                     #   bertopic/           BERTopic runs and selection
                     #   bertopic_mapping/   canvas edge → topic mappings
                     #   bertopic_mapping_llm/ LLM edge → topic mappings
                     #   llm_extractions/    node/edge JSON from GPT
  fig/               # All generated figures and tables
```

## Data flow

Private raw data flows through three phases to produce public outputs and figures:

```
Phase 0 (sensitive)     Phase 1 (preprocessing)     Phase 2 (analysis + figures)
─────────────────────   ─────────────────────────   ────────────────────────────
00 pre_cleaning ──┐
01 curation ──────┤
02 post_cleaning ─┤
03 llm_nodes ─────┘──── 10 create_safe_json ──┐
                        11 distractors ───────┤
                        12 prep_data ─────────┘──── 20–29 analysis scripts ──── fig/
                                                     30 prepare_nodes ─────┐
                                                     31 bertopic_fit ──────┤
                                                     32 bertopic_select ───┤
                                                     33 bertopic_map ──────┘──── 34–39 scripts ──── fig/
```

## Scripts

Scripts are numbered by phase. The first digit indicates the phase; scripts within a phase run in order.

### Phase 0: Raw data processing (requires private data)

| Script                  | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `00_pre_cleaning_w1.py` | Cleans raw oTree CSV export for wave 1 (gitignored)            |
| `00_pre_cleaning_w2.py` | Cleans raw oTree CSV export for wave 2 (gitignored)            |
| `01_curation.py`        | Parses cleaned CSV into structured JSON per participant        |
| `02_post_cleaning.py`   | Applies manual corrections to curated JSON                     |
| `03_llm_nodes.py`       | Runs GPT-4.1 node and edge extraction on interview transcripts |

### Phase 1: Sanitization and preprocessing

| Script                   | Description                                                                                                                                                    |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `10_create_safe_json.py` | Strips sensitive fields (interviews, demographics, codes, LLM prompts, etc.) from participant JSON; produces public `curation_w*.json` and private transcripts |
| `11_distractors.py`      | Filters to participants passing distractor attention checks; produces `distractors_w*.json`                                                                    |
| `12_prep_data.py`        | Extracts flat CSVs from filtered JSON: `edges_*.csv`, `interviews_w*.csv`                                                                                      |

### Phase 2: Analysis and figures

| Script                     | Output directory                                                     | Manuscript figure(s)    | Description                                                              |
| -------------------------- | -------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `20_pagetimes.py`          | `fig/pagetimes/`                                                     | —                       | Survey completion time distributions (diagnostic)                        |
| `21_rating_interviews.py`  | `fig/ratings/`                                                       | SI Figure S10 (top)     | Interview rating distributions (overall, relevance, ease, etc.)          |
| `22_rating_nodes.py`       | `fig/ratings/`                                                       | SI Figure S10 (bottom)  | Node accuracy ratings (real vs fake distractor summaries)                |
| `23_rating_networks.py`    | `fig/ratings/`                                                       | Figure 5A, 5B           | Network comparison ratings: canvas vs random, LLM vs random              |
| `24_training.py`           | `fig/training/`                                                      | Figure 9                | Training task accuracy across trials (support vs conflict edges)         |
| `25_concurrent.py`         | `fig/concurrent/`                                                    | Figure 6                | Concurrent validity: canvas vs pairwise and canvas vs LLM heatmaps       |
| `27_reliability.py`        | `fig/reliability/`                                                   | Figure 3, Table S4      | Test-retest reliability: scatter panels (words, nodes, edges) + stats    |
| `28_canvas_distance.py`    | `fig/canvas_distance/`                                               | SI Figure S11           | Canvas distance analysis: connected nodes placed closer?                 |
| `29_network_plots.py`      | `fig/networks/`                                                      | Figure 2                | Individual belief network visualizations for all participants            |
| `30_prepare_nodes.py`      | `data/public/`                                                       | —                       | Prepares `nodes.csv` for BERTopic (flags canvas presence)                |
| `31_bertopic_fit.py`       | `data/public/bertopic/`                                              | —                       | BERTopic grid search across 4 embedding models and ~18 parameter combos  |
| `32_bertopic_select.py`    | `data/public/bertopic/selection/`                                    | —                       | Selects top-10 BERTopic runs by DBCV score                               |
| `33_bertopic_map.py`       | `data/public/bertopic_mapping/`, `data/public/bertopic_mapping_llm/` | —                       | Maps top-10 topics onto canvas and LLM edges separately                  |
| `34_bertopic_plot.py`      | `fig/bertopic_mapping/`                                              | —                       | 2×2 per-participant stance/topic network plots (canvas and LLM versions) |
| `35_bertopic_tables.py`    | `fig/bertopic_mapping/`                                              | SI Table S1.6           | BERTopic topic overview (keywords, examples) as LaTeX longtable          |
| `36_node_reliability.py`   | `fig/node_reliability/`                                              | Figure 4, Table S6      | Node-topic test-retest reliability: phi coefficients, all 10 models      |
| `37_edge_reliability.py`   | `fig/edge_reliability/`                                              | Figure 4, Table S7      | Edge-topic test-retest reliability: phi coefficients + contingency table |
| `38_criterion.py`          | `fig/criterion/`                                                     | Figure 7, SI Figure S12 | Criterion validity: topic persistence by W1 topic degree                 |
| `39_collective_network.py` | `fig/collective_network/`                                            | Figure 8                | Population-level topic network (spectral ring layout)                    |

### Shared modules

| File               | Used by                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `utilities.py`     | All scripts (constants, path helpers, embedding model specs)                                 |
| `helpers.py`       | `01`, `12`, `23`, `25`, `27`, `28`, `29`, `36`, `37`, `38`, `39` (data extraction, plotting) |
| `llm_utilities.py` | `03` (OpenAI API calls, prompt templates, Pydantic response models)                          |

## Reproduction

All scripts are designed to be run from the `analysis/` directory. Phase 0 requires private data and API keys. Phases 1–2 can be rerun from the public sanitized JSON files.

```bash
cd OtreeAnalysis/analysis
python 20_pagetimes.py   # example: regenerates fig/pagetimes/
```

Phase 2 scripts (31–39) for topic modelling require `sentence-transformers`, `bertopic`, `umap-learn`, and `hdbscan`. The BERTopic grid search (`31`) takes approximately 30–60 minutes depending on hardware.
