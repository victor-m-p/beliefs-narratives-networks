
# questionnaire
VEMI_ITEMS = [
        ("I want to be healthy", "H"),
        ("Plant-based diets are better for the environment", "E"),
        ("Animals do not have to suffer", "A"),
        ("Animals’ rights are respected", "A"),
        ("I want to live a long time", "H"),
        ("Plant-based diets are more sustainable", "E"),
        ("I care about my body", "H"),
        ("Eating meat is bad for the planet", "E"),
        ("Animal rights are important to me", "A"),
        ("Plant-based diets are environmentally-friendly", "E"),
        ("It does not seem right to exploit animals", "A"),
        ("Plants have less of an impact on the environment than animal products", "E"),
        ("I am concerned about animal rights", "A"),
        ("My health is important to me", "H"),
        ("I don’t want animals to suffer", "A"),
    ]

MEMI_ITEMS = [
    ("It goes against nature to eat only plants.", "Natural"),
    ("Our bodies need the protein.", "Necessary"),
    ("I want to fit in.", "Normal"),
    ("It is delicious.", "Nice"),
    ("It makes people strong and vigorous.", "Necessary"),
    ("I don’t want other people to be uncomfortable.", "Normal"),
    ("It is in all of the best tasting food.", "Nice"),
    ("It could be unnatural not to eat meat.", "Natural"),
    ("It is necessary for good health.", "Necessary"),
    ("It is just one of the things people do.", "Normal"),
    ("It gives me pleasure.", "Nice"),
    ("I want to be sure I get all of the vitamins and minerals I need.", "Necessary"),
    ("Everybody does it.", "Normal"),
    ("It has good flavor.", "Nice"),
    ("It gives me strength and endurance.", "Necessary"),
    ("I don’t want to stand out.", "Normal"),
    ("Meals without it don’t taste good.", "Nice"),
    ("It is human nature to eat meat.", "Natural"),
    ("Eating meat is part of our biology.", "Natural"),
]

five_point_scale = {
    1: 'Not at all',
    2: 'Slightly',
    3: 'Moderately',
    4: 'Very well',
    5: 'Extremely well'
}

polarity_conversion = {
    0: "No Influence",
    1: "Positive Influence",
    2: "Negative Influence" 
}

# set for individual runs
date_w1 = "2025-12-09" # w1
date_w2 = "2025-12-16" # w2
wave_1 = 1
wave_2 = 2

# ---- NEW DATA PATHS (2026-02-06 refactor) ----
# Helper functions for new private/public data structure

def get_private_path(filename_pattern, wave=None):
    """Get path to private (sensitive) data file.

    Args:
        filename_pattern: e.g., 'curation_w{wave}.json', 'llm_prompts/node_prompts_w{wave}'
        wave: Wave number (1 or 2), will be substituted into {wave} if present

    Returns:
        Path relative to analysis/ folder
    """
    if wave is not None:
        filename_pattern = filename_pattern.format(wave=wave)
    return f"../data/private/{filename_pattern}"

def get_public_path(filename_pattern, wave=None):
    """Get path to public (safe) data file.

    Args:
        filename_pattern: e.g., 'curation_w{wave}.json', 'edges_canvas_w{wave}.csv'
        wave: Wave number (1 or 2), will be substituted into {wave} if present

    Returns:
        Path relative to analysis/ folder
    """
    if wave is not None:
        filename_pattern = filename_pattern.format(wave=wave)
    return f"../data/public/{filename_pattern}"

def get_llm_extraction_path(wave, extraction_type="node_extraction", model="gpt-4.1-2025-04-14"):
    """Get path to LLM extraction output (in public folder).

    Args:
        wave: Wave number (1 or 2)
        extraction_type: 'node_extraction' or 'edge_extraction'
        model: Model name

    Returns:
        Path relative to analysis/ folder
    """
    return f"../data/public/llm_extractions/{extraction_type}_w{wave}/{model}"


# ---- EMBEDDINGS -----
# Embedding model specs used across the project
EMB_MODELS = [
     dict(
        name="qwen3-embedding-0.6b",
        hf="Qwen/Qwen3-Embedding-0.6B",
        prefix=None,
        encode_normalize=False
        ),

    # https://huggingface.co/BAAI/bge-large-en-v1.5
    dict(
        name="bge-large-en-v1.5",
        hf="BAAI/bge-large-en-v1.5",
        prefix=None,
        encode_normalize=False
        ),

    # historical baselines
    # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
     dict(
        name="all-MiniLM-L6-v2",
        hf="sentence-transformers/all-MiniLM-L6-v2",
        prefix=None,
        encode_normalize=False
        ),

    # https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
    dict(
        name="bert-base-nli-mean-tokens",
        hf="sentence-transformers/bert-base-nli-mean-tokens",
        prefix=None,
        encode_normalize=False
        ),
]
