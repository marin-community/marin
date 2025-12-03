"""
Various long context datasets for post-training experiments.
"""
from experiments.defaults import default_download


# Institutional Books 1.0 dataset
institutional_books_raw = default_download(
    name="raw/institutional-books-1.0",
    hf_dataset_id="institutional/institutional-books-1.0",
    revision="d2f504a",
    override_output_path="raw/institutional-books-d2f504a",
)

# Longmino (whole pool

# (this is everything)
dolma3_longmino_pool_raw = default_download(
    name="dolma3_longmino_pool",
    hf_dataset_id="allenai/dolma3_longmino_pool",
    revision="bb7828777019d4a2a0bfd81412a11395d09f705f",
    override_output_path="dolma3_longmino_pool",
)


longmino_by_bucket = {
    name: dolma3_longmino_pool_raw / f"*-{desc}/*.jsonl.zst"
    for name, desc in {
        # they use 2eX notation but seem to mean powers of two
        "8k-16k": "2e13",
        "16k-32k": "2e14",
        "32k-64k": "2e15",
        "64k-128k": "2e16",
        "128k-256k": "2e17",
        "256k-512k": "2e18",
        "512k-1M": "2e19",
        "1M+": "2e20",
    }.items()
}


# longmino has a *ton* of metadata which makes the usual "comprsesed bytes ≈ tokens" heuristic not great
# instead, we rely on Olmo 3's token assessments
# https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf
longmino_bucket_token_counts = {
    "8k-16k": 144e9,
    "16k-32k": 118e9,
    "32k-64k": (8.77e9 + 24.1e9 + 106e9),  # 138.87B
    "64k-128k": 96e9,
    "128k-256k": 60.8e9,
    "256k-512k": 35.1e9,
    "512k-1M": 21.5e9,
    "1M+": 26.9e9,
}


