# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    name: dolma3_longmino_pool_raw / f"data/*-{desc}/*.jsonl.zst"
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


# longmino has a *ton* of metadata which makes the usual "compressed bytes ≈ tokens" heuristic not great
# instead, we rely on Olmo 3's token assessments
# https://www.datocms-assets.com/64837/1763662397-1763646865-olmo_3_technical_report-1.pdf
longmino_bucket_token_counts = {
    "8k-16k": 144e9,
    "16k-32k": 118e9,
    "32k-64k": 8.77e9 + 24.1e9 + 106e9,  # 138.87B
    "64k-128k": 96e9,
    "128k-256k": 60.8e9,
    "256k-512k": 35.1e9,
    "512k-1M": 21.5e9,
    "1M+": 26.9e9,
}


# finepdfs (filtered to English for now)
# HF dataset requires a config; use eng_Latn. Revision hash shortened to 7 chars.
finepdfs_eng_raw = default_download(
    name="finepdfs_eng_Latn",
    hf_dataset_id="HuggingFaceFW/finepdfs",
    revision="d8e8544",
    hf_urls_glob=["data/eng_Latn/train/*.parquet"],
    override_output_path="finepdfs_eng_Latn",
)

finepdfs_by_language = {"eng_Latn": finepdfs_eng_raw / "data/eng_Latn/train/*.parquet"}
finepdfs_validation_by_language = {"eng_Latn": finepdfs_eng_raw.cd("data/eng_Latn/test/*.parquet")}

# Approx tokens: 206,917,202 rows * ~400 tokens/page ≈ 83B (rough, from schema sample).
finepdfs_token_counts = {
    # ~206,917,202 docs * ~3,600 tokens/doc from manual audit ≈ 7.45e11 tokens
    "eng_Latn": 7.45e11
}

# finepdfs-edu (higher quality educational PDFs)
finepdfs_edu_eng_raw = default_download(
    name="finepdfs_edu_eng_Latn",
    hf_dataset_id="HuggingFaceFW/finepdfs-edu",
    revision="9cfabe2",
    hf_urls_glob=["data/eng_Latn/train/*.parquet"],
    override_output_path="finepdfs_edu_eng_Latn",
)

finepdfs_edu_by_language = {"eng_Latn": finepdfs_edu_eng_raw / "data/eng_Latn/train/*.parquet"}
# ~140B tokens for English
finepdfs_edu_token_counts = {"eng_Latn": 140e9}
