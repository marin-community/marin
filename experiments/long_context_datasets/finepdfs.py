# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
FinePDFs and FinePDFs-edu dataset definitions for long-context experiments.

Uses the same HuggingFace repo (HuggingFaceFW/finepdfs) as pretraining_datasets/finepdfs.py
but exposes path-based dicts suitable for mixture configs rather than executor tokenize steps.
"""

from experiments.defaults import default_download

FINEPDFS_HF_ID = "HuggingFaceFW/finepdfs"
FINEPDFS_REVISION = "89f5411"

# --- finepdfs (filtered to English for now) ---

finepdfs_eng_raw = default_download(
    name="finepdfs_eng_Latn",
    hf_dataset_id=FINEPDFS_HF_ID,
    revision=FINEPDFS_REVISION,
    hf_urls_glob=["data/eng_Latn/*/*.parquet"],
    override_output_path="finepdfs_eng_Latn",
)

finepdfs_by_language = {"eng_Latn": finepdfs_eng_raw / "data/eng_Latn/train/*.parquet"}
finepdfs_validation_by_language = {"eng_Latn": finepdfs_eng_raw.cd("data/eng_Latn/test/*.parquet")}

# ~206,917,202 docs * ~3,600 tokens/doc from manual audit ≈ 7.45e11 tokens
finepdfs_token_counts = {
    "eng_Latn": 7.45e11,
}

# --- finepdfs-edu (higher quality educational PDFs) ---

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
