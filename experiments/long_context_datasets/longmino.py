# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Longmino (dolma3_longmino_pool) dataset with length-bucket breakdowns."""

from marin.execution.artifact import Dataset
from marin.execution.lazy import Lazy
from marin.experiment.data import hf_download, tokenized

from experiments.llama import llama3_tokenizer

# Bucket name to HuggingFace dataset shard suffix. The repo uses 2eX notation (powers of two).
_LONGMINO_BUCKET_DESCS: dict[str, str] = {
    # they use 2eX notation but seem to mean powers of two
    "8k-16k": "2e13",
    "16k-32k": "2e14",
    "32k-64k": "2e15",
    "64k-128k": "2e16",
    "128k-256k": "2e17",
    "256k-512k": "2e18",
    "512k-1M": "2e19",
    "1M+": "2e20",
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


def longmino_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, Lazy[Dataset]]:
    """Tokenized Longmino buckets, keyed by length-bucket name."""
    raw = hf_download(
        "dolma3_longmino_pool",
        hf_id="allenai/dolma3_longmino_pool",
        revision="bb7828777019d4a2a0bfd81412a11395d09f705f",
        pin="dolma3_longmino_pool",
    )
    return {
        name: tokenized(
            f"dolma3_longmino/{name}",
            tokenizer=tokenizer,
            raw=raw,
            glob=f"data/*-{desc}/*.jsonl.zst",
        )
        for name, desc in _LONGMINO_BUCKET_DESCS.items()
    }
