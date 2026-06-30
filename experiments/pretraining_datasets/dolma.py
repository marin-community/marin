# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DOLMA 1.7 dataset as lazy ``Dataset`` handles (the flat dataset catalog).

One handle per quality split, tokenizing each split's glob from the fixed raw
path. The llama3-tokenized splits are pinned to their existing caches so
referencing them never re-tokenizes the multi-TiB corpus. This is the catalog
only — handles, no weights; the mixture weights are policy and live in the
experiment that chooses them.
"""

from marin.datakit.download.dolma import DOLMA_DATASETS
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# Dolma 1.7 is stored at a fixed path that predates the versioning system.
_DOLMA_V1_7 = "raw/dolma/v1.7"

# Sampling proportions from https://huggingface.co/datasets/allenai/dolma
DOLMA_OLMO_MIXTURE_WEIGHTS = {
    "dolma/algebraic-stack": 12.6,
    "dolma/arxiv": 28.0,
    "dolma/gutenberg": 5.3,
    "dolma/c4": 124.95,
    "dolma/cc": 597.75,
    "dolma/cc-news": 14.3,
    "dolma/falcon": 456.4,
    "dolma/megawika": 4.6,
    "dolma/open-web-math": 12.6,
    "dolma/pes2o": 57.2,
    "dolma/reddit": 79.9,
    "dolma/stackexchange": 19.6,
    "dolma/starcoder": 263.8,
    "dolma/flan": 16.5,
    "dolma/wiki": 7.4,
}

# NB: hashes predate a hashing change; never recompute these caches.
DOLMA_LLAMA3_OVERRIDES = {
    "c4": "tokenized/dolma/c4-e0e5ec",
    "cc": "tokenized/dolma/cc-74b017",
    "cc-news": "tokenized/dolma/cc-news-625d3e",
    "falcon": "tokenized/dolma/falcon-da8fd0",
    "flan": "tokenized/dolma/flan-a99cb2",
    "gutenberg": "tokenized/dolma/gutenberg-f9eb99",
    "reddit": "tokenized/dolma/reddit-62a64a",
    "starcoder": "tokenized/dolma/starcoder-8b6089",
    "algebraic-stack": "tokenized/dolma/algebraic-stack-cc00cf",
    "arxiv": "tokenized/dolma/arxiv-07a51f",
    "megawika": "tokenized/dolma/megawika-34abf2",
    "open-web-math": "tokenized/dolma/open-web-math-79823d",
    "pes2o": "tokenized/dolma/pes2o-538363",
    "stackexchange": "tokenized/dolma/stackexchange-adfc49",
    "wiki": "tokenized/dolma/wiki-212315",
}


def tokenize_dolma(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """One :class:`Dataset` handle per Dolma 1.7 split, keyed by ``dolma/<split>``."""
    return {
        f"dolma/{dataset}": tokenized(
            f"dolma/{dataset}",
            tokenizer=tokenizer,
            paths=[f"{_DOLMA_V1_7}/{file}" for file in files],
            pin=DOLMA_LLAMA3_OVERRIDES.get(dataset) if tokenizer == llama3_tokenizer else None,
            version="2026.06.28",
        )
        for dataset, files in DOLMA_DATASETS.items()
    }
