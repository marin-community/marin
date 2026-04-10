# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Canonical raw and tokenized pretraining datasets.

This module provides a unified registry of raw dataset downloads and their canonical
tokenizations. All datasets support a consistent FAMILY:SPLIT syntax in the CLI.

Dataset families are organized in separate modules:
- dolma: DOLMA 1.7 (15 splits)
- dolmino: DOLMINO (12 splits + combined math)
- nemotron: NEMOTRON CC v1 (7 quality-based splits)
- nemotron_v2: Nemotron v2 collection (CC v2/v2.1, Code, Math, Specialized, SFT)
- simple: Single-corpus datasets

Use `python -m experiments.pretraining_datasets list` to see all available datasets.
"""

# Import downloads and tokenized dicts from each module
from experiments.pretraining_datasets.dolma import (
    DOLMA_LLAMA3_OVERRIDES,
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    downloads as dolma_downloads,
    tokenize_dolma,
)
from marin.datakit.download.dolma import DOLMA_DATASETS
from experiments.pretraining_datasets.dolmino import (
    DOLMINO_DATASETS,
    DOLMINO_LLAMA3_OVERRIDES,
    downloads as dolmino_downloads,
    tokenize_dolmino,
    tokenize_dolmino_math,
    tokenize_dolmino_subset,
)
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_WEIGHTS,
    nemotron_cc_download,
    nemotron_mix,
    nemotron_mix_block_shuffle,
    tokenize_nemotron,
    tokenize_nemotron_subset,
)
from experiments.pretraining_datasets.nemotron_v2 import (
    NEMOTRON_V2_DATASETS,
    downloads as nemotron_v2_downloads,
    tokenize_nemotron_v2_family,
)
from experiments.pretraining_datasets.common_corpus import (
    common_corpus_download,
    tokenize_common_corpus,
)
from experiments.pretraining_datasets.nsf_awards import (
    nsf_awards_download,
    nsf_awards_tokenized,
)
from experiments.pretraining_datasets.simple import downloads as simple_downloads, tokenized as simple_tokenized

# Re-export constants
__all__ = [
    "DATASETS",
    "DOLMA_DATASETS",
    "DOLMA_LLAMA3_OVERRIDES",
    "DOLMA_OLMO_MIXTURE_WEIGHTS",
    "DOLMINO_DATASETS",
    "DOLMINO_LLAMA3_OVERRIDES",
    "NEMOTRON_DATASETS",
    "NEMOTRON_LLAMA3_OVERRIDES",
    "NEMOTRON_V2_DATASETS",
    "NEMOTRON_WEIGHTS",
    "nemotron_mix",
    "nemotron_mix_block_shuffle",
    "nsf_awards_download",
    "nsf_awards_tokenized",
    "tokenize_dolma",
    "tokenize_dolmino",
    "tokenize_dolmino_math",
    "tokenize_dolmino_subset",
    "tokenize_nemotron",
    "tokenize_nemotron_subset",
    "tokenize_nemotron_v2_family",
]


# ============================================================================
# UNIFIED DATASET REGISTRY
# ============================================================================

DATASETS = {
    # Simple datasets (single "all" subset)
    "dclm_baseline": {
        "subsets": ["all"],
        "download": simple_downloads["dclm_baseline"],
        "tokenize_fn": lambda: {"dclm_baseline/all": simple_tokenized["dclm_baseline"]},
    },
    "starcoderdata": {
        "subsets": ["all"],
        "download": simple_downloads["starcoderdata"],
        "tokenize_fn": lambda: {"starcoderdata/all": simple_tokenized["starcoderdata"]},
    },
    "proofpile_2": {
        "subsets": ["all"],
        "download": simple_downloads["proofpile_2"],
        "tokenize_fn": lambda: {"proofpile_2/all": simple_tokenized["proofpile_2"]},
    },
    "slimpajama_6b": {
        "subsets": ["all"],
        "download": simple_downloads["slimpajama_6b"],
        "tokenize_fn": lambda: {"slimpajama_6b/all": simple_tokenized["slimpajama_6b"]},
    },
    "fineweb_edu": {
        "subsets": ["all"],
        "download": simple_downloads["fineweb_edu"],
        "tokenize_fn": lambda: {"fineweb_edu/all": simple_tokenized["fineweb_edu"]},
    },
    # Special combined dataset
    "dolmino_math": {
        "subsets": ["all"],
        "download": dolmino_downloads["dolmino"],
        "tokenize_fn": lambda: {"dolmino_math/all": tokenize_dolmino_math()},
    },
    "common_corpus": {
        "subsets": ["all"],
        "download": common_corpus_download,
        "tokenize_fn": lambda: {"common_corpus/all": tokenize_common_corpus()},
    },
    "nsf_awards": {
        "subsets": ["all"],
        "download": nsf_awards_download,
        "tokenize_fn": lambda: {"nsf_awards/all": nsf_awards_tokenized},
    },
    # Multi-subset datasets
    "dolmino": {
        "subsets": list(DOLMINO_DATASETS.keys()),
        "download": dolmino_downloads["dolmino"],
        "tokenize_fn": tokenize_dolmino,
    },
    "nemotron_cc": {
        "subsets": list(NEMOTRON_DATASETS.keys()),
        "download": nemotron_cc_download(),
        "tokenize_fn": tokenize_nemotron,
    },
    "dolma": {
        "subsets": list(DOLMA_DATASETS.keys()),
        "download": dolma_downloads["dolma"],
        "tokenize_fn": tokenize_dolma,
    },
    # Nemotron v2 datasets (from nvidia/Nemotron-Pre-Training-Datasets collection)
    **{
        family: {
            "subsets": list(info.subsets.keys()),
            "download": nemotron_v2_downloads[family],
            "tokenize_fn": lambda f=family: tokenize_nemotron_v2_family(f),
        }
        for family, info in NEMOTRON_V2_DATASETS.items()
    },
}
