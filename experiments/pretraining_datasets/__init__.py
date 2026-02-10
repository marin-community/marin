# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Canonical raw and tokenized pretraining datasets.

This module provides a unified registry of raw dataset downloads and their canonical
tokenizations. All datasets support a consistent FAMILY:SPLIT syntax in the CLI.

Dataset families are organized in separate modules:
- dolma: DOLMA 1.7 (15 splits)
- dolmino: DOLMINO (12 splits + combined math)
- nemotron: NEMOTRON CC (7 quality-based splits)
- simple: Single-corpus datasets

Use `python -m experiments.pretraining_datasets list` to see all available datasets.
"""

# Import downloads and tokenized dicts from each module
from experiments.pretraining_datasets.dolma import (
    DOLMA_DATASETS,
    DOLMA_LLAMA3_OVERRIDES,
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    downloads as dolma_downloads,
    tokenize_dolma,
)
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
    downloads as nemotron_downloads,
    tokenize_nemotron,
    tokenize_nemotron_subset,
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
    "NEMOTRON_WEIGHTS",
    "tokenize_dolma",
    "tokenize_dolmino",
    "tokenize_dolmino_math",
    "tokenize_dolmino_subset",
    "tokenize_nemotron",
    "tokenize_nemotron_subset",
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
    # Multi-subset datasets
    "dolmino": {
        "subsets": list(DOLMINO_DATASETS.keys()),
        "download": dolmino_downloads["dolmino"],
        "tokenize_fn": tokenize_dolmino,
    },
    "nemotron_cc": {
        "subsets": list(NEMOTRON_DATASETS.keys()),
        "download": nemotron_downloads["nemotron_cc"],
        "tokenize_fn": tokenize_nemotron,
    },
    "dolma": {
        "subsets": list(DOLMA_DATASETS.keys()),
        "download": dolma_downloads["dolma"],
        "tokenize_fn": tokenize_dolma,
    },
}
