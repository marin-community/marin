# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Canonical raw and tokenized pretraining datasets.

Dataset families are organized in separate modules:
- dolma: DOLMA 1.7 (15 splits)
- dolmino: DOLMINO (12 splits + combined math)
- nemotron: NEMOTRON CC v1 (7 quality-based splits)
- nemotron_v2: Nemotron v2 collection (CC v2/v2.1, Code, Math, Specialized, SFT)
- simple: Single-corpus datasets
"""

from experiments.pretraining_datasets.dolma import (
    DOLMA_LLAMA3_OVERRIDES,
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    tokenize_dolma,
)
from marin.datakit.download.dolma import DOLMA_DATASETS
from experiments.pretraining_datasets.dolmino import (
    DOLMINO_DATASETS,
    DOLMINO_LLAMA3_OVERRIDES,
    tokenize_dolmino,
    tokenize_dolmino_math,
    tokenize_dolmino_subset,
)
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_WEIGHTS,
    nemotron_mix,
    nemotron_mix_block_shuffle,
    tokenize_nemotron,
    tokenize_nemotron_subset,
)
from experiments.pretraining_datasets.nemotron_v2 import (
    NEMOTRON_V2_DATASETS,
    tokenize_nemotron_v2_family,
)
from experiments.pretraining_datasets.nsf_awards import (
    nsf_awards_download,
    nsf_awards_tokenized,
)

__all__ = [
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
