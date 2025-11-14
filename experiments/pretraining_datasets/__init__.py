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
Canonical raw and tokenized pretraining datasets.

This module provides a registry of raw dataset downloads and their canonical
tokenizations. For multi-split datasets (dolmino, nemotron_cc, dolma), helper
functions generate tokenized versions of individual splits.

Dataset families are organized in separate modules:
- dolma: DOLMA 1.7 (15 splits)
- dolmino: DOLMINO (12 splits + combined math)
- nemotron: NEMOTRON CC (7 quality-based splits)
- simple: Single-corpus datasets

Use `python -m experiments.pretraining_datasets --list` to see all available datasets.
"""

# Import dataset families
from experiments.pretraining_datasets.dolma import (
    DOLMA_DATASETS,
    DOLMA_LLAMA3_OVERRIDES,
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    dolma,
    tokenize_dolma_steps,
)
from experiments.pretraining_datasets.dolmino import (
    DOLMINO_DATASETS,
    DOLMINO_LLAMA3_OVERRIDES,
    dolmino,
    dolmino_math_tokenized_llama3,
    get_dolmino_step,
    tokenize_dolmino_steps,
)
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_WEIGHTS,
    get_nemotron_step,
    nemotron_cc,
    tokenize_nemotron_steps,
)
from experiments.pretraining_datasets.simple import (
    dclm_baseline,
    dclm_baseline_tokenized_llama3,
    dclm_baseline_wrong,
    fineweb,
    fineweb_edu,
    fineweb_edu_tokenized_llama3,
    proofpile_2,
    proofpile_2_tokenized_llama3,
    slimpajama,
    slimpajama_6b,
    slimpajama_6b_tokenized_llama3,
    starcoderdata,
    starcoderdata_tokenized_llama3,
    the_pile_openwebtext2,
    the_stack_dedup,
)

# Re-export all commonly used items for backward compatibility
__all__ = [
    "DOLMA_DATASETS",
    "DOLMA_LLAMA3_OVERRIDES",
    "DOLMA_OLMO_MIXTURE_WEIGHTS",
    "DOLMINO_DATASETS",
    "DOLMINO_LLAMA3_OVERRIDES",
    "MULTI_SPLIT_DATASETS",
    "NEMOTRON_DATASETS",
    "NEMOTRON_LLAMA3_OVERRIDES",
    "NEMOTRON_WEIGHTS",
    "SIMPLE_TOKENIZED_DATASETS",
    "dclm_baseline",
    "dclm_baseline_tokenized_llama3",
    "dclm_baseline_wrong",
    "dolma",
    "dolmino",
    "dolmino_math_tokenized_llama3",
    "fineweb",
    "fineweb_edu",
    "fineweb_edu_tokenized_llama3",
    "get_dolmino_step",
    "get_nemotron_step",
    "nemotron_cc",
    "proofpile_2",
    "proofpile_2_tokenized_llama3",
    "slimpajama",
    "slimpajama_6b",
    "slimpajama_6b_tokenized_llama3",
    "starcoderdata",
    "starcoderdata_tokenized_llama3",
    "the_pile_openwebtext2",
    "the_stack_dedup",
    "tokenize_dolma_steps",
    "tokenize_dolmino_steps",
    "tokenize_nemotron_steps",
]


# ============================================================================
# DATASET REGISTRY
# ============================================================================
# Organize all datasets for easy lookup and CLI access

# Simple tokenized datasets (single corpus, already tokenized)
SIMPLE_TOKENIZED_DATASETS = {
    "dclm_baseline": dclm_baseline_tokenized_llama3,
    "starcoderdata": starcoderdata_tokenized_llama3,
    "proofpile_2": proofpile_2_tokenized_llama3,
    "slimpajama_6b": slimpajama_6b_tokenized_llama3,
    "fineweb_edu": fineweb_edu_tokenized_llama3,
}

# Multi-split dataset metadata
MULTI_SPLIT_DATASETS = {
    "dolmino": {
        "splits": DOLMINO_DATASETS,
        "tokenize_fn": tokenize_dolmino_steps,
    },
    "nemotron_cc": {
        "splits": NEMOTRON_DATASETS,
        "tokenize_fn": tokenize_nemotron_steps,
    },
    "dolma": {
        "splits": DOLMA_DATASETS,
        "tokenize_fn": tokenize_dolma_steps,
    },
}
