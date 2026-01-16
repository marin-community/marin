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

This module provides a unified registry of raw dataset downloads and their canonical
tokenizations. All datasets support a consistent FAMILY:SPLIT syntax in the CLI.

Dataset families are organized in separate modules:
- dolma: DOLMA 1.7 (15 splits)
- dolmino: DOLMINO (12 splits + combined math)
- nemotron: NEMOTRON CC (7 quality-based splits)
- simple: Single-corpus datasets

Use `python -m experiments.pretraining_datasets list` to see all available datasets.
"""

# Import download functions and tokenization helpers from each module
from experiments.pretraining_datasets.dolma import (
    DOLMA_DATASETS,
    DOLMA_LLAMA3_OVERRIDES,
    DOLMA_OLMO_MIXTURE_WEIGHTS,
    dolma_download,
    tokenize_dolma,
)
from experiments.pretraining_datasets.dolmino import (
    DOLMINO_DATASETS,
    DOLMINO_LLAMA3_OVERRIDES,
    dolmino_download,
    tokenize_dolmino,
    tokenize_dolmino_math,
    tokenize_dolmino_subset,
)
from experiments.pretraining_datasets.nemotron import (
    NEMOTRON_DATASETS,
    NEMOTRON_LLAMA3_OVERRIDES,
    NEMOTRON_WEIGHTS,
    nemotron_cc_download,
    tokenize_nemotron,
    tokenize_nemotron_subset,
)
from experiments.pretraining_datasets.simple import (
    dclm_baseline_download,
    fineweb_edu_download,
    proofpile_2_download,
    slimpajama_6b_download,
    starcoderdata_download,
    tokenize_simple,
)
from experiments.llama import llama3_tokenizer

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
        "download": dclm_baseline_download,
        "tokenize_fn": lambda: {
            "dclm_baseline/all": tokenize_simple("dclm_baseline", dclm_baseline_download(), tokenizer=llama3_tokenizer)
        },
    },
    "starcoderdata": {
        "subsets": ["all"],
        "download": starcoderdata_download,
        "tokenize_fn": lambda: {
            "starcoderdata/all": tokenize_simple("starcoderdata", starcoderdata_download(), tokenizer=llama3_tokenizer)
        },
    },
    "proofpile_2": {
        "subsets": ["all"],
        "download": proofpile_2_download,
        "tokenize_fn": lambda: {
            "proofpile_2/all": tokenize_simple("proofpile_2", proofpile_2_download(), tokenizer=llama3_tokenizer)
        },
    },
    "slimpajama_6b": {
        "subsets": ["all"],
        "download": slimpajama_6b_download,
        "tokenize_fn": lambda: {
            "slimpajama_6b/all": tokenize_simple("slimpajama_6b", slimpajama_6b_download(), tokenizer=llama3_tokenizer)
        },
    },
    "fineweb_edu": {
        "subsets": ["all"],
        "download": fineweb_edu_download,
        "tokenize_fn": lambda: {
            "fineweb_edu/all": tokenize_simple("fineweb_edu", fineweb_edu_download(), tokenizer=llama3_tokenizer)
        },
    },
    # Special combined dataset
    "dolmino_math": {
        "subsets": ["all"],
        "download": dolmino_download,
        "tokenize_fn": lambda: {"dolmino_math/all": tokenize_dolmino_math()},
    },
    # Multi-subset datasets
    "dolmino": {
        "subsets": list(DOLMINO_DATASETS.keys()),
        "download": dolmino_download,
        "tokenize_fn": tokenize_dolmino,
    },
    "nemotron_cc": {
        "subsets": list(NEMOTRON_DATASETS.keys()),
        "download": nemotron_cc_download,
        "tokenize_fn": tokenize_nemotron,
    },
    "dolma": {
        "subsets": list(DOLMA_DATASETS.keys()),
        "download": dolma_download,
        "tokenize_fn": tokenize_dolma,
    },
}
