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

"""Dataset definitions for three-phase data mixture swarm experiments.

Defines three data partitions:
1. Pretrain: Nemotron CC high-quality splits (no synthetic)
2. Midtrain: FineWeb-Edu
3. SFT: Math-focused instruction datasets with ChatLmDatasetFormat

IMPORTANT: All datasets use marin_tokenizer for consistency, which includes a chat template
required for ChatLmDatasetFormat in SFT datasets.
"""

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.pretraining_datasets.nemotron import tokenize_nemotron, NEMOTRON_WEIGHTS
from experiments.pretraining_datasets.simple import downloads as simple_downloads, _tokenize_simple
from marin.execution.executor import ExecutorStep


# ============================================================================
# PARTITION 1: PRETRAIN (Nemotron CC - high quality, no synthetic)
# ============================================================================

# Lazy initialization to avoid import-time execution
_nemotron_tokenized = None


def get_nemotron_tokenized():
    """Get Nemotron tokenized datasets (lazy initialization).

    Uses marin_tokenizer for consistency with SFT datasets (which require chat template).
    """
    global _nemotron_tokenized
    if _nemotron_tokenized is None:
        _nemotron_tokenized = tokenize_nemotron(tokenizer=marin_tokenizer)
    return _nemotron_tokenized


# Pretrain components: high-quality Nemotron splits (no synthetic)
PRETRAIN_SPLIT_NAMES = ["hq_actual", "medium_high", "medium"]

PRETRAIN_COMPONENTS = {
    f"nemotron_cc/{name}": lambda n=name: get_nemotron_tokenized()[f"nemotron_cc/{n}"]
    for name in PRETRAIN_SPLIT_NAMES
}

# Weights for pretrain partition (TiB-based from NEMOTRON_WEIGHTS)
PRETRAIN_WEIGHTS = {
    f"nemotron_cc/{name}": NEMOTRON_WEIGHTS[f"nemotron_cc/{name}"] for name in PRETRAIN_SPLIT_NAMES
}

# Total pretrain weight for normalization
PRETRAIN_TOTAL_WEIGHT = sum(PRETRAIN_WEIGHTS.values())


# ============================================================================
# PARTITION 2: MIDTRAIN (FineWeb-Edu)
# ============================================================================

# Lazy initialization for FineWeb-Edu with marin_tokenizer
_fineweb_edu_tokenized = None


def get_fineweb_edu_tokenized() -> ExecutorStep:
    """Get FineWeb-Edu tokenized with marin_tokenizer (lazy initialization).

    Uses marin_tokenizer for consistency with SFT datasets (which require chat template).
    """
    global _fineweb_edu_tokenized
    if _fineweb_edu_tokenized is None:
        _fineweb_edu_tokenized = _tokenize_simple(
            "fineweb-edu-marin",
            simple_downloads["fineweb_edu"],
            tokenizer=marin_tokenizer,
        )
    return _fineweb_edu_tokenized


MIDTRAIN_COMPONENTS = {
    "fineweb_edu": get_fineweb_edu_tokenized,
}

# FineWeb-Edu is ~1.3T tokens, but we use weight 1.0 since it's a single component
MIDTRAIN_WEIGHTS = {
    "fineweb_edu": 1.0,
}


# ============================================================================
# PARTITION 3: SFT (Math-focused instruction datasets)
# ============================================================================

# SFT dataset definitions: HuggingFace repo IDs and pre-tokenized paths
# Using already-tokenized datasets to avoid rate limiting on HuggingFace
SFT_DATASET_IDS = {
    "tulu_3_sft_mixture": "allenai/tulu-3-sft-mixture",
    "openthoughts_114k_math": "open-r1/OpenThoughts-114k-math",
    "verifiable_math_problems": "PrimeIntellect/verifiable-math-problems",
}

# Pre-tokenized paths for SFT datasets (with marin_tokenizer)
SFT_TOKENIZED_PATHS = {
    "tulu_3_sft_mixture": "tokenized/tulu_3_sft_mixture_marin_tokenizer-c0f545",
    "openthoughts_114k_math": "tokenized/openthoughts_114k_math_marin_tokenizer-2ec574",
    "verifiable_math_problems": "tokenized/verifiable_math_problems_marin_tokenizer-a665df",
}


def create_sft_tokenization_step(dataset_name: str, hf_dataset_id: str) -> ExecutorStep:
    """Create a tokenization step for an SFT dataset using ChatLmDatasetFormat.

    Uses pre-tokenized paths when available to avoid HuggingFace rate limiting.

    Args:
        dataset_name: Short name for the dataset (used in output path).
        hf_dataset_id: HuggingFace dataset ID.

    Returns:
        ExecutorStep for tokenizing the dataset.
    """
    dataset = get_instruction_dataset(hf_dataset_id, splits=["train"])
    step = default_tokenize(
        name=f"three_phase_swarm/sft/{dataset_name}",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),  # Masks user turns by default
    )

    # Use pre-tokenized path if available
    if dataset_name in SFT_TOKENIZED_PATHS:
        step = step.with_output_path(SFT_TOKENIZED_PATHS[dataset_name])

    return step


# Lazy initialization for SFT components
_sft_components = None


def get_sft_components() -> dict[str, ExecutorStep]:
    """Get SFT tokenization steps (lazy initialization)."""
    global _sft_components
    if _sft_components is None:
        _sft_components = {
            name: create_sft_tokenization_step(name, hf_id)
            for name, hf_id in SFT_DATASET_IDS.items()
        }
    return _sft_components


SFT_COMPONENTS = {name: lambda n=name: get_sft_components()[n] for name in SFT_DATASET_IDS}

# Sample counts for weighting within SFT partition
# These are approximate counts from the datasets
SFT_SAMPLE_COUNTS = {
    "tulu_3_sft_mixture": 939343,
    "openthoughts_114k_math": 89120,
    "verifiable_math_problems": 777457,
}

# Total SFT weight for normalization
SFT_TOTAL_WEIGHT = sum(SFT_SAMPLE_COUNTS.values())


# ============================================================================
# ALL COMPONENTS COMBINED
# ============================================================================


def get_all_components() -> dict[str, ExecutorStep]:
    """Get all dataset components for the mixture.

    Returns:
        Dictionary mapping component names to their ExecutorSteps.
    """
    components = {}

    # Pretrain components
    nemotron_tok = get_nemotron_tokenized()
    for name in PRETRAIN_SPLIT_NAMES:
        key = f"nemotron_cc/{name}"
        components[key] = nemotron_tok[key]

    # Midtrain components
    components["fineweb_edu"] = get_fineweb_edu_tokenized()

    # SFT components
    sft_comps = get_sft_components()
    components.update(sft_comps)

    return components


def expand_partition_weights(partition_weights: dict[str, float]) -> dict[str, float]:
    """Expand high-level partition weights to individual component weights.

    Args:
        partition_weights: Dictionary with keys "pretrain", "midtrain", "sft"
            and values summing to 1.0.

    Returns:
        Dictionary mapping individual component names to their weights.
    """
    weights = {}

    # Pretrain partition - distribute by Nemotron split sizes
    pretrain_w = partition_weights.get("pretrain", 0.0)
    for name in PRETRAIN_SPLIT_NAMES:
        key = f"nemotron_cc/{name}"
        weights[key] = pretrain_w * PRETRAIN_WEIGHTS[key] / PRETRAIN_TOTAL_WEIGHT

    # Midtrain partition - single component
    midtrain_w = partition_weights.get("midtrain", 0.0)
    weights["fineweb_edu"] = midtrain_w

    # SFT partition - distribute by sample counts
    sft_w = partition_weights.get("sft", 0.0)
    for name in SFT_DATASET_IDS:
        weights[name] = sft_w * SFT_SAMPLE_COUNTS[name] / SFT_TOTAL_WEIGHT

    return weights


# Expose lazy-evaluated ALL_COMPONENTS for backward compatibility
ALL_COMPONENTS = get_all_components
