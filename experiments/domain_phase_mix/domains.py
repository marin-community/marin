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

"""Reusable domain definitions for mixture experiments.

This module provides a registry of pre-defined data domains that can be
composed into different experiments. Each domain groups related datasets
and defines how weights should be distributed within the domain.

Domains are defined lazily to avoid import-time execution of expensive
tokenization/download operations.
"""

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.pretraining_datasets.dolmino import tokenize_dolmino_subset
from experiments.pretraining_datasets.nemotron import tokenize_nemotron, NEMOTRON_WEIGHTS
from experiments.pretraining_datasets.simple import downloads as simple_downloads, _tokenize_simple

from experiments.domain_phase_mix.config import Domain, DatasetComponent

# ============================================================================
# DOMAIN REGISTRY
# ============================================================================

_domain_registry: dict[str, Domain] = {}


def register_domain(domain: Domain) -> Domain:
    """Register a domain in the global registry.

    Args:
        domain: The domain to register.

    Returns:
        The registered domain (for chaining).
    """
    _domain_registry[domain.name] = domain
    return domain


def get_domain(name: str) -> Domain:
    """Get a domain from the registry by name.

    Args:
        name: The domain name.

    Returns:
        The registered domain.

    Raises:
        KeyError: If the domain is not registered.
    """
    if name not in _domain_registry:
        raise KeyError(f"Domain '{name}' not found. Available domains: {list(_domain_registry.keys())}")
    return _domain_registry[name]


def list_domains() -> list[str]:
    """List all registered domain names."""
    return list(_domain_registry.keys())


# ============================================================================
# NEMOTRON CC PRETRAINING DOMAIN
# ============================================================================

# Lazy cache for tokenized datasets
_nemotron_tokenized_cache = None


def _get_nemotron_tokenized():
    """Get Nemotron tokenized datasets with llama3_tokenizer (lazy).

    Uses the default llama3_tokenizer which has pre-existing caches at
    gs://marin-us-central1/tokenized/nemotron_cc/. The llama3_tokenizer
    is equivalent to marin_tokenizer per _are_tokenizers_equivalent.
    """
    global _nemotron_tokenized_cache
    if _nemotron_tokenized_cache is None:
        _nemotron_tokenized_cache = tokenize_nemotron()
    return _nemotron_tokenized_cache


def _nemotron_hq_actual():
    return _get_nemotron_tokenized()["nemotron_cc/hq_actual"]


def _nemotron_medium_high():
    return _get_nemotron_tokenized()["nemotron_cc/medium_high"]


def _nemotron_medium():
    return _get_nemotron_tokenized()["nemotron_cc/medium"]


def _nemotron_hq_synth():
    return _get_nemotron_tokenized()["nemotron_cc/hq_synth"]


def _nemotron_medium_low():
    return _get_nemotron_tokenized()["nemotron_cc/medium_low"]


def _nemotron_low_actual():
    return _get_nemotron_tokenized()["nemotron_cc/low_actual"]


# High-quality Nemotron splits (no synthetic)
NEMOTRON_HQ_DOMAIN = register_domain(
    Domain(
        name="nemotron_hq",
        components=[
            DatasetComponent(
                name="nemotron_cc/hq_actual",
                step_fn=_nemotron_hq_actual,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/hq_actual", 0.91),
            ),
            DatasetComponent(
                name="nemotron_cc/medium_high",
                step_fn=_nemotron_medium_high,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/medium_high", 0.82),
            ),
            DatasetComponent(
                name="nemotron_cc/medium",
                step_fn=_nemotron_medium,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/medium", 3.38),
            ),
        ],
        natural_proportion=0.70,
        description="High-quality Nemotron CC splits (hq_actual, medium_high, medium) - no synthetic data",
    )
)

# Full Nemotron domain (including synthetic and lower quality)
NEMOTRON_FULL_DOMAIN = register_domain(
    Domain(
        name="nemotron_full",
        components=[
            DatasetComponent(
                name="nemotron_cc/hq_actual",
                step_fn=_nemotron_hq_actual,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/hq_actual", 0.91),
            ),
            DatasetComponent(
                name="nemotron_cc/hq_synth",
                step_fn=_nemotron_hq_synth,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/hq_synth", 0.5),
            ),
            DatasetComponent(
                name="nemotron_cc/medium_high",
                step_fn=_nemotron_medium_high,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/medium_high", 0.82),
            ),
            DatasetComponent(
                name="nemotron_cc/medium",
                step_fn=_nemotron_medium,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/medium", 3.38),
            ),
            DatasetComponent(
                name="nemotron_cc/medium_low",
                step_fn=_nemotron_medium_low,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/medium_low", 1.0),
            ),
            DatasetComponent(
                name="nemotron_cc/low_actual",
                step_fn=_nemotron_low_actual,
                weight=NEMOTRON_WEIGHTS.get("nemotron_cc/low_actual", 0.5),
            ),
        ],
        natural_proportion=0.70,
        description="Full Nemotron CC dataset including synthetic and lower quality splits",
    )
)


# ============================================================================
# FINEWEB-EDU DOMAIN
# ============================================================================

_fineweb_edu_cache = None


def _get_fineweb_edu():
    """Get FineWeb-Edu tokenized with marin_tokenizer (lazy)."""
    global _fineweb_edu_cache
    if _fineweb_edu_cache is None:
        _fineweb_edu_cache = _tokenize_simple(
            "fineweb-edu-marin",
            simple_downloads["fineweb_edu"],
            tokenizer=marin_tokenizer,
        )
    return _fineweb_edu_cache


FINEWEB_EDU_DOMAIN = register_domain(
    Domain(
        name="fineweb_edu",
        components=[
            DatasetComponent(
                name="fineweb_edu",
                step_fn=_get_fineweb_edu,
                weight=1.0,
            ),
        ],
        natural_proportion=0.25,
        description="FineWeb-Edu dataset (~1.3T tokens of educational web content)",
    )
)


# ============================================================================
# DOLMINO DOMAIN (Mid-training)
# ============================================================================

# Dolmino is a mid-training corpus with multiple splits, tokenized with llama3_tokenizer
# Caches exist at gs://marin-us-central1/tokenized/dolmino/
# llama3_tokenizer is equivalent to marin_tokenizer per _are_tokenizers_equivalent

_dolmino_cache: dict = {}


def _get_dolmino_split(split: str):
    """Get a Dolmino split tokenized dataset (lazy)."""
    if split not in _dolmino_cache:
        _dolmino_cache[split] = tokenize_dolmino_subset(split)
    return _dolmino_cache[split]


def _dolmino_dclm():
    return _get_dolmino_split("dclm")


def _dolmino_flan():
    return _get_dolmino_split("flan")


def _dolmino_pes2o():
    return _get_dolmino_split("pes2o")


def _dolmino_stackexchange():
    return _get_dolmino_split("stackexchange")


def _dolmino_wiki():
    return _get_dolmino_split("wiki")


# Weights based on token counts from https://huggingface.co/datasets/allenai/dolmino-mix-1124
# Using token counts in billions as weights (proportional sampling)
DOLMINO_WEIGHTS = {
    "dclm": 752.0,  # 752B tokens - HQ web pages
    "flan": 17.0,  # 17B tokens - instruction-tuning data
    "pes2o": 58.6,  # 58.6B tokens - STEM papers
    "stackexchange": 1.26,  # 1.26B tokens - Q&A
    "wiki": 3.7,  # 3.7B tokens - encyclopedic
}
# Total: ~832.56B tokens

# Full Dolmino domain with all non-math splits
DOLMINO_DOMAIN = register_domain(
    Domain(
        name="dolmino",
        components=[
            DatasetComponent(
                name="dolmino/dclm",
                step_fn=_dolmino_dclm,
                weight=DOLMINO_WEIGHTS["dclm"],
            ),
            DatasetComponent(
                name="dolmino/flan",
                step_fn=_dolmino_flan,
                weight=DOLMINO_WEIGHTS["flan"],
            ),
            DatasetComponent(
                name="dolmino/pes2o",
                step_fn=_dolmino_pes2o,
                weight=DOLMINO_WEIGHTS["pes2o"],
            ),
            DatasetComponent(
                name="dolmino/stackexchange",
                step_fn=_dolmino_stackexchange,
                weight=DOLMINO_WEIGHTS["stackexchange"],
            ),
            DatasetComponent(
                name="dolmino/wiki",
                step_fn=_dolmino_wiki,
                weight=DOLMINO_WEIGHTS["wiki"],
            ),
        ],
        natural_proportion=0.25,
        description="Full Dolmino dataset (dclm, flan, pes2o, stackexchange, wiki) for mid-training",
    )
)


# ============================================================================
# SFT DOMAINS
# ============================================================================

# SFT dataset definitions
SFT_DATASETS = {
    "tulu_3_sft_mixture": {
        "hf_id": "allenai/tulu-3-sft-mixture",
        "sample_count": 939343,
        "description": "General instruction tuning mixture",
    },
    "openthoughts_114k_math": {
        "hf_id": "open-r1/OpenThoughts-114k-math",
        "sample_count": 89120,
        "description": "Math reasoning with chain-of-thought",
    },
    "verifiable_math_problems": {
        "hf_id": "PrimeIntellect/verifiable-math-problems",
        "sample_count": 777457,
        "description": "Verifiable math problem solving",
    },
}

# Pre-tokenized paths (if available)
SFT_TOKENIZED_PATHS = {
    "tulu_3_sft_mixture": "tokenized/tulu_3_sft_mixture_marin_tokenizer-c0f545",
    "openthoughts_114k_math": "tokenized/openthoughts_114k_math_marin_tokenizer-2ec574",
    "verifiable_math_problems": "tokenized/verifiable_math_problems_marin_tokenizer-a665df",
}

_sft_cache: dict = {}


def _create_sft_step(dataset_name: str):
    """Create a tokenization step for an SFT dataset."""
    if dataset_name not in _sft_cache:
        hf_id = SFT_DATASETS[dataset_name]["hf_id"]
        dataset = get_instruction_dataset(hf_id, splits=["train"])
        step = default_tokenize(
            name=f"sft/{dataset_name}",
            dataset=dataset / "**/*.jsonl.gz",
            tokenizer=marin_tokenizer,
            format=ChatLmDatasetFormat(),
        )
        # Use pre-tokenized path if available
        if dataset_name in SFT_TOKENIZED_PATHS:
            step = step.with_output_path(SFT_TOKENIZED_PATHS[dataset_name])
        _sft_cache[dataset_name] = step
    return _sft_cache[dataset_name]


def _tulu_3_sft():
    return _create_sft_step("tulu_3_sft_mixture")


def _openthoughts_math():
    return _create_sft_step("openthoughts_114k_math")


def _verifiable_math():
    return _create_sft_step("verifiable_math_problems")


# Math-focused SFT domain
MATH_SFT_DOMAIN = register_domain(
    Domain(
        name="math_sft",
        components=[
            DatasetComponent(
                name="tulu_3_sft_mixture",
                step_fn=_tulu_3_sft,
                weight=SFT_DATASETS["tulu_3_sft_mixture"]["sample_count"],
            ),
            DatasetComponent(
                name="openthoughts_114k_math",
                step_fn=_openthoughts_math,
                weight=SFT_DATASETS["openthoughts_114k_math"]["sample_count"],
            ),
            DatasetComponent(
                name="verifiable_math_problems",
                step_fn=_verifiable_math,
                weight=SFT_DATASETS["verifiable_math_problems"]["sample_count"],
            ),
        ],
        natural_proportion=0.05,
        description="Math-focused SFT datasets (Tulu-3 + math reasoning)",
    )
)

# General SFT domain (just Tulu-3)
GENERAL_SFT_DOMAIN = register_domain(
    Domain(
        name="general_sft",
        components=[
            DatasetComponent(
                name="tulu_3_sft_mixture",
                step_fn=_tulu_3_sft,
                weight=1.0,
            ),
        ],
        natural_proportion=0.05,
        description="General instruction tuning with Tulu-3 mixture",
    )
)


# ============================================================================
# COMPOSITE DOMAIN SETS
# ============================================================================


def get_three_partition_domains() -> list[Domain]:
    """Get the standard 3-partition domain set (pretrain, midtrain, sft).

    This is the default domain configuration for three-phase experiments.
    Uses full Dolmino for mid-training (dclm, flan, pes2o, stackexchange, wiki).

    Returns:
        List of [NEMOTRON_HQ_DOMAIN, DOLMINO_DOMAIN, MATH_SFT_DOMAIN]
    """
    return [NEMOTRON_HQ_DOMAIN, DOLMINO_DOMAIN, MATH_SFT_DOMAIN]


def get_two_partition_domains() -> list[Domain]:
    """Get a simple 2-partition domain set (pretrain + sft).

    Returns:
        List of [NEMOTRON_HQ_DOMAIN, MATH_SFT_DOMAIN]
    """
    return [NEMOTRON_HQ_DOMAIN, MATH_SFT_DOMAIN]
