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

from functools import partial

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.marin_models import MARIN_CHAT_TEMPLATE, marin_tokenizer
from experiments.posttrain.instruction_datasets import get_instruction_dataset
from experiments.pretraining_datasets.dolmino import tokenize_dolmino_subset
from experiments.pretraining_datasets.nemotron import tokenize_nemotron
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


def _nemotron_split(split: str):
    """Get a specific Nemotron split (lazy)."""
    return _get_nemotron_tokenized()[split]


# Token counts from GCS caches (queried via count_tokens.py)
NEMOTRON_TOKENS = {
    "nemotron_cc/hq_actual": 537620495374,  # 537.62B tokens
    "nemotron_cc/hq_synth": 1497529159716,  # 1497.53B tokens
    "nemotron_cc/medium_high": 489053720257,  # 489.05B tokens
    "nemotron_cc/medium": 1960603657130,  # 1960.60B tokens
    "nemotron_cc/medium_low": 860999424951,  # 861.00B tokens
    "nemotron_cc/low_actual": 384102407349,  # 384.10B tokens
}

# High-quality Nemotron splits (no synthetic)
NEMOTRON_HQ_DOMAIN = register_domain(
    Domain(
        name="nemotron_hq",
        components=[
            DatasetComponent(
                name="nemotron_cc/hq_actual",
                step_fn=partial(_nemotron_split, "nemotron_cc/hq_actual"),
                weight=NEMOTRON_TOKENS["nemotron_cc/hq_actual"],
            ),
            DatasetComponent(
                name="nemotron_cc/medium_high",
                step_fn=partial(_nemotron_split, "nemotron_cc/medium_high"),
                weight=NEMOTRON_TOKENS["nemotron_cc/medium_high"],
            ),
            DatasetComponent(
                name="nemotron_cc/medium",
                step_fn=partial(_nemotron_split, "nemotron_cc/medium"),
                weight=NEMOTRON_TOKENS["nemotron_cc/medium"],
            ),
        ],
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
                step_fn=partial(_nemotron_split, "nemotron_cc/hq_actual"),
                weight=NEMOTRON_TOKENS["nemotron_cc/hq_actual"],
            ),
            DatasetComponent(
                name="nemotron_cc/hq_synth",
                step_fn=partial(_nemotron_split, "nemotron_cc/hq_synth"),
                weight=NEMOTRON_TOKENS["nemotron_cc/hq_synth"],
            ),
            DatasetComponent(
                name="nemotron_cc/medium_high",
                step_fn=partial(_nemotron_split, "nemotron_cc/medium_high"),
                weight=NEMOTRON_TOKENS["nemotron_cc/medium_high"],
            ),
            DatasetComponent(
                name="nemotron_cc/medium",
                step_fn=partial(_nemotron_split, "nemotron_cc/medium"),
                weight=NEMOTRON_TOKENS["nemotron_cc/medium"],
            ),
            DatasetComponent(
                name="nemotron_cc/medium_low",
                step_fn=partial(_nemotron_split, "nemotron_cc/medium_low"),
                weight=NEMOTRON_TOKENS["nemotron_cc/medium_low"],
            ),
            DatasetComponent(
                name="nemotron_cc/low_actual",
                step_fn=partial(_nemotron_split, "nemotron_cc/low_actual"),
                weight=NEMOTRON_TOKENS["nemotron_cc/low_actual"],
            ),
        ],
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


# Weight in billions of tokens
FINEWEB_EDU_TOKENS_B = 1300.0  # ~1.3T tokens

FINEWEB_EDU_DOMAIN = register_domain(
    Domain(
        name="fineweb_edu",
        components=[
            DatasetComponent(
                name="fineweb_edu",
                step_fn=_get_fineweb_edu,
                weight=FINEWEB_EDU_TOKENS_B,
            ),
        ],
        # natural_proportion computed from total_weight (~1.3T tokens)
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


def _dolmino_split(split: str):
    """Get a Dolmino split tokenized dataset (lazy)."""
    if split not in _dolmino_cache:
        _dolmino_cache[split] = tokenize_dolmino_subset(split)
    return _dolmino_cache[split]


# Token counts from GCS caches (queried via count_tokens.py)
DOLMINO_TOKENS = {
    "dclm": 746292199610,  # 746.29B tokens - HQ web pages
    "flan": 16971415899,  # 16.97B tokens - instruction-tuning data
    "pes2o": 58517080692,  # 58.52B tokens - STEM papers
    "stackexchange": 1265589187,  # 1.27B tokens - Q&A
    "wiki": 3669138258,  # 3.67B tokens - encyclopedic
}
# Total: ~827B tokens

# Full Dolmino domain with all non-math splits
DOLMINO_DOMAIN = register_domain(
    Domain(
        name="dolmino",
        components=[
            DatasetComponent(
                name="dolmino/dclm",
                step_fn=partial(_dolmino_split, "dclm"),
                weight=DOLMINO_TOKENS["dclm"],
            ),
            DatasetComponent(
                name="dolmino/flan",
                step_fn=partial(_dolmino_split, "flan"),
                weight=DOLMINO_TOKENS["flan"],
            ),
            DatasetComponent(
                name="dolmino/pes2o",
                step_fn=partial(_dolmino_split, "pes2o"),
                weight=DOLMINO_TOKENS["pes2o"],
            ),
            DatasetComponent(
                name="dolmino/stackexchange",
                step_fn=partial(_dolmino_split, "stackexchange"),
                weight=DOLMINO_TOKENS["stackexchange"],
            ),
            DatasetComponent(
                name="dolmino/wiki",
                step_fn=partial(_dolmino_split, "wiki"),
                weight=DOLMINO_TOKENS["wiki"],
            ),
        ],
        description="Full Dolmino dataset (dclm, flan, pes2o, stackexchange, wiki) for mid-training",
    )
)


# ============================================================================
# SFT DOMAINS
# ============================================================================

# Token counts from GCS caches (queried via count_tokens.py)
SFT_TOKENS = {
    # Original datasets (~1.2B total)
    "tulu_3_sft_mixture": 749008790,  # 0.75B tokens
    "openthoughts_114k_math": 72964948,  # 0.07B tokens
    "verifiable_math_problems": 382056624,  # 0.38B tokens
    # Additional SFT datasets (~3B total)
    "acecode_89k": 26032149,  # 0.03B tokens
    "smoltalk": 883494479,  # 0.88B tokens
    "natural_reasoning": 966484170,  # 0.97B tokens
    "dolphin_r1_nonreasoning": 319820708,  # 0.32B tokens
    "dolphin_r1_reasoning": 508743187,  # 0.51B tokens
    "bespoke_stratos_17k": 85724829,  # 0.09B tokens
    # Large reasoning dataset (~17B)
    "openthoughts3_1.2m": 17449811417,  # 17.45B tokens
}

# SFT dataset HuggingFace IDs
# For datasets with custom splits, the HF ID includes the split name (e.g., smoltalk2/split_name)
# For standard datasets, just use the dataset ID and splits=["train"] will be used
SFT_HF_IDS = {
    "tulu_3_sft_mixture": "allenai/tulu-3-sft-mixture",
    "openthoughts_114k_math": "open-r1/OpenThoughts-114k-math",
    "verifiable_math_problems": "PrimeIntellect/verifiable-math-problems",
    "acecode_89k": "TIGER-Lab/AceCode-89K",
    "smoltalk": "HuggingFaceTB/smoltalk",
    "natural_reasoning": "facebook/natural_reasoning",
    "dolphin_r1_nonreasoning": "cognitivecomputations/dolphin-r1-nonreasoning",
    "dolphin_r1_reasoning": "cognitivecomputations/dolphin-r1-reasoning",
    "bespoke_stratos_17k": "bespokelabs/Bespoke-Stratos-17k",
    # smoltalk2 uses named splits, not "train" - the HF ID includes the split name
    "openthoughts3_1.2m": "HuggingFaceTB/smoltalk2/OpenThoughts3_1.2M_think",
}

# Datasets that use named splits instead of "train"
# These are registered in INSTRUCTION_DATASET_NAME_TO_CONFIG with their split as part of the key
SFT_CUSTOM_SPLIT_DATASETS = {
    "openthoughts3_1.2m",  # Uses split name "OpenThoughts3_1.2M_think"
}

# Pre-tokenized paths (if available)
SFT_TOKENIZED_PATHS = {
    "tulu_3_sft_mixture": "tokenized/tulu_3_sft_mixture_marin_tokenizer-c0f545",
    "openthoughts_114k_math": "tokenized/openthoughts_114k_math_marin_tokenizer-2ec574",
    "verifiable_math_problems": "tokenized/verifiable_math_problems_marin_tokenizer-a665df",
    "acecode_89k": "tokenized/acecode_89k_marin_tokenizer-95b190",
    "smoltalk": "tokenized/smoltalk_marin_tokenizer-051688",
    "natural_reasoning": "tokenized/natural_reasoning_marin_tokenizer-63db8d",
    "dolphin_r1_nonreasoning": "tokenized/dolphin_r1_nonreasoning_marin_tokenizer-ea7cd3",
    "dolphin_r1_reasoning": "tokenized/dolphin_r1_reasoning_marin_tokenizer-4938f5",
    "bespoke_stratos_17k": "tokenized/bespoke_stratos_17k_marin_tokenizer-bc8ca4",
    "openthoughts3_1.2m": "tokenized/openthoughts3_1.2m_marin_tokenizer-96c8fa",
}

_sft_cache: dict = {}


def _create_pretokenized_step(cache_path: str, tokenizer: str, dataset_format: ChatLmDatasetFormat):
    """Create a step that references an existing pre-tokenized cache.

    This creates an ExecutorStep with no dependencies that points to
    an existing tokenized cache. The step will not trigger any re-tokenization
    or downloads because:
    1. with_output_path() sets the output to the existing cache location
    2. The executor sees STATUS_SUCCESS at that location and skips execution
    3. The placeholder train_paths won't be used since execution is skipped

    Args:
        cache_path: Path to the pre-tokenized cache (relative to MARIN_PREFIX)
        tokenizer: Tokenizer identifier used for the cache
        dataset_format: Dataset format used during tokenization
    """
    from marin.execution.executor import ExecutorStep, this_output_path, versioned
    from marin.processing.tokenize import TokenizeConfig, tokenize

    config = TokenizeConfig(
        # Placeholder path - won't be used since cache exists with SUCCESS status.
        # We need at least one path due to validation in TokenizeConfig.__post_init__.
        train_paths=["__pretokenized_cache_reference__"],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(tokenizer),
        format=dataset_format,
    )
    # Create step and use with_output_path to point to the existing cache.
    # The executor will find STATUS_SUCCESS at this path and skip execution.
    step = ExecutorStep(
        name=f"sft_pretokenized/{cache_path.split('/')[-1]}",  # Clean name for logging
        fn=tokenize,
        config=config,
    )
    return step.with_output_path(cache_path)


def _sft_step(dataset_name: str):
    """Create a tokenization step for an SFT dataset (lazy).

    If a pre-tokenized cache exists (in SFT_TOKENIZED_PATHS), creates a lightweight
    step that references it without triggering HuggingFace downloads.
    Otherwise, creates the full tokenization pipeline.
    """
    if dataset_name not in _sft_cache:
        if dataset_name not in SFT_HF_IDS:
            raise ValueError(f"SFT dataset {dataset_name} not found in SFT_HF_IDS")

        # Check if pre-tokenized cache exists - if so, use direct reference
        if dataset_name in SFT_TOKENIZED_PATHS:
            step = _create_pretokenized_step(
                cache_path=SFT_TOKENIZED_PATHS[dataset_name],
                tokenizer=marin_tokenizer,
                dataset_format=ChatLmDatasetFormat(chat_template=MARIN_CHAT_TEMPLATE),
            )
        else:
            # Full tokenization pipeline (will download from HF)
            hf_id = SFT_HF_IDS[dataset_name]

            # For datasets with custom splits (like smoltalk2), don't pass splits=["train"]
            # The split is already part of the HF ID
            if dataset_name in SFT_CUSTOM_SPLIT_DATASETS:
                dataset = get_instruction_dataset(hf_id)  # Uses default splits from config
            else:
                dataset = get_instruction_dataset(hf_id, splits=["train"])

            step = default_tokenize(
                name=f"sft/{dataset_name}",
                dataset=dataset / "**/*.jsonl.gz",
                tokenizer=marin_tokenizer,
                format=ChatLmDatasetFormat(chat_template=MARIN_CHAT_TEMPLATE),
            )

        _sft_cache[dataset_name] = step
    return _sft_cache[dataset_name]


# Math-focused SFT domain (original ~1.2B tokens)
MATH_SFT_DOMAIN = register_domain(
    Domain(
        name="math_sft",
        components=[
            DatasetComponent(
                name="tulu_3_sft_mixture",
                step_fn=partial(_sft_step, "tulu_3_sft_mixture"),
                weight=SFT_TOKENS["tulu_3_sft_mixture"],
            ),
            DatasetComponent(
                name="openthoughts_114k_math",
                step_fn=partial(_sft_step, "openthoughts_114k_math"),
                weight=SFT_TOKENS["openthoughts_114k_math"],
            ),
            DatasetComponent(
                name="verifiable_math_problems",
                step_fn=partial(_sft_step, "verifiable_math_problems"),
                weight=SFT_TOKENS["verifiable_math_problems"],
            ),
        ],
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
                step_fn=partial(_sft_step, "tulu_3_sft_mixture"),
                weight=SFT_TOKENS["tulu_3_sft_mixture"],
            ),
        ],
        description="General instruction tuning with Tulu-3 mixture",
    )
)

# OpenThoughts3 SFT domain (~17.5B tokens) - single large reasoning dataset
# This provides a simpler, cleaner SFT setup focused on reasoning
OPENTHOUGHTS_SFT_DOMAIN = register_domain(
    Domain(
        name="openthoughts_sft",
        components=[
            DatasetComponent(
                name="openthoughts3_1.2m",
                step_fn=partial(_sft_step, "openthoughts3_1.2m"),
                weight=SFT_TOKENS["openthoughts3_1.2m"],
            ),
        ],
        description="OpenThoughts3 1.2M reasoning dataset (~17.5B tokens)",
    )
)

# Expanded SFT domain (~21B tokens) - includes all available SFT datasets
# This is ~18x larger than the original MATH_SFT_DOMAIN
EXPANDED_SFT_DOMAIN = register_domain(
    Domain(
        name="expanded_sft",
        components=[
            # Original datasets
            DatasetComponent(
                name="tulu_3_sft_mixture",
                step_fn=partial(_sft_step, "tulu_3_sft_mixture"),
                weight=SFT_TOKENS["tulu_3_sft_mixture"],
            ),
            DatasetComponent(
                name="openthoughts_114k_math",
                step_fn=partial(_sft_step, "openthoughts_114k_math"),
                weight=SFT_TOKENS["openthoughts_114k_math"],
            ),
            DatasetComponent(
                name="verifiable_math_problems",
                step_fn=partial(_sft_step, "verifiable_math_problems"),
                weight=SFT_TOKENS["verifiable_math_problems"],
            ),
            # Additional datasets
            DatasetComponent(
                name="acecode_89k",
                step_fn=partial(_sft_step, "acecode_89k"),
                weight=SFT_TOKENS["acecode_89k"],
            ),
            DatasetComponent(
                name="smoltalk",
                step_fn=partial(_sft_step, "smoltalk"),
                weight=SFT_TOKENS["smoltalk"],
            ),
            DatasetComponent(
                name="natural_reasoning",
                step_fn=partial(_sft_step, "natural_reasoning"),
                weight=SFT_TOKENS["natural_reasoning"],
            ),
            DatasetComponent(
                name="dolphin_r1_nonreasoning",
                step_fn=partial(_sft_step, "dolphin_r1_nonreasoning"),
                weight=SFT_TOKENS["dolphin_r1_nonreasoning"],
            ),
            DatasetComponent(
                name="dolphin_r1_reasoning",
                step_fn=partial(_sft_step, "dolphin_r1_reasoning"),
                weight=SFT_TOKENS["dolphin_r1_reasoning"],
            ),
            DatasetComponent(
                name="bespoke_stratos_17k",
                step_fn=partial(_sft_step, "bespoke_stratos_17k"),
                weight=SFT_TOKENS["bespoke_stratos_17k"],
            ),
            # Large reasoning dataset
            DatasetComponent(
                name="openthoughts3_1.2m",
                step_fn=partial(_sft_step, "openthoughts3_1.2m"),
                weight=SFT_TOKENS["openthoughts3_1.2m"],
            ),
        ],
        description="Expanded SFT datasets (~21B tokens): Tulu-3 + math + code + reasoning",
    )
)


# ============================================================================
# DOLMA DOMAINS (for two-stage experiment replication)
# ============================================================================

# Token counts from GCS caches (gs://marin-us-central1/tokenized/dolma/...)
# Queried using levanter TreeCache on 2025-01-28
DOLMA_TOKENS = {
    "dolma/c4": 134062553328,  # 134.06B tokens
    "dolma/starcoder": 216567300822,  # 216.57B tokens
}

_dolma_cache: dict = {}


def _dolma_split(split: str):
    """Get a Dolma split tokenized dataset (lazy).

    Uses existing caches at gs://marin-us-central1/tokenized/dolma/
    created by experiments/pretraining_datasets/dolma.py.
    """
    if split not in _dolma_cache:
        from experiments.pretraining_datasets import tokenize_dolma

        dolma_components = tokenize_dolma()
        _dolma_cache[split] = dolma_components[split]
    return _dolma_cache[split]


# C4 Common Crawl domain (~134B tokens)
C4_DOMAIN = register_domain(
    Domain(
        name="c4",
        components=[
            DatasetComponent(
                name="dolma/c4",
                step_fn=partial(_dolma_split, "dolma/c4"),
                weight=DOLMA_TOKENS["dolma/c4"],
            ),
        ],
        description="C4 Common Crawl web text from Dolma (~134B tokens)",
    )
)

# StarCoder code domain (~217B tokens)
STARCODER_DOMAIN = register_domain(
    Domain(
        name="starcoder",
        components=[
            DatasetComponent(
                name="dolma/starcoder",
                step_fn=partial(_dolma_split, "dolma/starcoder"),
                weight=DOLMA_TOKENS["dolma/starcoder"],
            ),
        ],
        description="StarCoder code data from Dolma (~217B tokens)",
    )
)


# ============================================================================
# COMPOSITE DOMAIN SETS
# ============================================================================


def get_two_stage_replication_domains() -> list[Domain]:
    """Get domains for replicating the two-stage code experiment.

    Uses C4 (common web data) and StarCoder (code data) from Dolma,
    matching the data sources used in experiments/two_stage/.

    Returns:
        List of [C4_DOMAIN, STARCODER_DOMAIN]
    """
    return [C4_DOMAIN, STARCODER_DOMAIN]


def get_three_partition_domains() -> list[Domain]:
    """Get the standard 3-partition domain set (pretrain, midtrain, sft).

    This is the default domain configuration for three-phase experiments.
    Uses full Nemotron CC (~5.7T tokens) for pretraining and OpenThoughts3 SFT (~17.5B tokens).

    Data ratio: ~5.7T : ~827B : ~17.5B = ~328 : 47 : 1
    Target ratio: 15T : 1T : 10B = 1500 : 100 : 1

    Returns:
        List of [NEMOTRON_FULL_DOMAIN, DOLMINO_DOMAIN, OPENTHOUGHTS_SFT_DOMAIN]
    """
    return [NEMOTRON_FULL_DOMAIN, DOLMINO_DOMAIN, OPENTHOUGHTS_SFT_DOMAIN]


def get_three_partition_domains_expanded_sft() -> list[Domain]:
    """Get a 3-partition domain set with expanded SFT (~21B tokens).

    Uses all available SFT datasets for maximum diversity.

    Data ratio: ~5.7T : ~827B : ~21B = ~270 : 39 : 1

    Returns:
        List of [NEMOTRON_FULL_DOMAIN, DOLMINO_DOMAIN, EXPANDED_SFT_DOMAIN]
    """
    return [NEMOTRON_FULL_DOMAIN, DOLMINO_DOMAIN, EXPANDED_SFT_DOMAIN]


def get_three_partition_domains_small_sft() -> list[Domain]:
    """Get a 3-partition domain set with smaller SFT (for comparison).

    Uses only high-quality Nemotron (~3T tokens) and original small SFT (~1.2B tokens).

    Returns:
        List of [NEMOTRON_HQ_DOMAIN, DOLMINO_DOMAIN, MATH_SFT_DOMAIN]
    """
    return [NEMOTRON_HQ_DOMAIN, DOLMINO_DOMAIN, MATH_SFT_DOMAIN]


def get_two_partition_domains() -> list[Domain]:
    """Get a simple 2-partition domain set (pretrain + sft).

    Returns:
        List of [NEMOTRON_FULL_DOMAIN, OPENTHOUGHTS_SFT_DOMAIN]
    """
    return [NEMOTRON_FULL_DOMAIN, OPENTHOUGHTS_SFT_DOMAIN]
