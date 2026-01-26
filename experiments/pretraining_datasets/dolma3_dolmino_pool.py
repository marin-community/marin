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

"""Dolma 3 Dolmino Pool dataset definitions and tokenization.

This module defines 133 partitions from the Dolma 3 Dolmino Pool dataset
(https://huggingface.co/datasets/allenai/dolma3_dolmino_pool).

The dataset contains 2.19T tokens organized into several categories:
- Common Crawl HQ: 48 partitions (24 topics x 2 years: 2019, 2020)
- olmOCR Science PDFs HQ: 23 partitions (23 topics, with 2e12/2e13 size variants merged)
- Stack-Edu FIM: 15 partitions (15 programming languages, shards merged)
- STEM Heavy Crawl: 24 partitions (24 topics)
- Synthetic datasets: 23 partitions

Total: 133 partitions

Note: The original dataset card lists ~135 distinct data sources. The discrepancy of 2
is due to counting methodology:
1. We merge verifiable math into 2 partitions (gpt41, o4mini) instead of 1
2. We include r1_reasoning which may not be in the original count
3. Some data sources in the card may be counted differently (shards vs merged)

Usage:
    from experiments.pretraining_datasets.dolma3_dolmino_pool import (
        download_dolmino_pool,
        tokenize_dolmino_pool,
        tokenize_dolmino_pool_subset,
        DOLMINO_POOL_PARTITIONS,
    )
"""

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

# =============================================================================
# CONSTANTS
# =============================================================================

HF_DATASET_ID = "allenai/dolma3_dolmino_pool"
HF_REVISION = "091589c58ab6acc180d71017ecea8201776f05b2"

# 24 topic categories used across Common Crawl HQ and olmOCR PDFs
TOPICS = [
    "adult_content",
    "art_and_design",
    "crime_and_law",
    "education_and_jobs",
    "electronics_and_hardware",
    "entertainment",
    "fashion_and_beauty",
    "finance_and_business",
    "food_and_dining",
    "games",
    "health",
    "history_and_geography",
    "home_and_hobbies",
    "industrial",
    "literature",
    "politics",
    "religion",
    "science_math_and_technology",
    "social_life",
    "software",
    "software_development",
    "sports_and_fitness",
    "transportation",
    "travel_and_tourism",
]

# Stack-Edu FIM programming languages
STACK_EDU_LANGUAGES = [
    "C",
    "CSharp",
    "Cpp",
    "Go",
    "Java",
    "JavaScript",
    "Markdown",
    "PHP",
    "Python",
    "Ruby",
    "Rust",
    "SQL",
    "Shell",
    "Swift",
    "TypeScript",
]

# =============================================================================
# PARTITION DEFINITIONS
# =============================================================================

# Each partition maps to one or more directories in the HF dataset.
# Format: partition_name -> list of directory globs under data/
# We use glob patterns to handle sharded directories (e.g., stack_edu_fim-Python_*)

DOLMINO_POOL_PARTITIONS: dict[str, list[str]] = {}

# --- Common Crawl HQ (48 partitions: 24 topics x 2 years) ---
# Token estimate: 1.32T total
for year in ["19", "20"]:
    for topic in TOPICS:
        partition_name = f"common_crawl_hq/{year}_{topic}"
        dir_pattern = f"common_crawl-high-quality_{year}_{topic}"
        DOLMINO_POOL_PARTITIONS[partition_name] = [dir_pattern]

# --- olmOCR Science PDFs HQ (23 partitions: 23 topics, merging 2e12/2e13 variants) ---
# Token estimate: 240B total
# Note: "social_life" topic doesn't exist in olmOCR, only 23 topics
OLMOCR_TOPICS = [t for t in TOPICS if t != "social_life"]
for topic in OLMOCR_TOPICS:
    partition_name = f"olmocr_pdfs_hq/{topic}"
    # Merge both 2e12 and 2e13 variants
    DOLMINO_POOL_PARTITIONS[partition_name] = [
        f"olmocr_science_pdfs-high_quality-{topic}-2e12",
        f"olmocr_science_pdfs-high_quality-{topic}-2e13",
    ]

# --- Stack-Edu FIM (15 partitions: 15 programming languages) ---
# Token estimate: 21.4B total
# Each language has multiple sharded directories (e.g., stack_edu_fim-Python_0000 to _0018)
for lang in STACK_EDU_LANGUAGES:
    partition_name = f"stack_edu_fim/{lang}"
    # Use glob pattern to match all shards for this language
    DOLMINO_POOL_PARTITIONS[partition_name] = [f"stack_edu_fim-{lang}_*"]

# --- STEM Heavy Crawl (24 partitions: 24 topics) ---
# Token estimate: 5.21B total
# Files are named by topic: {topic}.{shard}.jsonl.zst (e.g., adult_content.00000000.jsonl.zst)
for topic in TOPICS:
    partition_name = f"stem_heavy_crawl/{topic}"
    # Match files starting with the topic name
    DOLMINO_POOL_PARTITIONS[partition_name] = [f"stem-heavy-crawl/{topic}.*"]

# --- Synthetic Datasets (24 partitions) ---
# QA (synthetic) - 3 sources
DOLMINO_POOL_PARTITIONS["synth_qa/nemotron_synth_qa"] = ["nemotron-synth-qa"]  # 487B
DOLMINO_POOL_PARTITIONS["synth_qa/reddit_to_flashcards"] = ["reddit_to_flashcards"]  # 21.6B
DOLMINO_POOL_PARTITIONS["synth_qa/wiki_to_rcqa"] = [
    "wiki_to_rcqa-part1",
    "wiki_to_rcqa-part2",
    "wiki_to_rcqa-part3",
]  # 4.22B

# Code (synthetic) - 1 source
DOLMINO_POOL_PARTITIONS["synth_code/cranecode"] = ["cranecode"]  # 18.8B

# Thinking/Reasoning (synthetic) - 9 sources
DOLMINO_POOL_PARTITIONS["synth_thinking/llama_nemotron_reasoning"] = ["llama_nemotron-reasoning-traces"]  # 20.9B
DOLMINO_POOL_PARTITIONS["synth_thinking/openthoughts2_reasoning"] = ["openthoughts2-reasoning-traces"]  # 5.6B
DOLMINO_POOL_PARTITIONS["synth_thinking/qwq_reasoning"] = ["qwq-reasoning-traces"]  # 4.77B
DOLMINO_POOL_PARTITIONS["synth_thinking/general_reasoning_mix"] = ["general_reasoning_mix"]  # 2.48B
DOLMINO_POOL_PARTITIONS["synth_thinking/code_meta_reasoning"] = ["code-meta-reasoning"]  # 1.27B
DOLMINO_POOL_PARTITIONS["synth_thinking/math_meta_reasoning"] = ["math-meta-reasoning"]  # 1.05B
DOLMINO_POOL_PARTITIONS["synth_thinking/omr_rewrite_fullthoughts"] = ["omr-rewrite-fullthoughts"]  # 850M
DOLMINO_POOL_PARTITIONS["synth_thinking/program_verifiable"] = ["program_verifiable"]  # 438M
DOLMINO_POOL_PARTITIONS["synth_thinking/gemini_reasoning"] = ["gemini-reasoning-traces"]  # 246M
# Note: r1-reasoning-traces also exists but wasn't in the original table
DOLMINO_POOL_PARTITIONS["synth_thinking/r1_reasoning"] = ["r1-reasoning-traces"]

# Math (synthetic) - 5 sources
DOLMINO_POOL_PARTITIONS["synth_math/dolmino_math"] = ["dolmino-math"]  # 10.7B
DOLMINO_POOL_PARTITIONS["synth_math/cranemath"] = ["cranemath"]  # 5.62B
DOLMINO_POOL_PARTITIONS["synth_math/megamatt"] = ["megamatt"]  # 3.88B
DOLMINO_POOL_PARTITIONS["synth_math/tinymath_mind"] = ["tinymath-mind"]  # 899M
DOLMINO_POOL_PARTITIONS["synth_math/tinymath_pot"] = ["tinymath-pot"]  # 241M
# Verifiable math problems (split into two sources in HF)
DOLMINO_POOL_PARTITIONS["synth_math/verifiable_gpt41"] = ["verifiable-gpt41"]
DOLMINO_POOL_PARTITIONS["synth_math/verifiable_o4mini"] = ["verifiable-o4mini"]

# Instruction (synthetic) - 2 sources
DOLMINO_POOL_PARTITIONS["synth_instruction/dolmino_flan"] = ["dolmino_1-flan"]  # 16.8B
DOLMINO_POOL_PARTITIONS["synth_instruction/tulu_3_sft"] = ["tulu-3-sft"]  # 1.61B

# =============================================================================
# TOKEN COUNTS (estimated from dataset card)
# =============================================================================

# Token counts in billions, from the dataset documentation
# These are approximate and should be verified by running count_tokens.py after tokenization
DOLMINO_POOL_TOKEN_COUNTS_B: dict[str, float] = {
    # Common Crawl HQ - total ~1.32T, distributed across 48 partitions
    # Average ~27.5B per partition, but varies by topic
    # We'll use placeholder values - actual counts should be measured
}

# Populate approximate token counts for Common Crawl HQ
# Total: 1.32T / 48 = ~27.5B average per partition
for partition in DOLMINO_POOL_PARTITIONS:
    if partition.startswith("common_crawl_hq/"):
        DOLMINO_POOL_TOKEN_COUNTS_B[partition] = 27.5  # Approximate, to be measured

# olmOCR PDFs HQ - total ~240B / 23 = ~10.4B average
for partition in DOLMINO_POOL_PARTITIONS:
    if partition.startswith("olmocr_pdfs_hq/"):
        DOLMINO_POOL_TOKEN_COUNTS_B[partition] = 10.4  # Approximate

# Stack-Edu FIM - total ~21.4B / 15 = ~1.4B average
for partition in DOLMINO_POOL_PARTITIONS:
    if partition.startswith("stack_edu_fim/"):
        DOLMINO_POOL_TOKEN_COUNTS_B[partition] = 1.4  # Approximate

# STEM Heavy Crawl - total ~5.21B / 24 = ~0.217B average per topic
for partition in DOLMINO_POOL_PARTITIONS:
    if partition.startswith("stem_heavy_crawl/"):
        DOLMINO_POOL_TOKEN_COUNTS_B[partition] = 0.217  # Approximate

# Individual synthetic datasets (from documentation)
DOLMINO_POOL_TOKEN_COUNTS_B.update(
    {
        "synth_qa/nemotron_synth_qa": 487.0,
        "synth_qa/reddit_to_flashcards": 21.6,
        "synth_qa/wiki_to_rcqa": 4.22,
        "synth_code/cranecode": 18.8,
        "synth_thinking/llama_nemotron_reasoning": 20.9,
        "synth_thinking/openthoughts2_reasoning": 5.6,
        "synth_thinking/qwq_reasoning": 4.77,
        "synth_thinking/general_reasoning_mix": 2.48,
        "synth_thinking/code_meta_reasoning": 1.27,
        "synth_thinking/math_meta_reasoning": 1.05,
        "synth_thinking/omr_rewrite_fullthoughts": 0.85,
        "synth_thinking/program_verifiable": 0.438,
        "synth_thinking/gemini_reasoning": 0.246,
        "synth_thinking/r1_reasoning": 1.0,  # Approximate
        "synth_math/dolmino_math": 10.7,
        "synth_math/cranemath": 5.62,
        "synth_math/megamatt": 3.88,
        "synth_math/tinymath_mind": 0.899,
        "synth_math/tinymath_pot": 0.241,
        "synth_math/verifiable_gpt41": 0.5,  # Approximate
        "synth_math/verifiable_o4mini": 0.5,  # Approximate
        "synth_instruction/dolmino_flan": 16.8,
        "synth_instruction/tulu_3_sft": 1.61,
    }
)

# =============================================================================
# DOWNLOAD STEP
# =============================================================================

# Single download step for the entire dataset
_download_step = ExecutorStep(
    name="raw/dolma3_dolmino_pool",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

# Cache for download step (will be populated after first download with correct hash)
_download_cache: ExecutorStep | None = None


def download_dolmino_pool() -> ExecutorStep:
    """Get the download step for the Dolma 3 Dolmino Pool dataset.

    Returns:
        ExecutorStep that downloads the full dataset to GCS.
    """
    global _download_cache
    if _download_cache is None:
        _download_cache = _download_step
    return _download_cache


def _get_partition_base_dir():
    """Get the base directory for partition data."""
    # Note: append_sha_to_path=False in DownloadConfig, so files are directly under output_path/data/
    return download_dolmino_pool().cd("data")


# =============================================================================
# TOKENIZATION
# =============================================================================

_tokenize_cache: dict[str, ExecutorStep] = {}


def _resolve_partition_paths(partition_name: str) -> list:
    """Resolve partition directory patterns to actual paths.

    Args:
        partition_name: Name of the partition (e.g., "common_crawl_hq/19_adult_content")

    Returns:
        List of InputName paths for the partition's data files.
    """
    base_dir = _get_partition_base_dir()
    dir_patterns = DOLMINO_POOL_PARTITIONS[partition_name]

    paths = []
    for pattern in dir_patterns:
        if "*" in pattern:
            # For glob patterns, we need to expand them
            # The pattern will be resolved at execution time
            paths.append(base_dir / pattern / "**/*.jsonl.zst")
        else:
            # Direct directory reference
            paths.append(base_dir / pattern / "**/*.jsonl.zst")

    return paths


def tokenize_dolmino_pool_subset(
    partition_name: str,
    tokenizer: str | None = None,
) -> ExecutorStep:
    """Create a tokenization step for a specific partition.

    Args:
        partition_name: Name of the partition (e.g., "common_crawl_hq/19_adult_content")
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.

    Returns:
        ExecutorStep for tokenizing the partition.
    """
    if partition_name not in DOLMINO_POOL_PARTITIONS:
        raise KeyError(
            f"Partition '{partition_name}' not found. " f"Available partitions: {list(DOLMINO_POOL_PARTITIONS.keys())}"
        )

    cache_key = f"{partition_name}:{tokenizer}"
    if cache_key in _tokenize_cache:
        return _tokenize_cache[cache_key]

    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    # Create output path
    safe_name = partition_name.replace("/", "_")
    output_path = f"tokenized/dolma3_dolmino_pool/{safe_name}"

    # Get data paths
    train_paths = _resolve_partition_paths(partition_name)

    step = ExecutorStep(
        name=output_path,
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
        ),
    )

    _tokenize_cache[cache_key] = step
    return step


def tokenize_dolmino_pool(
    tokenizer: str | None = None,
    partitions: list[str] | None = None,
) -> dict[str, ExecutorStep]:
    """Create tokenization steps for Dolmino Pool partitions.

    Args:
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.
        partitions: List of partition names to tokenize. If None, tokenizes all.

    Returns:
        Dictionary mapping partition names to ExecutorSteps.
    """
    if partitions is None:
        partitions = list(DOLMINO_POOL_PARTITIONS.keys())

    steps = {}
    for partition in partitions:
        steps[partition] = tokenize_dolmino_pool_subset(partition, tokenizer=tokenizer)

    return steps


# =============================================================================
# PARTITION GROUPINGS (for domain-level experiments)
# =============================================================================


def get_common_crawl_hq_partitions() -> list[str]:
    """Get all Common Crawl HQ partition names (48 partitions)."""
    return [p for p in DOLMINO_POOL_PARTITIONS if p.startswith("common_crawl_hq/")]


def get_olmocr_pdfs_hq_partitions() -> list[str]:
    """Get all olmOCR PDFs HQ partition names (23 partitions)."""
    return [p for p in DOLMINO_POOL_PARTITIONS if p.startswith("olmocr_pdfs_hq/")]


def get_stack_edu_fim_partitions() -> list[str]:
    """Get all Stack-Edu FIM partition names (15 partitions)."""
    return [p for p in DOLMINO_POOL_PARTITIONS if p.startswith("stack_edu_fim/")]


def get_stem_heavy_crawl_partitions() -> list[str]:
    """Get all STEM Heavy Crawl partition names (24 partitions)."""
    return [p for p in DOLMINO_POOL_PARTITIONS if p.startswith("stem_heavy_crawl/")]


def get_synthetic_partitions() -> list[str]:
    """Get all synthetic dataset partition names (23 partitions)."""
    return [p for p in DOLMINO_POOL_PARTITIONS if p.startswith("synth_")]


def get_all_partition_names() -> list[str]:
    """Get all partition names."""
    return list(DOLMINO_POOL_PARTITIONS.keys())


# Verify partition count
# 48 (CC HQ) + 23 (olmOCR) + 15 (Stack-Edu) + 24 (STEM) + 23 (synthetic) = 133
_expected_partitions = 48 + 23 + 15 + 24 + 23
assert (
    len(DOLMINO_POOL_PARTITIONS) == _expected_partitions
), f"Expected {_expected_partitions} partitions, got {len(DOLMINO_POOL_PARTITIONS)}"
