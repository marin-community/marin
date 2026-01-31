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

"""Dolma 3 Pool dataset definitions and tokenization.

This module defines 333 partitions for the Dolma 3 Pool pretraining corpus.

The Dolma 3 Pool is assembled from multiple HuggingFace datasets:
- allenai/dolma3_pool: Common Crawl (290) + olmOCR PDFs (25) = 315 partitions
- HuggingFaceTB/stack-edu: Stack-Edu code (15 partitions)
- HuggingFaceTB/finemath: FineMath 3+ (1 partition)
- allenai/dolma (v1.7): arXiv (1) + Wikipedia (1) = 2 partitions

Total: 333 partitions, ~9.31T tokens

Per the OLMo 3 technical report:
- Common Crawl is partitioned by topic (18 categories) and quality tier (vigintile buckets)
- Stack-Edu provides high-quality code in 15 programming languages
- FineMath provides mathematical content with quality score >= 3
- arXiv and Wikipedia from Dolma v1.7 provide scientific/encyclopedic content

Usage:
    from experiments.pretraining_datasets.dolma3_pool import (
        download_dolma3_pool,
        tokenize_dolma3_pool,
        tokenize_dolma3_pool_subset,
        DOLMA3_POOL_PARTITIONS,
    )
"""

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

# =============================================================================
# CONSTANTS
# =============================================================================

HF_DATASET_ID = "allenai/dolma3_pool"
HF_REVISION = "main"  # TODO: Pin to specific commit hash after verification

# 18 topic categories present in Common Crawl
# Note: 6 topics from the full 24 are missing (social_life, software, software_development,
# sports_and_fitness, transportation, travel_and_tourism)
COMMON_CRAWL_TOPICS = [
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
]

# 24 topic categories in olmOCR Science PDFs (includes all 24 topics)
OLMOCR_TOPICS = [
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

# Quality tiers per Common Crawl topic (derived from HF repo)
# These correspond to vigintile buckets (5-percentile intervals) of quality scores
# The naming convention is common_crawl-{topic}-{tier:04d} where tier is the quality bucket
# Higher tier numbers = higher quality
COMMON_CRAWL_QUALITY_TIERS = {
    "adult_content": list(range(7, 20)),  # 0007-0019 (13 tiers)
    "art_and_design": list(range(5, 20)),  # 0005-0019 (15 tiers)
    "crime_and_law": list(range(2, 20)),  # 0002-0019 (18 tiers)
    "education_and_jobs": list(range(3, 20)),  # 0003-0019 (17 tiers)
    "electronics_and_hardware": list(range(5, 20)),  # 0005-0019 (15 tiers)
    "entertainment": list(range(5, 20)),  # 0005-0019 (15 tiers)
    "fashion_and_beauty": list(range(6, 20)),  # 0006-0019 (14 tiers)
    "finance_and_business": list(range(2, 20)),  # 0002-0019 (18 tiers)
    "food_and_dining": list(range(4, 20)),  # 0004-0019 (16 tiers)
    "games": list(range(4, 20)),  # 0004-0019 (16 tiers)
    "health": list(range(2, 20)),  # 0002-0019 (18 tiers)
    "history_and_geography": list(range(4, 20)),  # 0004-0019 (16 tiers)
    "home_and_hobbies": list(range(5, 20)),  # 0005-0019 (15 tiers)
    "industrial": list(range(3, 20)),  # 0003-0019 (17 tiers)
    "literature": list(range(4, 20)),  # 0004-0019 (16 tiers)
    "politics": list(range(2, 20)),  # 0002-0019 (18 tiers)
    "religion": list(range(3, 20)),  # 0003-0019 (17 tiers)
    "science_math_and_technology": list(range(2, 18)),  # 0002-0017 (16 tiers)
}

# Stack-Edu programming languages (from HuggingFaceTB/stack-edu)
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

# External dataset IDs
HF_STACK_EDU_ID = "HuggingFaceTB/stack-edu"
HF_FINEMATH_ID = "HuggingFaceTB/finemath"
HF_DOLMA_V17_ID = "allenai/dolma"

# =============================================================================
# PARTITION DEFINITIONS
# =============================================================================

# Each partition maps to one directory in the HF dataset.
# Partitions are defined at the shard level (topic * quality tier)
DOLMA3_POOL_PARTITIONS: dict[str, list[str]] = {}

# --- Common Crawl (290 partitions: 18 topics * varying quality tiers) ---
# Token estimate: 8.14T total
# Each partition is a specific topic + quality tier combination
for topic in COMMON_CRAWL_TOPICS:
    for tier in COMMON_CRAWL_QUALITY_TIERS[topic]:
        partition_name = f"common_crawl/{topic}/{tier:04d}"
        dir_name = f"common_crawl-{topic}-{tier:04d}"
        DOLMA3_POOL_PARTITIONS[partition_name] = [dir_name]

# --- olmOCR Science PDFs (25 partitions: 24 topics, science_math split into 2) ---
# Token estimate: 972B total
for topic in OLMOCR_TOPICS:
    if topic == "science_math_and_technology":
        # This topic is split into part1 and part2
        DOLMA3_POOL_PARTITIONS[f"olmocr_pdfs/{topic}/part1"] = ["olmocr_science_pdfs-science_math_and_technology-part1"]
        DOLMA3_POOL_PARTITIONS[f"olmocr_pdfs/{topic}/part2"] = ["olmocr_science_pdfs-science_math_and_technology-part2"]
    else:
        partition_name = f"olmocr_pdfs/{topic}"
        DOLMA3_POOL_PARTITIONS[partition_name] = [f"olmocr_science_pdfs-{topic}"]

# --- Stack-Edu (15 partitions: 15 programming languages) ---
# Source: HuggingFaceTB/stack-edu
# Token estimate: 137B total
for lang in STACK_EDU_LANGUAGES:
    partition_name = f"stack_edu/{lang}"
    # Each language is a top-level directory in the HF repo
    DOLMA3_POOL_PARTITIONS[partition_name] = [lang]

# --- FineMath 3+ (1 partition) ---
# Source: HuggingFaceTB/finemath (finemath-3plus subset)
# Token estimate: 34.1B
DOLMA3_POOL_PARTITIONS["finemath_3plus"] = ["finemath-3plus"]

# --- arXiv (1 partition) ---
# Source: allenai/dolma v1.7
# Token estimate: 21.4B (28B in Dolma, selecting subset)
DOLMA3_POOL_PARTITIONS["arxiv"] = ["arxiv"]

# --- Wikipedia & Wikibooks (1 partition) ---
# Source: allenai/dolma v1.7
# Token estimate: 3.69B (7.4B in Dolma = wiki * 2)
DOLMA3_POOL_PARTITIONS["wikipedia"] = ["wiki"]

# =============================================================================
# TOKEN COUNTS (estimated from dataset card)
# =============================================================================

# Token counts in billions, from the dataset documentation
# These are approximate and should be verified by running count_tokens.py after tokenization
DOLMA3_POOL_TOKEN_COUNTS_B: dict[str, float] = {}

# Common Crawl - total ~8.14T / 290 = ~28.1B average per partition
# (actual distribution varies significantly by topic and quality tier)
for partition in DOLMA3_POOL_PARTITIONS:
    if partition.startswith("common_crawl/"):
        DOLMA3_POOL_TOKEN_COUNTS_B[partition] = 28.1  # Approximate, to be measured

# olmOCR PDFs - total ~972B / 25 = ~38.9B average per partition
for partition in DOLMA3_POOL_PARTITIONS:
    if partition.startswith("olmocr_pdfs/"):
        DOLMA3_POOL_TOKEN_COUNTS_B[partition] = 38.9  # Approximate

# Stack-Edu - total ~137B / 15 = ~9.1B average per language
for partition in DOLMA3_POOL_PARTITIONS:
    if partition.startswith("stack_edu/"):
        DOLMA3_POOL_TOKEN_COUNTS_B[partition] = 9.1  # Approximate

# Individual datasets
DOLMA3_POOL_TOKEN_COUNTS_B["finemath_3plus"] = 34.1
DOLMA3_POOL_TOKEN_COUNTS_B["arxiv"] = 21.4
DOLMA3_POOL_TOKEN_COUNTS_B["wikipedia"] = 3.69

# =============================================================================
# DOWNLOAD STEPS
# =============================================================================

# Main Dolma 3 Pool dataset (Common Crawl + olmOCR)
_download_dolma3_pool_step = ExecutorStep(
    name="raw/dolma3_pool",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

# Stack-Edu (15 programming languages)
_download_stack_edu_step = ExecutorStep(
    name="raw/stack_edu",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=HF_STACK_EDU_ID,
        revision="main",  # TODO: Pin to specific commit
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

# FineMath
_download_finemath_step = ExecutorStep(
    name="raw/finemath",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=HF_FINEMATH_ID,
        revision="main",  # TODO: Pin to specific commit
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

# Dolma v1.7 (for arXiv and Wikipedia)
# Note: This is also defined in dolma.py, but we define here for self-containment
_download_dolma_v17_step = ExecutorStep(
    name="raw/dolma_v1.7",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=HF_DOLMA_V17_ID,
        revision="main",  # TODO: Pin to specific commit
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)


def download_dolma3_pool() -> ExecutorStep:
    """Get the download step for the main Dolma 3 Pool dataset (CC + olmOCR)."""
    return _download_dolma3_pool_step


def download_stack_edu() -> ExecutorStep:
    """Get the download step for Stack-Edu dataset."""
    return _download_stack_edu_step


def download_finemath() -> ExecutorStep:
    """Get the download step for FineMath dataset."""
    return _download_finemath_step


def download_dolma_v17() -> ExecutorStep:
    """Get the download step for Dolma v1.7 dataset (arXiv, Wikipedia)."""
    return _download_dolma_v17_step


def download_all_dolma3_pool_sources() -> list[ExecutorStep]:
    """Get all download steps needed for the complete Dolma 3 Pool.

    Returns:
        List of ExecutorSteps for all source datasets.
    """
    return [
        download_dolma3_pool(),
        download_stack_edu(),
        download_finemath(),
        download_dolma_v17(),
    ]


def _get_dolma3_pool_base_dir():
    """Get the base directory for main Dolma 3 Pool data (CC + olmOCR)."""
    # Note: append_sha_to_path=False in DownloadConfig, so files are directly under output_path/data/
    return download_dolma3_pool().cd("data")


def _get_stack_edu_base_dir():
    """Get the base directory for Stack-Edu data."""
    return download_stack_edu().cd("data")


def _get_finemath_base_dir():
    """Get the base directory for FineMath data."""
    return download_finemath().cd("data")


def _get_dolma_v17_base_dir():
    """Get the base directory for Dolma v1.7 data."""
    return download_dolma_v17().cd("data")


# =============================================================================
# TOKENIZATION
# =============================================================================

_tokenize_cache: dict[str, ExecutorStep] = {}


def _resolve_partition_paths(partition_name: str) -> list:
    """Resolve partition directory patterns to actual paths.

    Args:
        partition_name: Name of the partition (e.g., "common_crawl/adult_content/0005")

    Returns:
        List of InputName paths for the partition's data files.
    """
    dir_patterns = DOLMA3_POOL_PARTITIONS[partition_name]

    # Determine which source dataset this partition comes from
    if partition_name.startswith("common_crawl/") or partition_name.startswith("olmocr_pdfs/"):
        base_dir = _get_dolma3_pool_base_dir()
        file_pattern = "**/*.jsonl.zst"
    elif partition_name.startswith("stack_edu/"):
        base_dir = _get_stack_edu_base_dir()
        file_pattern = "**/*.parquet"  # Stack-Edu uses parquet format
    elif partition_name == "finemath_3plus":
        base_dir = _get_finemath_base_dir()
        file_pattern = "**/*.parquet"  # FineMath uses parquet format
    elif partition_name == "arxiv":
        base_dir = _get_dolma_v17_base_dir()
        file_pattern = "**/*.json.gz"  # Dolma v1.7 uses json.gz
    elif partition_name == "wikipedia":
        base_dir = _get_dolma_v17_base_dir()
        file_pattern = "**/*.json.gz"  # Dolma v1.7 uses json.gz
    else:
        raise ValueError(f"Unknown partition source: {partition_name}")

    paths = []
    for pattern in dir_patterns:
        paths.append(base_dir / pattern / file_pattern)

    return paths


def tokenize_dolma3_pool_subset(
    partition_name: str,
    tokenizer: str | None = None,
) -> ExecutorStep:
    """Create a tokenization step for a specific partition.

    Args:
        partition_name: Name of the partition (e.g., "common_crawl/adult_content")
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.

    Returns:
        ExecutorStep for tokenizing the partition.
    """
    if partition_name not in DOLMA3_POOL_PARTITIONS:
        raise KeyError(
            f"Partition '{partition_name}' not found. " f"Available partitions: {list(DOLMA3_POOL_PARTITIONS.keys())}"
        )

    cache_key = f"{partition_name}:{tokenizer}"
    if cache_key in _tokenize_cache:
        return _tokenize_cache[cache_key]

    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    # Create output path
    safe_name = partition_name.replace("/", "_")
    output_path = f"tokenized/dolma3_pool/{safe_name}"

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


def tokenize_dolma3_pool(
    tokenizer: str | None = None,
    partitions: list[str] | None = None,
) -> dict[str, ExecutorStep]:
    """Create tokenization steps for Dolma 3 Pool partitions.

    Args:
        tokenizer: Tokenizer to use. Defaults to marin_tokenizer.
        partitions: List of partition names to tokenize. If None, tokenizes all.

    Returns:
        Dictionary mapping partition names to ExecutorSteps.
    """
    if partitions is None:
        partitions = list(DOLMA3_POOL_PARTITIONS.keys())

    steps = {}
    for partition in partitions:
        steps[partition] = tokenize_dolma3_pool_subset(partition, tokenizer=tokenizer)

    return steps


# =============================================================================
# PARTITION GROUPINGS (for domain-level experiments)
# =============================================================================


def get_common_crawl_partitions() -> list[str]:
    """Get all Common Crawl partition names (290 partitions)."""
    return [p for p in DOLMA3_POOL_PARTITIONS if p.startswith("common_crawl/")]


def get_common_crawl_partitions_by_topic(topic: str) -> list[str]:
    """Get Common Crawl partitions for a specific topic.

    Args:
        topic: Topic name (e.g., "adult_content", "science_math_and_technology")

    Returns:
        List of partition names for that topic across all quality tiers.
    """
    prefix = f"common_crawl/{topic}/"
    return [p for p in DOLMA3_POOL_PARTITIONS if p.startswith(prefix)]


def get_common_crawl_partitions_by_tier(min_tier: int = 0, max_tier: int = 19) -> list[str]:
    """Get Common Crawl partitions within a quality tier range.

    Args:
        min_tier: Minimum quality tier (inclusive). Higher = higher quality.
        max_tier: Maximum quality tier (inclusive).

    Returns:
        List of partition names within the specified tier range.
    """
    result = []
    for topic in COMMON_CRAWL_TOPICS:
        for tier in COMMON_CRAWL_QUALITY_TIERS[topic]:
            if min_tier <= tier <= max_tier:
                result.append(f"common_crawl/{topic}/{tier:04d}")
    return result


def get_olmocr_pdfs_partitions() -> list[str]:
    """Get all olmOCR PDFs partition names (25 partitions)."""
    return [p for p in DOLMA3_POOL_PARTITIONS if p.startswith("olmocr_pdfs/")]


def get_stack_edu_partitions() -> list[str]:
    """Get all Stack-Edu partition names (15 partitions)."""
    return [p for p in DOLMA3_POOL_PARTITIONS if p.startswith("stack_edu/")]


def get_finemath_partitions() -> list[str]:
    """Get FineMath partition names (1 partition)."""
    return [p for p in DOLMA3_POOL_PARTITIONS if p.startswith("finemath_")]


def get_arxiv_partitions() -> list[str]:
    """Get arXiv partition names (1 partition)."""
    return [p for p in DOLMA3_POOL_PARTITIONS if p == "arxiv"]


def get_wikipedia_partitions() -> list[str]:
    """Get Wikipedia partition names (1 partition)."""
    return [p for p in DOLMA3_POOL_PARTITIONS if p == "wikipedia"]


def get_all_partition_names() -> list[str]:
    """Get all partition names."""
    return list(DOLMA3_POOL_PARTITIONS.keys())


def get_web_partitions_by_topic() -> dict[str, list[str]]:
    """Get CC + olmOCR PDF partitions grouped by topic (24 topics).

    This groups all web-scraped content (Common Crawl and olmOCR PDFs) by their
    topic category. For topics that exist in both CC and olmOCR, the partitions
    are combined. For topics only in olmOCR (6 topics), only PDF partitions are
    included.

    Returns:
        Dictionary mapping topic names to lists of partition names.
        - 18 topics have both CC (multiple quality tiers) + olmOCR partitions
        - 6 topics have only olmOCR PDF partitions
    """
    result: dict[str, list[str]] = {}

    for topic in OLMOCR_TOPICS:
        partitions = []

        # Add Common Crawl partitions if this topic exists in CC
        if topic in COMMON_CRAWL_TOPICS:
            partitions.extend(get_common_crawl_partitions_by_topic(topic))

        # Add olmOCR PDF partitions
        if topic == "science_math_and_technology":
            # This topic is split into part1 and part2
            partitions.append(f"olmocr_pdfs/{topic}/part1")
            partitions.append(f"olmocr_pdfs/{topic}/part2")
        else:
            partitions.append(f"olmocr_pdfs/{topic}")

        result[topic] = partitions

    return result


def get_web_topics() -> list[str]:
    """Get all 24 topic names for web content (CC + olmOCR PDFs)."""
    return list(OLMOCR_TOPICS)


def get_web_partitions() -> list[str]:
    """Get all Common Crawl partitions (290 partitions).

    This returns only the CC web content, excluding olmOCR PDFs which have
    known data quality issues.

    Returns:
        List of all Common Crawl partition names.
    """
    return get_common_crawl_partitions()


def get_web_with_ocr_partitions() -> list[str]:
    """Get all CC + olmOCR PDF partitions (315 partitions total).

    Note: olmOCR PDFs have known data quality issues (malformed files).
    Consider using get_web_partitions() instead for only CC content.

    Returns:
        List of all partition names for web content (CC + olmOCR).
    """
    return get_common_crawl_partitions() + get_olmocr_pdfs_partitions()


# =============================================================================
# VERIFICATION
# =============================================================================

# Verify partition count:
# - Common Crawl: 290 (sum of all quality tiers across 18 topics)
# - olmOCR PDFs: 25 (23 topics + 2 for science_math split)
# - Stack-Edu: 15 (15 programming languages)
# - FineMath: 1 (finemath-3plus)
# - arXiv: 1
# - Wikipedia: 1
# Total: 333 partitions
_expected_cc_partitions = sum(len(tiers) for tiers in COMMON_CRAWL_QUALITY_TIERS.values())
_expected_olmocr_partitions = len(OLMOCR_TOPICS) + 1  # +1 for science_math split into 2
_expected_stack_edu_partitions = len(STACK_EDU_LANGUAGES)
_expected_finemath_partitions = 1
_expected_arxiv_partitions = 1
_expected_wikipedia_partitions = 1

_expected_partitions = (
    _expected_cc_partitions
    + _expected_olmocr_partitions
    + _expected_stack_edu_partitions
    + _expected_finemath_partitions
    + _expected_arxiv_partitions
    + _expected_wikipedia_partitions
)

assert len(DOLMA3_POOL_PARTITIONS) == _expected_partitions, (
    f"Expected {_expected_partitions} partitions "
    f"({_expected_cc_partitions} CC + {_expected_olmocr_partitions} olmOCR + "
    f"{_expected_stack_edu_partitions} Stack-Edu + {_expected_finemath_partitions} FineMath + "
    f"{_expected_arxiv_partitions} arXiv + {_expected_wikipedia_partitions} Wikipedia), "
    f"got {len(DOLMA3_POOL_PARTITIONS)}"
)
