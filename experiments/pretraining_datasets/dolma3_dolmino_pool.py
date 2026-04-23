# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dolma 3 Dolmino Pool dataset definitions and tokenization.

This module defines 133 partitions from the Dolma 3 Dolmino Pool dataset
(https://huggingface.co/datasets/allenai/dolma3_dolmino_pool).

The raw dataset is documented at roughly 2.19T tokens organized into several categories:
- Common Crawl HQ: 48 partitions (24 topics x 2 years: 2019, 2020)
- olmOCR Science PDFs HQ: 23 partitions (23 topics, with 2e12/2e13 size variants merged)
- Stack-Edu FIM: 15 partitions (15 programming languages, shards merged)
- STEM Heavy Crawl: 24 partitions (24 topics)
- Synthetic datasets: 23 partitions

Completed tokenized caches on GCS in `us-central1` total 2,288,298,862,641
tokens with the Marin tokenizer across all 133 declared partitions. The
`dolmino_1-flan` source directory contains three known-corrupted shards
(`tulu_flan-0064`, `0122`, and `0163`), which are stored on GCS with a
`.corrupted` suffix and therefore excluded from tokenization inputs.

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

from fray.cluster import ResourceConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
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
# RAW DATA AUDIT NOTES
# =============================================================================

KNOWN_CORRUPTED_DOLMINO_SOURCE_FILES: tuple[str, ...] = (
    "dolmino_1-flan/tulu_flan-0064.jsonl.zst.corrupted",
    "dolmino_1-flan/tulu_flan-0122.jsonl.zst.corrupted",
    "dolmino_1-flan/tulu_flan-0163.jsonl.zst.corrupted",
)

# =============================================================================
# TOKEN COUNTS
# =============================================================================

# Token counts in billions, measured from GCS `train/.stats.json` files via
# `count_dolmino_pool_tokens.py`.
DOLMINO_POOL_TOKEN_COUNTS_B: dict[str, float] = {
    "common_crawl_hq/19_adult_content": 8.3810,  # 8,381,037,655 tokens
    "common_crawl_hq/19_art_and_design": 15.9377,  # 15,937,673,069 tokens
    "common_crawl_hq/19_crime_and_law": 20.3919,  # 20,391,917,102 tokens
    "common_crawl_hq/19_education_and_jobs": 30.6153,  # 30,615,268,029 tokens
    "common_crawl_hq/19_electronics_and_hardware": 16.5263,  # 16,526,336,822 tokens
    "common_crawl_hq/19_entertainment": 56.9343,  # 56,934,318,529 tokens
    "common_crawl_hq/19_fashion_and_beauty": 20.6187,  # 20,618,718,869 tokens
    "common_crawl_hq/19_finance_and_business": 58.2013,  # 58,201,301,469 tokens
    "common_crawl_hq/19_food_and_dining": 19.2608,  # 19,260,774,360 tokens
    "common_crawl_hq/19_games": 29.2777,  # 29,277,676,675 tokens
    "common_crawl_hq/19_health": 54.3096,  # 54,309,609,972 tokens
    "common_crawl_hq/19_history_and_geography": 20.0166,  # 20,016,567,948 tokens
    "common_crawl_hq/19_home_and_hobbies": 54.8049,  # 54,804,861,525 tokens
    "common_crawl_hq/19_industrial": 9.5765,  # 9,576,492,233 tokens
    "common_crawl_hq/19_literature": 44.6384,  # 44,638,386,888 tokens
    "common_crawl_hq/19_politics": 54.8123,  # 54,812,285,796 tokens
    "common_crawl_hq/19_religion": 31.0000,  # 30,999,999,638 tokens
    "common_crawl_hq/19_science_math_and_technology": 44.9178,  # 44,917,823,508 tokens
    "common_crawl_hq/19_social_life": 31.3450,  # 31,345,026,701 tokens
    "common_crawl_hq/19_software": 16.5842,  # 16,584,201,955 tokens
    "common_crawl_hq/19_software_development": 16.1923,  # 16,192,310,540 tokens
    "common_crawl_hq/19_sports_and_fitness": 40.2728,  # 40,272,829,283 tokens
    "common_crawl_hq/19_transportation": 16.7638,  # 16,763,804,995 tokens
    "common_crawl_hq/19_travel_and_tourism": 25.0952,  # 25,095,232,417 tokens
    "common_crawl_hq/20_adult_content": 0.4796,  # 479,608,373 tokens
    "common_crawl_hq/20_art_and_design": 12.0004,  # 12,000,384,386 tokens
    "common_crawl_hq/20_crime_and_law": 16.6171,  # 16,617,087,516 tokens
    "common_crawl_hq/20_education_and_jobs": 25.0890,  # 25,088,986,752 tokens
    "common_crawl_hq/20_electronics_and_hardware": 13.5571,  # 13,557,052,917 tokens
    "common_crawl_hq/20_entertainment": 52.2176,  # 52,217,644,862 tokens
    "common_crawl_hq/20_fashion_and_beauty": 16.5287,  # 16,528,701,064 tokens
    "common_crawl_hq/20_finance_and_business": 50.1801,  # 50,180,102,412 tokens
    "common_crawl_hq/20_food_and_dining": 15.7125,  # 15,712,460,719 tokens
    "common_crawl_hq/20_games": 25.7855,  # 25,785,481,876 tokens
    "common_crawl_hq/20_health": 40.4163,  # 40,416,327,553 tokens
    "common_crawl_hq/20_history_and_geography": 13.0872,  # 13,087,198,319 tokens
    "common_crawl_hq/20_home_and_hobbies": 44.6649,  # 44,664,885,954 tokens
    "common_crawl_hq/20_industrial": 8.3613,  # 8,361,279,734 tokens
    "common_crawl_hq/20_literature": 32.9940,  # 32,994,013,793 tokens
    "common_crawl_hq/20_politics": 44.2785,  # 44,278,458,224 tokens
    "common_crawl_hq/20_religion": 22.0074,  # 22,007,365,839 tokens
    "common_crawl_hq/20_science_math_and_technology": 28.7562,  # 28,756,164,218 tokens
    "common_crawl_hq/20_social_life": 25.8833,  # 25,883,299,728 tokens
    "common_crawl_hq/20_software": 12.7998,  # 12,799,804,093 tokens
    "common_crawl_hq/20_software_development": 12.0337,  # 12,033,705,145 tokens
    "common_crawl_hq/20_sports_and_fitness": 33.4371,  # 33,437,104,070 tokens
    "common_crawl_hq/20_transportation": 14.3233,  # 14,323,250,744 tokens
    "common_crawl_hq/20_travel_and_tourism": 19.9014,  # 19,901,437,416 tokens
    "olmocr_pdfs_hq/adult_content": 0.0189,  # 18,861,709 tokens
    "olmocr_pdfs_hq/art_and_design": 1.4688,  # 1,468,806,367 tokens
    "olmocr_pdfs_hq/crime_and_law": 7.5491,  # 7,549,087,174 tokens
    "olmocr_pdfs_hq/education_and_jobs": 23.8538,  # 23,853,758,087 tokens
    "olmocr_pdfs_hq/electronics_and_hardware": 1.7216,  # 1,721,626,962 tokens
    "olmocr_pdfs_hq/entertainment": 0.7015,  # 701,489,491 tokens
    "olmocr_pdfs_hq/fashion_and_beauty": 0.1206,  # 120,615,956 tokens
    "olmocr_pdfs_hq/finance_and_business": 11.7713,  # 11,771,251,347 tokens
    "olmocr_pdfs_hq/food_and_dining": 0.5393,  # 539,297,159 tokens
    "olmocr_pdfs_hq/games": 0.3146,  # 314,604,548 tokens
    "olmocr_pdfs_hq/health": 28.1396,  # 28,139,614,976 tokens
    "olmocr_pdfs_hq/history_and_geography": 3.3578,  # 3,357,787,565 tokens
    "olmocr_pdfs_hq/home_and_hobbies": 0.8420,  # 841,956,470 tokens
    "olmocr_pdfs_hq/industrial": 5.8769,  # 5,876,949,084 tokens
    "olmocr_pdfs_hq/literature": 4.4739,  # 4,473,927,184 tokens
    "olmocr_pdfs_hq/politics": 7.8518,  # 7,851,757,650 tokens
    "olmocr_pdfs_hq/religion": 3.3180,  # 3,318,027,266 tokens
    "olmocr_pdfs_hq/science_math_and_technology": 93.8071,  # 93,807,115,671 tokens
    "olmocr_pdfs_hq/software": 0.7758,  # 775,792,563 tokens
    "olmocr_pdfs_hq/software_development": 4.4800,  # 4,479,963,800 tokens
    "olmocr_pdfs_hq/sports_and_fitness": 1.1666,  # 1,166,579,038 tokens
    "olmocr_pdfs_hq/transportation": 3.1957,  # 3,195,687,882 tokens
    "olmocr_pdfs_hq/travel_and_tourism": 0.5578,  # 557,827,150 tokens
    "stack_edu_fim/C": 4.7944,  # 4,794,378,523 tokens
    "stack_edu_fim/CSharp": 7.2804,  # 7,280,376,665 tokens
    "stack_edu_fim/Cpp": 12.5377,  # 12,537,737,094 tokens
    "stack_edu_fim/Go": 1.4168,  # 1,416,818,637 tokens
    "stack_edu_fim/Java": 31.1296,  # 31,129,627,881 tokens
    "stack_edu_fim/JavaScript": 8.9608,  # 8,960,779,585 tokens
    "stack_edu_fim/Markdown": 25.9952,  # 25,995,237,784 tokens
    "stack_edu_fim/PHP": 7.4808,  # 7,480,845,163 tokens
    "stack_edu_fim/Python": 17.7957,  # 17,795,732,931 tokens
    "stack_edu_fim/Ruby": 1.4206,  # 1,420,554,277 tokens
    "stack_edu_fim/Rust": 1.4312,  # 1,431,212,270 tokens
    "stack_edu_fim/SQL": 6.9786,  # 6,978,561,755 tokens
    "stack_edu_fim/Shell": 2.5910,  # 2,590,981,737 tokens
    "stack_edu_fim/Swift": 1.5320,  # 1,531,963,169 tokens
    "stack_edu_fim/TypeScript": 2.5383,  # 2,538,259,767 tokens
    "stem_heavy_crawl/adult_content": 0.0015,  # 1,526,613 tokens
    "stem_heavy_crawl/art_and_design": 0.0352,  # 35,160,664 tokens
    "stem_heavy_crawl/crime_and_law": 0.0468,  # 46,837,223 tokens
    "stem_heavy_crawl/education_and_jobs": 0.1632,  # 163,161,047 tokens
    "stem_heavy_crawl/electronics_and_hardware": 0.0745,  # 74,515,890 tokens
    "stem_heavy_crawl/entertainment": 0.0848,  # 84,849,784 tokens
    "stem_heavy_crawl/fashion_and_beauty": 0.0132,  # 13,220,186 tokens
    "stem_heavy_crawl/finance_and_business": 0.2849,  # 284,875,709 tokens
    "stem_heavy_crawl/food_and_dining": 0.0298,  # 29,849,150 tokens
    "stem_heavy_crawl/games": 0.0909,  # 90,938,673 tokens
    "stem_heavy_crawl/health": 0.2426,  # 242,591,532 tokens
    "stem_heavy_crawl/history_and_geography": 0.1098,  # 109,839,314 tokens
    "stem_heavy_crawl/home_and_hobbies": 0.0425,  # 42,531,797 tokens
    "stem_heavy_crawl/industrial": 0.0216,  # 21,554,048 tokens
    "stem_heavy_crawl/literature": 0.2250,  # 224,991,359 tokens
    "stem_heavy_crawl/politics": 0.1417,  # 141,669,175 tokens
    "stem_heavy_crawl/religion": 0.0634,  # 63,377,894 tokens
    "stem_heavy_crawl/science_math_and_technology": 2.2053,  # 2,205,270,315 tokens
    "stem_heavy_crawl/social_life": 0.0411,  # 41,101,470 tokens
    "stem_heavy_crawl/software": 0.1725,  # 172,534,368 tokens
    "stem_heavy_crawl/software_development": 1.0403,  # 1,040,276,580 tokens
    "stem_heavy_crawl/sports_and_fitness": 0.0439,  # 43,859,279 tokens
    "stem_heavy_crawl/transportation": 0.0302,  # 30,166,406 tokens
    "stem_heavy_crawl/travel_and_tourism": 0.0091,  # 9,054,760 tokens
    "synth_code/cranecode": 18.8608,  # 18,860,808,823 tokens
    "synth_instruction/dolmino_flan": 16.4424,  # 16,442,404,921 tokens
    "synth_instruction/tulu_3_sft": 1.5380,  # 1,538,031,598 tokens
    "synth_math/cranemath": 5.6329,  # 5,632,856,885 tokens
    "synth_math/dolmino_math": 10.7086,  # 10,708,619,773 tokens
    "synth_math/megamatt": 3.8993,  # 3,899,255,174 tokens
    "synth_math/tinymath_mind": 0.8992,  # 899,172,352 tokens
    "synth_math/tinymath_pot": 0.2423,  # 242,255,476 tokens
    "synth_math/verifiable_gpt41": 0.3648,  # 364,813,694 tokens
    "synth_math/verifiable_o4mini": 0.0739,  # 73,921,022 tokens
    "synth_qa/nemotron_synth_qa": 488.3019,  # 488,301,926,682 tokens
    "synth_qa/reddit_to_flashcards": 34.5974,  # 34,597,359,285 tokens
    "synth_qa/wiki_to_rcqa": 4.2541,  # 4,254,057,981 tokens
    "synth_thinking/code_meta_reasoning": 1.2675,  # 1,267,452,019 tokens
    "synth_thinking/gemini_reasoning": 0.2458,  # 245,803,085 tokens
    "synth_thinking/general_reasoning_mix": 2.4688,  # 2,468,826,907 tokens
    "synth_thinking/llama_nemotron_reasoning": 20.8186,  # 20,818,607,573 tokens
    "synth_thinking/math_meta_reasoning": 1.0515,  # 1,051,507,567 tokens
    "synth_thinking/omr_rewrite_fullthoughts": 0.8499,  # 849,884,028 tokens
    "synth_thinking/openthoughts2_reasoning": 5.5798,  # 5,579,818,653 tokens
    "synth_thinking/program_verifiable": 0.3916,  # 391,614,940 tokens
    "synth_thinking/qwq_reasoning": 4.7556,  # 4,755,570,038 tokens
    "synth_thinking/r1_reasoning": 2.4688,  # 2,468,826,907 tokens
}

DOLMINO_POOL_COMPLETED_PARTITIONS: tuple[str, ...] = tuple(DOLMINO_POOL_TOKEN_COUNTS_B)

# =============================================================================
# DOWNLOAD STEP
# =============================================================================

# Cache for download step (will be populated after first download with correct hash)
_download_cache: dict[str, ExecutorStep] = {}


def _worker_resources_cache_key(worker_resources: ResourceConfig | None) -> str:
    return "default" if worker_resources is None else repr(worker_resources)


def download_dolmino_pool(*, worker_resources: ResourceConfig | None = None) -> ExecutorStep:
    """Get the download step for the Dolma 3 Dolmino Pool dataset.

    Returns:
        ExecutorStep that downloads the full dataset to GCS.
    """
    cache_key = _worker_resources_cache_key(worker_resources)
    if cache_key not in _download_cache:
        _download_cache[cache_key] = ExecutorStep(
            name="raw/dolma3_dolmino_pool",
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id=HF_DATASET_ID,
                revision=HF_REVISION,
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
                worker_resources=worker_resources,
            ),
        )
    return _download_cache[cache_key]


def _get_partition_base_dir(*, worker_resources: ResourceConfig | None = None):
    """Get the base directory for partition data."""
    # Note: append_sha_to_path=False in DownloadConfig, so files are directly under output_path/data/
    return download_dolmino_pool(worker_resources=worker_resources).cd("data")


# =============================================================================
# TOKENIZATION
# =============================================================================

_tokenize_cache: dict[str, ExecutorStep] = {}


def _resolve_partition_paths(partition_name: str, *, worker_resources: ResourceConfig | None = None) -> list:
    """Resolve partition directory patterns to actual paths.

    Args:
        partition_name: Name of the partition (e.g., "common_crawl_hq/19_adult_content")

    Returns:
        List of InputName paths for the partition's data files.
    """
    base_dir = _get_partition_base_dir(worker_resources=worker_resources)
    dir_patterns = DOLMINO_POOL_PARTITIONS[partition_name]

    paths = []
    for pattern in dir_patterns:
        # stem-heavy-crawl files are flat (topic.NNNNN.jsonl.zst), not in subdirectories
        if pattern.startswith("stem-heavy-crawl/"):
            paths.append(base_dir / pattern)
        elif "*" in pattern:
            # For glob patterns matching subdirectories (e.g., stack_edu_fim-C_*)
            paths.append(base_dir / pattern / "**/*.jsonl.zst")
        else:
            # Direct directory reference
            paths.append(base_dir / pattern / "**/*.jsonl.zst")

    return paths


def tokenize_dolmino_pool_subset(
    partition_name: str,
    tokenizer: str | None = None,
    worker_resources: ResourceConfig | None = None,
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

    if tokenizer is None:
        from experiments.marin_models import marin_tokenizer

        tokenizer = marin_tokenizer

    cache_key = f"{partition_name}:{tokenizer}:{_worker_resources_cache_key(worker_resources)}"
    if cache_key in _tokenize_cache:
        return _tokenize_cache[cache_key]

    # Create output path
    safe_name = partition_name.replace("/", "_")
    output_path = f"tokenized/dolma3_dolmino_pool/{safe_name}"

    # Get data paths
    train_paths = _resolve_partition_paths(partition_name, worker_resources=worker_resources)

    config_kwargs = {}
    if worker_resources is not None:
        config_kwargs["worker_resources"] = worker_resources

    step = ExecutorStep(
        name=output_path,
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=train_paths,
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(tokenizer),
            **config_kwargs,
        ),
    )

    _tokenize_cache[cache_key] = step
    return step


def tokenize_dolmino_pool(
    tokenizer: str | None = None,
    partitions: list[str] | None = None,
    worker_resources: ResourceConfig | None = None,
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
        steps[partition] = tokenize_dolmino_pool_subset(
            partition,
            tokenizer=tokenizer,
            worker_resources=worker_resources,
        )

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
