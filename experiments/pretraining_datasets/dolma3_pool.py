# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dolma 3 Pool dataset definitions and tokenization.

This module defines 333 declared partitions for the Dolma 3 Pool pretraining corpus.

The Dolma 3 Pool is assembled from multiple HuggingFace datasets:
- allenai/dolma3_pool: Common Crawl (290) + olmOCR PDFs (25) = 315 partitions
- HuggingFaceTB/stack-edu: Stack-Edu code (15 partitions)
- HuggingFaceTB/finemath: FineMath 3+ (1 partition)
- allenai/dolma (v1.7): arXiv (1) + Wikipedia (1) = 2 partitions

Total raw dataset: 333 partitions, ~9.31T tokens

As of the 2026-03 GCS audit in `us-central1`, completed tokenized caches exist for
293 partitions totaling 6,391,092,209,185 tokens with the Marin/Llama 3 tokenizer:
- 290 Common Crawl partitions under `tokenized/dolma3_pool/`
- `finemath_3plus` via `tokenized/finemath_3_plus-a26b0f`
- `arxiv` via `tokenized/dolma/arxiv-07a51f`
- `wikipedia` via `tokenized/dolma/wiki-212315`

The remaining 40 incomplete partitions are:
- 25 known-broken `olmocr_pdfs/*` partitions
- 15 `stack_edu/*` partitions; the current Hugging Face export contains only
  Software Heritage blob IDs and metadata, so a content hydration step is
  required before tokenization

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

from fray.cluster import ResourceConfig
from experiments.llama import llama3_tokenizer
from experiments.marin_models import marin_tokenizer
from experiments.midtraining_datasets import finemath_3_plus_tokenized
from experiments.pretraining_datasets.dolma import tokenize_dolma
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.transform.stack_edu.hydrate import StackEduHydrationConfig, hydrate_stack_edu as hydrate_stack_edu_text

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
# TOKEN COUNTS
# =============================================================================

# Token counts in billions, measured from GCS `train/.stats.json` files via
# `count_dolma3_tokens.py`. Only completed tokenized partitions are included here.
DOLMA3_POOL_TOKEN_COUNTS_B: dict[str, float] = {
    # Common Crawl — 290 partitions, 6325.18B total (measured)
    "common_crawl/adult_content/0007": 8.1102,
    "common_crawl/adult_content/0008": 2.9688,
    "common_crawl/adult_content/0009": 3.8694,
    "common_crawl/adult_content/0010": 5.1352,
    "common_crawl/adult_content/0011": 4.6115,
    "common_crawl/adult_content/0012": 4.9450,
    "common_crawl/adult_content/0013": 5.5435,
    "common_crawl/adult_content/0014": 6.0113,
    "common_crawl/adult_content/0015": 7.0451,
    "common_crawl/adult_content/0016": 8.8337,
    "common_crawl/adult_content/0017": 10.1601,
    "common_crawl/adult_content/0018": 8.4126,
    "common_crawl/adult_content/0019": 0.4833,
    "common_crawl/art_and_design/0005": 22.1560,
    "common_crawl/art_and_design/0006": 5.7819,
    "common_crawl/art_and_design/0007": 5.6660,
    "common_crawl/art_and_design/0008": 6.0852,
    "common_crawl/art_and_design/0009": 6.6512,
    "common_crawl/art_and_design/0010": 7.3557,
    "common_crawl/art_and_design/0011": 8.1254,
    "common_crawl/art_and_design/0012": 8.9590,
    "common_crawl/art_and_design/0013": 9.9741,
    "common_crawl/art_and_design/0014": 11.2792,
    "common_crawl/art_and_design/0015": 12.2810,
    "common_crawl/art_and_design/0016": 13.4797,
    "common_crawl/art_and_design/0017": 14.7230,
    "common_crawl/art_and_design/0018": 15.9689,
    "common_crawl/art_and_design/0019": 12.0241,
    "common_crawl/crime_and_law/0002": 12.5606,
    "common_crawl/crime_and_law/0003": 6.3222,
    "common_crawl/crime_and_law/0004": 6.7977,
    "common_crawl/crime_and_law/0005": 7.4133,
    "common_crawl/crime_and_law/0006": 8.0124,
    "common_crawl/crime_and_law/0007": 8.6653,
    "common_crawl/crime_and_law/0008": 9.2402,
    "common_crawl/crime_and_law/0009": 10.0198,
    "common_crawl/crime_and_law/0010": 10.8546,
    "common_crawl/crime_and_law/0011": 11.6719,
    "common_crawl/crime_and_law/0012": 12.6170,
    "common_crawl/crime_and_law/0013": 13.6897,
    "common_crawl/crime_and_law/0014": 14.7700,
    "common_crawl/crime_and_law/0015": 16.1054,
    "common_crawl/crime_and_law/0016": 17.7379,
    "common_crawl/crime_and_law/0017": 19.1881,
    "common_crawl/crime_and_law/0018": 20.3942,
    "common_crawl/crime_and_law/0019": 16.6459,
    "common_crawl/education_and_jobs/0003": 32.2744,
    "common_crawl/education_and_jobs/0004": 13.6612,
    "common_crawl/education_and_jobs/0005": 14.0447,
    "common_crawl/education_and_jobs/0006": 14.6714,
    "common_crawl/education_and_jobs/0007": 15.3110,
    "common_crawl/education_and_jobs/0008": 16.0249,
    "common_crawl/education_and_jobs/0009": 17.0860,
    "common_crawl/education_and_jobs/0010": 18.3647,
    "common_crawl/education_and_jobs/0011": 19.7452,
    "common_crawl/education_and_jobs/0012": 21.3299,
    "common_crawl/education_and_jobs/0013": 23.4303,
    "common_crawl/education_and_jobs/0014": 25.3901,
    "common_crawl/education_and_jobs/0015": 27.7475,
    "common_crawl/education_and_jobs/0016": 29.3825,
    "common_crawl/education_and_jobs/0017": 30.1109,
    "common_crawl/education_and_jobs/0018": 30.6513,
    "common_crawl/education_and_jobs/0019": 25.1086,
    "common_crawl/electronics_and_hardware/0005": 30.0584,
    "common_crawl/electronics_and_hardware/0006": 7.2109,
    "common_crawl/electronics_and_hardware/0007": 7.8654,
    "common_crawl/electronics_and_hardware/0008": 8.3556,
    "common_crawl/electronics_and_hardware/0009": 8.8236,
    "common_crawl/electronics_and_hardware/0010": 9.5023,
    "common_crawl/electronics_and_hardware/0011": 10.1811,
    "common_crawl/electronics_and_hardware/0012": 11.0022,
    "common_crawl/electronics_and_hardware/0013": 11.8641,
    "common_crawl/electronics_and_hardware/0014": 13.0754,
    "common_crawl/electronics_and_hardware/0015": 14.3992,
    "common_crawl/electronics_and_hardware/0016": 15.8921,
    "common_crawl/electronics_and_hardware/0017": 16.5342,
    "common_crawl/electronics_and_hardware/0018": 16.5437,
    "common_crawl/electronics_and_hardware/0019": 13.5781,
    "common_crawl/entertainment/0005": 66.5515,
    "common_crawl/entertainment/0006": 19.7846,
    "common_crawl/entertainment/0007": 20.6280,
    "common_crawl/entertainment/0008": 22.5809,
    "common_crawl/entertainment/0009": 24.5401,
    "common_crawl/entertainment/0010": 27.0682,
    "common_crawl/entertainment/0011": 29.7409,
    "common_crawl/entertainment/0012": 32.8153,
    "common_crawl/entertainment/0013": 36.5312,
    "common_crawl/entertainment/0014": 40.3075,
    "common_crawl/entertainment/0015": 44.7550,
    "common_crawl/entertainment/0016": 49.5566,
    "common_crawl/entertainment/0017": 53.4605,
    "common_crawl/entertainment/0018": 56.9462,
    "common_crawl/entertainment/0019": 52.2313,
    "common_crawl/fashion_and_beauty/0006": 39.5778,
    "common_crawl/fashion_and_beauty/0007": 8.6125,
    "common_crawl/fashion_and_beauty/0008": 8.3534,
    "common_crawl/fashion_and_beauty/0009": 8.9981,
    "common_crawl/fashion_and_beauty/0010": 9.7861,
    "common_crawl/fashion_and_beauty/0011": 10.7182,
    "common_crawl/fashion_and_beauty/0012": 11.7751,
    "common_crawl/fashion_and_beauty/0013": 13.1222,
    "common_crawl/fashion_and_beauty/0014": 14.4607,
    "common_crawl/fashion_and_beauty/0015": 16.4295,
    "common_crawl/fashion_and_beauty/0016": 19.1731,
    "common_crawl/fashion_and_beauty/0017": 22.3389,
    "common_crawl/fashion_and_beauty/0018": 20.6438,
    "common_crawl/fashion_and_beauty/0019": 16.5288,
    "common_crawl/finance_and_business/0002": 41.6317,
    "common_crawl/finance_and_business/0003": 23.2957,
    "common_crawl/finance_and_business/0004": 24.9435,
    "common_crawl/finance_and_business/0005": 27.7572,
    "common_crawl/finance_and_business/0006": 30.8444,
    "common_crawl/finance_and_business/0007": 34.1214,
    "common_crawl/finance_and_business/0008": 37.0690,
    "common_crawl/finance_and_business/0009": 39.5314,
    "common_crawl/finance_and_business/0010": 42.7319,
    "common_crawl/finance_and_business/0011": 46.1030,
    "common_crawl/finance_and_business/0012": 49.9955,
    "common_crawl/finance_and_business/0013": 55.1379,
    "common_crawl/finance_and_business/0014": 57.2928,
    "common_crawl/finance_and_business/0015": 60.6668,
    "common_crawl/finance_and_business/0016": 62.3372,
    "common_crawl/finance_and_business/0017": 61.2711,
    "common_crawl/finance_and_business/0018": 58.2317,
    "common_crawl/finance_and_business/0019": 50.1958,
    "common_crawl/food_and_dining/0004": 32.0076,
    "common_crawl/food_and_dining/0005": 9.7650,
    "common_crawl/food_and_dining/0006": 9.9031,
    "common_crawl/food_and_dining/0007": 10.1236,
    "common_crawl/food_and_dining/0008": 10.7505,
    "common_crawl/food_and_dining/0009": 11.5404,
    "common_crawl/food_and_dining/0010": 12.4586,
    "common_crawl/food_and_dining/0011": 13.4664,
    "common_crawl/food_and_dining/0012": 14.6765,
    "common_crawl/food_and_dining/0013": 16.2203,
    "common_crawl/food_and_dining/0014": 17.8311,
    "common_crawl/food_and_dining/0015": 19.5507,
    "common_crawl/food_and_dining/0016": 20.6675,
    "common_crawl/food_and_dining/0017": 20.9818,
    "common_crawl/food_and_dining/0018": 19.2611,
    "common_crawl/food_and_dining/0019": 15.7252,
    "common_crawl/games/0004": 26.7228,
    "common_crawl/games/0005": 9.5143,
    "common_crawl/games/0006": 9.8985,
    "common_crawl/games/0007": 11.2803,
    "common_crawl/games/0008": 12.8891,
    "common_crawl/games/0009": 14.4052,
    "common_crawl/games/0010": 15.8904,
    "common_crawl/games/0011": 17.3876,
    "common_crawl/games/0012": 19.2004,
    "common_crawl/games/0013": 20.8170,
    "common_crawl/games/0014": 22.2655,
    "common_crawl/games/0015": 23.5868,
    "common_crawl/games/0016": 24.9254,
    "common_crawl/games/0017": 26.7686,
    "common_crawl/games/0018": 29.2831,
    "common_crawl/games/0019": 25.8175,
    "common_crawl/health/0002": 31.7340,
    "common_crawl/health/0003": 15.8529,
    "common_crawl/health/0004": 17.5073,
    "common_crawl/health/0005": 19.7607,
    "common_crawl/health/0006": 22.1476,
    "common_crawl/health/0007": 23.9675,
    "common_crawl/health/0008": 25.9938,
    "common_crawl/health/0009": 28.6333,
    "common_crawl/health/0010": 30.9711,
    "common_crawl/health/0011": 34.1369,
    "common_crawl/health/0012": 37.3391,
    "common_crawl/health/0013": 41.0083,
    "common_crawl/health/0014": 44.8591,
    "common_crawl/health/0015": 47.6130,
    "common_crawl/health/0016": 50.9879,
    "common_crawl/health/0017": 53.7696,
    "common_crawl/health/0018": 54.3364,
    "common_crawl/health/0019": 40.4470,
    "common_crawl/history_and_geography/0004": 11.8555,
    "common_crawl/history_and_geography/0005": 3.9521,
    "common_crawl/history_and_geography/0006": 4.3099,
    "common_crawl/history_and_geography/0007": 4.8783,
    "common_crawl/history_and_geography/0008": 5.6145,
    "common_crawl/history_and_geography/0009": 6.5718,
    "common_crawl/history_and_geography/0010": 7.6621,
    "common_crawl/history_and_geography/0011": 8.6438,
    "common_crawl/history_and_geography/0012": 9.8335,
    "common_crawl/history_and_geography/0013": 11.3812,
    "common_crawl/history_and_geography/0014": 13.1203,
    "common_crawl/history_and_geography/0015": 14.9103,
    "common_crawl/history_and_geography/0016": 16.9694,
    "common_crawl/history_and_geography/0017": 19.4308,
    "common_crawl/history_and_geography/0018": 20.0286,
    "common_crawl/history_and_geography/0019": 13.1317,
    "common_crawl/home_and_hobbies/0005": 89.6744,
    "common_crawl/home_and_hobbies/0006": 21.5748,
    "common_crawl/home_and_hobbies/0007": 21.8290,
    "common_crawl/home_and_hobbies/0008": 22.7672,
    "common_crawl/home_and_hobbies/0009": 24.3445,
    "common_crawl/home_and_hobbies/0010": 26.2217,
    "common_crawl/home_and_hobbies/0011": 28.7261,
    "common_crawl/home_and_hobbies/0012": 31.5137,
    "common_crawl/home_and_hobbies/0013": 35.0505,
    "common_crawl/home_and_hobbies/0014": 39.1018,
    "common_crawl/home_and_hobbies/0015": 44.0560,
    "common_crawl/home_and_hobbies/0016": 50.7280,
    "common_crawl/home_and_hobbies/0017": 56.6170,
    "common_crawl/home_and_hobbies/0018": 54.8373,
    "common_crawl/home_and_hobbies/0019": 44.7009,
    "common_crawl/industrial/0003": 10.8213,
    "common_crawl/industrial/0004": 5.3171,
    "common_crawl/industrial/0005": 4.7163,
    "common_crawl/industrial/0006": 4.9923,
    "common_crawl/industrial/0007": 5.4092,
    "common_crawl/industrial/0008": 5.8075,
    "common_crawl/industrial/0009": 6.1795,
    "common_crawl/industrial/0010": 6.5338,
    "common_crawl/industrial/0011": 6.7950,
    "common_crawl/industrial/0012": 7.0944,
    "common_crawl/industrial/0013": 7.4960,
    "common_crawl/industrial/0014": 8.0981,
    "common_crawl/industrial/0015": 8.6191,
    "common_crawl/industrial/0016": 9.2965,
    "common_crawl/industrial/0017": 9.7640,
    "common_crawl/industrial/0018": 9.5947,
    "common_crawl/industrial/0019": 8.3809,
    "common_crawl/literature/0004": 21.1829,
    "common_crawl/literature/0005": 7.6674,
    "common_crawl/literature/0006": 8.2332,
    "common_crawl/literature/0007": 9.2788,
    "common_crawl/literature/0008": 10.6653,
    "common_crawl/literature/0009": 11.8540,
    "common_crawl/literature/0010": 13.3324,
    "common_crawl/literature/0011": 15.0354,
    "common_crawl/literature/0012": 17.0215,
    "common_crawl/literature/0013": 19.5059,
    "common_crawl/literature/0014": 22.4556,
    "common_crawl/literature/0015": 26.0208,
    "common_crawl/literature/0016": 30.0368,
    "common_crawl/literature/0017": 37.1372,
    "common_crawl/literature/0018": 44.7503,
    "common_crawl/literature/0019": 33.0448,
    "common_crawl/politics/0002": 32.6222,
    "common_crawl/politics/0003": 16.6911,
    "common_crawl/politics/0004": 18.7018,
    "common_crawl/politics/0005": 21.0305,
    "common_crawl/politics/0006": 23.3714,
    "common_crawl/politics/0007": 25.6847,
    "common_crawl/politics/0008": 27.9326,
    "common_crawl/politics/0009": 30.4821,
    "common_crawl/politics/0010": 32.6118,
    "common_crawl/politics/0011": 35.7281,
    "common_crawl/politics/0012": 39.0471,
    "common_crawl/politics/0013": 42.2582,
    "common_crawl/politics/0014": 46.2495,
    "common_crawl/politics/0015": 49.6156,
    "common_crawl/politics/0016": 52.0312,
    "common_crawl/politics/0017": 54.9430,
    "common_crawl/politics/0018": 54.8526,
    "common_crawl/politics/0019": 44.3075,
    "common_crawl/religion/0003": 16.4091,
    "common_crawl/religion/0004": 6.7081,
    "common_crawl/religion/0005": 7.3877,
    "common_crawl/religion/0006": 8.3482,
    "common_crawl/religion/0007": 9.3263,
    "common_crawl/religion/0008": 10.4817,
    "common_crawl/religion/0009": 11.7045,
    "common_crawl/religion/0010": 13.1056,
    "common_crawl/religion/0011": 14.8457,
    "common_crawl/religion/0012": 16.9208,
    "common_crawl/religion/0013": 19.3030,
    "common_crawl/religion/0014": 21.9973,
    "common_crawl/religion/0015": 24.4685,
    "common_crawl/religion/0016": 26.9195,
    "common_crawl/religion/0017": 29.4571,
    "common_crawl/religion/0018": 31.0457,
    "common_crawl/religion/0019": 22.0503,
    "common_crawl/science_math_and_technology/0002": 18.1809,
    "common_crawl/science_math_and_technology/0003": 9.6556,
    "common_crawl/science_math_and_technology/0004": 10.5078,
    "common_crawl/science_math_and_technology/0005": 11.8670,
    "common_crawl/science_math_and_technology/0006": 12.8776,
    "common_crawl/science_math_and_technology/0007": 14.3939,
    "common_crawl/science_math_and_technology/0008": 15.8998,
    "common_crawl/science_math_and_technology/0009": 17.5606,
    "common_crawl/science_math_and_technology/0010": 19.9104,
    "common_crawl/science_math_and_technology/0011": 22.5940,
    "common_crawl/science_math_and_technology/0012": 25.6444,
    "common_crawl/science_math_and_technology/0013": 28.5507,
    "common_crawl/science_math_and_technology/0014": 31.8361,
    "common_crawl/science_math_and_technology/0015": 35.8001,
    "common_crawl/science_math_and_technology/0016": 40.4560,
    "common_crawl/science_math_and_technology/0017": 38.2456,
    # Reused equivalent-tokenizer caches outside tokenized/dolma3_pool.
    "finemath_3plus": 34.0019,  # 34,001,855,255 tokens
    "arxiv": 28.2376,  # 28,237,567,983 tokens
    # Stack-Edu — 15 partitions, 134.07B total (measured from hydrated/tokenized caches).
    "stack_edu/C": 4.7191,  # 4,719,072,920 tokens
    "stack_edu/CSharp": 7.1877,  # 7,187,732,301 tokens
    "stack_edu/Cpp": 12.4844,  # 12,484,367,348 tokens
    "stack_edu/Go": 1.3993,  # 1,399,268,360 tokens
    "stack_edu/Java": 31.2876,  # 31,287,573,657 tokens
    "stack_edu/JavaScript": 8.8594,  # 8,859,382,448 tokens
    "stack_edu/Markdown": 26.5106,  # 26,510,597,487 tokens
    "stack_edu/PHP": 7.3828,  # 7,382,771,394 tokens
    "stack_edu/Python": 17.9413,  # 17,941,260,722 tokens
    "stack_edu/Ruby": 1.3882,  # 1,388,223,902 tokens
    "stack_edu/Rust": 1.4190,  # 1,419,025,899 tokens
    "stack_edu/SQL": 6.9486,  # 6,948,605,401 tokens
    "stack_edu/Shell": 2.5468,  # 2,546,795,208 tokens
    "stack_edu/Swift": 1.5048,  # 1,504,820,240 tokens
    "stack_edu/TypeScript": 2.4916,  # 2,491,556,983 tokens
    "wikipedia": 3.6691,  # 3,669,138,258 tokens
}

DOLMA3_POOL_REUSED_EQUIVALENT_CACHE_PARTITIONS = frozenset({"finemath_3plus", "arxiv", "wikipedia"})

DOLMA3_POOL_COMPLETED_PARTITIONS: tuple[str, ...] = tuple(DOLMA3_POOL_TOKEN_COUNTS_B)

DOLMA3_POOL_KNOWN_BROKEN_PARTITIONS: tuple[str, ...] = tuple(
    partition for partition in DOLMA3_POOL_PARTITIONS if partition.startswith("olmocr_pdfs/")
)

DOLMA3_POOL_INCOMPLETE_PARTITIONS: tuple[str, ...] = tuple(
    partition for partition in DOLMA3_POOL_PARTITIONS if partition not in DOLMA3_POOL_TOKEN_COUNTS_B
)

# =============================================================================
# DOWNLOAD STEPS
# =============================================================================

_download_step_cache: dict[tuple[str, str], ExecutorStep] = {}


def _worker_resources_cache_key(worker_resources: ResourceConfig | None) -> str:
    return "default" if worker_resources is None else repr(worker_resources)


def _download_step(
    *,
    name: str,
    hf_dataset_id: str,
    revision: str,
    worker_resources: ResourceConfig | None = None,
) -> ExecutorStep:
    cache_key = (name, _worker_resources_cache_key(worker_resources))
    if cache_key not in _download_step_cache:
        _download_step_cache[cache_key] = ExecutorStep(
            name=name,
            fn=download_hf,
            config=DownloadConfig(
                hf_dataset_id=hf_dataset_id,
                revision=revision,
                gcs_output_path=this_output_path(),
                wait_for_completion=True,
                worker_resources=worker_resources,
            ),
        )
    return _download_step_cache[cache_key]


def download_dolma3_pool(*, worker_resources: ResourceConfig | None = None) -> ExecutorStep:
    """Get the download step for the main Dolma 3 Pool dataset (CC + olmOCR)."""
    return _download_step(
        name="raw/dolma3_pool",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        worker_resources=worker_resources,
    )


def download_stack_edu(*, worker_resources: ResourceConfig | None = None) -> ExecutorStep:
    """Get the download step for Stack-Edu dataset."""
    return _download_step(
        name="raw/stack_edu",
        hf_dataset_id=HF_STACK_EDU_ID,
        revision="main",
        worker_resources=worker_resources,
    )


def download_finemath(*, worker_resources: ResourceConfig | None = None) -> ExecutorStep:
    """Get the download step for FineMath dataset."""
    return _download_step(
        name="raw/finemath",
        hf_dataset_id=HF_FINEMATH_ID,
        revision="main",
        worker_resources=worker_resources,
    )


def download_dolma_v17(*, worker_resources: ResourceConfig | None = None) -> ExecutorStep:
    """Get the download step for Dolma v1.7 dataset (arXiv, Wikipedia)."""
    return _download_step(
        name="raw/dolma_v1.7",
        hf_dataset_id=HF_DOLMA_V17_ID,
        revision="main",
        worker_resources=worker_resources,
    )


def _uses_reused_equivalent_cache(partition_name: str, tokenizer: str) -> bool:
    """Return whether a partition reuses an existing equivalent-tokenizer cache."""
    return (
        tokenizer in {marin_tokenizer, llama3_tokenizer}
        and partition_name in DOLMA3_POOL_REUSED_EQUIVALENT_CACHE_PARTITIONS
    )


def download_all_dolma3_pool_sources(
    partitions: list[str] | None = None,
    tokenizer: str | None = None,
    worker_resources: ResourceConfig | None = None,
) -> list[ExecutorStep]:
    """Get the download steps required for the requested Dolma 3 Pool partitions.

    Returns:
        List of ExecutorSteps for the source datasets that still need raw inputs.
    """
    if partitions is None:
        partitions = list(DOLMA3_POOL_PARTITIONS)

    if tokenizer is None:
        tokenizer = marin_tokenizer

    download_steps: dict[str, ExecutorStep] = {}
    for partition_name in partitions:
        if _uses_reused_equivalent_cache(partition_name, tokenizer):
            continue

        if partition_name.startswith("common_crawl/") or partition_name.startswith("olmocr_pdfs/"):
            step = download_dolma3_pool(worker_resources=worker_resources)
        elif partition_name.startswith("stack_edu/"):
            step = download_stack_edu(worker_resources=worker_resources)
        elif partition_name == "finemath_3plus":
            step = download_finemath(worker_resources=worker_resources)
        elif partition_name in {"arxiv", "wikipedia"}:
            step = download_dolma_v17(worker_resources=worker_resources)
        else:
            raise ValueError(f"Unknown partition source: {partition_name}")

        download_steps[step.name] = step

    return list(download_steps.values())


def _get_dolma3_pool_base_dir(*, worker_resources: ResourceConfig | None = None):
    """Get the base directory for main Dolma 3 Pool data (CC + olmOCR)."""
    # Note: append_sha_to_path=False in DownloadConfig, so files are directly under output_path/data/
    return download_dolma3_pool(worker_resources=worker_resources).cd("data")


def _get_stack_edu_base_dir(*, worker_resources: ResourceConfig | None = None):
    """Get the base directory for Stack-Edu data."""
    # Stack-Edu raw downloads are written directly under the dataset root
    # (e.g. gs://.../raw/stack_edu-<sha>/Python/train-*.parquet), not under
    # an intermediate data/ directory.
    return download_stack_edu(worker_resources=worker_resources)


def _get_finemath_base_dir(*, worker_resources: ResourceConfig | None = None):
    """Get the base directory for FineMath data."""
    return download_finemath(worker_resources=worker_resources).cd("data")


def _get_dolma_v17_base_dir(*, worker_resources: ResourceConfig | None = None):
    """Get the base directory for Dolma v1.7 data."""
    return download_dolma_v17(worker_resources=worker_resources).cd("data")


# =============================================================================
# TOKENIZATION
# =============================================================================

_tokenize_cache: dict[str, ExecutorStep] = {}
_stack_edu_hydration_cache: dict[str, ExecutorStep] = {}
STACK_EDU_HYDRATION_MAX_WORKERS = 200


def hydrate_stack_edu_subset(
    partition_name: str,
    *,
    worker_resources: ResourceConfig | None = None,
) -> ExecutorStep:
    """Create a hydration step for a Stack-Edu partition."""
    if not partition_name.startswith("stack_edu/"):
        raise ValueError(f"Expected a Stack-Edu partition, got {partition_name}")

    cache_key = f"{partition_name}:{_worker_resources_cache_key(worker_resources)}"
    if cache_key in _stack_edu_hydration_cache:
        return _stack_edu_hydration_cache[cache_key]

    language = partition_name.removeprefix("stack_edu/")
    step = ExecutorStep(
        name=f"documents/stack_edu/{language}",
        fn=hydrate_stack_edu_text,
        config=StackEduHydrationConfig(
            input_path=_get_stack_edu_base_dir(worker_resources=worker_resources) / language,
            output_path=this_output_path(),
            language=language,
            max_rows_per_task=versioned(20_000),
            max_workers=STACK_EDU_HYDRATION_MAX_WORKERS,
            worker_resources=worker_resources,
            max_retries_per_blob=8,
            pipeline_version=versioned("v1"),
        ),
    )
    _stack_edu_hydration_cache[cache_key] = step
    return step


def get_stack_edu_hydration_steps(
    partitions: list[str] | None = None,
    *,
    worker_resources: ResourceConfig | None = None,
) -> dict[str, ExecutorStep]:
    """Create hydration steps for the requested Stack-Edu partitions."""
    if partitions is None:
        partitions = get_stack_edu_partitions()

    return {
        partition_name: hydrate_stack_edu_subset(partition_name, worker_resources=worker_resources)
        for partition_name in partitions
        if partition_name.startswith("stack_edu/")
    }


def _resolve_partition_paths(
    partition_name: str,
    *,
    worker_resources: ResourceConfig | None = None,
) -> list:
    """Resolve partition directory patterns to actual paths.

    Args:
        partition_name: Name of the partition (e.g., "common_crawl/adult_content/0005")

    Returns:
        List of InputName paths for the partition's data files.
    """
    dir_patterns = DOLMA3_POOL_PARTITIONS[partition_name]

    # Determine which source dataset this partition comes from
    if partition_name.startswith("common_crawl/") or partition_name.startswith("olmocr_pdfs/"):
        base_dir = _get_dolma3_pool_base_dir(worker_resources=worker_resources)
        file_pattern = "**/*.jsonl.zst"
    elif partition_name.startswith("stack_edu/"):
        return [hydrate_stack_edu_subset(partition_name, worker_resources=worker_resources).cd("train") / "*.jsonl.zst"]
    elif partition_name == "finemath_3plus":
        base_dir = _get_finemath_base_dir(worker_resources=worker_resources)
        file_pattern = "**/*.parquet"  # FineMath uses parquet format
    elif partition_name == "arxiv":
        base_dir = _get_dolma_v17_base_dir(worker_resources=worker_resources)
        file_pattern = "**/*.json.gz"  # Dolma v1.7 uses json.gz
    elif partition_name == "wikipedia":
        base_dir = _get_dolma_v17_base_dir(worker_resources=worker_resources)
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
    *,
    worker_resources: ResourceConfig | None = None,
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

    if tokenizer is None:
        tokenizer = marin_tokenizer

    cache_key = f"{partition_name}:{tokenizer}:{_worker_resources_cache_key(worker_resources)}"
    if cache_key in _tokenize_cache:
        return _tokenize_cache[cache_key]

    if _uses_reused_equivalent_cache(partition_name, tokenizer):
        if partition_name == "finemath_3plus":
            step = finemath_3_plus_tokenized
        elif partition_name == "arxiv":
            step = tokenize_dolma(tokenizer=llama3_tokenizer)["dolma/arxiv"]
        elif partition_name == "wikipedia":
            step = tokenize_dolma(tokenizer=llama3_tokenizer)["dolma/wiki"]
        else:
            raise ValueError(f"Unexpected reused-cache partition: {partition_name}")

        _tokenize_cache[cache_key] = step
        return step

    # Create output path
    safe_name = partition_name.replace("/", "_")
    output_path = f"tokenized/dolma3_pool/{safe_name}"

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


def tokenize_dolma3_pool(
    tokenizer: str | None = None,
    partitions: list[str] | None = None,
    *,
    worker_resources: ResourceConfig | None = None,
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
        steps[partition] = tokenize_dolma3_pool_subset(
            partition,
            tokenizer=tokenizer,
            worker_resources=worker_resources,
        )

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
