#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["google-cloud-storage"]
# ///
"""Count exact token counts for Dolma 3 Dolmino Pool partitions from GCS caches.

This script scans completed tokenized caches under
`gs://marin-us-central1/tokenized/dolma3_dolmino_pool/`, reads the
`train/.stats.json` files, and emits Python code for
`experiments/pretraining_datasets/dolma3_dolmino_pool.py`.

Usage:
    uv run experiments/domain_phase_mix/count_dolmino_pool_tokens.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from google.cloud import storage

GCS_BUCKET = "marin-us-central1"
GCS_PREFIX = "tokenized/dolma3_dolmino_pool/"

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
OLMOCR_TOPICS = [topic for topic in TOPICS if topic != "social_life"]
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
SYNTHETIC_PARTITIONS = {
    "synth_qa_nemotron_synth_qa": "synth_qa/nemotron_synth_qa",
    "synth_qa_reddit_to_flashcards": "synth_qa/reddit_to_flashcards",
    "synth_qa_wiki_to_rcqa": "synth_qa/wiki_to_rcqa",
    "synth_code_cranecode": "synth_code/cranecode",
    "synth_thinking_llama_nemotron_reasoning": "synth_thinking/llama_nemotron_reasoning",
    "synth_thinking_openthoughts2_reasoning": "synth_thinking/openthoughts2_reasoning",
    "synth_thinking_qwq_reasoning": "synth_thinking/qwq_reasoning",
    "synth_thinking_general_reasoning_mix": "synth_thinking/general_reasoning_mix",
    "synth_thinking_code_meta_reasoning": "synth_thinking/code_meta_reasoning",
    "synth_thinking_math_meta_reasoning": "synth_thinking/math_meta_reasoning",
    "synth_thinking_omr_rewrite_fullthoughts": "synth_thinking/omr_rewrite_fullthoughts",
    "synth_thinking_program_verifiable": "synth_thinking/program_verifiable",
    "synth_thinking_gemini_reasoning": "synth_thinking/gemini_reasoning",
    "synth_thinking_r1_reasoning": "synth_thinking/r1_reasoning",
    "synth_math_dolmino_math": "synth_math/dolmino_math",
    "synth_math_cranemath": "synth_math/cranemath",
    "synth_math_megamatt": "synth_math/megamatt",
    "synth_math_tinymath_mind": "synth_math/tinymath_mind",
    "synth_math_tinymath_pot": "synth_math/tinymath_pot",
    "synth_math_verifiable_gpt41": "synth_math/verifiable_gpt41",
    "synth_math_verifiable_o4mini": "synth_math/verifiable_o4mini",
    "synth_instruction_dolmino_flan": "synth_instruction/dolmino_flan",
    "synth_instruction_tulu_3_sft": "synth_instruction/tulu_3_sft",
}

DOLMINO_POOL_PARTITIONS = {
    **{f"common_crawl_hq/{year}_{topic}": [] for year in ("19", "20") for topic in TOPICS},
    **{f"olmocr_pdfs_hq/{topic}": [] for topic in OLMOCR_TOPICS},
    **{f"stack_edu_fim/{language}": [] for language in STACK_EDU_LANGUAGES},
    **{f"stem_heavy_crawl/{topic}": [] for topic in TOPICS},
    **{partition: [] for partition in SYNTHETIC_PARTITIONS.values()},
}


def find_completed_caches(
    bucket_name: str,
    prefix: str,
) -> tuple[dict[str, int], dict[str, list[tuple[str, int]]], list[str]]:
    """Find completed caches and exact token counts for each partition."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    safe_to_partition = {partition.replace("/", "_"): partition for partition in DOLMINO_POOL_PARTITIONS}

    results: dict[str, int] = {}
    duplicates: dict[str, list[tuple[str, int]]] = defaultdict(list)
    unmapped: list[str] = []

    stats_blobs = list(bucket.list_blobs(prefix=prefix, match_glob="**/.stats.json"))
    print(f"Found {len(stats_blobs)} .stats.json files", file=sys.stderr)

    for blob in stats_blobs:
        parts = blob.name.split("/")
        if len(parts) < 5 or parts[-1] != ".stats.json" or parts[-2] != "train":
            continue

        dir_with_hash = parts[2]
        safe_name = dir_with_hash.rsplit("-", 1)[0]
        partition = safe_to_partition.get(safe_name)
        if partition is None:
            unmapped.append(dir_with_hash)
            continue

        tokens = json.loads(blob.download_as_string()).get("total_tokens", 0)
        duplicates[partition].append((dir_with_hash, tokens))
        if partition not in results or tokens > results[partition]:
            results[partition] = tokens

    duplicate_summary = {partition: entries for partition, entries in duplicates.items() if len(entries) > 1}
    return results, duplicate_summary, unmapped


def _print_group_summary(results: dict[str, int], prefix: str, label: str) -> None:
    group = {partition: tokens for partition, tokens in results.items() if partition.startswith(prefix)}
    total = sum(group.values())
    print(f"# {label}: {len(group)} partitions, {total:,} tokens ({total / 1e9:.2f}B)")


def _print_top_level_summary(results: dict[str, int]) -> None:
    top_level = {
        "dolmino_common_crawl_hq": sum(
            tokens for partition, tokens in results.items() if partition.startswith("common_crawl_hq/")
        ),
        "dolmino_olmocr_pdfs_hq": sum(
            tokens for partition, tokens in results.items() if partition.startswith("olmocr_pdfs_hq/")
        ),
        "dolmino_stack_edu_fim": sum(
            tokens for partition, tokens in results.items() if partition.startswith("stack_edu_fim/")
        ),
        "dolmino_stem_heavy_crawl": sum(
            tokens for partition, tokens in results.items() if partition.startswith("stem_heavy_crawl/")
        ),
        "dolmino_synth_code": sum(
            tokens for partition, tokens in results.items() if partition.startswith("synth_code/")
        ),
        "dolmino_synth_instruction": sum(
            tokens for partition, tokens in results.items() if partition.startswith("synth_instruction/")
        ),
        "dolmino_synth_math": sum(
            tokens for partition, tokens in results.items() if partition.startswith("synth_math/")
        ),
        "dolmino_synth_qa": sum(tokens for partition, tokens in results.items() if partition.startswith("synth_qa/")),
        "dolmino_synth_thinking": sum(
            tokens for partition, tokens in results.items() if partition.startswith("synth_thinking/")
        ),
    }

    print("\n# Top-level Dolmino domains:")
    for name, total in top_level.items():
        print(f'#   "{name}": {total:,} tokens ({total / 1e9:.2f}B)')


def main() -> None:
    print("Querying GCS for completed Dolmino Pool tokenized caches...", file=sys.stderr)
    results, duplicates, unmapped = find_completed_caches(GCS_BUCKET, GCS_PREFIX)
    missing = sorted(set(DOLMINO_POOL_PARTITIONS) - set(results))

    print("\n# ===== RESULTS =====\n")
    print(f"# Completed partitions: {len(results)} / {len(DOLMINO_POOL_PARTITIONS)}")
    _print_group_summary(results, "common_crawl_hq/", "Common Crawl HQ")
    _print_group_summary(results, "olmocr_pdfs_hq/", "olmOCR PDFs HQ")
    _print_group_summary(results, "stack_edu_fim/", "Stack-Edu FIM")
    _print_group_summary(results, "stem_heavy_crawl/", "STEM Heavy Crawl")
    synth_total = sum(tokens for partition, tokens in results.items() if partition.startswith("synth_"))
    synth_partition_count = sum(1 for partition in results if partition.startswith("synth_"))
    print(f"# Synthetic: {synth_partition_count} partitions, {synth_total:,} tokens " f"({synth_total / 1e9:.2f}B)")
    _print_top_level_summary(results)
    grand_total = sum(results.values())
    print(f"# Grand total: {grand_total:,} tokens ({grand_total / 1e9:.2f}B)")

    if duplicates:
        print("\n# Partitions with multiple completed caches:")
        for partition, entries in sorted(duplicates.items()):
            print(f"#   {partition}: {entries}")

    if unmapped:
        print("\n# Unmapped cache directories:")
        for dirname in sorted(unmapped):
            print(f"#   {dirname}")

    if missing:
        print("\n# Missing partitions:")
        for partition in missing:
            print(f"#   {partition}")

    print("\n# ===== Python code for dolma3_dolmino_pool.py =====\n")
    print("DOLMINO_POOL_TOKEN_COUNTS_B: dict[str, float] = {")
    for partition in sorted(results):
        tokens = results[partition]
        print(f'    "{partition}": {tokens / 1e9:.4f},  # {tokens:,} tokens')
    print("}")
    print()
    print("DOLMINO_POOL_COMPLETED_PARTITIONS: tuple[str, ...] = tuple(DOLMINO_POOL_TOKEN_COUNTS_B)")


if __name__ == "__main__":
    main()
