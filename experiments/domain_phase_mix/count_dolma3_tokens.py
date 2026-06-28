#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["google-cloud-storage"]
# ///
"""Count actual token counts for all Dolma 3 Pool partitions from GCS .stats.json files.

Finds completed tokenized caches (those with .stats.json in train/) and reads
the total_tokens field. Outputs Python code to update DOLMA3_POOL_TOKEN_COUNTS_B
in dolma3_pool.py.

Usage:
    uv run experiments/domain_phase_mix/count_dolma3_tokens.py
"""

import json
import sys

from google.cloud import storage

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import TOP_LEVEL_DOMAIN_TOKEN_COUNTS

GCS_BUCKET = "marin-us-central1"
GCS_PREFIX = "tokenized/dolma3_pool/"
EXTERNAL_CACHE_COUNTS = {
    # These three partitions reuse equivalent-tokenizer caches outside tokenized/dolma3_pool.
    # Their exact token counts were measured directly from the caches during the 2026-03 audit.
    "finemath_3plus": 34_001_855_255,
    "arxiv": 28_237_567_983,
    "wikipedia": 3_669_138_258,
}


def find_completed_caches(bucket_name: str, prefix: str) -> dict[str, int]:
    """Find all completed caches and their token counts.

    Returns:
        Dict mapping partition name -> total_tokens
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all .stats.json files under the prefix
    stats_blobs = list(bucket.list_blobs(prefix=prefix, match_glob="**/.stats.json"))
    print(f"Found {len(stats_blobs)} .stats.json files", file=sys.stderr)

    # Group by partition name (strip hash suffix)
    # Path format: tokenized/dolma3_pool/{partition_name}-{hash}/train/.stats.json
    results: dict[str, int] = {}
    duplicates: dict[str, list[tuple[str, int]]] = {}

    for blob in stats_blobs:
        # e.g. tokenized/dolma3_pool/common_crawl_adult_content_0007-fc8625/train/.stats.json
        path = blob.name
        parts = path.split("/")
        # parts = ["tokenized", "dolma3_pool", "common_crawl_adult_content_0007-fc8625", "train", ".stats.json"]
        if len(parts) < 5 or parts[-1] != ".stats.json" or parts[-2] != "train":
            continue

        dir_with_hash = parts[2]  # e.g. "common_crawl_adult_content_0007-fc8625"
        # Strip the hash suffix (last 7 chars after the last dash)
        last_dash = dir_with_hash.rfind("-")
        if last_dash == -1:
            continue
        partition_key = dir_with_hash[:last_dash]  # e.g. "common_crawl_adult_content_0007"

        # Convert back to partition name format: common_crawl_adult_content_0007 -> common_crawl/adult_content/0007
        # But we need to handle different partition types differently
        # For now just store the raw key and token count

        data = json.loads(blob.download_as_string())
        tokens = data.get("total_tokens", 0)

        if partition_key in results:
            if partition_key not in duplicates:
                duplicates[partition_key] = [(dir_with_hash, results[partition_key])]
            duplicates[partition_key].append((dir_with_hash, tokens))
            # Keep the larger count (more likely to be the complete one)
            if tokens > results[partition_key]:
                results[partition_key] = tokens
        else:
            results[partition_key] = tokens

    if duplicates:
        print(f"\nWARNING: {len(duplicates)} partitions have multiple completed caches:", file=sys.stderr)
        for key, entries in sorted(duplicates.items())[:10]:
            print(f"  {key}: {entries}", file=sys.stderr)
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more", file=sys.stderr)

    return results


def safe_name_to_partition(safe_name: str) -> str | None:
    """Convert GCS safe name back to partition name.

    Examples:
        common_crawl_adult_content_0007 -> common_crawl/adult_content/0007
        olmocr_pdfs_health -> olmocr_pdfs/health
        stack_edu_Python -> stack_edu/Python
        finemath_3plus -> finemath_3plus
        arxiv -> arxiv
        wikipedia -> wikipedia
    """
    # Common crawl: common_crawl_{topic}_{tier}
    if safe_name.startswith("common_crawl_"):
        rest = safe_name[len("common_crawl_") :]
        # The tier is always a 4-digit number at the end
        if len(rest) >= 5 and rest[-4:].isdigit() and rest[-5] == "_":
            topic = rest[:-5]
            tier = rest[-4:]
            return f"common_crawl/{topic}/{tier}"

    # olmOCR: olmocr_pdfs_{topic} or olmocr_pdfs_{topic}_part{n}
    # Actual format: olmocr_science_pdfs_{topic}
    # Wait, the safe_name comes from partition_name.replace("/", "_")
    # partition_name = "olmocr_pdfs/{topic}" -> safe_name = "olmocr_pdfs_{topic}"
    if safe_name.startswith("olmocr_pdfs_"):
        rest = safe_name[len("olmocr_pdfs_") :]
        # Check for part1/part2 suffix
        if rest.endswith("_part1"):
            topic = rest[:-6]
            return f"olmocr_pdfs/{topic}/part1"
        elif rest.endswith("_part2"):
            topic = rest[:-6]
            return f"olmocr_pdfs/{topic}/part2"
        else:
            return f"olmocr_pdfs/{rest}"

    # Stack-edu: stack_edu_{language}
    if safe_name.startswith("stack_edu_"):
        lang = safe_name[len("stack_edu_") :]
        return f"stack_edu/{lang}"

    # Simple names
    if safe_name in ("finemath_3plus", "arxiv", "wikipedia"):
        return safe_name

    return None


def main():
    print("Querying GCS for completed tokenized caches...", file=sys.stderr)
    raw_counts = find_completed_caches(GCS_BUCKET, GCS_PREFIX)
    print(f"\nFound {len(raw_counts)} completed partitions", file=sys.stderr)

    # Convert safe names to partition names
    partition_counts: dict[str, int] = {}
    unmapped: list[str] = []

    for safe_name, tokens in sorted(raw_counts.items()):
        partition = safe_name_to_partition(safe_name)
        if partition is not None:
            partition_counts[partition] = tokens
        else:
            unmapped.append(safe_name)

    partition_counts.update(EXTERNAL_CACHE_COUNTS)

    if unmapped:
        print(f"\nWARNING: Could not map {len(unmapped)} safe names to partitions:", file=sys.stderr)
        for name in unmapped:
            print(f"  {name}", file=sys.stderr)

    # Output results grouped by type
    print("\n# ===== RESULTS =====\n")

    # Common Crawl
    cc_partitions = {k: v for k, v in partition_counts.items() if k.startswith("common_crawl/")}
    cc_total = sum(cc_partitions.values())
    print(f"# Common Crawl: {len(cc_partitions)} partitions, {cc_total:,} tokens ({cc_total / 1e9:.2f}B)")

    # Group CC by topic for summary
    topic_totals: dict[str, tuple[int, int]] = {}  # topic -> (count, total_tokens)
    for partition, tokens in sorted(cc_partitions.items()):
        parts = partition.split("/")
        topic = parts[1]
        count, total = topic_totals.get(topic, (0, 0))
        topic_totals[topic] = (count + 1, total + tokens)

    print("\n# Per-topic CC totals:")
    for topic in sorted(topic_totals.keys()):
        count, total = topic_totals[topic]
        print(f"#   {topic}: {count} tiers, {total:,} tokens ({total / 1e9:.2f}B)")

    print("\n# Nextgen Dolma 3 CC domains:")
    nextgen_cc_domains = {
        domain_name: tokens
        for domain_name, tokens in TOP_LEVEL_DOMAIN_TOKEN_COUNTS.items()
        if domain_name.startswith("dolma3_cc/")
    }
    for domain_name, total in nextgen_cc_domains.items():
        print(f'#   "{domain_name}": {total:,} tokens ({total / 1e9:.2f}B)')

    # olmOCR
    olmocr_partitions = {k: v for k, v in partition_counts.items() if k.startswith("olmocr_pdfs/")}
    if olmocr_partitions:
        olmocr_total = sum(olmocr_partitions.values())
        print(f"\n# olmOCR: {len(olmocr_partitions)} partitions, {olmocr_total:,} tokens ({olmocr_total / 1e9:.2f}B)")

    # Stack-Edu
    stack_partitions = {k: v for k, v in partition_counts.items() if k.startswith("stack_edu/")}
    if stack_partitions:
        stack_total = sum(stack_partitions.values())
        print(f"\n# Stack-Edu: {len(stack_partitions)} partitions, {stack_total:,} tokens ({stack_total / 1e9:.2f}B)")
        print(f'#   "dolma3_stack_edu": {stack_total:,} tokens ({stack_total / 1e9:.2f}B)')

    # Others
    for name in ("finemath_3plus", "arxiv", "wikipedia"):
        if name in partition_counts:
            tokens = partition_counts[name]
            print(f"\n# {name}: {tokens:,} tokens ({tokens / 1e9:.2f}B)")
            print(f'#   "dolma3_{name}": {tokens:,} tokens ({tokens / 1e9:.2f}B)')

    # Output Python dict for dolma3_pool.py
    print("\n\n# ===== Python code for dolma3_pool.py =====\n")
    print("DOLMA3_POOL_TOKEN_COUNTS_B: dict[str, float] = {")
    for partition in sorted(partition_counts.keys()):
        tokens = partition_counts[partition]
        tokens_b = tokens / 1e9
        print(f'    "{partition}": {tokens_b:.4f},  # {tokens:,} tokens')
    print("}")

    # Summary
    grand_total = sum(partition_counts.values())
    print(f"\n# Grand total: {len(partition_counts)} partitions, {grand_total:,} tokens ({grand_total / 1e9:.2f}B)")


if __name__ == "__main__":
    main()
