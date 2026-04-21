# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample documents from Nemotron CC quality buckets and Dolma source splits.

Produces JSONL files with stratified train/test splits for downstream
oracle labeling and embedding evaluation.
"""

import logging
import os
import random
from itertools import islice

from rigging.filesystem import url_to_fs
from zephyr.readers import load_jsonl
from zephyr.writers import write_parquet_file

from marin.datakit.normalize import generate_id

logger = logging.getLogger(__name__)

# Nemotron CC quality buckets (actual only, skip synthetic per issue discussion).
# Files live under kind2=actual/ (not directly under kind=actual/).
NEMOTRON_QUALITY_BUCKETS = {
    "high": "quality=high/kind=actual/kind2=actual",
    "medium_high": "quality=medium-high/kind=actual/kind2=actual",
    "medium": "quality=medium/kind=actual/kind2=actual",
    "medium_low": "quality=medium-low/kind=actual/kind2=actual",
    "low": "quality=low/kind=actual/kind2=actual",
}

# Dolma 1.7 source splits
DOLMA_SOURCES = [
    "algebraic-stack",
    "arxiv",
    "gutenberg",
    "c4",
    "cc",
    "cc-news",
    "falcon",
    "megawika",
    "open-web-math",
    "pes2o",
    "reddit",
    "stackexchange",
    "starcoder",
    "flan",
    "wiki",
]

TRAIN_FRACTION = 0.8
RANDOM_SEED = 42
MAX_FILES_PER_STRATUM = 42

# Mapping from Dolma source names to their file glob patterns
DOLMA_SOURCE_PATTERNS: dict[str, list[str]] = {
    "algebraic-stack": ["algebraic-stack-train-*.json.gz"],
    "arxiv": ["arxiv-*.json.gz"],
    "gutenberg": ["books-*.json.gz"],
    "c4": ["c4-*.json.gz"],
    "cc": ["cc_en_head-*.json.gz", "cc_en_middle-*.json.gz", "cc_en_tail-*.json.gz"],
    "cc-news": ["cc_news_head-*.json.gz", "cc_news_middle-*.json.gz", "cc_news_tail-*.json.gz"],
    "falcon": ["falcon-*.json.gz"],
    "megawika": ["megawika-*.json.gz"],
    "open-web-math": ["open-web-math-train-*.json.gz"],
    "pes2o": ["pes2o-*.json.gz"],
    "reddit": ["reddit-*.json.gz"],
    "stackexchange": ["stackexchange-*.json.gz"],
    "starcoder": ["starcoder-*.json.gz"],
    "flan": ["tulu_flan-*.json.gz"],
    "wiki": ["wiki-*.json.gz"],
}

# Binary quality strategy (Dolma): known-high = arxiv + wiki, known-low = cc_en_tail.
# Complements the Nemotron graduated buckets by providing an unambiguous
# high/low contrast. "Low" uses cc_en_tail rather than the full cc split
# because Dolma's tail shard is its lowest-quality CC slice.
DOLMA_QUALITY_BINARY_PATTERNS: dict[str, list[str]] = {
    "high": ["arxiv-*.json.gz", "wiki-*.json.gz"],
    "low": ["cc_en_tail-*.json.gz"],
}


def _sample_from_files(
    file_patterns: list[str],
    n_per_stratum: int,
    label: str,
    label_field: str,
    seed: int,
    max_files: int = MAX_FILES_PER_STRATUM,
) -> list[dict]:
    """Sample n documents from a set of file patterns.

    Uses reservoir sampling. Limits the number of files globbed per pattern
    to avoid slow recursive listings on GCS.
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    count = 0

    for pattern in file_patterns:
        fs, path = url_to_fs(pattern)
        matched = sorted(fs.glob(path))
        if not matched:
            logger.warning("No files matched pattern: %s", pattern)
            continue

        # Only read a few files — we only need n_per_stratum docs total
        rng_files = random.Random(seed)
        rng_files.shuffle(matched)
        matched = matched[:max_files]

        for filepath in matched:
            full_path = f"gs://{filepath}" if not filepath.startswith("gs://") else filepath
            # Read at most 10x the target sample size per file to bound memory/time
            max_per_file = n_per_stratum * 10
            for doc in islice(load_jsonl(full_path), max_per_file):
                count += 1
                text = doc.get("text", "")
                if not text:
                    continue
                record = {
                    "doc_id": doc.get("id", generate_id(text)),
                    "text": text,
                    "source": doc.get("source", "unknown"),
                    label_field: label,
                }
                # Reservoir sampling
                if len(reservoir) < n_per_stratum:
                    reservoir.append(record)
                else:
                    j = rng.randint(0, count - 1)
                    if j < n_per_stratum:
                        reservoir[j] = record

    logger.info("Sampled %d documents from %d total for label=%s", len(reservoir), count, label)
    return reservoir


def _train_test_split(docs: list[dict], train_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split documents into train/test with deterministic shuffle."""
    rng = random.Random(seed)
    shuffled = list(docs)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_fraction)
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]
    for doc in train:
        doc["split"] = "train"
    for doc in test:
        doc["split"] = "test"
    return train, test


def sample_quality_documents_nemotron(
    output_path: str,
    nemotron_base_path: str,
    n_per_bucket: int = 25,
    seed: int = RANDOM_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> None:
    """Sample documents from each Nemotron CC quality bucket (5-bucket graduated)."""
    all_docs: list[dict] = []

    for bucket_name, bucket_path in NEMOTRON_QUALITY_BUCKETS.items():
        pattern = os.path.join(nemotron_base_path, bucket_path, "*.jsonl.gz")
        docs = _sample_from_files(
            file_patterns=[pattern],
            n_per_stratum=n_per_bucket,
            label=bucket_name,
            label_field="quality_bucket",
            seed=seed + hash(bucket_name),
        )
        all_docs.extend(docs)

    train, test = _train_test_split(all_docs, train_fraction, seed)

    write_parquet_file(train + test, os.path.join(output_path, "quality_samples.parquet"))
    write_parquet_file(train, os.path.join(output_path, "quality_train.parquet"))
    write_parquet_file(test, os.path.join(output_path, "quality_test.parquet"))

    logger.info(
        "Sampled %d quality documents (%d train, %d test) to %s",
        len(all_docs),
        len(train),
        len(test),
        output_path,
    )


def sample_quality_documents_binary(
    output_path: str,
    dolma_base_path: str,
    n_per_bucket: int = 25,
    seed: int = RANDOM_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> None:
    """Sample documents for a 2-bucket quality strategy from Dolma.

    "high" draws from arxiv + wiki; "low" draws from cc_en_tail. Pairs with
    sample_quality_documents_nemotron (5-bucket graduated) so downstream
    evaluation can compare a clean high/low contrast against a graded signal.
    """
    all_docs: list[dict] = []

    for bucket_name, patterns in DOLMA_QUALITY_BINARY_PATTERNS.items():
        full_patterns = [os.path.join(dolma_base_path, p) for p in patterns]
        docs = _sample_from_files(
            file_patterns=full_patterns,
            n_per_stratum=n_per_bucket,
            label=bucket_name,
            label_field="quality_bucket",
            seed=seed + hash(bucket_name),
        )
        all_docs.extend(docs)

    train, test = _train_test_split(all_docs, train_fraction, seed)

    write_parquet_file(train + test, os.path.join(output_path, "quality_samples.parquet"))
    write_parquet_file(train, os.path.join(output_path, "quality_train.parquet"))
    write_parquet_file(test, os.path.join(output_path, "quality_test.parquet"))

    logger.info(
        "Sampled %d binary-quality documents (%d train, %d test) to %s",
        len(all_docs),
        len(train),
        len(test),
        output_path,
    )


def sample_topic_documents(
    output_path: str,
    dolma_base_path: str,
    n_per_source: int = 25,
    seed: int = RANDOM_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> None:
    """Sample documents from each Dolma source split for topic evaluation."""
    all_docs: list[dict] = []

    for source_name in DOLMA_SOURCES:
        patterns = DOLMA_SOURCE_PATTERNS[source_name]
        full_patterns = [os.path.join(dolma_base_path, p) for p in patterns]
        docs = _sample_from_files(
            file_patterns=full_patterns,
            n_per_stratum=n_per_source,
            label=source_name,
            label_field="source_label",
            seed=seed + hash(source_name),
        )
        all_docs.extend(docs)

    train, test = _train_test_split(all_docs, train_fraction, seed)

    write_parquet_file(train + test, os.path.join(output_path, "topic_samples.parquet"))
    write_parquet_file(train, os.path.join(output_path, "topic_train.parquet"))
    write_parquet_file(test, os.path.join(output_path, "topic_test.parquet"))

    logger.info(
        "Sampled %d topic documents (%d train, %d test) to %s",
        len(all_docs),
        len(train),
        len(test),
        output_path,
    )
