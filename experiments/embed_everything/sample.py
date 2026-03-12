# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample documents from Nemotron CC quality buckets and Dolma source splits.

Produces JSONL files with stratified train/test splits for downstream
oracle labeling and embedding evaluation.
"""

import gzip
import hashlib
import json
import logging
import os
import random

from iris.marin_fs import open_url, url_to_fs

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
MAX_FILES_PER_STRATUM = 3

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


def _stable_doc_id(text: str, source: str) -> str:
    """Generate a stable document ID from content hash."""
    return hashlib.sha256(f"{source}:{text[:1000]}".encode()).hexdigest()[:16]


def _read_gzipped_jsonl(gcs_path: str, max_records: int = 0):
    """Stream JSONL records from a .json.gz or .jsonl.gz file on GCS.

    Streams decompression to avoid loading multi-GiB files into memory.
    If max_records > 0, stops after yielding that many records.
    """
    count = 0
    with open_url(gcs_path, "rb") as raw:
        with gzip.GzipFile(fileobj=raw) as gz:
            for raw_line in gz:
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)
                    count += 1
                    if max_records > 0 and count >= max_records:
                        return


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
            try:
                for doc in _read_gzipped_jsonl(full_path, max_records=max_per_file):
                    count += 1
                    text = doc.get("text", "")
                    if not text:
                        continue
                    record = {
                        "doc_id": doc.get("id", _stable_doc_id(text, doc.get("source", "unknown"))),
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
            except Exception:
                logger.exception("Error reading %s", full_path)
                continue

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


def _write_jsonl(docs: list[dict], output_path: str) -> None:
    """Write documents to a JSONL file."""
    with open_url(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


def sample_quality_documents(
    output_path: str,
    nemotron_base_path: str,
    n_per_bucket: int = 25,
    seed: int = RANDOM_SEED,
    train_fraction: float = TRAIN_FRACTION,
) -> None:
    """Sample documents from each Nemotron CC quality bucket."""
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

    _write_jsonl(train + test, os.path.join(output_path, "quality_samples.jsonl"))
    _write_jsonl(train, os.path.join(output_path, "quality_train.jsonl"))
    _write_jsonl(test, os.path.join(output_path, "quality_test.jsonl"))

    logger.info(
        "Sampled %d quality documents (%d train, %d test) to %s",
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

    _write_jsonl(train + test, os.path.join(output_path, "topic_samples.jsonl"))
    _write_jsonl(train, os.path.join(output_path, "topic_train.jsonl"))
    _write_jsonl(test, os.path.join(output_path, "topic_test.jsonl"))

    logger.info(
        "Sampled %d topic documents (%d train, %d test) to %s",
        len(all_docs),
        len(train),
        len(test),
        output_path,
    )
