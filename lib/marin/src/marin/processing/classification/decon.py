# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Decontamination using rbloom bloom filters.

Builds a bloom filter from a contamination source (eval data) and marks paragraphs
in the input data that appear in the filter.
"""

import hashlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass

import draccus
import dupekit
import msgspec
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file

from marin.execution.executor import THIS_OUTPUT_PATH
from marin.processing.classification.deduplication.dedup_commons import (
    DEFAULT_FILETYPES,
    _collect_input_files,
    _get_extension,
)
from marin.utils import rebase_file_path

logger = logging.getLogger(__name__)


@dataclass
class NGramConfig:
    """
    Configuration class for deduplication n-gram settings.

    Attributes:
        ngram_length (int): Size of the ngram (e.g. 8)
        stride (int): Step size when moving through string to generate ngrams
        overlap_threshold (float): Minimum fraction of matching ngrams for a paragraph
            to be recorded as a contaminated span
    """

    ngram_length: int = 8
    stride: int = 0
    overlap_threshold: float = 0.7


@dataclass(frozen=True)
class DeconConfig:
    """
    Configuration class for running decontamination.

    Attributes:
        input_path (str | list[str]): Path(s) of files to apply decontamination to.
        output_path (str): Path for storing results
        attribute_name (str): Name for key to store duplicate span info in json
        estimated_doc_count (int): estimated number of docs to deduplicate
        false_positive_rate (float): false positive rate for Bloom filter
        ngram (NGramConfig): settings for ngram matching including length, match threshold, and stride
        processes (int): number of processes to use
        decontaminate_source (str | None): source to seed bloom filter when decontaminating
        text_field (str): field to use for text content in Parquet files
    """

    input_path: str | list[str]
    output_path: str = THIS_OUTPUT_PATH
    attribute_name: str = "duplicate_text"
    estimated_doc_count: int = 1000000
    false_positive_rate: float = 0.001
    ngram: NGramConfig | None = None
    processes: int = 1
    decontaminate_source: str | None = None
    text_field: str = "text"


def _bloom_hash(x: str) -> int:
    if isinstance(x, bytes):
        return int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "big")
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "big")


def extract_ngrams(text: str, n: int, stride: int) -> Iterator[str]:
    """
    Extract n-grams from text based on config.
    """
    tokens: list[str] = text.split()

    for i in range(0, len(tokens) - n + 1, stride + 1):
        yield " ".join(tokens[i : i + n])


def extract_features(text: str, ngram_config: NGramConfig | None) -> Iterator[str]:
    """
    Extract features (paragraphs or n-grams) from text.
    """
    paragraphs = text.split("\n")

    for para in paragraphs:
        if ngram_config:
            yield from extract_ngrams(para, ngram_config.ngram_length, ngram_config.stride)
        else:
            # Exact paragraph matching
            yield para


def _record_id(record: dict) -> str:
    if "id" in record:
        return record["id"]
    else:
        # compute hash of the msgspec serialization of the record
        s = msgspec.msgpack.encode(record, order="deterministic")
        return str(_bloom_hash(s))


def build_filter(
    input_path: str | list[str],
    bloom_path: str,
    config: DeconConfig,
) -> str:
    """
    Build a bloom filter from input dataset.
    """

    def build_shard_bloom(records: Iterator[dict], _) -> Iterator[bytes]:
        """Build bloom filter from a shard of records and yield serialized bytes."""
        bf = dupekit.Bloom(config.estimated_doc_count, config.false_positive_rate)

        for record in records:
            text = record.get(config.text_field, "")
            for feature in extract_features(text, config.ngram):
                bf.add(_bloom_hash(feature))

        yield bf.save_bytes()

    all_files = _collect_input_files(input_paths=input_path, filetypes=DEFAULT_FILETYPES)
    logger.info(f"Building bloom filter from {all_files} into {bloom_path}")

    def _merge_bloom(bloom_files: Iterator[str], _):
        merged_bloom = dupekit.Bloom(config.estimated_doc_count, config.false_positive_rate)
        for bloom_file_path in bloom_files:
            fs, path = url_to_fs(bloom_file_path)
            with fs.open(path, "rb") as f:
                bloom_bytes = f.read()
            shard_bloom = dupekit.Bloom.load_bytes(bloom_bytes)
            merged_bloom.update(shard_bloom)
        yield merged_bloom.save_bytes()

    ctx = ZephyrContext(name="decon-build")
    # Build bloom filters for all shards in parallel
    shard_blooms_data = ctx.execute(
        Dataset.from_iterable(all_files)
        .reshard(num_shards=config.processes)
        .load_file()
        .select(config.text_field)
        .map_shard(build_shard_bloom)
        .write_binary(f"{bloom_path}-{{shard:05d}}-of-{{total:05d}}.bin", skip_existing=True),
    ).results

    if len(shard_blooms_data) == 1:
        return shard_blooms_data[0]

    logger.info(f"Merging {len(shard_blooms_data)} shard bloom filters...")
    merged_bloom = ctx.execute(
        Dataset.from_iterable(shard_blooms_data)
        .reshard(num_shards=1)
        .map_shard(_merge_bloom)
        .write_binary(bloom_path, skip_existing=True),
    ).results

    return merged_bloom[0]


def calculate_paragraph_overlap(
    paragraph: str, bloom_filter: "dupekit.Bloom", ngram_config: NGramConfig | None
) -> float:
    """
    Calculate overlap score for a paragraph against a bloom filter.
    """
    if ngram_config:
        ngrams = list(extract_ngrams(paragraph, ngram_config.ngram_length, ngram_config.stride))
        if not ngrams:
            # Paragraph too short for n-grams - fall back to exact paragraph matching
            return 1.0 if _bloom_hash(paragraph) in bloom_filter else 0.0
        else:
            # N-gram matching
            matches = sum(1 for ng in ngrams if _bloom_hash(ng) in bloom_filter)
            return matches / len(ngrams)
    else:
        # Exact paragraph matching
        return 1.0 if _bloom_hash(paragraph) in bloom_filter else 0.0


def mark_duplicates_bloom(
    input_path: str | list[str],
    bloom_path: str,
    output_path: str,
    config: DeconConfig,
) -> list[str]:
    """
    Apply bloom filter to input data, marking duplicate spans.
    """

    # Determine base path for rebasing
    base_path = input_path[0] if isinstance(input_path, list) else input_path
    all_files = _collect_input_files(input_paths=input_path, filetypes=DEFAULT_FILETYPES)

    # Threshold gates which paragraphs get recorded as contaminated spans.
    # Only meaningful for n-gram mode (score is a fraction in [0, 1]); in exact
    # paragraph mode score is 0 or 1 so any non-zero match is always recorded.
    threshold = config.ngram.overlap_threshold if config.ngram else 0.0

    def process_shard_with_bloom(records: Iterator[dict], _) -> Iterator[dict]:
        """Load bloom filter once per shard and mark duplicates."""
        # Load bloom filter from storage
        fs, path = url_to_fs(bloom_path)
        with fs.open(path, "rb") as f:
            bloom_bytes = f.read()
        bf = dupekit.Bloom.load_bytes(bloom_bytes)

        # Process each record
        for record in records:
            text = record.get(config.text_field, "")
            paragraphs = text.split("\n")
            duplicate_spans = []

            offset = 0
            for para in paragraphs:
                if not para:
                    offset += 1  # Just the newline
                    continue

                overlap_score = calculate_paragraph_overlap(para, bf, config.ngram)
                if overlap_score > 0 and overlap_score >= threshold:
                    duplicate_spans.append([offset, offset + len(para), overlap_score])
                offset += len(para) + 1  # +1 for newline

            yield {
                "id": _record_id(record),
                "attributes": {config.attribute_name: duplicate_spans},
            }

    # Use write_jsonl with callable output pattern
    zephyr_ctx = ZephyrContext(name="decon-mark")
    return zephyr_ctx.execute(
        Dataset.from_iterable(all_files)
        .flat_map(load_file)
        .map_shard(process_shard_with_bloom)
        .write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                all_files[shard_idx],
                output_path,
                new_extension=_get_extension(all_files[shard_idx]),
                old_extension=_get_extension(all_files[shard_idx]),
            ),
            skip_existing=True,
        ),
    ).results


def decontaminate(config: DeconConfig):
    """Build a bloom filter from ``decontaminate_source`` and mark matching paragraphs in ``input_path``."""
    if not config.decontaminate_source:
        raise ValueError("decontaminate_source is required")

    bloom_path = os.path.join(config.output_path, "bloom", "filter.bin")
    bloom_path = build_filter(config.decontaminate_source, bloom_path, config)
    mark_duplicates_bloom(config.input_path, bloom_path, config.output_path, config)

    return {"success": True}


@draccus.wrap()
def main(config: DeconConfig):

    configure_logging(level=logging.INFO)

    result = decontaminate(config)
    print(f"Decontamination completed: {result}")


if __name__ == "__main__":
    main()
