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

"""
Decontamination using rbloom bloom filters.

This module provides two workflows:
1. DECONTAMINATE: Mark paragraphs that appear in a contamination source
2. TRAIN_TEST_OVERLAP: Detect train-test overlap using n-gram matching
"""

import hashlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum, auto
import typing

from marin.execution.executor import THIS_OUTPUT_PATH
import draccus
import fsspec
import msgspec
import wandb

from marin.utilities.wandb_utils import WANDB_PROJECT, WANDB_ENTITY

from marin.utils import fsspec_glob, rebase_file_path
from zephyr import Dataset, flow_backend
from zephyr.readers import load_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dupekit import Bloom


class DeconMode(StrEnum):
    DECONTAMINATE = auto()
    TRAIN_TEST_OVERLAP = auto()


@dataclass
class NGramConfig:
    """
    Configuration class for deduplication n-gram settings.

    Attributes:
        ngram_length (int | list[int]): Size of the ngram (e.g. 8) or list of sizes (e.g. [10, 15])
        stride (int): Step size when moving through string to generate ngrams
        overlap_threshold (float): Percentage of duplicate ngrams for a paragraph to be considered duplicate
    """

    ngram_length: int | list[int] = 8
    stride: int = 0
    overlap_threshold: float = 0.7


@dataclass(frozen=True)
class DeconConfig:
    """
    Configuration class for running decontamination/overlap checks.

    Attributes:
        input_path (str | list[str]): Path(s) of files to apply decontamination to.
        output_path (str): Path for storing results
        attribute_name (str): Name for key to store duplicate span info in json
        estimated_doc_count (int): estimated number of docs to deduplicate
        false_positive_rate (float): false positive rate for Bloom filter
        ngram (NGramConfig): settings for ngram matching including length, match threshold, and stride
        processes (int): number of processes to use
        mode (DeconMode): switch between decontamination (build filter) and regular deduplication
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
    mode: DeconMode = DeconMode.DECONTAMINATE
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


def _collect_input_files(input_path: str | list[str]) -> list[str]:
    """
    Given an input path or list of paths, collect all matching files (jsonl, parquet, etc).
    """
    input_paths = input_path if isinstance(input_path, list) else [input_path]
    all_files = []
    for path in input_paths:
        logger.info(f"Collecting files from path: {path}")
        files = fsspec_glob(f"{path.rstrip('/')}/**/*.{{jsonl,jsonl.gz,jsonl.zst,parquet}}")
        if files:
            all_files.extend(files)
        else:
            if not path.endswith(("jsonl", "jsonl.gz", "jsonl.zst", "parquet")):
                raise FileNotFoundError(f"No files found in path: {path}")
            all_files.append(path)  # Assume it's a single file
    assert all_files, "No input files found for deduplication."
    return all_files


def _init_wandb(config: DeconConfig, tags: list[str] | None = None):
    """
    Initialize wandb if configured.
    """
    if "WANDB_API_KEY" not in os.environ:
        return

    run_name = os.environ.get("WANDB_RUN_NAME")
    if not run_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_name = f"{config.mode}-{timestamp}"

    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        tags=[str(config.mode)] + (tags or []),
        config={
            "mode": str(config.mode),
            "input_path": config.input_path,
            "processes": config.processes,
        },
    )


def _record_id(record: dict) -> str:
    if "id" in record:
        return record["id"]
    else:
        # compute hash of the msgspec serialization of the record
        s = msgspec.msgpack.encode(record, order="deterministic")
        return str(_bloom_hash(s))


def _get_extension(file_path: str) -> str:
    for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if file_path.endswith(ext):
            return ext
    raise ValueError(f"Unsupported extension: {file_path}.")


def build_filter(
    input_path: str | list[str],
    bloom_path: str,
    config: DeconConfig,
) -> str:
    """
    Build a bloom filter from input dataset.
    """
    from dupekit import Bloom

    def build_shard_bloom(records: Iterator[dict]) -> Iterator[bytes]:
        """Build bloom filter from a shard of records and yield serialized bytes."""
        bf = Bloom(config.estimated_doc_count, config.false_positive_rate)

        for record in records:
            text = record.get(config.text_field, "")
            for feature in extract_features(text, config.ngram):
                bf.add(_bloom_hash(feature))

        yield bf.save_bytes()

    all_files = _collect_input_files(input_path)
    logger.info(f"Building bloom filter from {all_files} into {bloom_path}")

    # Build bloom filters for all shards in parallel
    shard_blooms_data = flow_backend().execute(
        Dataset.from_iterable(all_files)
        .reshard(num_shards=config.processes)
        .load_file()
        .select(config.text_field)
        .map_shard(build_shard_bloom)
        .write_binary(f"{bloom_path}-{{shard:05d}}-of-{{total:05d}}.bin", skip_existing=True)
    )

    if len(shard_blooms_data) == 1:
        return shard_blooms_data[0]

    logger.info(f"Merging {len(shard_blooms_data)} shard bloom filters...")

    def _merge_bloom(bloom_files: Iterator[str]):
        merged_bloom = Bloom(config.estimated_doc_count, config.false_positive_rate)
        for bloom_file_path in bloom_files:
            fs, path = fsspec.url_to_fs(bloom_file_path)
            with fs.open(path, "rb") as f:
                bloom_bytes = f.read()
            shard_bloom = Bloom.load_bytes(bloom_bytes)
            merged_bloom.update(shard_bloom)
        yield merged_bloom.save_bytes()

    merged_bloom = flow_backend().execute(
        Dataset.from_iterable(shard_blooms_data)
        .reshard(num_shards=1)
        .map_shard(_merge_bloom)
        .write_binary(bloom_path, skip_existing=True)
    )

    return merged_bloom[0]


def calculate_paragraph_overlap(paragraph: str, bloom_filter: "Bloom", ngram_config: NGramConfig | None) -> float:
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
    from dupekit import Bloom

    # Determine base path for rebasing
    base_path = input_path[0] if isinstance(input_path, list) else input_path
    all_files = _collect_input_files(input_path)

    def process_shard_with_bloom(records: Iterator[dict]) -> Iterator[dict]:
        """Load bloom filter once per shard and mark duplicates."""
        # Load bloom filter from storage
        fs, path = fsspec.url_to_fs(bloom_path)
        with fs.open(path, "rb") as f:
            bloom_bytes = f.read()
        bf = Bloom.load_bytes(bloom_bytes)

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
                if overlap_score > 0:
                    duplicate_spans.append([offset, offset + len(para), overlap_score])
                offset += len(para) + 1  # +1 for newline

            yield {
                "id": _record_id(record),
                "attributes": {config.attribute_name: duplicate_spans},
            }

    # Use write_jsonl with callable output pattern
    result = list(
        flow_backend(max_parallelism=config.processes).execute(
            Dataset.from_iterable(all_files)
            .flat_map(load_file)
            .map_shard(process_shard_with_bloom)
            .write_jsonl(
                output_pattern=lambda shard_idx, total: rebase_file_path(
                    base_path, all_files[shard_idx], output_path, old_extension=_get_extension(all_files[shard_idx])
                ),
                skip_existing=True,
            )
        )
    )
    return result


def _run_decontamination(config: DeconConfig):
    """
    Decontamination: build filter from contamination source, apply to input (read-only)
    """
    if not config.decontaminate_source:
        raise ValueError("decontaminate_source is required in DECONTAMINATE mode")

    bloom_path = os.path.join(config.output_path, "bloom", "filter.bin")
    bloom_path = build_filter(config.decontaminate_source, bloom_path, config)
    mark_duplicates_bloom(config.input_path, bloom_path, config.output_path, config)

    return {
        "success": True,
        "mode": "decontamination",
    }


def _run_train_test_overlap(config: DeconConfig):
    """
    Train-test overlap: build filter from training data, apply to test data for each n-gram size
    """
    if not config.decontaminate_source:
        raise ValueError("decontaminate_source is required in TRAIN_TEST_OVERLAP mode")

    if not config.ngram:
        raise ValueError("ngram config is required in TRAIN_TEST_OVERLAP mode")

    # Handle multiple n-gram sizes
    ngram_lengths = (
        config.ngram.ngram_length if isinstance(config.ngram.ngram_length, list) else [config.ngram.ngram_length]
    )

    for ngram_len in ngram_lengths:
        current_ngram_config = NGramConfig(
            ngram_length=ngram_len,
            stride=config.ngram.stride,
            overlap_threshold=config.ngram.overlap_threshold,
        )

        # Create config for this n-gram size
        train_config = DeconConfig(
            input_path=config.decontaminate_source,
            output_path=config.output_path,
            ngram=current_ngram_config,
            text_field=config.text_field,
            estimated_doc_count=config.estimated_doc_count,
            false_positive_rate=config.false_positive_rate,
            processes=config.processes,
            attribute_name=config.attribute_name,
        )

        bloom_path = os.path.join(config.output_path, "bloom", f"{ngram_len}.bin")
        bloom_path = build_filter(config.decontaminate_source, bloom_path, train_config)

        # Step 2: Apply filter to test data
        test_config = DeconConfig(
            input_path=config.input_path,
            output_path=os.path.join(config.output_path, str(ngram_len)),
            attribute_name=f"{config.attribute_name}_{ngram_len}",
            ngram=current_ngram_config,
            text_field=config.text_field,
            estimated_doc_count=config.estimated_doc_count,
            false_positive_rate=config.false_positive_rate,
            processes=config.processes,
        )

        mark_duplicates_bloom(config.input_path, bloom_path, test_config.output_path, test_config)

    return {
        "success": True,
        "mode": "train_test_overlap",
        "ngram_lengths_processed": ngram_lengths,
    }


def decontaminate(config: DeconConfig):
    """Main entry point for decontamination workflows."""
    if config.mode == DeconMode.DECONTAMINATE:
        return _run_decontamination(config)
    elif config.mode == DeconMode.TRAIN_TEST_OVERLAP:
        return _run_train_test_overlap(config)
    else:
        raise ValueError(f"Unknown mode {config.mode}")


@draccus.wrap()
def main(config: DeconConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    result = decontaminate(config)
    print(f"Decontamination completed: {result}")


if __name__ == "__main__":
    main()
