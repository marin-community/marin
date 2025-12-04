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
Deduplication using rbloom bloom filters and zephyr streaming.

This module provides three deduplication workflows:
1. DEDUPLICATE: Remove duplicate paragraphs within a dataset
2. EXACT_DOC_DEDUPLICATE: Remove duplicate documents based on full text hash
3. DECONTAMINATE: Mark paragraphs that appear in a contamination source
4. TRAIN_TEST_OVERLAP: Detect train-test overlap using n-gram matching

All workflows use rbloom bloom filters for efficient duplicate detection.
"""

import hashlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum, auto
import typing

from marin.execution.executor import THIS_OUTPUT_PATH

import pyarrow as pa
import pyarrow.json as pa_json
import draccus
import fsspec
from marin.utilities.time_logger import log_time
import msgspec

from marin.utils import fsspec_glob, rebase_file_path
from zephyr import Dataset, flow_backend, load_parquet
from zephyr.readers import load_file, open_file, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dupekit import Bloom


def _bloom_hash(x: str) -> int:
    if isinstance(x, bytes):
        return int.from_bytes(hashlib.blake2b(x, digest_size=8).digest(), "big")
    return int.from_bytes(hashlib.blake2b(x.encode(), digest_size=8).digest(), "big")


class DedupMode(StrEnum):
    DECONTAMINATE = auto()
    DEDUPLICATE = auto()
    EXACT_DOC_DEDUPLICATE = auto()
    TRAIN_TEST_OVERLAP = auto()


@dataclass
class NGramConfig:
    """
    Configuration class for Dolma deduplication n-gram settings.
    Dolma dedupe pipeline has an ngram match mode which is an alternative to exact match.
    Paragraphs are newline delimited text in the document.
    For each paragraph, all ngrams are produced with a given stride.
    So for 3-gram with 0 stride, 'The cat sat on the mat.' produces:
    'The cat sat', 'cat sat on', 'sat on the', 'on the mat', and 'the mat.'
    If you don't want the ngrams to overlap, you can increase stride.
    Stride is how many tokens to skip when moving through the string to produce ngrams.
    The ngrams are run through a bloom filter which contains all seen ngrams.
    The paragraph is considered a duplicate if the percentage of found ngrams is above a threshold.
    In short, a paragraph is considered a duplicate if its ngrams are typically duplicates.

    Attributes:
        ngram_length (int | list[int]): Size of the ngram (e.g. 8) or list of sizes (e.g. [10, 15])
        stride (int): Step size when moving through string to generate ngrams
        overlap_threshold (float): Percentage of duplicate ngrams for a paragraph to be considered duplicate
    """

    ngram_length: int | list[int] = 8
    stride: int = 0
    overlap_threshold: float = 0.7


@dataclass(frozen=True)
class DedupeConfig:
    """
    Configuration class for running deduplication on docs using Zephyr.

    Deduplication will identify spans of text in documents that are duplicate.

    Attributes:
        input_path (str | list[str]): Path(s) of files to apply deduplication to.
        output_path (str): Path for storing results of deduplication (char spans in docs that are duplicate)
        attribute_name (str): Name for key to store duplicate span info in json
        min_length (int): min length of document to be deduplicated
        min_words (int): min number of words to be deduplicated
        bloom_filter_size (int): set size of Bloom filter in bytes
        estimated_doc_count (int): estimated number of docs to deduplicate
        false_positive_rate (float): false positive rate for Bloom filter
        ngram (NGramConfig): settings for ngram matching including length, match threshold, and stride
        processes (int): number of processes to use for deduplication
        mode (DedupMode): switch between decontamination (build filter) and regular deduplication
        decontaminate_source (str | None): source to seed bloom filter when decontaminating
        bloom_filter_path (str): path to write or read the bloom filter file
        text_field (str): field to use for text content in Parquet files
    """

    # TODO (rav): had to make this optional to avoid default argument issues in dataclass, what is the
    #   best way to handle this in marin and draccus?
    input_path: str | list[str]
    output_path: str = THIS_OUTPUT_PATH
    attribute_name: str = "duplicate_text"
    min_length: int = 0
    min_words: int = 0
    bloom_filter_size: int | None = None  # default to 0 to use estimated_doc_count and false_positive_rate
    estimated_doc_count: int = 1000000
    false_positive_rate: float = 0.001
    ngram: NGramConfig | None = None  # use ngram matching if ngram settings provided
    processes: int = 1
    # mode switch between decontamination (build filter) and regular deduplication
    mode: DedupMode = DedupMode.DEDUPLICATE
    # source to seed bloom filter when decontaminating
    decontaminate_source: str | None = None
    # path to write or read the bloom filter file
    bloom_filter_path: str = "deduper_bloom_filter.bin"
    # field to use for text content in Parquet files
    text_field: str = "text"


def extract_ngrams(text: str, n: int, stride: int) -> Iterator[str]:
    """
    Extract n-grams from text based on config.

    Args:
        text: Input text to extract n-grams from
        n: Size of the n-gram
        stride: Step size when moving through string to generate ngrams

    Yields:
        N-gram strings
    """
    tokens: list[str] = text.split()

    for i in range(0, len(tokens) - n + 1, stride + 1):
        yield " ".join(tokens[i : i + n])


def extract_features(text: str, ngram_config: NGramConfig | None) -> Iterator[str]:
    """
    Extract features (paragraphs or n-grams) from text.

    Args:
        text: Input text to extract features from
        ngram_config: If provided, extract n-grams; otherwise extract paragraphs

    Yields:
        Feature strings (either paragraphs or n-grams)
    """
    paragraphs = text.split("\n")

    for para in paragraphs:
        if ngram_config:
            yield from extract_ngrams(para, ngram_config.ngram_length, ngram_config.stride)
        else:
            # Exact paragraph matching
            yield para


def collect_input_files(input_path: str | list[str]) -> list[str]:
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
    return all_files


def build_filter(
    input_path: str | list[str],
    bloom_path: str,
    config: DedupeConfig,
) -> str:
    """
    Build a bloom filter from input dataset.

    Phase 1: Build per-shard bloom filters in parallel
    Phase 2: Merge all shard blooms and save to bloom_path

    Args:
        input_path: Path(s) to input data
        bloom_path: Where to save the merged bloom filter
        config: Configuration (contains ngram settings, text_field, etc.)

    Returns:
        Path to saved bloom filter
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

    all_files = collect_input_files(input_path)
    logger.info(f"Building bloom filter from {all_files} into {bloom_path}")

    # Build bloom filters for all shards in parallel
    shard_blooms_data = flow_backend().execute(
        Dataset.from_iterable(all_files)
        .reshard(num_shards=config.processes)
        .flat_map(lambda path: load_file(path, columns=[config.text_field]))
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

    Uses n-gram matching if ngram_config is provided, otherwise exact paragraph matching.
    For paragraphs too short for n-grams, falls back to exact matching.

    Args:
        paragraph: Text paragraph to check
        bloom_filter: Bloom filter to check against
        ngram_config: N-gram configuration, or None for exact paragraph matching

    Returns:
        Overlap score between 0.0 and 1.0
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


def mark_duplicates_bloom(
    input_path: str | list[str],
    bloom_path: str,
    output_path: str,
    config: DedupeConfig,
) -> list[str]:
    """
    Apply bloom filter to input data, marking duplicate spans.

    Output files will mirror the structure of input files using rebase_file_path,
    making them discoverable by consolidate.py.

    Args:
        input_path: Path(s) to input data
        bloom_path: Path to saved bloom filter
        output_path: Where to write output
        config: Configuration (contains attribute_name, ngram settings, etc.)

    Returns:
        List of output file paths
    """
    from dupekit import Bloom

    # Determine base path for rebasing
    base_path = input_path[0] if isinstance(input_path, list) else input_path
    all_files = collect_input_files(input_path)

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


def _str_hash(s: str) -> str:
    return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()


def _load_batches(file_path: str, columns: list[str] | None = None, **parquet_kwargs) -> Iterator[pa.RecordBatch]:
    # Private function for now to isolate the `pa.RecordBatch` experiment
    if not file_path.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported extension: {file_path}.")
    with open_file(file_path, "rb") as f:
        if file_path.endswith(".parquet"):
            import pyarrow.parquet as pq

            if columns is not None:
                parquet_kwargs = {**parquet_kwargs, "columns": columns}

            parquet_file = pq.ParquetFile(f)
            yield from parquet_file.iter_batches(**parquet_kwargs)
        else:
            yield from pa_json.read_json(f).to_batches()


def _run_deduplication(config: DedupeConfig):
    import dupekit
    from dupekit import Transformation

    input_files = collect_input_files(config.input_path)

    def compute_paragraph_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="resolved_id"),
            Transformation.SplitParagraphs(text_col=config.text_field, id_col="resolved_id"),
            Transformation.Hash(input_col="paragraph_text", output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
            Transformation.SelectColumns(columns=["hash", "doc_id"]),
        ]
        return dupekit.transform(batch, pipeline)

    def _count_reduce(key: str, items: Iterator[pa.StructScalar]) -> dict[str, str] | None:
        try:
            head = next(items)
            _ = next(items)
        except StopIteration:
            return None
        # A dupe exists
        return {
            "hash": key,
            "canonical": head["doc_id"],
        }

    backend = flow_backend(max_parallelism=config.processes)
    # first compute the full set of duplicate keys.
    duplicate_key_shards = list(
        backend.execute(
            Dataset.from_list(input_files).flat_map(_load_batches)
            # NOTE: when do we want to trigger reshard. Keep in mind that reshard will materialize the
            #   text field!
            # TODO: the resharding logic should be improved, based on size and/or max_parallelism
            .reshard(num_shards=config.processes if len(input_files) > 3 and len(input_files) < 42 else None)
            .map(compute_paragraph_hashes)
            .flat_map(lambda batch: batch.to_pylist())
            .group_by(
                lambda key_fn: key_fn["hash"],
                _count_reduce,
            )
            .filter(lambda record: record)
            .reshard(1)
            .write_parquet(f"{config.output_path}/metadata/dup-key-{{shard:05d}}-of-{{total:05d}}.parquet"),
            verbose=True,
        )
    )

    # Determine base path for rebasing
    base_path = config.input_path[0] if isinstance(config.input_path, list) else config.input_path

    def mark_exact_dups_paragraphs(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Mark duplicate paragraphs in a single record using exact hash matching."""

        with log_time("Load duplicate map"):
            dup_map = {}
            for record in load_parquet(duplicate_key_shards[0]):
                dup_map[record["hash"]] = {
                    "canonical": record["canonical"],
                }

        logger.info(f"There are {len(dup_map)} duplicate paragraphs.")
        for batch in batches:
            yield dupekit.mark_paragraph_duplicates(
                batch,
                dup_map,
                config.attribute_name,
                algorithm=dupekit.HashAlgorithm.Xxh3_128,
            )

    # Use write_jsonl with callable output pattern
    backend.execute(
        Dataset.from_list(input_files)
        .flat_map(_load_batches)
        .map_shard(mark_exact_dups_paragraphs)
        .flat_map(lambda batch: batch.to_pylist())
        .write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                input_files[shard_idx],
                f"{config.output_path}/data",
                old_extension=_get_extension(input_files[shard_idx]),
            ),
            skip_existing=True,
        )
    )

    return {
        "success": True,
        "mode": "deduplication",
    }


def _run_exact_doc_deduplication(config: DedupeConfig):
    """
    Exact document deduplication: identify duplicate documents based on full text hash.
    This is a temporary implementation, primarily to compare directly with the Ai2 duplodocus.
    """
    import dupekit
    from dupekit import Transformation

    input_files = collect_input_files(config.input_path)

    def compute_document_hashes(batch: pa.RecordBatch) -> pa.RecordBatch:
        pipeline = [
            Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="resolved_id"),
            Transformation.Hash(input_col=config.text_field, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128),
            Transformation.SelectColumns(columns=["hash", "resolved_id"]),
        ]
        return dupekit.transform(batch, pipeline)

    def _count_reduce(key: str, items: Iterator[pa.StructScalar]) -> dict[str, str] | None:
        try:
            head = next(items)
            _ = next(items)
        except StopIteration:
            return None
        # A dupe exists
        return {
            "hash": key,
            "canonical": head["resolved_id"],
        }

    # first compute the full set of duplicate keys.
    backend = flow_backend(max_parallelism=config.processes)
    duplicate_key_shards = list(
        backend.execute(
            Dataset.from_list(input_files).flat_map(_load_batches)
            # NOTE: when do we want to trigger reshard. Keep in mind that reshard will materialize the
            #   text field!
            # TODO: the resharding logic should be improved, based on size and/or max_parallelism
            .reshard(num_shards=config.processes if len(input_files) > 3 and len(input_files) < 42 else None)
            .map(compute_document_hashes)
            .flat_map(lambda batch: batch.to_pylist())
            .group_by(
                lambda key_fn: key_fn["hash"],
                _count_reduce,
            )
            .filter(lambda record: record)
            .reshard(1)
            .write_parquet(f"{config.output_path}/metadata/dup-key-{{shard:05d}}-of-{{total:05d}}.parquet"),
            verbose=True,
        )
    )

    # Determine base path for rebasing
    base_path = config.input_path[0] if isinstance(config.input_path, list) else config.input_path
    if base_path in input_files:
        # NOTE: if the base_path is in the input_files, means it's a specific file, so rebase to its directory
        base_path = os.path.dirname(base_path)

    def mark_exact_dups_documents(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        """Mark exact duplicate documents using exact hash matching."""

        with log_time("Load duplicate map"):
            dup_map = {}
            for record in load_parquet(duplicate_key_shards[0]):
                dup_map[record["hash"]] = {
                    "canonical": record["canonical"],
                }

        logger.info(f"There are {len(dup_map)} duplicate documents.")

        for batch in batches:
            prepared_batch = dupekit.transform(
                batch,
                [
                    Transformation.ResolveIds(text_col=config.text_field, id_col="id", output_col="id"),
                    Transformation.Hash(
                        input_col=config.text_field, output_col="hash", algo=dupekit.HashAlgorithm.Xxh3_128
                    ),
                ],
            )
            yield dupekit.mark_document_duplicates(prepared_batch, dup_map, config.attribute_name, hash_col="hash")

    # Use write_jsonl with callable output pattern
    backend.execute(
        Dataset.from_list(input_files).flat_map(_load_batches)
        # NOTE/TODO: we can't reshard here to increase parallelism because afaiu we want to match
        # the shards of the input files for rebase_file_path to work correctly.
        .map_shard(mark_exact_dups_documents)
        .flat_map(lambda batch: batch.to_pylist())
        .write_jsonl(
            output_pattern=lambda shard_idx, total: rebase_file_path(
                base_path,
                input_files[shard_idx],
                f"{config.output_path}/data",
                old_extension=_get_extension(input_files[shard_idx]),
            ),
            skip_existing=True,
        ),
        verbose=True,
    )

    return {
        "success": True,
        "mode": "exact_doc_deduplication",
    }


def _run_decontamination(config: DedupeConfig):
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


def _run_train_test_overlap(config: DedupeConfig):
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
        train_config = DedupeConfig(
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
        test_config = DedupeConfig(
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


def dedupe(config: DedupeConfig):
    """Main entry point: dispatch between decontamination and deduplication workflows."""
    if config.mode == DedupMode.DECONTAMINATE:
        return _run_decontamination(config)
    elif config.mode == DedupMode.DEDUPLICATE:
        return _run_deduplication(config)
    elif config.mode == DedupMode.EXACT_DOC_DEDUPLICATE:
        return _run_exact_doc_deduplication(config)
    elif config.mode == DedupMode.TRAIN_TEST_OVERLAP:
        return _run_train_test_overlap(config)
    else:
        raise ValueError(f"Unknown mode {config.mode}")


@draccus.wrap()
def main(config: DedupeConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    result = dedupe(config)
    print(f"Deduplication completed: {result}")


if __name__ == "__main__":
    main()
