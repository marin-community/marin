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
Major problems with getting dedupe to work currently.

Dolma requires some pretty old dependencies:
1. tokenizers <=0.19.1 means that no modern transsformers can be used hence why we have to use
transformers==4.44.0.
2. s3fs==2023.06 means that a pretty old version of s3fs needs to be used which means
an old fsspec needs to be used. This is a problem because this version will not recognize
the recursive glob pattern **/*.jsonl.gz correctly!

We circumvent this using a custom runtime environment with just the required packages for
just the Dolma package. We then schedule it directly on the node that is used to
copy the files in to make sure there is data locality.
"""

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum

import draccus
import fsspec
import ray
from tqdm import tqdm

from marin.core.runtime import cached_or_construct_output, workflow_cached
from marin.utils import (
    fsspec_exists,
    fsspec_glob,
    fsspec_isdir,
    fsspec_mkdirs,
    fsspec_rm,
    fsspec_size,
    parquet_to_jsonl_gz,
    rebase_file_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported input formats for JSON/JSONL documents that can be staged for
# Dolma deduplication.  All of these will be normalised to GZip-compressed
# ``*.jsonl.gz`` files inside the temporary ``documents/`` directory.
# ---------------------------------------------------------------------------
SUPPORTED_JSONL_EXTENSIONS: list[str] = [
    ".jsonl",  # uncompressed JSON Lines
    ".jsonl.gz",  # gzip-compressed JSON Lines
    ".jsonl.zst",  # zstd-compressed JSON Lines
    ".jsonl.gs",  # Google-storage compressed JSON Lines
    ".json.gz",  # gzip-compressed single JSON object per line
    ".json.zst",  # zstd-compressed single JSON object per line
]


def _normalise_to_gz(path: str) -> str:
    """Return *path* with its extension rewritten to ``.jsonl.gz``.
        Necessary because Dolma requires the input files to be in .jsonl.gz format.

    The input path must end with one of ``SUPPORTED_JSONL_EXTENSIONS``.
    If the extension is already ``.jsonl.gz`` it is returned unchanged.
    """
    for ext in SUPPORTED_JSONL_EXTENSIONS:
        if path.endswith(ext):
            if ext == ".jsonl.gz":
                return path  # already normalised
            return path[: -len(ext)] + ".jsonl.gz"
    # caller must ensure extension is supported
    return path


def _get_path_stats(path: str) -> tuple[int, int]:
    """Get size and file count for a single path (file or directory)."""
    if fsspec_isdir(path):
        extensions_pattern = "{" + ",".join([*SUPPORTED_JSONL_EXTENSIONS, ".parquet"]) + "}"
        pattern = os.path.join(path, f"**/*{extensions_pattern}")
        files = fsspec_glob(pattern)
        total_size = sum(fsspec_size(f) for f in files)
        file_count = len(files)
    else:
        # Single file
        total_size = fsspec_size(path)
        file_count = 1 if total_size > 0 else 0
    return total_size, file_count


def _check_input_size(input_path: str | list[str], debug: bool = False) -> dict:
    """Check if input has processable content before downloading"""

    # Handle list of paths
    if isinstance(input_path, list):
        total_size = 0
        total_file_count = 0
        has_any_content = True
        errors = []

        for path in input_path:
            try:
                path_size, path_file_count = _get_path_stats(path)

                total_size += path_size
                total_file_count += path_file_count

                # If ANY path has no content, mark has_content as False
                if path_size <= 100:
                    has_any_content = False

            except Exception as e:
                has_any_content = False
                errors.append(f"{path}: {e!s}")

        if debug:
            print(f"total_size: {total_size}, file_count: {total_file_count}, paths: {len(input_path)}\n\n", flush=True)
        result = {
            "has_content": has_any_content,
            "total_size_bytes": total_size,
            "file_count": total_file_count,
            "path_count": len(input_path),
        }
        if errors:
            result["errors"] = errors
        return result

    # Handle single path (original logic)
    try:
        total_size, file_count = _get_path_stats(input_path)
        if debug:
            print(f"total_size: {total_size}, file_count: {file_count}\n\n", flush=True)
        return {
            "has_content": total_size > 100,  # 100 bytes is a reasonable minimum for a shard
            "total_size_bytes": total_size,
            "file_count": file_count,
        }
    except Exception as e:
        return {"has_content": False, "total_size_bytes": 0, "file_count": 0, "error": str(e)}


class DedupMode(str, Enum):
    DECONTAMINATE = "decontaminate"
    DEDUPLICATE = "deduplicate"
    TRAIN_TEST_OVERLAP = "train_test_overlap"


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


@dataclass
class DedupeConfig:
    """
    Configuration class for running dolma deduplication on docs.

    Deduplication will identify spans of text in documents that are duplicate.

    Empty file handling: If input_path or decontaminate_source (when required) contains
    no processable files or has 0 total bytes, the workflow will return early with
    success=True and detailed metadata about why no processing occurred. This prevents
    expensive downloads and processing of empty datasets.

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
        temp_dir (str | None): directory for temporary files (defaults to /dev/shm for performance)
        num_cpus (int): number of CPUs to allocate for Ray remote function
        memory (int): memory in bytes to allocate for Ray remote function (16GB default)
        resources (dict[str, float] | None): custom resources for Ray remote function
    """

    input_path: str | list[str]
    output_path: str
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
    # directory for temporary files (defaults to /dev/shm for performance)
    temp_dir: str | None = "/dev/shm"
    # Ray resource configuration
    num_cpus: int = 16
    memory: int = 16 * 1024 * 1024 * 1024  # 16GB in bytes
    resources: dict[str, float] | None = None
    # Debug flag to control verbose print statements
    debug: bool = False


def _format_input_paths_for_error(input_paths: str | list[str], max_paths: int = 3) -> str:
    """Helper function to format input paths for error messages."""
    if isinstance(input_paths, list):
        if len(input_paths) <= max_paths:
            return f"[{', '.join(input_paths)}]"
        else:
            shown_paths = ", ".join(input_paths[:max_paths])
            remaining = len(input_paths) - max_paths
            return f"[{shown_paths}, ... and {remaining} more]"
    else:
        return input_paths


def _copy_multiple_inputs(
    input_paths: str | list[str], local_base_dir: str, text_field: str = "text", debug: bool = False
) -> None:
    """Helper function to copy files from single path or list of paths."""
    if isinstance(input_paths, list):
        for path in input_paths:
            copy_files_in(path, local_base_dir, text_field, debug)
    else:
        copy_files_in(input_paths, local_base_dir, text_field, debug)


def copy_files_in(input_path, local_base_dir, text_field: str = "text", debug: bool = False):
    # Ensure input_path doesn't end with a slash
    input_path = input_path.rstrip("/")

    # Auto-convert Parquet inputs into .jsonl.gz under documents/
    parquet_pattern = os.path.join(input_path, "**", "*.parquet")
    if input_path.endswith(".parquet") or fsspec_glob(parquet_pattern):
        docs_dir = os.path.join(local_base_dir, "documents")
        parquet_to_jsonl_gz(input_path, docs_dir, text_field)
        if debug:
            print(f"Converted Parquet → JSONL.gz into {docs_dir}", flush=True)
        return

    # If input_path is a single file, copy only that file
    if fsspec_exists(input_path) and not fsspec_isdir(input_path):
        # Extract the filename and normalise extension to .jsonl.gz
        relative_path = os.path.basename(input_path)
        relative_path = _normalise_to_gz(relative_path)

        # Construct the output path under 'documents/'
        output_file = os.path.join(local_base_dir, "documents", relative_path)

        # Ensure the output directory exists
        fsspec_mkdirs(os.path.dirname(output_file))

        # Copy the file using fsspec with gzip compression (always write gzip)
        with fsspec.open(input_path, "rb", compression="infer") as f_remote:
            with fsspec.open(output_file, "wb", compression="gzip") as f_local:
                f_local.write(f_remote.read())

        if debug:
            print(f"Copied 1 file from {input_path} to {output_file}", flush=True)
        return

    # Create a single pattern that matches all supported extensions recursively
    extensions_pattern = "{" + ",".join(SUPPORTED_JSONL_EXTENSIONS) + "}"
    recursive_pattern = os.path.join(input_path, "**", f"*{extensions_pattern}")
    input_files = fsspec_glob(recursive_pattern)

    # ------------------------------------------------------------------
    # Copy each discovered file, normalising extension to .jsonl.gz
    # ------------------------------------------------------------------
    for input_file in tqdm(input_files, desc="Copying files"):
        # Determine destination relative path and make sure extension is .jsonl.gz
        relative_path = os.path.relpath(input_file, input_path)
        relative_path = _normalise_to_gz(relative_path)

        output_file = os.path.join(local_base_dir, "documents", relative_path)
        fsspec_mkdirs(os.path.dirname(output_file))

        # Read with automatic decompression and write as gzip
        with fsspec.open(input_file, "rb", compression="infer") as f_remote:
            with fsspec.open(output_file, "wb", compression="gzip") as f_local:
                f_local.write(f_remote.read())

    if debug:
        print(
            f"Copied {len(input_files)} files to {os.path.join(local_base_dir, 'documents')}",
            flush=True,
        )


def copy_files_out(local_base_dir, output_path, attribute_name, debug: bool = False):
    # Ensure output_path doesn't end with a slash
    output_path = output_path.rstrip("/")

    local_attribute_dir = os.path.join(local_base_dir, "attributes", attribute_name)

    # Get all .jsonl.gz files in the local attribute directory (recursive)
    glob_path = os.path.join(local_attribute_dir, "**", "*.jsonl.gz")
    local_files = fsspec_glob(glob_path)

    files_uploaded = 0
    for local_file in tqdm(local_files, desc="Uploading files"):
        # Use rebase_file_path to get the correct output path
        output_file = rebase_file_path(local_attribute_dir, local_file, output_path)

        # Ensure the output directory exists
        output_file_dir = os.path.dirname(output_file)
        fsspec_mkdirs(output_file_dir)

        # Copy the file using fsspec
        with fsspec.open(local_file, "rb") as f_local:
            with fsspec.open(output_file, "wb") as f_remote:
                f_remote.write(f_local.read())

        files_uploaded += 1

    if debug:
        print(f"Uploaded {files_uploaded} files to {output_path}", flush=True)


@cached_or_construct_output(success_suffix="SUCCESS")
def copy_file_out(input_file_path, output_file_path, debug: bool = False):

    # Ensure the output directory exists
    output_file_dir = os.path.dirname(output_file_path)
    fsspec_mkdirs(output_file_dir)

    # Copy the file using fsspec
    with fsspec.open(input_file_path, "rb") as f_local:
        with fsspec.open(output_file_path, "wb") as f_remote:
            f_remote.write(f_local.read())

    if debug:
        print(f"Uploaded file to {output_file_path}", flush=True)


def delete_jsonl_files(dir_path, debug: bool = False):
    """
    Delete all JSONL files (both .jsonl and .jsonl.gz) in the specified directory and its subdirectories.

    Args:
        dir_path (str): The path to the directory containing JSONL files.

    Returns:
        int: The number of files deleted.
    """
    # Ensure dir_path doesn't end with a slash
    dir_path = dir_path.rstrip("/")

    glob_path = os.path.join(dir_path, "**", "*.jsonl{,.gz}")
    files_to_delete = fsspec_glob(glob_path)

    files_deleted = 0
    for file_path in tqdm(files_to_delete, desc="Deleting files"):
        if fsspec_rm(file_path):
            files_deleted += 1

    if debug:
        print(f"Deleted {files_deleted} JSONL files from {dir_path}", flush=True)
    return files_deleted


# Fast estimate of total n-grams by sampling a subset of documents
def estimate_total_ngrams_fast(
    local_base_dir, ngram_lengths: list[int], sample_lines: int = 1000, debug: bool = False
) -> dict[int, int]:
    """
    Estimate *unique* n-gram count quickly by reading at most `sample_lines`
    JSONL lines across all `<documents_dir>/**/*.jsonl.gz`.

    Args:
        local_base_dir: Base directory containing documents/ subdirectory
        ngram_lengths: List of n-gram lengths to estimate for
        sample_lines: Maximum number of lines to sample for estimation

    Returns:
        Dict mapping ngram_length -> estimated_doc_count
    """
    pattern = os.path.join(local_base_dir, "documents", "**", "*.jsonl.gz")

    total_lines = 0  # total records in corpus
    sample_lines_seen = 0  # lines actually included in the sample
    ngram_totals = {n: 0 for n in ngram_lengths}  # Σ n-grams over sampled lines for each n
    malformed_lines = 0

    for file in fsspec_glob(pattern):
        with fsspec.open(file, "rt", compression="infer") as f:
            for line_num, raw_line in enumerate(f):
                total_lines += 1
                if sample_lines_seen >= sample_lines:
                    continue

                try:
                    data = json.loads(raw_line)
                    text = data.get("text", "")
                    if not isinstance(text, str):
                        if debug:
                            print(
                                f"Warning: 'text' field is not a string in {file}:{line_num}, type: {type(text)}",
                                flush=True,
                            )
                        malformed_lines += 1
                        continue
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"JSON decode error in {file}:{line_num}: {e}", flush=True)
                    malformed_lines += 1
                    continue

                tokens = text.split()
                L = len(tokens)

                # Calculate n-grams for each requested length
                for ngram_length in ngram_lengths:
                    if L < ngram_length:
                        ngram_totals[ngram_length] += 1
                    else:
                        # Using stride=0 for estimation (most conservative)
                        ngram_totals[ngram_length] += (L - ngram_length) + 1

                sample_lines_seen += 1

    if malformed_lines > 0:
        if debug:
            print(f"Warning: Found {malformed_lines} malformed lines during sampling", flush=True)

    if sample_lines_seen == 0:
        # nothing decoded ⇒ fall back to "one n-gram per line" heuristic
        return {n: max(1, total_lines) for n in ngram_lengths}

    # Calculate final estimates
    estimates = {}
    for ngram_length in ngram_lengths:
        avg_ngrams_per_line = ngram_totals[ngram_length] / sample_lines_seen
        estimates[ngram_length] = max(1, int(avg_ngrams_per_line * total_lines))
        # overestimate by 1.5x to be conservative
        estimates[ngram_length] = int(estimates[ngram_length] * 1.5)

    return estimates


def do_dedup(
    local_base_dir,
    attribute_name,
    min_length,
    min_words,
    bloom_filter_size,
    estimated_doc_count,
    false_positive_rate,
    ngram,
    processes,
    read_only=False,
    bloom_filter_file="deduper_bloom_filter.bin",
    train_test_overlap=False,
    pre_estimated_counts: dict[int, int] | None = None,
    debug: bool = False,
):
    bloom_filter_file = os.path.join(local_base_dir, bloom_filter_file)

    # If n-gram mode and no explicit bloom_filter_size, use pre-computed estimates
    if ngram is not None and not bloom_filter_size and pre_estimated_counts is not None:
        ngram_length = ngram.ngram_length
        estimated_doc_count = pre_estimated_counts[ngram_length]
        if debug:
            print(f"Using pre-computed estimate for {ngram_length}-grams: {estimated_doc_count}", flush=True)
        if estimated_doc_count < 100:
            if debug:
                print(
                    f"Warning: Pre-computed estimate for {ngram_length}-grams is too low: {estimated_doc_count}",
                    flush=True,
                )

    command = [
        "RUST_BACKTRACE=full",
        "dolma",
        "dedupe",
        "--documents",
        os.path.join(local_base_dir, "documents", "**", "*.jsonl.gz"),
        "--dedupe.paragraphs.attribute_name",
        attribute_name,
        "--dedupe.skip_empty",
        "--dedupe.min_length",
        str(min_length),
        "--dedupe.min_words",
        str(min_words),
        "--bloom_filter.file",
        bloom_filter_file,
        "--processes",
        str(processes),
        "--bloom_filter.estimated_doc_count",
        str(max(1, estimated_doc_count)),
        "--bloom_filter.desired_false_positive_rate",
        str(false_positive_rate),
    ]

    if bloom_filter_size:
        command.extend(
            [
                "--bloom_filter.size_in_bytes",
                str(bloom_filter_size),
            ]
        )

    # for decontamination bloom filter is read only
    command.append("--bloom_filter.read_only" if read_only else "--no-bloom_filter.read_only")

    # add ngram settings to dolma dedupe command if in ngram matching mode
    if ngram is not None:
        command.extend(
            [
                "--dedupe.paragraphs.by_ngram.ngram_length",
                str(ngram.ngram_length),
                "--dedupe.paragraphs.by_ngram.overlap_threshold",
                str(ngram.overlap_threshold),
                "--dedupe.paragraphs.by_ngram.stride",
                str(ngram.stride),
            ]
        )

    # ONLY set special paragraph separator for train-test overlap
    if train_test_overlap:
        # chatgpt says the separator below is extrememly unlikely to ever occur, so we count all n-grams in document
        # instead of per paragraph, giving us overlap if any n-grams match
        command.extend(
            [
                "--dedupe.paragraphs.paragraph_separator",
                "\u001e\u001e",
            ]
        )

    process = subprocess.Popen(
        " ".join(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    for line in process.stdout:
        if debug:
            print(line, end="", flush=True)
    process.wait()

    # Rename temporary files
    attr_dir = os.path.join(local_base_dir, "attributes", "duplicate_documents")
    if debug:
        print(f"Checking for temporary files in: {attr_dir}", flush=True)
    for root, _, files in os.walk(attr_dir):
        for file in files:
            if file.endswith(".jsonl.gz.tmp"):

                old_path = os.path.join(root, file)
                new_path = old_path.rsplit(".tmp", 1)[0]
                os.rename(old_path, new_path)

    if process.returncode != 0:
        # Common failure cases
        error_messages = {
            127: "Dolma command not found. Please install dolma: pip install dolma",
            126: "Dolma command found but not executable. Check permissions.",
            2: "Dolma command failed due to missing file or invalid arguments",
            1: "Dolma command failed with general error",
        }

        error_msg = error_messages.get(
            process.returncode, f"Dolma deduplication failed with return code {process.returncode}"
        )
        raise RuntimeError(error_msg)
    return process.returncode


def _shard_jsonl_source(jsonl_path: str, writers: list) -> None:
    """Stream JSONL file and distribute lines across writers.
    We do this to write the bloom filters faster for large training shards."""
    with fsspec.open(jsonl_path, "rt", compression="infer") as fr:
        for idx, line in enumerate(fr):
            writers[idx % len(writers)].write(line)


def _shard_documents_for_train_test_overlap(local_base_dir: str, processes: int, debug: bool = False) -> bool:
    """
    Shard documents in the documents/ directory for train-test overlap processing.
    Works in-place to avoid symlink issues.

    Args:
        local_base_dir: Base directory containing documents/ subdirectory
        processes: Number of shards to create
        debug: If True, print debug information with flush

    Returns:
        True if sharding was performed, False if no files found
    """
    if debug:
        print(f"Sharding documents for train-test overlap processing with {processes} shards", flush=True)

    documents_dir = os.path.join(local_base_dir, "documents")

    # Find all jsonl.gz files in documents directory
    docs_pattern = os.path.join(documents_dir, "**", "*.jsonl.gz")
    jsonl_files = fsspec_glob(docs_pattern)

    if debug:
        print(f"Found {len(jsonl_files)} JSONL files to shard", flush=True)

    # Create shard writers directly in the documents directory
    writers = []
    shard_paths = []
    for i in range(processes):
        shard_path = os.path.join(documents_dir, f"shard_{i}.jsonl.gz")
        shard_paths.append(shard_path)
        writers.append(fsspec.open(shard_path, "wt", compression="gzip").open())

    try:
        # Shard each jsonl.gz file
        for jsonl_file in jsonl_files:
            if debug:
                print(f"Sharding file: {jsonl_file}", flush=True)
            _shard_jsonl_source(jsonl_file, writers)
    finally:
        for w in writers:
            w.close()

    # Delete the original files (but keep the documents directory intact)
    for original_file in jsonl_files:
        if debug:
            print(f"Removing original file: {original_file}", flush=True)
        fsspec_rm(original_file)

    if debug:
        print(f"Successfully sharded {len(jsonl_files)} files into {processes} shards in-place", flush=True)
        print(f"Shard files created: {shard_paths}", flush=True)

    return True


# Helpers for the two workflows
@workflow_cached(success_suffix="SUCCESS")
def _run_decontamination(config: DedupeConfig):
    # Check input sizes first before any processing
    input_check = _check_input_size(config.input_path, config.debug)
    if not input_check["has_content"]:
        return {
            "success": True,
            "reason": "empty_input",
            "input_path": _format_input_paths_for_error(config.input_path),
            **input_check,
        }

    if not config.decontaminate_source:
        raise ValueError("decontaminate_source is required in DECONTAMINATE mode")

    source_check = _check_input_size(config.decontaminate_source, config.debug)
    if not source_check["has_content"]:
        if config.debug:
            print(f"Empty decontaminate source: {config.decontaminate_source}", flush=True)
        logger.info(f"Empty decontaminate source: {config.decontaminate_source}")
        return {
            "success": True,
            "reason": "empty_decontaminate_source",
            "decontaminate_source": config.decontaminate_source,
            **source_check,
        }

    if config.debug:
        print(f" content: {source_check['has_content']}", flush=True)
    with tempfile.TemporaryDirectory(dir=config.temp_dir, prefix="marin_dedupe_") as tmpdir:
        # 1) build the filter
        copy_files_in(config.decontaminate_source, tmpdir, text_field=config.text_field, debug=config.debug)
        do_dedup(
            tmpdir,
            config.attribute_name,
            config.min_length,
            config.min_words,
            config.bloom_filter_size,
            config.estimated_doc_count,
            config.false_positive_rate,
            config.ngram,
            config.processes,
            read_only=False,
            bloom_filter_file=config.bloom_filter_path,
            train_test_overlap=False,
            pre_estimated_counts=None,
            debug=config.debug,
        )
        # 2) clear out JSONLs
        delete_jsonl_files(tmpdir, config.debug)
        # 3) apply filter to real input
        _copy_multiple_inputs(config.input_path, tmpdir, text_field=config.text_field, debug=config.debug)
        do_dedup(
            tmpdir,
            config.attribute_name,
            config.min_length,
            config.min_words,
            config.bloom_filter_size,
            config.estimated_doc_count,
            config.false_positive_rate,
            config.ngram,
            config.processes,
            read_only=True,
            bloom_filter_file=config.bloom_filter_path,
            train_test_overlap=False,
            pre_estimated_counts=None,
            debug=config.debug,
        )
        # 4) write results
        copy_files_out(tmpdir, config.output_path, config.attribute_name, config.debug)

        return {
            "success": True,
            "reason": "completed_normally",
            "mode": "decontamination",
            **input_check,
            "decontaminate_source_info": source_check,
        }


@workflow_cached(success_suffix="SUCCESS")
def _run_deduplication(config: DedupeConfig):
    # Check input size first before any processing
    input_check = _check_input_size(config.input_path, config.debug)
    if not input_check["has_content"]:
        return {
            "success": True,
            "reason": "empty_input",
            "input_path": _format_input_paths_for_error(config.input_path),
            **input_check,
        }

    with tempfile.TemporaryDirectory(dir=config.temp_dir, prefix="marin_dedupe_") as tmpdir:
        # run standard deduplication
        _copy_multiple_inputs(config.input_path, tmpdir, text_field=config.text_field, debug=config.debug)
        do_dedup(
            tmpdir,
            config.attribute_name,
            config.min_length,
            config.min_words,
            config.bloom_filter_size,
            config.estimated_doc_count,
            config.false_positive_rate,
            config.ngram,
            config.processes,
            read_only=False,
            bloom_filter_file=config.bloom_filter_path,
            train_test_overlap=False,
            pre_estimated_counts=None,
            debug=config.debug,
        )
        copy_files_out(tmpdir, config.output_path, config.attribute_name, config.debug)

        return {"success": True, "reason": "completed_normally", "mode": "deduplication", **input_check}


@workflow_cached(success_suffix="SUCCESS")
def _run_train_test_overlap(config: DedupeConfig):
    """
    Run train-test overlap detection, supporting multiple n-gram sizes.

    This workflow performs, for each n-gram size N:
      1) Build a Bloom filter from a training (decontamination) source.
      2) Apply that filter in read-only mode against a test set to detect overlaps.
      3) Upload only the test overlaps (duplicate spans) to the remote output directory.

    Inputs in `config`:
      - decontaminate_source: path (file or directory) of training docs to build the filter.
      - input_path: path (file or directory) of test docs to scan for overlaps.
      - ngram.ngram_length: int or list[int], the N-gram sizes to run.
      - output_path: remote base directory; results will go under `<output_path>/<N>/...`.
      - bloom_filter_path, bloom_filter_size, estimated_doc_count, false_positive_rate: Bloom filter parameters.
      - processes: number of parallel shards to use.

    Key details / Dolma requirements:
      * Dolma always expects `--documents <...>/documents/**/*.jsonl.gz`.
      * We stage the training shards in `documents_seed/` and test files in `documents_test/`.
      * Before each Dolma invocation, we symlink one of those into `documents/` so Dolma sees exactly that path.
      * We never delete the seed or test source trees; we only swap the `documents/` symlink.
      * After building the filter, we remove the seed symlink; after testing, we remove the test symlink.

    Expected outputs:
      * Under `<output_path>/<N>/`, you will find a mirror of the test-set directory structure,
        each `.jsonl.gz` containing the duplicate-span attributes for that doc.
      * Only test overlaps are uploaded; seed-shard attribute files are removed before upload.

    Empty file handling:
      * If input_path or decontaminate_source contains no processable files (0 bytes total),
        the function will return early with success=True and reason="empty_input" or
        "empty_decontaminate_source", avoiding any expensive processing or downloads.

    The sharding logic is now handled inside do_dedup when train_test_overlap=True.
    """
    # Check input sizes first before any processing
    input_check = _check_input_size(config.input_path, config.debug)
    if not input_check["has_content"]:
        return {
            "success": True,
            "reason": "empty_input",
            "input_path": _format_input_paths_for_error(config.input_path),
            **input_check,
        }
    if not config.decontaminate_source:
        raise ValueError("decontaminate_source is required in TRAIN_TEST_OVERLAP mode")
    source_check = _check_input_size(config.decontaminate_source, config.debug)
    if not source_check["has_content"]:
        return {
            "success": True,
            "reason": "empty_decontaminate_source",
            "decontaminate_source": config.decontaminate_source,
            **source_check,
        }
    if not config.ngram:
        raise ValueError("ngram config is required in TRAIN_TEST_OVERLAP mode")
    with tempfile.TemporaryDirectory(dir=config.temp_dir, prefix="marin_dedupe_") as tmpdir:

        # Handle single or multiple ngram_lengths
        ngram_lengths = (
            config.ngram.ngram_length if isinstance(config.ngram.ngram_length, list) else [config.ngram.ngram_length]
        )

        # 1) Convert decontaminate source to jsonl.gz format using copy_files_in
        copy_files_in(config.decontaminate_source, tmpdir, text_field=config.text_field, debug=config.debug)
        seed_dir = os.path.join(tmpdir, "documents_seed")
        os.rename(os.path.join(tmpdir, "documents"), seed_dir)

        # 2) Pre-estimate ngram counts for ALL sizes before sharding
        if config.debug:
            print(f"Pre-estimating ngram counts for sizes: {ngram_lengths}", flush=True)
        if not config.bloom_filter_size:
            # Temporarily symlink for estimation
            os.symlink(seed_dir, os.path.join(tmpdir, "documents"))
            pre_estimated_counts = estimate_total_ngrams_fast(tmpdir, ngram_lengths, debug=config.debug)
            os.remove(os.path.join(tmpdir, "documents"))
            if config.debug:
                print(f"Pre-computed estimates: {pre_estimated_counts}", flush=True)
        else:
            pre_estimated_counts = None

        # 3) Shard the training data once for parallel processing
        if config.debug:
            print(f"Sharding training data into {config.processes} shards...", flush=True)
        os.symlink(seed_dir, os.path.join(tmpdir, "documents"))
        _shard_documents_for_train_test_overlap(tmpdir, config.processes, debug=config.debug)
        os.remove(os.path.join(tmpdir, "documents"))
        if config.debug:
            print("Training data sharded and originals removed", flush=True)

        # 4) Stage test data under a separate directory
        _copy_multiple_inputs(config.input_path, tmpdir, debug=config.debug)
        test_dir = os.path.join(tmpdir, "documents_test")
        os.rename(os.path.join(tmpdir, "documents"), test_dir)

        # 5) Iterate over n-gram sizes
        for ngram_len in ngram_lengths:
            current_ngram_config = NGramConfig(
                ngram_length=ngram_len,
                stride=config.ngram.stride,
                overlap_threshold=config.ngram.overlap_threshold,
            )
            current_attr_name = f"{config.attribute_name}_{ngram_len}"

            # Determine bloom-filter filename for this size
            current_bloom_filter = f"{config.bloom_filter_path}_{ngram_len}"

            # a) Build bloom filter on seed data
            os.symlink(seed_dir, os.path.join(tmpdir, "documents"))
            _ = do_dedup(
                tmpdir,
                current_attr_name,
                config.min_length,
                config.min_words,
                config.bloom_filter_size,
                config.estimated_doc_count,
                config.false_positive_rate,
                current_ngram_config,
                config.processes,
                read_only=False,
                bloom_filter_file=current_bloom_filter,
                pre_estimated_counts=pre_estimated_counts,
                train_test_overlap=True,
                debug=config.debug,
            )
            os.remove(os.path.join(tmpdir, "documents"))

            # b) Apply bloom filter to test data
            os.symlink(test_dir, os.path.join(tmpdir, "documents"))
            _ = do_dedup(
                tmpdir,
                current_attr_name,
                config.min_length,
                config.min_words,
                config.bloom_filter_size,
                config.estimated_doc_count,
                config.false_positive_rate,
                current_ngram_config,
                config.processes,
                read_only=True,
                bloom_filter_file=current_bloom_filter,
                pre_estimated_counts=pre_estimated_counts,
                train_test_overlap=True,
                debug=config.debug,
            )
            os.remove(os.path.join(tmpdir, "documents"))

            # c) Upload results for this n-gram size
            ngram_output = os.path.join(config.output_path, str(ngram_len))
            # Remove any seed-shard attribute files (at root of attribute dir)
            attr_dir = os.path.join(tmpdir, "attributes", current_attr_name)
            if os.path.isdir(attr_dir):
                for fname in os.listdir(attr_dir):
                    fpath = os.path.join(attr_dir, fname)
                    # top-level attribute files from seed run end in .jsonl.gz and have no subdir
                    if fname.endswith(".jsonl.gz") and os.path.isfile(fpath):
                        os.remove(fpath)
            # Now upload only test overlap files
            copy_files_out(tmpdir, ngram_output, current_attr_name, config.debug)

            # d) Clean up the bloom-filter file to save space
            bf_path = os.path.join(tmpdir, current_bloom_filter)
            if os.path.exists(bf_path):
                os.remove(bf_path)
        fsspec_rm(tmpdir)

        return {
            "success": True,
            "reason": "completed_normally",
            "mode": "train_test_overlap",
            "ngram_lengths_processed": ngram_lengths,
            **input_check,
            "decontaminate_source_info": source_check,
        }


def _dedupe_impl(config: DedupeConfig):
    """Top-level Ray task: dispatch between decontamination and deduplication workflows."""
    if config.mode == DedupMode.DECONTAMINATE:
        _run_decontamination(config)
    elif config.mode == DedupMode.DEDUPLICATE:
        _run_deduplication(config)
    elif config.mode == DedupMode.TRAIN_TEST_OVERLAP:
        _run_train_test_overlap(config)
    else:
        raise ValueError(f"Unknown mode {config.mode}")


def dedupe_with_config_resources(config: DedupeConfig):
    """Create and return a Ray remote function configured with the specified resources."""
    # Create the remote function with the configured resources
    remote_options = {
        "num_cpus": config.num_cpus,
        "memory": config.memory,
        # "runtime_env": build_runtime_env_for_packages(extra=["deduplication"]),
    }

    if config.resources is not None:
        remote_options["resources"] = config.resources

    # Return the configured remote function (not the result)
    return ray.remote(**remote_options)(_dedupe_impl)


# Default remote function with standard resources for backward compatibility
dedupe = ray.remote(num_cpus=16, memory=16 * 1024 * 1024 * 1024)(_dedupe_impl)


@draccus.wrap()
def main(config: DedupeConfig):
    remote_func = dedupe_with_config_resources(config)
    ray.get(remote_func.remote(config))


if __name__ == "__main__":
    main()
