import gzip
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum

import draccus
import fsspec
import pandas as pd
import pyarrow.parquet as pq
import ray
from tqdm import tqdm

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir, fsspec_mkdirs, fsspec_rm, rebase_file_path, fsspec_size

logger = logging.getLogger(__name__)


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

    Attributes:
        input_path (str): Path of files to apply deduplication to.
        output_path (str): Path for storing results of deduplication (char spans in docs that are duplicate)
        attribute_name (str): Name for key to store duplicate span info in json
        min_length (int): min length of document to be deduplicated
        min_words (int): min number of words to be deduplicated
        bloom_filter_size (int): set size of Bloom filter in bytes
        estimated_doc_count (int): estimated number of docs to deduplicate
        false_positive_rate (float): false positive rate for Bloom filter
        ngram (NGramConfig): settings for ngram matching including length, match threshold, and stride
        processes (int): number of processes to use for deduplication
        # mode switch between decontamination (build filter) and regular deduplication
        mode: DedupMode = DedupMode.DEDUPLICATE
        # source to seed bloom filter when decontaminating
        decontaminate_source: str | None = None
        # path to write or read the bloom filter file
        bloom_filter_path: str = "deduper_bloom_filter.bin"
    """

    input_path: str
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


# Helper: convert Parquet files (single or shards) directly into .jsonl.gz under docs_dir
def _parquet_to_jsonl_gz(input_path: str, docs_dir: str) -> None:
    os.makedirs(docs_dir, exist_ok=True)
    # find all Parquet files
    if input_path.endswith(".parquet"):
        parquet_files = [input_path]
    else:
        parquet_files = fsspec_glob(f"{input_path.rstrip('/')}/*.parquet")
    
    for pq in parquet_files:
        df = pd.read_parquet(pq)
        # Skip empty Parquet files gracefully
        if df.empty:
            logger.error(f"Parquet file {pq} contains 0 rows, skipping conversion")
            continue
        out_name = os.path.splitext(os.path.basename(pq))[0] + ".jsonl.gz"
        out_path = os.path.join(docs_dir, out_name)
        
        print(f"Converting {pq} with columns: {list(df.columns)}")
        
        # Determine text field strategy
        has_text = "text" in df.columns
        has_content = "content" in df.columns
        
        if not has_text and not has_content:
            logger.error(f"CRITICAL: Parquet file {pq} has neither 'text' nor 'content' fields! Available columns: {list(df.columns)}")
            logger.error(f"Cannot convert {pq} - skipping this file")
            continue
        elif not has_text and has_content:
            logger.warning(f"Parquet file {pq} missing 'text' field, using 'content' as fallback")
        
        with gzip.open(out_path, "wt") as f:
            for rec in df.to_dict(orient="records"):
                # Robust text field handling
                if "text" not in rec:
                    if "content" in rec:
                        rec["text"] = rec.pop("content")
                    else:
                        logger.error(f"Record in {pq} missing both 'text' and 'content' fields, has keys: {list(rec.keys())}")
                        continue
                
                # Validate text field is actually a string
                if not isinstance(rec["text"], str):
                    logger.warning(f"Text field in {pq} is not a string (type: {type(rec['text'])}), converting to string")
                    rec["text"] = str(rec["text"])
                    
                if "id" not in rec:
                    # Generate a synthetic ID if missing
                    logger.warning(f"Adding synthetic id to {pq}")
                    rec["id"] = f"synthetic_{hash(str(rec))}"
                
                f.write(json.dumps(rec) + "\n")


def copy_files_in(input_path, local_base_dir):
    # Ensure input_path doesn't end with a slash
    input_path = input_path.rstrip("/")

    # Auto-convert Parquet inputs into .jsonl.gz under documents/
    parquet_pattern = f"{input_path}/**/*.parquet"
    if input_path.endswith(".parquet") or fsspec_glob(parquet_pattern):
        docs_dir = os.path.join(local_base_dir, "documents")
        _parquet_to_jsonl_gz(input_path, docs_dir)
        print(f"Converted Parquet → JSONL.gz into {docs_dir}")
        return

    # If input_path is a single file, copy only that file
    if fsspec_exists(input_path) and not fsspec_isdir(input_path):
        # Extract the filename and normalize extension
        relative_path = os.path.basename(input_path)
        if relative_path.endswith(".jsonl.zst"):
            relative_path = relative_path[: -len(".jsonl.zst")] + ".jsonl.gz"
        elif relative_path.endswith(".jsonl.gs"):
            relative_path = relative_path[: -len(".jsonl.gs")] + ".jsonl.gz"
        elif relative_path.endswith(".json.zst"):
            relative_path = relative_path[: -len(".json.zst")] + ".jsonl.gz"
        elif relative_path.endswith(".json.gz"):
            relative_path = relative_path[: -len(".json.gz")] + ".jsonl.gz"

        # Construct the output path under 'documents/'
        output_file = os.path.join(local_base_dir, "documents", relative_path)

        # Ensure the output directory exists
        fsspec_mkdirs(os.path.dirname(output_file))

        # Copy the file using fsspec with gzip compression
        with fsspec.open(input_path, "rb", compression="infer") as f_remote:
            with fsspec.open(output_file, "wb", compression="gzip") as f_local:
                f_local.write(f_remote.read())

        print(f"Copied 1 file from {input_path} to {output_file}")
        return

    # Get all .jsonl.gz, .jsonl.zst, .jsonl.gs, .json.gz, and .json.zst files in the input directory
    jsonl_gz_pattern = f"{input_path}/**/*.jsonl.gz"
    jsonl_zst_pattern = f"{input_path}/**/*.jsonl.zst"
    jsonl_gs_pattern = f"{input_path}/**/*.jsonl.gs"
    json_gz_pattern = f"{input_path}/**/*.json.gz"
    json_zst_pattern = f"{input_path}/**/*.json.zst"
    # First attempt recursive glob patterns
    input_files = (
        fsspec_glob(jsonl_gz_pattern)
        + fsspec_glob(jsonl_zst_pattern)
        + fsspec_glob(jsonl_gs_pattern)
        + fsspec_glob(json_gz_pattern)
        + fsspec_glob(json_zst_pattern)
    )
    fallback = False
    # Fallback to shallow glob if none found
    if not input_files:
        fallback = True
        shallow_patterns = [
            f"{input_path}/*.jsonl.gz",
            f"{input_path}/*.jsonl.zst",
            f"{input_path}/*.jsonl.gs",
            f"{input_path}/*.json.gz",
            f"{input_path}/*.json.zst",
        ]
        input_files = []
        for pattern in shallow_patterns:
            input_files.extend(fsspec_glob(pattern))
    # Log the result
    if fallback:
        print(f"Found {len(input_files)} input files in {input_path} (shallow), first five: {input_files[:5]}")
    else:
        print(f"Found {len(input_files)} input files in {input_path}, first five: {input_files[:5]}")

    for input_file in tqdm(input_files, desc="Copying files"):
        # Extract the relative path from the input file
        relative_path = os.path.relpath(input_file, input_path)
        # Normalize extension: convert .jsonl.zst or .jsonl.gs to .jsonl.gz for local documents
        if relative_path.endswith(".jsonl.zst"):
            relative_path = relative_path[: -len(".jsonl.zst")] + ".jsonl.gz"
        elif relative_path.endswith(".jsonl.gs"):  # Handle .jsonl.gs
            relative_path = relative_path[: -len(".jsonl.gs")] + ".jsonl.gz"
        elif relative_path.endswith(".json.zst"):
            relative_path = relative_path[: -len(".json.zst")] + ".jsonl.gz"
        elif relative_path.endswith(".json.gz"):
            relative_path = relative_path[: -len(".json.gz")] + ".jsonl.gz"

        # Construct the output path, ensuring it's under the 'documents' directory
        output_file = os.path.join(local_base_dir, "documents", relative_path)

        # Ensure the output directory exists
        output_path = os.path.dirname(output_file)
        fsspec_mkdirs(output_path)

        # Copy the file using fsspec
        with fsspec.open(input_file, "rb", compression="infer") as f_remote:
            # always compress local documents as gzip for downstream dedupe
            with fsspec.open(output_file, "wb", compression="gzip") as f_local:
                f_local.write(f_remote.read())

    # Dolma deduplicator requires 'documents/' as a subdir
    print(f"Copied {len(input_files)} files to {os.path.join(local_base_dir, 'documents')}")


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
):
    bloom_filter_file = os.path.join(local_base_dir, bloom_filter_file)

    # If n-gram mode and no explicit bloom_filter_size, override estimated_doc_count
    if ngram is not None and not bloom_filter_size:
        total_ngrams = estimate_total_ngrams_fast(local_base_dir, ngram)
        estimated_doc_count = total_ngrams

    command = [
        "RUST_BACKTRACE=full",
        "dolma",
        "dedupe",
        "--documents",
        f"{local_base_dir}/documents/**/*.jsonl.gz",
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
    ]

    if bloom_filter_size:
        command.extend([
            "--bloom_filter.size_in_bytes",
            str(bloom_filter_size),
            "--bloom_filter.estimated_doc_count",
            str(max(1, estimated_doc_count)),
            "--bloom_filter.desired_false_positive_rate",
            str(false_positive_rate),
        ])
    else:
        command.extend(
            [
                "--bloom_filter.estimated_doc_count",
                str(estimated_doc_count),
                "--bloom_filter.desired_false_positive_rate",
                str(false_positive_rate),
            ]
        )

    # for decontamination bloom filter is read only
    command.append("--bloom_filter.read_only" if read_only else "--no-bloom_filter.read_only")

    # add ngram settings to dolma dedupe command if in ngram matching mode
    if ngram is not None:
        # chatgpt says the separator below is extrememly unlikely to ever occur, so we count all n-grams in document
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

    if train_test_overlap:
        # For train-test overlap, we need to shard the documents for parallel processing
        # when constructing bloom filter (not when applying it)
        if not read_only:
            _shard_documents_for_train_test_overlap(local_base_dir, processes, debug=False)
        
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
        print(line, end="", flush=True)
    process.wait()

    # Rename temporary files
    attr_dir = os.path.join(local_base_dir, "attributes/duplicate_documents")
    print(f"Checking for temporary files in: {attr_dir}")
    for root, _, files in os.walk(attr_dir):
        for file in files:
            if file.endswith(".jsonl.gz.tmp"):

                old_path = os.path.join(root, file)
                new_path = old_path.rsplit(".tmp", 1)[0]
                os.rename(old_path, new_path)

    return process.returncode


@cached_or_construct_output(success_suffix="SUCCESS")
def copy_files_out(local_base_dir, output_path, attribute_name):
    # Ensure output_path doesn't end with a slash
    output_path = output_path.rstrip("/")

    local_attribute_dir = os.path.join(local_base_dir, "attributes", attribute_name)

    # Get all .jsonl.gz files in the local attribute directory (recursive)
    glob_path = f"{local_attribute_dir}/**/*.jsonl.gz"
    local_files = fsspec_glob(glob_path)
    # Fallback to shallow glob if no files found
    if not local_files:
        shallow_glob = f"{local_attribute_dir}/*.jsonl.gz"
        local_files = fsspec_glob(shallow_glob)

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

    print(f"Uploaded {files_uploaded} files to {output_path}")


def delete_jsonl_files(dir_path):
    """
    Delete all JSONL files (both .jsonl and .jsonl.gz) in the specified directory and its subdirectories.

    Args:
        dir_path (str): The path to the directory containing JSONL files.

    Returns:
        int: The number of files deleted.
    """
    # Ensure dir_path doesn't end with a slash
    dir_path = dir_path.rstrip("/")

    # Get all .jsonl and .jsonl.gz files in the directory and its subdirectories
    glob_path_jsonl = f"{dir_path}/**/*.jsonl"
    glob_path_jsonl_gz = f"{dir_path}/**/*.jsonl.gz"

    files_to_delete = fsspec_glob(glob_path_jsonl) + fsspec_glob(glob_path_jsonl_gz)

    files_deleted = 0
    for file_path in tqdm(files_to_delete, desc="Deleting files"):
        if fsspec_rm(file_path):
            files_deleted += 1

    print(f"Deleted {files_deleted} JSONL files from {dir_path}")
    return files_deleted


# Fast estimate of total n-grams by sampling a subset of documents
def estimate_total_ngrams_fast(local_base_dir, ngram, sample_lines: int = 1000):
    """
    Estimate *unique* n-gram count quickly by reading at most `sample_lines`
    JSONL lines across all `<documents_dir>/**/*.jsonl.gz`.
    """
    pattern = os.path.join(local_base_dir, "documents", "**", "*.jsonl.gz")

    total_lines = 0  # total records in corpus
    sample_lines_seen = 0  # lines actually included in the sample
    ngram_total = 0  # Σ n-grams over sampled lines
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
                        print(f"Warning: 'text' field is not a string in {file}:{line_num}, type: {type(text)}")
                        malformed_lines += 1
                        continue
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in {file}:{line_num}: {e}")
                    malformed_lines += 1
                    continue

                tokens = text.split()
                L = len(tokens)

                step = max(1, ngram.stride)
                if L < ngram.ngram_length:
                    ngram_total += 1
                else:
                    ngram_total += (L - ngram.ngram_length) // step + 1
                sample_lines_seen += 1

    if malformed_lines > 0:
        print(f"Warning: Found {malformed_lines} malformed lines during sampling")

    if sample_lines_seen == 0:
        # nothing decoded ⇒ fall back to "one n-gram per line" heuristic
        return total_lines

    avg_ngrams_per_line = ngram_total / sample_lines_seen
    return int(avg_ngrams_per_line * total_lines)


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
    
    if not jsonl_files:
        # Fallback to shallow search
        docs_pattern = os.path.join(documents_dir, "*.jsonl.gz")
        jsonl_files = fsspec_glob(docs_pattern)
    
    if not jsonl_files:
        if debug:
            print("No JSONL files found for sharding", flush=True)
        return False
    
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
def _run_decontamination(config: DedupeConfig):
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="marin_dedupe_") as tmpdir:
        if not config.decontaminate_source:
            raise ValueError("decontaminate_source is required in DECONTAMINATE mode")
        # 1) build the filter
        copy_files_in(config.decontaminate_source, tmpdir)
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
        )
        # 2) clear out JSONLs
        delete_jsonl_files(tmpdir)
        # 3) apply filter to real input
        copy_files_in(config.input_path, tmpdir)
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
        )
        # 4) write results
        copy_files_out(tmpdir, config.output_path, config.attribute_name)


def _run_deduplication(config: DedupeConfig):
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="marin_dedupe_") as tmpdir:
        # run standard deduplication
        copy_files_in(config.input_path, tmpdir)
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
        )
        copy_files_out(tmpdir, config.output_path, config.attribute_name)


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

    The sharding logic is now handled inside do_dedup when train_test_overlap=True.
    """
    with tempfile.TemporaryDirectory(dir="/dev/shm", prefix="marin_dedupe_") as tmpdir:
        if not config.decontaminate_source:
            raise ValueError("decontaminate_source is required in TRAIN_TEST_OVERLAP mode")
        if not config.ngram:
            raise ValueError("ngram config is required in TRAIN_TEST_OVERLAP mode")

        # Handle single or multiple ngram_lengths
        ngram_lengths = (
            config.ngram.ngram_length if isinstance(config.ngram.ngram_length, list) else [config.ngram.ngram_length]
        )

        # 1) Convert decontaminate source to jsonl.gz format using copy_files_in
        copy_files_in(config.decontaminate_source, tmpdir)
        seed_dir = os.path.join(tmpdir, "documents_seed") 
        os.rename(os.path.join(tmpdir, "documents"), seed_dir)

        # 2) Stage test data under a separate directory
        copy_files_in(config.input_path, tmpdir)
        test_dir = os.path.join(tmpdir, "documents_test")
        os.rename(os.path.join(tmpdir, "documents"), test_dir)

        # 3) Iterate over n-gram sizes
        for ngram_len in ngram_lengths:
            current_ngram_config = NGramConfig(
                ngram_length=ngram_len,
                stride=config.ngram.stride,
                overlap_threshold=config.ngram.overlap_threshold,
            )
            current_attr_name = (
                f"{config.attribute_name}_{ngram_len}" if len(ngram_lengths) > 1 else config.attribute_name
            )

            # Determine bloom-filter filename for this size
            current_bloom_filter = (
                f"{config.bloom_filter_path}_{ngram_len}" if len(ngram_lengths) > 1 else config.bloom_filter_path
            )

            # a) Build bloom filter on seed data
            os.symlink(seed_dir, os.path.join(tmpdir, "documents"))
            build_rc = do_dedup(
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
                train_test_overlap=True,
            )
            os.remove(os.path.join(tmpdir, "documents"))

            # b) Apply bloom filter to test data
            os.symlink(test_dir, os.path.join(tmpdir, "documents"))
            test_rc = do_dedup(
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
                train_test_overlap=True,
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
            copy_files_out(tmpdir, ngram_output, current_attr_name)

            # d) Clean up the bloom-filter file to save space
            bf_path = os.path.join(tmpdir, current_bloom_filter)
            if os.path.exists(bf_path):
                os.remove(bf_path)
        fsspec_rm(tmpdir)


@ray.remote(num_cpus=8, memory=1024 * 1024 * 1024 * 16)
def dedupe(config: DedupeConfig):
    """Top-level Ray task: dispatch between decontamination and deduplication workflows."""
    if config.mode == DedupMode.DECONTAMINATE:
        _run_decontamination(config)
    elif config.mode == DedupMode.DEDUPLICATE:
        _run_deduplication(config)
    elif config.mode == DedupMode.TRAIN_TEST_OVERLAP:
        _run_train_test_overlap(config)

    else:
        raise ValueError(f"Unknown mode {config.mode}")


@draccus.wrap()
def main(config: DedupeConfig):
    ray.get(dedupe.remote(config))


if __name__ == "__main__":
    main()
