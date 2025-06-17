import gzip
import json
import os
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
from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir, fsspec_mkdirs, fsspec_rm, rebase_file_path


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
        length (int): Size of the ngram (e.g. 8)
        stride (int): Step size when moving through string to generate ngrams
        threshold (float): Percentage of duplicate ngrams for a paragraph to be considered duplicate
    """

    ngram_length: int = 8
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
    print(f"DEBUG: Starting Parquet conversion for {input_path}", flush=True)
    os.makedirs(docs_dir, exist_ok=True)
    # find all Parquet files
    if input_path.endswith(".parquet"):
        parquet_files = [input_path]
    else:
        parquet_files = fsspec_glob(f"{input_path.rstrip('/')}/*.parquet")
    print(f"DEBUG: Found {len(parquet_files)} Parquet files: {parquet_files[:3]}...", flush=True)
    for pq in parquet_files:
        print(f"DEBUG: Converting {pq}", flush=True)
        df = pd.read_parquet(pq)
        print(f"DEBUG: Read Parquet with {len(df)} rows, columns: {list(df.columns)}", flush=True)
        out_name = os.path.splitext(os.path.basename(pq))[0] + ".jsonl.gz"
        out_path = os.path.join(docs_dir, out_name)
        print(f"DEBUG: Writing to {out_path}", flush=True)
        with gzip.open(out_path, "wt") as f:
            for rec in df.to_dict(orient="records"):
                f.write(json.dumps(rec) + "\n")
        print(f"DEBUG: Successfully wrote {out_path}", flush=True)
    print("DEBUG: Finished Parquet conversion", flush=True)


def copy_files_in(input_path, local_base_dir):
    # Ensure input_path doesn't end with a slash
    input_path = input_path.rstrip("/")
    print(f"DEBUG: copy_files_in called with input_path={input_path}, local_base_dir={local_base_dir}", flush=True)

    # Auto-convert Parquet inputs into .jsonl.gz under documents/
    parquet_pattern = f"{input_path}/**/*.parquet"
    print(f"DEBUG: Checking for Parquet files with pattern {parquet_pattern}", flush=True)
    if input_path.endswith(".parquet") or fsspec_glob(parquet_pattern):
        print("DEBUG: Detected Parquet input, converting to JSONL", flush=True)
        docs_dir = os.path.join(local_base_dir, "documents")
        _parquet_to_jsonl_gz(input_path, docs_dir)
        print(f"Converted Parquet → JSONL.gz into {docs_dir}")
        return
    else:
        print("DEBUG: No Parquet files detected, proceeding with JSONL logic", flush=True)

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
):
    bloom_filter_file = os.path.join(local_base_dir, bloom_filter_file)

    # If n-gram mode and no explicit bloom_filter_size, override estimated_doc_count
    if ngram is not None and not bloom_filter_size:
        total_ngrams = estimate_total_ngrams_fast(local_base_dir, ngram)
        # total_ngrams = estimate_ngram_count(local_base_dir, ngram)
        estimated_doc_count = total_ngrams
        print(f"Estimated total {ngram.ngram_length}-grams: {estimated_doc_count}")

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
        command.extend(["--bloom_filter.size_in_bytes", str(bloom_filter_size)])
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
    JSONL lines across all `documents/**/*.jsonl.gz`.
    """
    pattern = os.path.join(local_base_dir, "documents", "**", "*.jsonl.gz")

    total_lines = 0  # total records in corpus
    sample_lines_seen = 0  # lines actually included in the sample
    ngram_total = 0  # Σ n-grams over sampled lines

    for file in fsspec_glob(pattern):
        with fsspec.open(file, "rt", compression="infer") as f:
            for raw_line in f:
                total_lines += 1
                if sample_lines_seen >= sample_lines:
                    continue

                try:
                    text = json.loads(raw_line).get("text", "")
                except json.JSONDecodeError:
                    continue

                tokens = text.split()
                L = len(tokens)

                step = max(1, ngram.stride)
                if L < ngram.ngram_length:
                    ngram_total += 1
                else:
                    ngram_total += (L - ngram.ngram_length) // step + 1
                sample_lines_seen += 1

    if sample_lines_seen == 0:
        # nothing decoded ⇒ fall back to "one n-gram per line" heuristic
        return total_lines

    avg_ngrams_per_line = ngram_total / sample_lines_seen
    return int(avg_ngrams_per_line * total_lines)


def _shard_decontaminate_source(source_path: str, doc_dir: str, num_processes: int) -> None:
    """
    Shard the decontaminate source into num_processes files under doc_dir.
    Handles both Parquet and compressed JSONL efficiently.
    """
    print(f"DEBUG: Sharding {source_path} into {num_processes} shards", flush=True)

    # Create shard writers
    writers = []
    for i in range(num_processes):
        shard_path = os.path.join(doc_dir, f"shard_{i}.jsonl.gz")
        writers.append(fsspec.open(shard_path, "wt", compression="gzip").open())

    try:
        if source_path.endswith(".parquet"):
            print("DEBUG: Detected Parquet file, using streaming approach", flush=True)
            _shard_parquet_source(source_path, writers)
        else:
            print("DEBUG: Detected JSONL file, using line-by-line approach", flush=True)
            _shard_jsonl_source(source_path, writers)
    finally:
        for w in writers:
            w.close()
        print("DEBUG: Finished sharding, closed all writers", flush=True)


def _shard_parquet_source(parquet_path: str, writers: list) -> None:
    """Stream Parquet file and distribute records across writers."""
    print(f"DEBUG: Opening Parquet file for streaming: {parquet_path}", flush=True)

    with fsspec.open(parquet_path, "rb") as f:
        parquet_file = pq.ParquetFile(f)
        print(f"DEBUG: Parquet file has {parquet_file.metadata.num_rows} rows", flush=True)

        idx = 0
        # Read in batches to avoid loading entire file into memory
        for batch in parquet_file.iter_batches(batch_size=1000):
            # Convert batch to pandas for easier record iteration
            df = batch.to_pandas()
            for record in df.to_dict(orient="records"):
                json_line = json.dumps(record) + "\n"
                writers[idx % len(writers)].write(json_line)
                idx += 1

            if idx % 10000 == 0:
                print(f"DEBUG: Processed {idx} records from Parquet", flush=True)

        print(f"DEBUG: Finished processing {idx} records from Parquet", flush=True)


def _shard_jsonl_source(jsonl_path: str, writers: list) -> None:
    """Stream JSONL file and distribute lines across writers."""
    print(f"DEBUG: Opening JSONL file for streaming: {jsonl_path}", flush=True)

    with fsspec.open(jsonl_path, "rt", compression="infer") as fr:
        for idx, line in enumerate(fr):
            writers[idx % len(writers)].write(line)

            if idx % 10000 == 0:
                print(f"DEBUG: Processed {idx} lines from JSONL", flush=True)

        print(f"DEBUG: Finished processing {idx+1} lines from JSONL", flush=True)


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
        fsspec_rm(tmpdir)


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
        fsspec_rm(tmpdir)


def _run_train_test_overlap(config: DedupeConfig):
    """Run train-test overlap detection, supporting multiple n-gram sizes efficiently."""
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="marin_dedupe_") as tmpdir:
        print(f"DEBUG: _run_train_test_overlap starting with tmpdir={tmpdir}", flush=True)
        
        
        if not config.decontaminate_source:
            raise ValueError("decontaminate_source is required in TRAIN_TEST_OVERLAP mode")
        if not config.ngram:
            raise ValueError("ngram config is required in TRAIN_TEST_OVERLAP mode")
        
        print(f"DEBUG: decontaminate_source={config.decontaminate_source}", flush=True)

        # Handle both single int and list of ints for ngram_length
        ngram_lengths = (
            config.ngram.ngram_length 
            if isinstance(config.ngram.ngram_length, list) 
            else [config.ngram.ngram_length]
        )
        print(f"DEBUG: Processing n-gram sizes: {ngram_lengths}", flush=True)

        # 1) Shard the decontaminate source ONCE (reused for all n-gram sizes)
        doc_dir = os.path.join(tmpdir, "documents")
        fsspec_mkdirs(doc_dir)
        print(f"DEBUG: Created doc_dir={doc_dir}", flush=True)
        print(f"DEBUG: Sharding decontaminate source (once for all n-gram sizes)", flush=True)
        _shard_decontaminate_source(config.decontaminate_source, doc_dir, config.processes)
        
        # Process each n-gram size with the same sharded decontaminate source
        for i, ngram_len in enumerate(ngram_lengths):
            print(f"DEBUG: Processing n-gram size {ngram_len} ({i+1}/{len(ngram_lengths)})", flush=True)
            
            # Create n-gram specific configuration
            current_ngram_config = NGramConfig(
                ngram_length=ngram_len,
                stride=config.ngram.stride,
                overlap_threshold=config.ngram.overlap_threshold
            )
            
            # Create attribute name for this n-gram size
            current_attr_name = (
                f"{config.attribute_name}_{ngram_len}" 
                if len(ngram_lengths) > 1 
                else config.attribute_name
            )
            print(f"DEBUG: Using attribute name: {current_attr_name}", flush=True)
            
            # Create n-gram specific bloom filter path
            current_bloom_filter = f"{ngram_len}_{config.bloom_filter_path}" if len(ngram_lengths) > 1 else config.bloom_filter_path
            
            # 2) Build the bloom filter for this n-gram size using the sharded decontaminate source
            print(f"DEBUG: Building bloom filter for {ngram_len}-gram using existing sharded source", flush=True)
            do_dedup(
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
            )
            
            print(f"DEBUG: Completed building {ngram_len}-gram bloom filter", flush=True)
            
        # 3) Clear out the decontaminate JSONLs (keep bloom filters)
        print(f"DEBUG: Clearing decontaminate JSONLs after building all bloom filters", flush=True)
        delete_jsonl_files(tmpdir)
        
        # 4) Copy input files ONCE (reused for all n-gram sizes)
        print(f"DEBUG: Copying input files (once for all n-gram sizes)", flush=True)
        copy_files_in(config.input_path, tmpdir)
        
        # 5) Apply each bloom filter to the same input files
        for i, ngram_len in enumerate(ngram_lengths):
            print(f"DEBUG: Applying {ngram_len}-gram filter to input data ({i+1}/{len(ngram_lengths)})", flush=True)
            
            # Create n-gram specific configuration (same as before)
            current_ngram_config = NGramConfig(
                ngram_length=ngram_len,
                stride=config.ngram.stride,
                overlap_threshold=config.ngram.overlap_threshold
            )
            
            # Same attribute name and bloom filter as before
            current_attr_name = (
                f"{config.attribute_name}_{ngram_len}" 
                if len(ngram_lengths) > 1 
                else config.attribute_name
            )
            current_bloom_filter = f"{ngram_len}_{config.bloom_filter_path}" if len(ngram_lengths) > 1 else config.bloom_filter_path
            
            # Apply this n-gram's bloom filter to the input files
            do_dedup(
                tmpdir,
                current_attr_name,
                config.min_length,
                config.min_words,
                config.bloom_filter_size,
                config.estimated_doc_count,
                config.false_positive_rate,
                current_ngram_config,
                config.processes,
                read_only=True,  # Use existing bloom filter
                bloom_filter_file=current_bloom_filter,
            )
            
            # 6) Write results for this n-gram size
            print(f"DEBUG: Writing results for {ngram_len}-gram", flush=True)
            copy_files_out(tmpdir, config.output_path, current_attr_name)
            
            print(f"DEBUG: Completed processing {ngram_len}-gram", flush=True)
        
        print(f"DEBUG: Completed processing all {len(ngram_lengths)} n-gram sizes efficiently", flush=True)
        fsspec_rm(tmpdir)
        

@ray.remote(num_cpus=64, memory=1024 * 1024 * 1024 * 128)
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
