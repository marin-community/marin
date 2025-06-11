import functools
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

import draccus
import fsspec
import psutil
import ray
from tqdm import tqdm

# Import dolma deduper with PYTHONPATH approach
dolma_deduper = None

# Check PYTHONPATH
import sys
print(f"PYTHONPATH includes: {[p for p in sys.path if 'dolma' in p]}")

try:
    # Try direct import from dolma.cli first (most likely location)
    from dolma.cli import deduper as dolma_deduper
    print("SUCCESS: Imported dolma deduper from dolma.cli")
except ImportError as e:
    print(f"Failed to import from dolma.cli: {e}")
    
    try:
        # Fallback: try direct dolma import
        from dolma import deduper as dolma_deduper
        print("SUCCESS: Imported dolma deduper from dolma")
    except ImportError as e2:
        print(f"Failed to import from dolma: {e2}")
        
        # Debug what's available
        try:
            import dolma
            print(f"dolma location: {getattr(dolma, '__file__', 'unknown')}")
            print(f"dolma contents: {dir(dolma)}")
            
            # Check if cli module exists
            try:
                import dolma.cli
                print(f"dolma.cli contents: {dir(dolma.cli)}")
            except ImportError:
                print("dolma.cli module not found")
                
        except ImportError as e3:
            print(f"Cannot import dolma at all: {e3}")

if dolma_deduper is None:
    print("ERROR: dolma_deduper is None - will fail at runtime")

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir, fsspec_mkdirs, fsspec_rm, rebase_file_path

# Toggle manual partition for splitting single JSONL into multiple chunks
MANUAL_PARTITION = True


def profile(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        start_cpu = process.cpu_times()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        end_mem = process.memory_info().rss
        end_cpu = process.cpu_times()
        # CPU time consumed by this function (user + system)
        cpu_time = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
        duration = end_time - start_time
        total_cpus = psutil.cpu_count(logical=True)
        avg_cores = cpu_time / duration if duration > 0 else 0.0
        print(
            f"[PROFILE] {func.__name__} took {duration:.2f}s; mem delta {(end_mem - start_mem)/1024**2:.2f}MB;"
            f" CPU time {cpu_time:.2f}s; avg cores {avg_cores:.2f}/{total_cpus}",
            flush=True,
        )
        return result

    return wrapper


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
        ngram_length (int | list[int]): Size of the ngram (e.g. 8) or list of sizes
        stride (int): Step size when moving through string to generate ngrams
        overlap_threshold (float): Percentage of duplicate ngrams for a paragraph to be considered duplicate
    """

    ngram_length: int | list[int] = 8
    stride: int = 0
    overlap_threshold: float = 0.7

    def get_lengths(self) -> list[int]:
        """Helper method to always return a list of lengths"""
        if isinstance(self.ngram_length, int):
            return [self.ngram_length]
        return self.ngram_length


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
        decontaminate (bool): run in decontamination mode (don't add non benchmark text to Bloom filter)
        decontaminate_path (str): path of decontamination set text
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
    decontaminate: bool = False
    decontaminate_path: str | None = None


def copy_files_in(input_path, local_base_dir, splits=None):
    # Ensure input_path doesn't end with a slash
    input_path = input_path.rstrip("/")
    # Manual partition logic: split single JSONL into 'splits' parts
    if MANUAL_PARTITION and splits and fsspec_exists(input_path) and not fsspec_isdir(input_path):
        relative_name = os.path.basename(input_path)
        # Normalize stem and set output suffix to .jsonl.gz
        if relative_name.endswith(".jsonl.zst"):
            stem = relative_name[: -len(".jsonl.zst")]
        elif relative_name.endswith(".jsonl.gs"):
            stem = relative_name[: -len(".jsonl.gs")]
        elif relative_name.endswith(".json.zst"):
            stem = relative_name[: -len(".json.zst")]
        elif relative_name.endswith(".json.gz"):
            stem = relative_name[: -len(".json.gz")]
        elif relative_name.endswith(".jsonl.gz"):
            stem = relative_name[: -len(".jsonl.gz")]
        else:
            stem, _ = os.path.splitext(relative_name)
        suffix = ".jsonl.gz"
        # Count total lines
        total_lines = 0
        with fsspec.open(input_path, "rt", compression="infer") as reader:
            for _ in reader:
                total_lines += 1
        if total_lines == 0:
            print(f"No lines to split in {input_path}")
            return
        chunk_size = (total_lines + splits - 1) // splits
        print(f"Splitting {relative_name} ({total_lines} lines) into {splits} parts (chunk_size={chunk_size})")
        # Ensure base documents directory exists
        fsspec_mkdirs(os.path.join(local_base_dir, "documents"))
        part = None
        writer = None
        with fsspec.open(input_path, "rt", compression="infer") as reader:
            for idx, line in enumerate(reader):
                p = min(idx // chunk_size, splits - 1)
                if p != part:
                    if writer:
                        writer.close()
                    chunk_name = f"{stem}_part{p}{suffix}"
                    out_path = os.path.join(local_base_dir, "documents", chunk_name)
                    fsspec_mkdirs(os.path.dirname(out_path))
                    writer = gzip.open(out_path, "wt")
                    part = p
                writer.write(line)
            if writer:
                writer.close()
        print(f"Split into {splits} files under {local_base_dir}/documents")
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


@profile
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
        total_ngrams = estimate_total_ngrams_fast_fixed(local_base_dir, ngram)
        # total_ngrams = estimate_ngram_count(local_base_dir, ngram)
        estimated_doc_count = total_ngrams
        print(f"Estimated total {ngram.ngram_length}-grams: {estimated_doc_count}")

    # Run Dolma deduper directly via Python API instead of subprocess

    dolma_config = {
        "documents": [f"{local_base_dir}/documents/**/*.jsonl.gz"],
        "dedupe": {
            "name": attribute_name,
            "skip_empty": True,
            "min_length": min_length,
            "min_words": min_words,
            "paragraphs": {
                "attribute_name": attribute_name,
                "by_ngram": {
                    "ngram_length": ngram.ngram_length,
                    "overlap_threshold": ngram.overlap_threshold,
                    "stride": ngram.stride,
                },
                "paragraph_separator": "\u001e\u001e",
            },
        },
        "bloom_filter": {
            "file": bloom_filter_file,
            "read_only": read_only,
            "size_in_bytes": bloom_filter_size or 0,
            "estimated_doc_count": estimated_doc_count,
            "desired_false_positive_rate": false_positive_rate,
        },
        "work_dir": {"input": local_base_dir, "output": local_base_dir},
        "processes": processes,
        "compression": {"input": None, "output": None},
        "dryrun": False,
        "is_s3_volume": False,
    }
    print(f"Running Dolma deduper with config: {dolma_config}", flush=True)
    
    if dolma_deduper is None:
        raise ImportError("Could not import dolma deduper function. Make sure dolma is properly installed.")
    
    dolma_deduper(dolma_config)

    # Rename any temporary output files
    for root, _, files in os.walk(os.path.join(local_base_dir, "attributes")):
        for file in files:
            if file.endswith(".jsonl.gz.tmp"):
                old_path = os.path.join(root, file)
                new_path = old_path.rsplit(".tmp", 1)[0]
                os.rename(old_path, new_path)

    return 0


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
        print(f"No files found in {local_attribute_dir}, shallow globbing", flush=True)
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


@cached_or_construct_output(success_suffix="SUCCESS")
def copy_file_out(input_file_path, output_file_path):

    # Ensure the output directory exists
    output_file_dir = os.path.dirname(output_file_path)
    fsspec_mkdirs(output_file_dir)

    # Copy the file using fsspec
    with fsspec.open(input_file_path, "rb") as f_local:
        with fsspec.open(output_file_path, "wb") as f_remote:
            f_remote.write(f_local.read())

    print(f"Uploaded file to {output_file_path}")


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
def estimate_total_ngrams_fast_fixed(local_base_dir, ngram, sample_lines: int = 1000):
    """
    Estimate *unique* n-gram count quickly by reading at most `sample_lines`
    JSONL lines across all `documents/**/*.jsonl.gz`.

    ----------------------------------------------------------------------
    ⚠️  FIX: stride handling
    ----------------------------------------------------------------------
    In the original version `stride==0` (Dolma's default, meaning "advance
    one token") was special-cased as

        stride = ngram.stride if ngram.stride > 0 else 1

    but that value was *not* reused in the formula below, so the
    denominator differed between the two statements.  We instead compute
    a single variable `step = max(1, ngram.stride)` and use it
    consistently when counting n-grams.
    ----------------------------------------------------------------------
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

                step = max(1, ngram.stride)  # ← fixed
                if L < ngram.ngram_length:
                    ngram_total += 1
                else:
                    ngram_total += (L - ngram.ngram_length) // step + 1

                # one illustrative debug print per file
                if sample_lines_seen == 0:
                    print(f"[sample] L={L}, step={step}, ngrams_for_line=" f"{(L - ngram.ngram_length) // step + 1}")
                sample_lines_seen += 1

    if sample_lines_seen == 0:
        # nothing decoded ⇒ fall back to "one n-gram per line" heuristic
        return total_lines

    avg_ngrams_per_line = ngram_total / sample_lines_seen
    return int(avg_ngrams_per_line * total_lines)


# Fast estimate of total n-grams by sampling a subset of documents
def estimate_total_ngrams_fast(local_base_dir, ngram, sample_lines=1000):
    """
    Fast estimate of total n-grams by sampling up to sample_lines JSONL lines
    across all staged documents.
    """
    pattern = os.path.join(local_base_dir, "documents", "**", "*.jsonl.gz")
    total_docs = 0
    sample_count = 0
    ngram_sum = 0
    files = fsspec_glob(pattern)
    for file in files:
        with fsspec.open(file, "rt", compression="infer") as f:
            for idx, line in enumerate(f):
                total_docs += 1
                if sample_count < sample_lines:
                    try:
                        rec = json.loads(line)
                        text = rec.get("text", "")
                    except json.JSONDecodeError:
                        continue
                    tokens = text.split()
                    L = len(tokens)

                    if L < ngram.ngram_length:
                        ngram_sum += 1
                    else:
                        stride = ngram.stride if ngram.stride > 0 else 1
                        ngram_sum += (L - ngram.ngram_length) // stride + 1
                    sample_count += 1
    if sample_count == 0:
        return total_docs
    avg_ngrams = ngram_sum / sample_count
    return int(avg_ngrams * total_docs)


def get_dolma_attribute_name(base_attribute_name: str, ngram_length: int) -> str:
    """Generate attribute name with ngram suffix"""
    return f"{base_attribute_name}_{ngram_length}"


def get_bloom_filter_name(ngram_length: int) -> str:
    """Generate bloom filter filename with ngram suffix"""
    return f"decontaminated_bloom_filter_{ngram_length}.bin"


def get_local_organized_dir(tmpdir: str, base_attribute_name: str, ngram_length: int) -> str:
    """Target directory after reorganization"""
    return os.path.join(tmpdir, "attributes", base_attribute_name, str(ngram_length))


def get_local_dolma_dir(tmpdir: str, base_attribute_name: str, ngram_length: int) -> str:
    """Directory where dolma writes results with ngram suffix"""
    dolma_attr_name = get_dolma_attribute_name(base_attribute_name, ngram_length)
    return os.path.join(tmpdir, "attributes", dolma_attr_name)


@profile
def reorganize_attributes_by_ngram(tmpdir: str, base_attribute_name: str, ngram_lengths: list[int], job_id: str):
    """
    Reorganize dolma output from:
      attributes/mmlu_overlap_8_abc123/ -> attributes/mmlu_overlap/8/
      attributes/mmlu_overlap_13_abc123/ -> attributes/mmlu_overlap/13/
    """
    for ngram_length in ngram_lengths:
        dolma_dir = get_local_dolma_dir(tmpdir, base_attribute_name, ngram_length)
        target_dir = get_local_organized_dir(tmpdir, base_attribute_name, ngram_length)

        if os.path.exists(dolma_dir):
            # Ensure target parent exists
            os.makedirs(os.path.dirname(target_dir), exist_ok=True)
            # Move the directory
            shutil.move(dolma_dir, target_dir)
            print(f"Reorganized {dolma_dir} -> {target_dir}")


@ray.remote
def dolma_dedup(
    input_path,
    output_path,
    attribute_name,
    min_length,
    min_words,
    bloom_filter_size,
    estimated_doc_count,
    false_positive_rate,
    ngram,
    processes,
    decomtaminate_dir,
    decontaminate,
):
    # require ngram config for decontamination
    if decontaminate and ngram is None:
        raise ValueError("ngram configuration is required in decontamination mode")

    # Starting deduplication (ngram suffix distinguishes jobs)
    print("Starting deduplication job")

    # Get all n-gram lengths to process
    ngram_lengths = ngram.get_lengths() if ngram else [None]

    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
        print(f"Using temporary directory: {tmpdir}")

        if decontaminate:
            # PHASE 1: Build bloom filters
            copy_files_in(decomtaminate_dir, tmpdir, splits=processes)

            for ngram_length in ngram_lengths:
                print(f"Building bloom filter for {ngram_length}-grams...")

                # Create ngram config for this length
                length_ngram = NGramConfig(
                    ngram_length=ngram_length, stride=ngram.stride, overlap_threshold=ngram.overlap_threshold
                )

                # Use unique attribute name and bloom filter with job ID
                dolma_attr_name = get_dolma_attribute_name(attribute_name, ngram_length)
                bloom_filter_file = get_bloom_filter_name(ngram_length)

                do_dedup(
                    tmpdir,
                    dolma_attr_name,
                    min_length,
                    min_words,
                    bloom_filter_size,
                    0,  # estimated doc count will be computed in do_dedup
                    false_positive_rate,
                    length_ngram,
                    processes,
                    read_only=False,
                    bloom_filter_file=bloom_filter_file,
                )

            # Delete all JSONL files in the temporary directory since we have bloom filters
            delete_jsonl_files(tmpdir)

            # PHASE 2: Apply bloom filters
            copy_files_in(input_path, tmpdir, splits=processes)

            for ngram_length in ngram_lengths:
                print(f"Applying {ngram_length}-gram bloom filter...")

                length_ngram = NGramConfig(
                    ngram_length=ngram_length, stride=ngram.stride, overlap_threshold=ngram.overlap_threshold
                )

                dolma_attr_name = get_dolma_attribute_name(attribute_name, ngram_length)
                bloom_filter_file = get_bloom_filter_name(ngram_length)

                do_dedup(
                    tmpdir,
                    dolma_attr_name,
                    min_length,
                    min_words,
                    bloom_filter_size,
                    0,
                    false_positive_rate,
                    length_ngram,
                    processes,
                    read_only=True,
                    bloom_filter_file=bloom_filter_file,
                )
        else:
            # Standard deduplication mode
            copy_files_in(input_path, tmpdir, splits=processes)

            for ngram_length in ngram_lengths:
                if ngram_length is not None:
                    print(f"Running deduplication for {ngram_length}-grams...")

                    length_ngram = NGramConfig(
                        ngram_length=ngram_length, stride=ngram.stride, overlap_threshold=ngram.overlap_threshold
                    )
                    dolma_attr_name = get_dolma_attribute_name(attribute_name, ngram_length)
                    bloom_filter_file = f"bloom_filter_{ngram_length}.bin"
                else:
                    length_ngram = None
                    dolma_attr_name = attribute_name
                    bloom_filter_file = "bloom_filter.bin"

                do_dedup(
                    tmpdir,
                    dolma_attr_name,
                    min_length,
                    min_words,
                    bloom_filter_size,
                    estimated_doc_count,
                    false_positive_rate,
                    length_ngram,
                    processes,
                    bloom_filter_file=bloom_filter_file,
                )

        # PHASE 3: Upload all attribute files Dolma produced
        base_attr_dir = os.path.join(tmpdir, "attributes")
        all_files = fsspec_glob(f"{base_attr_dir}/**/*.jsonl.gz")
        for local_file in tqdm(all_files, desc="Uploading results"):
            # compute remote path relative to base_attr_dir
            rel_path = os.path.relpath(local_file, base_attr_dir)
            remote_file = os.path.join(output_path.rstrip("/"), rel_path)
            copy_file_out(local_file, remote_file)

    print("Deduplication completed successfully")
    return f"Deduplication process completed for n-grams: {ngram_lengths}"


@ray.remote(num_cpus=16, memory=1024 * 1024 * 1024 * 2)
def dedupe(config: DedupeConfig):
    # require directory if decontaminate is set
    if config.decontaminate and config.decontaminate_path is None:
        raise ValueError("decontaminate_path is required if decontaminate is set")

    input_path = config.input_path
    output_path = config.output_path
    result = ray.get(
        dolma_dedup.remote(
            input_path,
            output_path,
            config.attribute_name,
            config.min_length,
            config.min_words,
            config.bloom_filter_size,
            config.estimated_doc_count,
            config.false_positive_rate,
            config.ngram,
            config.processes,
            config.decontaminate_path,
            config.decontaminate,
        )
    )
    print(result)


@draccus.wrap()
def main(config: DedupeConfig):
    ray.get(dedupe.remote(config))


if __name__ == "__main__":
    main()
