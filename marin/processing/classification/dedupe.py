import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import draccus
import fsspec
import ray
from tqdm import tqdm

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir, fsspec_mkdirs, fsspec_rm, rebase_file_path


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


def copy_files_in(input_path, local_base_dir):
    # Ensure input_path doesn't end with a slash
    input_path = input_path.rstrip("/")

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
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
        if decontaminate:
            # First we copy the files to the temporary directory to get bloom filter
            copy_files_in(decomtaminate_dir, tmpdir)
            do_dedup(
                tmpdir,
                attribute_name,
                min_length,
                min_words,
                bloom_filter_size,
                # estimated doc count will be computed in do_dedup
                0,
                false_positive_rate,
                ngram,
                processes,
                read_only=False,
                bloom_filter_file="decontaminated_bloom_filter.bin",
            )

            # Delete all JSONL files in the temporary directory since we have bloom filter
            delete_jsonl_files(tmpdir)
            # Then copy files of interest and apply bloom filter read only
            copy_files_in(input_path, tmpdir)
            do_dedup(
                tmpdir,
                attribute_name,
                min_length,
                min_words,
                bloom_filter_size,
                0,
                false_positive_rate,
                ngram,
                processes,
                read_only=True,
                bloom_filter_file="decontaminated_bloom_filter.bin",
            )
        else:
            copy_files_in(input_path, tmpdir)
            do_dedup(
                tmpdir,
                attribute_name,
                min_length,
                min_words,
                bloom_filter_size,
                estimated_doc_count,
                false_positive_rate,
                ngram,
                processes,
            )

        # copy files out stays the same.
        # Ensure output_path doesn't end with a slash
        output_path = output_path.rstrip("/")

        local_attribute_dir = os.path.join(tmpdir, "attributes", attribute_name)

        # Get all .jsonl.gz files in the local attribute directory
        glob_path = f"{local_attribute_dir}/**/*.jsonl.gz"
        local_files = fsspec_glob(glob_path)
        for local_file in tqdm(local_files, desc="Uploading files"):
            output_file = rebase_file_path(local_attribute_dir, local_file, output_path)
            copy_file_out(local_file, output_file)

    return "Deduplication process completed"


@ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 2)
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
