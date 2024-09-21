import os
import subprocess
import tempfile
from dataclasses import dataclass

import draccus
import fsspec
import ray
from tqdm import tqdm

from marin.utils import fsspec_glob, fsspec_mkdirs, fsspec_rm, rebase_file_path, validate_marin_gcp_path


@dataclass
class DedupeConfig:
    input_path: str
    output_path: str
    attribute_name: str = "duplicate_text"
    min_length: int = 0
    min_words: int = 0
    bloom_filter_size: int | None = None  # default to 0 to use estimated_doc_count and false_positive_rate
    estimated_doc_count: int = 1000000
    false_positive_rate: float = 0.001
    processes: int = 1
    decontaminate: bool = False
    decontaminate_path: str | None = None


def copy_files_in(input_path, local_base_dir):
    # Ensure input_path doesn't end with a slash
    input_path = input_path.rstrip("/")

    # Get all .jsonl.gz files in the input directory
    glob_path = f"{input_path}/**/*.jsonl.gz"
    print(f"glob_path: {glob_path}")
    input_files = fsspec_glob(glob_path)

    print(f"printing first five input files: {input_files[:5]}")

    for input_file in tqdm(input_files, desc="Copying files"):
        # Extract the relative path from the input file
        relative_path = os.path.relpath(input_file, input_path)

        # Construct the output path, ensuring it's under the 'documents' directory
        output_file = os.path.join(local_base_dir, "documents", relative_path)

        # Ensure the output directory exists
        output_path = os.path.dirname(output_file)
        fsspec_mkdirs(output_path)

        # Copy the file using fsspec
        with fsspec.open(input_file, "rb", compression="infer") as f_remote:
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
    processes,
    read_only=False,
    bloom_filter_file="deduper_bloom_filter.bin",
):
    bloom_filter_file = os.path.join(local_base_dir, bloom_filter_file)
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


def copy_files_out(local_base_dir, output_path, attribute_name):
    # Ensure output_path doesn't end with a slash
    output_path = output_path.rstrip("/")

    local_attribute_dir = os.path.join(local_base_dir, "attributes", attribute_name)

    # Get all .jsonl.gz files in the local attribute directory
    glob_path = f"{local_attribute_dir}/**/*.jsonl.gz"
    local_files = fsspec_glob(glob_path)

    files_uploaded = 0
    for local_file in tqdm(local_files, desc="Uploading files"):
        # Use rebase_file_path to get the correct output path
        output_file = rebase_file_path(local_attribute_dir, local_file, output_path)

        print(f"[DEBUG] Uploading {local_file} to {output_file}")

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


@ray.remote(runtime_env={"pip": ["dolma"]})
def dolma_dedup(
    input_path,
    output_path,
    attribute_name,
    min_length,
    min_words,
    bloom_filter_size,
    estimated_doc_count,
    false_positive_rate,
    processes,
    decomtaminate_dir,
    decontaminate,
):

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if decontaminate:
                # First we copy the files to the temporary directory to get bloom filter
                copy_files_in(decomtaminate_dir, tmpdir)
                do_dedup(
                    tmpdir,
                    attribute_name,
                    min_length,
                    min_words,
                    bloom_filter_size,
                    estimated_doc_count,
                    false_positive_rate,
                    processes,
                    read_only=False,
                    bloom_filter_file="decotaminated_bloom_filter.bin",
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
                    estimated_doc_count,
                    false_positive_rate,
                    processes,
                    read_only=True,
                    bloom_filter_file="decotaminated_bloom_filter.bin",
                )
                copy_files_out(tmpdir, output_path, attribute_name)
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
                    processes,
                )
                copy_files_out(tmpdir, output_path, attribute_name)
        except Exception as e:
            print(f"An error occurred during deduplication: {e}")
    return "Deduplication process completed"


@ray.remote
def main_ray(config: DedupeConfig):
    # require directory if decontaminate is set
    if config.decontaminate and config.decontaminate_path is None:
        raise ValueError("decontaminate_path is required if decontaminate is set")

    input_path = validate_marin_gcp_path(config.input_path)
    output_path = validate_marin_gcp_path(config.output_path)
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
            config.processes,
            config.decontaminate_path,
            config.decontaminate,
        )
    )
    print(result)


@draccus.wrap()
def main(config: DedupeConfig):
    ray.init()
    ray.get(main_ray.remote(config))


if __name__ == "__main__":
    main()
