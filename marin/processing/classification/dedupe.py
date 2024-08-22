import argparse
import ray
import time
import psutil
import multiprocessing
import socket
import os
import tempfile
from google.cloud import storage
from tqdm import tqdm
import glob
import shutil
import subprocess
import fsspec
from marin.utils import validate_marin_gcp_path, fsspec_mkdirs, rebase_file_path, fsspec_get_curr_subdirectories, fsspec_isdir, fsspec_dir_only_contains_files, fsspec_glob

def copy_files_in(input_dir, local_base_dir):
    # Ensure input_dir doesn't end with a slash
    input_dir = input_dir.rstrip('/')
    
    # Get all .jsonl.gz files in the input directory
    glob_path = f"{input_dir}/**/*.jsonl.gz"
    print(f"glob_path: {glob_path}")
    input_files = fsspec_glob(glob_path)
    
    print(f"printing input files five: {input_files[:5]}")
    
    for input_file in tqdm(input_files, desc="Copying files"):
        # Extract the relative path from the input file
        relative_path = os.path.relpath(input_file, input_dir)
        
        # Construct the output path, ensuring it's under the 'documents' directory
        output_file = os.path.join(local_base_dir, 'documents', relative_path)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        fsspec_mkdirs(output_dir)
        
        # Copy the file using fsspec
        with fsspec.open(input_file, "rb", compression="infer") as f_remote:
            with fsspec.open(output_file, "wb", compression="gzip") as f_local:
                f_local.write(f_remote.read())

    # Dolma deduplicator requires 'documents/' as a subdirectory
    print(f"Copied {len(input_files)} files to {os.path.join(local_base_dir, 'documents')}")

def do_dedup(local_base_dir, attribute_name, min_length, min_words, bloom_filter_size, estimated_doc_count, false_positive_rate, processes):
    command = [
        "RUST_BACKTRACE=full",
        "dolma", "dedupe",
        "--documents", f"{local_base_dir}/documents/**/*.jsonl.gz",
        "--dedupe.paragraphs.attribute_name", attribute_name,
        "--dedupe.skip_empty" ,
        "--dedupe.min_length", str(min_length),
        "--dedupe.min_words", str(min_words),
        "--bloom_filter.file", f"{local_base_dir}/deduper_bloom_filter_test.bin",
        "--no-bloom_filter.read_only",
        "--processes", str(processes)
    ]

    if bloom_filter_size:
        command.extend(["--bloom_filter.size_in_bytes", str(bloom_filter_size)])
    else:
        command.extend([
            "--bloom_filter.estimated_doc_count", str(estimated_doc_count),
            "--bloom_filter.desired_false_positive_rate", str(false_positive_rate)
        ])
    
    
    process = subprocess.Popen(
        ' '.join(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   
        shell=True,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    
    # Rename temporary files
    attr_dir = os.path.join(local_base_dir, "attributes/duplicate_documents")
    print(f"Checking for temporary files in: {attr_dir}")
    for root, _, files in os.walk(attr_dir):
        for file in files:
            if file.endswith('.jsonl.gz.tmp'):

                old_path = os.path.join(root, file)
                new_path = old_path.rsplit('.tmp', 1)[0]
                os.rename(old_path, new_path)
    
    return process.returncode

def copy_files_out(local_base_dir, output_dir, attribute_name):
    # Ensure output_dir doesn't end with a slash
    output_dir = output_dir.rstrip('/')
    
    local_attribute_dir = os.path.join(local_base_dir, 'attributes', attribute_name)
    
    # Get all .jsonl.gz files in the local attribute directory
    glob_path = f"{local_attribute_dir}/**/*.jsonl.gz"
    local_files = fsspec_glob(glob_path)
    
    files_uploaded = 0
    for local_file in tqdm(local_files, desc="Uploading files"):
        # Use rebase_file_path to get the correct output path
        output_file = rebase_file_path(local_base_dir, local_file, output_dir)
        
        print(f"[DEBUG] Uploading {local_file} to {output_file}")
        
        # Ensure the output directory exists
        output_file_dir = os.path.dirname(output_file)
        fsspec_mkdirs(output_file_dir)
        
        # Copy the file using fsspec
        with fsspec.open(local_file, "rb") as f_local:
            with fsspec.open(output_file, "wb") as f_remote:
                f_remote.write(f_local.read())
        
        files_uploaded += 1
    
    print(f"Uploaded {files_uploaded} files to {output_dir}")



@ray.remote(runtime_env={"pip": ["dolma"]})
def dolma_dedup(input_dir, output_dir, attribute_name, min_length, min_words, bloom_filter_size, estimated_doc_count, false_positive_rate, processes):
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            copy_files_in(input_dir, tmpdir)
            do_dedup(tmpdir, attribute_name, min_length, min_words, bloom_filter_size, estimated_doc_count, false_positive_rate, processes)
            copy_files_out(tmpdir, output_dir, attribute_name)
        except Exception as e:
            print(f"An error occurred during deduplication: {e}")
    return "Deduplication process completed"

def main(config):
    ray.init()

    input_dir = validate_marin_gcp_path(config.input_dir)
    output_dir = validate_marin_gcp_path(config.output_dir)
    result = ray.get(dolma_dedup.remote(input_dir, output_dir, config.attribute_name, config.min_length, config.min_words, config.bloom_filter_size, config.estimated_doc_count, config.false_positive_rate, config.processes))
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dolma deduplication on a single Ray worker")
    parser.add_argument('--input_dir', type=str, required=True, help='GCP input directory path')
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory to save attributes for dedupe")
    parser.add_argument('--attribute_name', type=str, default='duplicate_text', help='Name of the attribute to set if the document is a duplicate')
    parser.add_argument('--min_length', type=int, default=0, help='Minimum length of documents to be deduplicated')
    parser.add_argument('--min_words', type=int, default=0, help='Minimum number of uniseg word units in documents to be deduplicated')
    parser.add_argument('--bloom_filter_size', type=int, default=None, help='Size of the Bloom filter in bytes')
    parser.add_argument('--estimated_doc_count', type=int, default=1000000, help='Estimated number of documents to dedupe')
    parser.add_argument('--false_positive_rate', type=float, default=0.001, help='Desired false positive rate for the Bloom filter')
    parser.add_argument('--processes', type=int, default=1, help='Number of processes to use for deduplication')
    args = parser.parse_args()
    config = argparse.Namespace(
    input_dir=args.input_dir,
    output_dir=args.output_dir,
    attribute_name=args.attribute_name,
    min_length=args.min_length,
    min_words=args.min_words,
    bloom_filter_size=args.bloom_filter_size,
    estimated_doc_count=args.estimated_doc_count,
    false_positive_rate=args.false_positive_rate,
    processes=args.processes
    )   
    main(config)