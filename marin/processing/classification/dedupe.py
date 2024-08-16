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
from marin.utils import validate_gcp_path, fsspec_mkdirs, rebase_file_path

def copy_files_in(input_dir, local_base_dir):
    print(f"\n[DEBUG] Starting copy_files_in function")
    print(f"[DEBUG] Input directory: {input_dir}")
    print(f"[DEBUG] Local base directory: {local_base_dir}")
    
    client = storage.Client()
    bucket_name = input_dir.split('/')[2]
    prefix = '/'.join(input_dir.split('/')[3:])
    print(f"[DEBUG] Bucket name: {bucket_name}")
    print(f"[DEBUG] Prefix: {prefix}")
    
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"[DEBUG] Number of blobs found: {len(blobs)}")
    
    for blob in tqdm(blobs, desc="Downloading files"):
        if blob.name.endswith('.jsonl.gz'):
            local_file_path = os.path.join(local_base_dir, 'documents', blob.name.split(prefix)[1].lstrip('/'))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            print(f"[DEBUG] Downloading {blob.name} to {local_file_path}", flush=True)
            blob.download_to_filename(local_file_path)
    
    print(f"[DEBUG] Files in local base directory after download:")
    for root, dirs, files in os.walk(local_base_dir):
        for file in files:
            print(f"[DEBUG] {os.path.join(root, file)}")

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
    print(f"[DEBUG] Dolma dedupe command: {' '.join(command)}")
    
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
    print(f"\n[DEBUG] Dolma dedupe command completed with return code: {process.returncode}")
    
    print(f"[DEBUG] Contents of local base directory after dedupe:")
    for root, dirs, files in os.walk(local_base_dir):
        for file in files:
            print(f"[DEBUG] {os.path.join(root, file)}")
    
    # Rename temporary files
    attr_dir = os.path.join(local_base_dir, "attributes/duplicate_documents")
    print(f"[DEBUG] Checking for temporary files in: {attr_dir}")
    tmp_files_found = False
    for root, _, files in os.walk(attr_dir):
        for file in files:
            if file.endswith('.jsonl.gz.tmp'):
                tmp_files_found = True
                old_path = os.path.join(root, file)
                new_path = old_path[:-4]  # Remove '.tmp'
                print(f"[DEBUG] Renaming {old_path} to {new_path}")
                os.rename(old_path, new_path)
    
    if not tmp_files_found:
        print("[DEBUG] No .tmp files found in the attributes directory")
    
    return process.returncode

def copy_files_out(local_base_dir, output_dir, attribute_name):
    print(f"\n[DEBUG] Starting copy_files_out function")
    print(f"[DEBUG] Local base directory: {local_base_dir}")
    print(f"[DEBUG] Output directory: {output_dir}")
    
    client = storage.Client()
    
    # Remove 'gs://' prefix if present
    bucket_name = output_dir.split('/')[2]
    gcs_base_path = '/'.join(output_dir.split('/')[3:])
    
    print(f"[DEBUG] Bucket name: {bucket_name}")
    print(f"[DEBUG] GCS base path: {gcs_base_path}")
    
    bucket = client.get_bucket(bucket_name)
    local_attribute_dir = os.path.join(local_base_dir, 'attributes', attribute_name)
    print(f"[DEBUG] Local attribute directory: {local_attribute_dir}")
    
    files_uploaded = 0
    for root, _, files in os.walk(local_attribute_dir):
        for file in files:
            if file.endswith('.jsonl.gz'):
                local_file_path = os.path.join(root, file)
                
                # Use rebase_file_path to get the correct GCS path
                gcs_file_path = rebase_file_path(local_attribute_dir, local_file_path, gcs_base_path)
                
                print(f"[DEBUG] Uploading {local_file_path} to gs://{bucket_name}/{gcs_file_path}")
                
                # Ensure the directory exists in GCS
                gcs_dir = os.path.dirname(f"gs://{bucket_name}/{gcs_file_path}")
                fsspec_mkdirs(gcs_dir)
                
                blob = bucket.blob(gcs_file_path)
                blob.upload_from_filename(local_file_path)
                files_uploaded += 1
    
    print(f"[DEBUG] Total files uploaded: {files_uploaded}")



@ray.remote(runtime_env={"pip": ["dolma"]})
def dolma_dedup(input_dir, output_dir, attribute_name, min_length, min_words, bloom_filter_size, estimated_doc_count, false_positive_rate, processes):
    print(f"\n[DEBUG] Starting dolma_dedup function")
    print(f"[DEBUG] Input directory: {input_dir}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[DEBUG] Created temporary directory: {tmpdir}")
        try:
            copy_files_in(input_dir, tmpdir)
            do_dedup(tmpdir, attribute_name, min_length, min_words, bloom_filter_size, estimated_doc_count, false_positive_rate, processes)
            copy_files_out(tmpdir, output_dir, attribute_name)
        except Exception as e:
            print(f"[DEBUG] An error occurred: {e}")
    return "Deduplication process completed"

def main(config):
    ray.init()

    print("[DEBUG] Starting Dolma deduplication process...")
    input_dir = validate_gcp_path(config.input_dir, path_type="documents")
    output_dir = validate_gcp_path(config.output_dir, path_type="attributes")
    result = ray.get(dolma_dedup.remote(input_dir, output_dir, config.attribute_name, config.min_length, config.min_words, config.bloom_filter_size, config.estimated_doc_count, config.false_positive_rate, config.processes))
    print(f"[DEBUG] Result: {result}")

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