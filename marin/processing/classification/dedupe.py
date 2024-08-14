import argparse
import ray
import time
import psutil
import multiprocessing
import socket
import os
from google.cloud import storage
from tqdm import tqdm
import glob
import shutil
import subprocess
from marin.processing.classification.config.inference_config import InferenceConfig

@ray.remote(runtime_env={"pip": ["dolma"]})
def read_gcp_and_write_local(input_dir):
    hostname = socket.gethostname()
    print(f"\nRunning write/copy command on {hostname}\n")
    client = storage.Client()
    bucket_name = input_dir.split('/')[2]
    prefix = '/'.join(input_dir.split('/')[3:])
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    local_base_dir = '/tmp/gcp_files/documents'
    os.makedirs(local_base_dir, exist_ok=True)
    for blob in tqdm(blobs, desc="Downloading files"):
        print(f"\nBlob name: {blob.name}")
        if blob.name.endswith('.jsonl.gz'):
            local_file_path = os.path.join(local_base_dir, blob.name.split(prefix)[1].lstrip('/'))
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            print(f"\n Downloading {blob.name} to {local_file_path}", flush=True)
            blob.download_to_filename(local_file_path)

    dedupe_dolma()
    write_dedupe_attributes(local_base_dir, input_dir)
    return local_base_dir

# @ray.remote(runtime_env={"pip": ["dolma"]})
def dedupe_dolma():
    hostname = socket.gethostname()
    print(f"\nRunning Dolma dedupe command on {hostname}\n")
    command = [
        "RUST_BACKTRACE=full",
        "dolma", "dedupe",
        "--documents", "/tmp/gcp_files/documents/**/*.jsonl.gz",
        "--dedupe.documents.attribute_name", "duplicate_documents",
        "--dedupe.documents.key", "$.id",
        "--dedupe.skip_empty",
        "--bloom_filter.file", "/tmp/deduper_bloom_filter_test.bin",
        "--no-bloom_filter.read_only",
        "--bloom_filter.estimated_doc_count", "1000000",
        "--bloom_filter.desired_false_positive_rate", "0.00001",
        "--processes", "1"
    ]
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
    print(f"\nDolma dedupe command completed on {hostname}\n")
    
    # Rename temporary files
    attr_dir = "/tmp/gcp_files/attributes/duplicate_documents"
    for root, _, files in os.walk(attr_dir):
        print(f"\nRenaming temporary files in {root}")
        print(f"Files: {files}")
        for file in files:
            if file.endswith('.jsonl.gz.tmp'):
                old_path = os.path.join(root, file)
                new_path = old_path[:-4]  # Remove '.tmp'
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} to {new_path}")
    
    print_system_info()
    return process.returncode

# @ray.remote
def write_dedupe_attributes(local_base_dir, gcp_documents_dir):
    client = storage.Client()
    bucket_name = gcp_documents_dir.split('/')[2]
    documents_prefix = '/'.join(gcp_documents_dir.split('/')[3:])
    attributes_prefix = 'attributes/' + '/'.join(documents_prefix.split('/')[1:])
    bucket = client.get_bucket(bucket_name)

    # Update the local attribute directory path
    local_attribute_dir = os.path.join(local_base_dir, 'attributes', 'duplicate_documents')
    print(f"\nLocal attribute directory: {local_attribute_dir}")

    # Recursively find all .jsonl.gz files
    attribute_files = []
    for root, dirs, files in os.walk(local_attribute_dir):
        for file in files:
            if file.endswith('.jsonl.gz'):
                attribute_files.append(os.path.join(root, file))

    print(f"\nFound {len(attribute_files)} attribute files")

    for local_file_path in attribute_files:
        print(f"Processing file: {local_file_path}")
        
        # Construct the GCP path
        relative_path = os.path.relpath(local_file_path, local_attribute_dir)
        gcp_file_path = os.path.join(attributes_prefix, 'duplicates', relative_path)

        # Ensure we're not uploading to the redundant path
        if 'attributes/duplicate_documents' not in gcp_file_path:
            blob = bucket.blob(gcp_file_path)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to {gcp_file_path}")
        else:
            print(f"Skipping redundant upload for {local_file_path}")

    print("Attribute files upload completed")

    # Print all files in local attribute directory for verification
    print("\nAll files in local attribute directory:")
    for root, dirs, files in os.walk(local_attribute_dir):
        for file in files:
            print(os.path.join(root, file))
@ray.remote
def print_file_system_info(local_base_dir):
    print("\nLocal file system structure:")
    for root, dirs, files in os.walk(local_base_dir):
        level = root.replace(local_base_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
    print_system_info()

def print_system_info():
    print(f"\nCurrent time: {time.ctime()}")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Number of CPUs: {multiprocessing.cpu_count()}")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"Disk Usage: {psutil.disk_usage('/').percent}%")

def main(config):
    print("Starting file download...")
    local_base_dir = ray.get(read_gcp_and_write_local.remote(config.input_dir))
    print(f"Files downloaded to: {local_base_dir}")
    print("Starting DOLMA dedupe work...")
    ray.get(busy_work.remote())
    print("Printing file system info...")
    ray.get(print_file_system_info.remote(local_base_dir))
    print("Writing dedupe attributes to GCP...")
    ray.get(write_dedupe_attributes.remote('/tmp/gcp_files', config.input_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run resource-intensive work on a single Ray worker")
    parser.add_argument('--input_dir', type=str, required=True, help='GCP input directory path')
    args = parser.parse_args()
    config = argparse.Namespace(input_dir=args.input_dir)
    ray.init()
    main(config)