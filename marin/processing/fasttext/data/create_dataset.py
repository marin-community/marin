"""
Code to load and preprocess data for fasttext training

Usage:
ray job submit --working-dir . --no-wait -- python marin/processing/fasttext/data/create_dataset.py --max-num-samples <max-num-samples>
"""

import argparse
import json

import fsspec
import ray

POSITIVE_JSON_PATHS = [
    "gs://marin-data/raw/dolma/dolma-v1.7/wiki-0000.json.gz",
    "gs://marin-data/raw/dolma/dolma-v1.7/wiki-0001.json.gz"
]

NEGATIVE_JSON_PATHS = [
    "gs://marin-data/raw/dolma/dolma-v1.7/c4-0000.json.gz",
]

OUTPUT_FILE_PATH = "gs://marin-data/scratch/chrisc/dataset.txt.gz"

@ray.remote
def process_file(json_path, label, max_num_samples=None):
    labeled_lines = []
    with fsspec.open(json_path, 'rt', compression="gzip") as f_in:
        for i, line in enumerate(f_in):
            if max_num_samples and i >= max_num_samples:
                break

            data = json.loads(line)
            text = data.get("text", "")
            text = text.replace("\n", " ")
            if text:
                labeled_lines.append(f"__label__{label} {text}")

    # with fsspec.open(f"{json_path}.success", 'w') as f_out:
    #     f_out.write("SUCCESS")

    return labeled_lines

@ray.remote
def write_lines(output_file, lines):
    with fsspec.open(output_file, 'wt', compression="gzip") as f:
        for line in lines:
            f.write(line + "\n")

def process_files(input_files, output_file, max_num_samples=None):
    # Remove the output file if it exists, to start fresh
    if fs.exists(output_file):
        fs.rm(output_file)

    labeled_lines = ray.get([process_file.remote(json_path, label, max_num_samples) for json_path, label in input_files])
    flattened_labeled_lines = [line for sublist in labeled_lines for line in sublist]
    
    print(len(flattened_labeled_lines))
    # Write lines in parallel
    ray.get(write_lines.remote(output_file, flattened_labeled_lines))
    # chunk_size = 1024  # Adjust chunk size as needed
    # chunks = [flattened_labeled_lines[i:i + chunk_size] for i in range(0, len(flattened_labeled_lines), chunk_size)]
    # ray.get([write_lines.remote(output_file, chunk) for chunk in chunks])

def main(max_num_samples):
    ray.init()
    print(f"[*] Cluster Statistics :: {len(ray.nodes())} nodes w/ {ray.cluster_resources().get('CPU', 0)} total CPUs")

    input_files = []
    for filename in POSITIVE_JSON_PATHS:
        input_files.append((filename, "positive"))
    
    for filename in NEGATIVE_JSON_PATHS:
        input_files.append((filename, "negative"))

    process_files(input_files, OUTPUT_FILE_PATH, max_num_samples)

    print(f"[*] Training file created at: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-samples", type=int, default=None, help="The maximum number of samples to process")

    args = parser.parse_args()

    fs = fsspec.filesystem('gcs')

    main(args.max_num_samples)
