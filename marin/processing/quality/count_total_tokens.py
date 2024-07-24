"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.quality.count_total_tokens --input_dir gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/
"""

import argparse
import json
import gzip
import fsspec
import ray
from marin.core.runtime import map_files_in_directory

def count_bytes_in_file(filename: str) -> int:
    total_bytes = 0
    with fsspec.open(filename, "rt", compression="gzip") as f:
        for line in f:
            data = json.loads(line)
            if "text" in data:
                total_bytes += len(data["text"].encode('utf-8'))
    return total_bytes

@ray.remote
class ByteCounter:
    def __init__(self):
        self.total_bytes = 0

    def add(self, count: int):
        self.total_bytes += count

    def get_total(self) -> int:
        return self.total_bytes

@ray.remote
def process_file(input_filename: str, output_filename: str, byte_counter: ray.actor.ActorHandle):
    file_bytes = count_bytes_in_file(input_filename)
    byte_counter.add.remote(file_bytes)

def count_total_bytes(input_dir: str) -> int:
    ray.init()

    byte_counter = ByteCounter.remote()

    responses = map_files_in_directory(
        process_file.remote,
        input_dir,
        "**/*.jsonl.gz",
        "gs://marin-data/scratch/chrisc/count-total-tokens/", # random output_dir, unused
        None,
        byte_counter
    )

    # Wait for all tasks to complete
    ray.get(responses)

    # Get the final count
    total_bytes = ray.get(byte_counter.get_total.remote())

    return total_bytes

def main():
    parser = argparse.ArgumentParser(description="Count total bytes in 'text' fields of jsonl.gz files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing jsonl.gz files")

    args = parser.parse_args()

    total_bytes = count_total_bytes(args.input_dir)
    print(f"Total bytes in 'text' fields: {total_bytes}")

if __name__ == "__main__":
    main()
