"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.validation.count_total_tokens --input_path gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/
"""

import argparse
import os
import time

import pandas as pd
import ray

from marin.utils import fsspec_glob

MAX_TASKS_IN_FLIGHT = 1000
NUM_DOWNLOAD_RETRIES = 5


def count_tokens_in_file(filename: str, tokenizer_name: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = None
    for _ in range(NUM_DOWNLOAD_RETRIES):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            break
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            time.sleep(1)

    if tokenizer is None:
        raise RuntimeError(f"Failed to load tokenizer {tokenizer_name} after {NUM_DOWNLOAD_RETRIES} retries")

    total_tokens = 0

    # Determine file format and read accordingly
    if filename.endswith(".parquet"):
        df = pd.read_parquet(filename)
    elif filename.endswith(".jsonl.zst"):
        df = pd.read_json(filename, lines=True, compression="zstd")
    elif filename.endswith(".jsonl.gz"):
        df = pd.read_json(filename, lines=True, compression="gzip")
    elif filename.endswith(".jsonl"):
        df = pd.read_json(filename, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    # Count tokens in 'text' column if it exists
    if "text" in df.columns:
        for text in df["text"].dropna():
            total_tokens += len(tokenizer.encode(str(text)))

    return total_tokens


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def process_file(input_filename: str, tokenizer_name: str):
    file_tokens = count_tokens_in_file(input_filename, tokenizer_name)
    return file_tokens


def count_total_tokens(input_path: str, tokenizer_name: str, filetype: str) -> int:

    responses = []
    tokens = 0
    input_paths = fsspec_glob(os.path.join(input_path, f"**/*.{filetype}"))
    for input_path in input_paths:
        while len(responses) >= MAX_TASKS_IN_FLIGHT:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            for response in ray.get(ready_refs):
                tokens += response

        result_ref = process_file.remote(input_path, tokenizer_name)
        responses.append(result_ref)

    # Wait for all tasks to complete
    for response in ray.get(responses):
        tokens += response

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Count total tokens in 'text' fields of supported file formats.")
    parser.add_argument("--input_path", type=str, required=True, help="Input directory containing data files")
    parser.add_argument(
        "--tokenizer_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the tokenizer to use"
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default="jsonl.gz",
        help="Filetype of the input files (jsonl.gz, jsonl.zst, parquet, jsonl)",
    )

    args = parser.parse_args()

    total_tokens = count_total_tokens(args.input_path, args.tokenizer_name, args.filetype)
    print(f"Total tokens in 'text' fields: {total_tokens}")


if __name__ == "__main__":
    main()
