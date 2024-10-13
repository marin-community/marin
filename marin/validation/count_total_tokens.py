"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.validation.count_total_tokens --input_path gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/
"""

import argparse
import json
import os

import fsspec
import ray

from marin.utils import fsspec_glob

MAX_TASKS_IN_FLIGHT = 1000


def count_tokens_in_file(filename: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    total_tokens = 0
    with fsspec.open(filename, "rt", compression="gzip") as f:
        for line in f:
            data = json.loads(line)
            if "text" in data:
                total_tokens += len(tokenizer.encode(data["text"]))
    return total_tokens


@ray.remote(memory=500 * 1024 * 1024)
def process_file(input_filename: str):
    file_tokens = count_tokens_in_file(input_filename)
    return file_tokens


def count_total_tokens(input_path: str) -> int:

    responses = []
    tokens = 0
    input_paths = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))
    for input_path in input_paths:
        while len(responses) >= MAX_TASKS_IN_FLIGHT:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            for response in ray.get(ready_refs):
                tokens += response

        result_ref = process_file.remote(input_path)
        responses.append(result_ref)

    # Wait for all tasks to complete
    for response in ray.get(responses):
        tokens += response

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Count total tokens in 'text' fields of jsonl.gz files.")
    parser.add_argument("--input_path", type=str, required=True, help="Input directory containing jsonl.gz files")

    args = parser.parse_args()

    total_tokens = count_total_tokens(args.input_path)
    print(f"Total tokens in 'text' fields: {total_tokens}")


if __name__ == "__main__":
    main()
