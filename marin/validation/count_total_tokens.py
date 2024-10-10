"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.validation.count_total_tokens --input_path gs://marin-data/filtered/dclm-fasttext-quality/fineweb/fw-v1.0/md/
"""

import argparse
import json

import fsspec
import ray

from marin.core.runtime import map_files_in_directory


@ray.remote
class TokenCounter:
    def __init__(self):
        self.total_tokens = 0

    def add(self, count: int):
        self.total_tokens += count

    def get_total(self) -> int:
        return self.total_tokens


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


@ray.remote
def process_file(input_filename: str, output_filename: str, token_counter: ray.actor.ActorHandle):
    file_tokens = count_tokens_in_file(input_filename)
    token_counter.add.remote(file_tokens)


def count_total_tokens(input_path: str) -> int:

    token_counter = TokenCounter.remote()

    responses = map_files_in_directory(
        process_file.remote,
        input_path,
        "**/*.jsonl.gz",
        "gs://marin-us-central2/scratch/chrisc/count-total-tokens/",  # random output_path, unused
        None,
        False,
        token_counter,
    )

    # Wait for all tasks to complete
    ray.get(responses)

    # Get the final count
    total_tokens = ray.get(token_counter.get_total.remote())

    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="Count total tokens in 'text' fields of jsonl.gz files.")
    parser.add_argument("--input_path", type=str, required=True, help="Input directory containing jsonl.gz files")

    args = parser.parse_args()

    total_tokens = count_total_tokens(args.input_path)
    print(f"Total tokens in 'text' fields: {total_tokens}")


if __name__ == "__main__":
    main()
