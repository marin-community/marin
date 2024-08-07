"""
Code to split the fasttext training file created using `create_dataset.py` into a train-test split
The output is two files: a train file and test file where the test file is a random sample of the input file
up to the test-ratio proportion of the input file.

Usage:
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.quality.fasttext.data.train_test_split --input-file <input_file> --output-train-file <output_train_file> --output-test-file <output_test_file> --test-ratio <test_ratio>
"""

import argparse
import random

import fsspec
import ray


@ray.remote
def process_file(input_file: str, output_train_file: str, output_test_file: str, test_ratio: float):
    with fsspec.open(input_file, "rt", compression="gzip") as f_in:
        with fsspec.open(output_train_file, "wt") as f_out_train:
            with fsspec.open(output_test_file, "wt") as f_out_test:
                for line in f_in:
                    if random.random() < test_ratio:
                        f_out_test.write(line)
                    else:
                        f_out_train.write(line)


def main(input_file: str, output_train_file: str, output_test_file: str, test_ratio: float):
    ray.init()

    response = process_file.options(memory=16 * 1024 * 1024 * 1024).remote(
        input_file, output_train_file, output_test_file, test_ratio
    )

    try:
        ray.get(response)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="The input file path")
    parser.add_argument("--output-train-file", type=str, required=True, help="The output train file path")
    parser.add_argument("--output-test-file", type=str, required=True, help="The output test file path")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="The size of the test set")
    parser.add_argument("--seed", type=int, default=42, help="The random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    main(args.input_file, args.output_train_file, args.output_test_file, args.test_ratio)
