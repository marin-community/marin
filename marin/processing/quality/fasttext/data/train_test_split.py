"""
Code to split the fasttext training file created using `create_dataset.py` into a train-val split

Usage:
python marin/processing/fasttext/data/train_test_split.py
"""

import argparse
import random

import fsspec

INPUT_FILE = "gs://marin-data/scratch/chrisc/dataset.txt.gz"
OUTPUT_TRAIN_FILE = "gs://marin-data/scratch/chrisc/fasttext_train.txt.gz"
OUTPUT_VAL_FILE = "gs://marin-data/scratch/chrisc/fasttext_test.txt.gz"


def read_file(file_path):
    with fsspec.open(file_path, "rt", compression="gzip") as f_in:
        lines = f_in.readlines()
    return lines


def shuffle_lines(lines):
    random.shuffle(lines)
    return lines


def split_data(lines, val_ratio=0.2):
    split_index = int(len(lines) * (1 - val_ratio))
    train_lines = lines[:split_index]
    val_lines = lines[split_index:]
    return train_lines, val_lines


def write_file(lines, file_path):
    with fsspec.open(file_path, "wt", compression="gzip") as f_out:
        for line in lines:
            f_out.write(line)


def main(input_file, output_train_file, output_val_file, val_ratio):
    lines = read_file(input_file)
    shuffled_lines = shuffle_lines(lines)
    train_lines, val_lines = split_data(shuffled_lines, val_ratio)
    write_file(train_lines, output_train_file)
    write_file(val_lines, output_val_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2, help="The size of the validation set")

    args = parser.parse_args()

    main(INPUT_FILE, OUTPUT_TRAIN_FILE, OUTPUT_VAL_FILE, args.val_ratio)
