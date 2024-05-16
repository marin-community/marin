"""
Code to split the fasttext training file created using `create_dataset.py` into a train-val split

Usage:
nlprun -m jagupard36 -c 16 -r 40G 'python marin/processing/fasttext/data/train_test_split.py --input-file /nlp/scr/cychou/fasttext.txt --train-file /nlp/scr/cychou/fasttext.train --val-file /nlp/scr/cychou/fasttext.val --val-ratio 0.2'
"""

import argparse
import random

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
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
    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="The input file to split into train and val set")
    parser.add_argument("--train-file", type=str, help="The output train file")
    parser.add_argument("--val-file", type=str, help="The output val file")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="The size of the validation set")

    args = parser.parse_args()

    lines = read_file(args.input_file)
    shuffled_lines = shuffle_lines(lines)
    train_lines, val_lines = split_data(shuffled_lines, args.val_ratio)
    write_file(train_lines, args.train_file)
    write_file(val_lines, args.val_file)

            