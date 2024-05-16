"""
Code to load and preprocess data for fasttext training

Usage:
nlprun -m jagupard36 -c 16 -r 40G 'python marin/processing/fasttext/data/create_dataset.py --positive-json-paths /nlp/scr/cychou/wiki-0000.json --negative-json-paths /nlp/scr/cychou/c4-0000.json --max-num-samples 5000 --output-file-path /nlp/scr/cychou/fasttext.txt'
"""

import argparse
import json

def process_file(json_path, label, max_num_samples=None):
    labeled_lines = []
    with open(json_path, 'r') as file:
        for i, line in enumerate(file):
            if max_num_samples and i >= max_num_samples:
                break

            data = json.loads(line)
            text = data.get("text", "")
            text = text.replace("\n", " ")
            if text:
                labeled_lines.append(f"__label__{label} {text}")
    return labeled_lines

def process_files(input_files, output_file, label, max_num_samples=None):
    for json_path in input_files:
        labeled_lines = process_file(json_path, label, max_num_samples)
        with open(output_file, 'a') as output_file:
            for line in labeled_lines:
                output_file.write(line + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-json-paths", nargs="+", help="The paths to the positive JSON files")
    parser.add_argument("--negative-json-paths", nargs="+", help="The paths to the negative JSON files")
    parser.add_argument("--output-file-path", type=str, help="The output file path")
    parser.add_argument("--max-num-samples", type=int, default=None, help="The maximum number of samples to process")

    args = parser.parse_args()

    process_files(args.positive_json_paths, args.output_file_path, "positive", args.max_num_samples)
    process_files(args.negative_json_paths, args.output_file_path, "negative", args.max_num_samples)

    # Combine and write to the output file
    print(f"Training file created at: {args.output_file_path}")