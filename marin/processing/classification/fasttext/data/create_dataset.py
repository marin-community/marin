"""
Code to load in Dolma formatted data and create a single file with data in fasttext format:
"__label__{label_name} {text}" for each row in the input files, where the label can be 
"hq" or "lq" depending on whether it is a high quality data source or low quality data source.
The final file format is a single text file with one line per label-text pair.

Usage:
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.fasttext.data.create_dataset --high-quality-files <input_files> --low-quality-files <input_files> --output-file <output_file>
"""

import argparse
import json
from typing import List
import time

import fsspec
import ray
from marin.utils import fsspec_glob, fsspec_isdir


@ray.remote
class QueueActor:
    def __init__(self):
        self.queue = []
        self.finished = False

    def add(self, items):
        self.queue.extend(items)

    def get(self):
        if self.queue:
            return self.queue.pop(0)
        return None

    def is_empty(self):
        return len(self.queue) == 0

    def set_finished(self):
        self.finished = True

    def is_finished(self):
        return self.finished

@ray.remote
def process_preprocessed_file(preprocessed_path: str) -> List[str]:
    """
    Process a file that is already in fasttext format. Assumes that the text is uncompressed.
    Used in the case such as combining DCLM's O-H 2.5 fasttext files with the rest of the data.
    """

    labeled_lines = []
    with fsspec.open(preprocessed_path, "rt") as f_in:
        for line in f_in:
            text = line.strip()
            labeled_lines.append(text)

    return labeled_lines

@ray.remote
def process_file(json_path: str, label: str) -> List[str]:
    """
    Process a file that is in Dolma format.
    """

    labeled_lines = []
    with fsspec.open(json_path, "rt", compression="gzip") as f_in:
        for line in f_in:
            data = json.loads(line)
            text = data.get("text", "")
            text = text.replace("\n", " ")
            if text:
                labeled_lines.append(f"__label__{label} {text}")
    return labeled_lines


@ray.remote
def write_to_file(output_file: str, queue_actor):
    with fsspec.open(output_file, "wt", compression="gzip") as f:
        while True:
            item = ray.get(queue_actor.get.remote())
            if item is not None:
                f.write(item + "\n")
            elif ray.get(queue_actor.is_finished.remote()) and ray.get(queue_actor.is_empty.remote()):
                break
            else:
                time.sleep(0.1)  # Short sleep to avoid busy waiting


# Given a list of filepaths, return a list of files to process
# If the filepath is a directory, recursively search for .jsonl.gz files
def get_files(filepaths: List[str]):
    files = []
    for filepath in filepaths:
        if fsspec_isdir(filepath):
            files.extend(fsspec_glob(filepath, "**/*.jsonl.gz"))
        else:
            files.append(filepath)
    return files


def main(high_quality_filepaths: List[str], low_quality_filepaths: List[str], preprocessed_filepaths: List[str], output_file: str):
    ray.init()

    # We set a large memory limit for the queue actor to avoid memory issues when
    # processing large files
    queue_actor = QueueActor.options(memory=100 * 1024 * 1024 * 1024).remote()

    # Start the writer process
    writer = write_to_file.remote(output_file, queue_actor)

    # Process files and add results to the queue
    tasks = []
    high_quality_files = get_files(high_quality_filepaths)
    low_quality_files = get_files(low_quality_filepaths)
    preprocessed_files = get_files(preprocessed_filepaths)
    for filename in high_quality_files:
        label = "hq"
        task = process_file.remote(filename, label)
        tasks.append(queue_actor.add.remote(task))

    for filename in low_quality_files:
        label = "lq"
        task = process_file.remote(filename, label)
        tasks.append(queue_actor.add.remote(task))
    
    for filename in preprocessed_files:
        task = process_preprocessed_file.remote(filename)
        tasks.append(queue_actor.add.remote(task))

    # Wait for all tasks to complete
    ray.get(tasks)

    # Signal that all files have been processed
    ray.get(queue_actor.set_finished.remote())

    # Wait for the writer to finish
    ray.get(writer)

    print(f"[*] Training file created at: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-quality-files", nargs="+", type=str, required=True)
    parser.add_argument("--low-quality-files", nargs="+", type=str, required=True)
    parser.add_argument("--preprocessed-files", nargs="+", type=str, required=False)
    parser.add_argument("--output-file", type=str, required=True)

    args = parser.parse_args()

    main(args.high_quality_files, args.low_quality_files, args.output_file)
