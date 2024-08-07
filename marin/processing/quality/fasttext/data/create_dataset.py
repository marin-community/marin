"""
Code to load and preprocess data for fasttext training

Usage:
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.quality.fasttext.data.create_dataset --high-quality-files <input_files> --low-quality-files <input_files> --output-file <output_file>
"""

import argparse
import json
from typing import List
import time

import fsspec
import ray


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
def process_file(json_path: str, label: str) -> List[str]:
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


def main(high_quality_files: List[str], low_quality_files: List[str], output_file: str):
    ray.init()

    # We set a large memory limit for the queue actor to avoid memory issues when
    # processing large files
    queue_actor = QueueActor.options(memory=100 * 1024 * 1024 * 1024).remote()

    # Start the writer process
    writer = write_to_file.remote(output_file, queue_actor)

    # Process files and add results to the queue
    tasks = []
    for filename in high_quality_files:
        label = "hq"
        task = process_file.remote(filename, label)
        tasks.append(queue_actor.add.remote(task))

    for filename in low_quality_files:
        label = "lq"
        task = process_file.remote(filename, label)
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
    parser.add_argument("--output-file", type=str, required=True)

    args = parser.parse_args()

    main(args.high_quality_files, args.low_quality_files, args.output_file)
