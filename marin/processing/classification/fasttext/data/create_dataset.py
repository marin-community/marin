"""
Code to load in Dolma formatted data and create a single file with data in fasttext format:
"__label__{label_name} {text}" for each row in the input files, where the label can be
"hq" or "lq" depending on whether it is a high quality data source or low quality data source.
The final file format is a single text file with one line per label-text pair.

Usage:
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.fasttext.data.create_dataset --config <config_file>
"""

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import fsspec
import ray
import tqdm
import yaml

from marin.processing.classification.fasttext.data.config import DatasetConfig
from marin.utils import fsspec_glob, fsspec_isdir


@ray.remote
class QueueActor:
    """
    A process that adds items to a queue and allows for a writer process to read from the queue
    to write to a final fasttext file in google cloud storage.
    """

    def __init__(self):
        self.queue: List[Dict[str, List[str]]] = []
        self.finished: bool = False

    def add(self, item: Dict[str, List[str]]):
        self.queue.append(item)

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


def get_files(filepath: str):
    """
    Given a list of filepaths, return a list of files to process
    If the filepath is a directory, recursively search for .jsonl.gz files
    """

    files = []
    if fsspec_isdir(filepath):
        files.extend(fsspec_glob(os.path.join(filepath, "**/*.jsonl.gz")))
    else:
        files.append(filepath)

    return files


@ray.remote
def process_file(config: DatasetConfig) -> List[str]:
    """
    Process a file that is in Dolma format or is a preprocessed fasttext-formatted text file.

    We support random sampling of files. If the filepath is a directory, then
    we search for all .jsonl.gz files and allow a random sampling of a certain chunk size
    per file. The number of examples are randomly sampled without replacement from the
    file.

    Output:
    A dictionary with the filepath as the key and the list of lines in fasttext format as the value.
    """

    filepaths = get_files(config.filepath)

    chunk_size: Optional[int] = None
    total_offset: int = 0
    if config.max_num_samples:
        chunk_size = config.max_num_samples // len(filepaths)

    filepaths_to_labelled_lines: Dict[str, List[str]] = {}
    for i, filepath in tqdm.tqdm(enumerate(filepaths), total=len(filepaths)):
        lines_per_file = []

        # If the file is in a fasttext format, we assume that the file is not compressed.
        if config.preprocessed:
            with fsspec.open(filepath, "rt") as f_in:
                for line in f_in:
                    text = line.strip()
                    text = text.replace("\n", " ")
                    if text:
                        lines_per_file.append(f"__label__{config.label} {text}")
        else:  # The file is in dolma format.
            with fsspec.open(filepath, "rt", compression="gzip") as f_in:
                for line in f_in:
                    data = json.loads(line)
                    text = data.get("text", "")
                    text = text.replace("\n", " ")
                    if text:
                        lines_per_file.append(f"__label__{config.label} {text}")

        # truncate to not exceed the number of lines in the file
        if config.max_num_samples:
            if i == len(filepaths) - 1:
                chunk_size = config.max_num_samples - total_offset
            maybe_truncated_chunk_size = min(chunk_size, len(lines_per_file))
            filepaths_to_labelled_lines[filepath] = random.sample(lines_per_file, maybe_truncated_chunk_size)
            total_offset += maybe_truncated_chunk_size
        else:
            filepaths_to_labelled_lines[filepath] = lines_per_file
            total_offset += len(lines_per_file)

    return filepaths_to_labelled_lines


@ray.remote
def write_to_file(output_file: str, queue_actor):
    """
    Writes the output file and metadata file.

    The metadata file contains the filepath and the number of examples sampled from that file.
    This allows us to track the provenance of the dataset composition of the file.
    """

    metadata = {}
    with fsspec.open(output_file, "wt", compression="gzip") as f:
        with fsspec.open(f"{output_file}.metadata", "wt") as f_metadata:
            while True:
                item: Dict[str, List[str]] = ray.get(queue_actor.get.remote())
                if item is not None:
                    for filepath, lines in item.items():
                        for line in lines:
                            f.write(line + "\n")

                        metadata[filepath] = len(lines)
                elif ray.get(queue_actor.is_finished.remote()) and ray.get(queue_actor.is_empty.remote()):
                    f_metadata.write(json.dumps(metadata))
                    break
                else:
                    time.sleep(0.1)  # Short sleep to avoid busy waiting


def main(config: Dict[str, Any]):
    ray.init()

    # We set a large memory limit for the queue actor to avoid memory issues when
    # processing large files
    queue_actor = QueueActor.options(memory=100 * 1024 * 1024 * 1024).remote()

    # Start the writer process
    writer = write_to_file.remote(config["output_file"], queue_actor)

    dataset_configs: List[DatasetConfig] = [DatasetConfig(**config) for config in config["dataset_configs"]]

    # Process files and add results to the queue
    tasks = []
    for dataset_config in dataset_configs:
        task = process_file.remote(dataset_config)
        tasks.append(queue_actor.add.remote(task))

    # Wait for all tasks to complete
    ray.get(tasks)

    # Signal that all files have been processed
    ray.get(queue_actor.set_finished.remote())

    # Wait for the writer to finish
    ray.get(writer)

    print(f"[*] Training file created at: {config['output_file']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
