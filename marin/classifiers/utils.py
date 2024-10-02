"""
utils.py

Utility functions for building quality classifiers.
"""

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import fsspec
import numpy as np
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.processing.classification.types import DatasetFormat, Example
from marin.utils import fsspec_glob, fsspec_isdir


@cached_or_construct_output(success_suffix="SUCCESS")
def write_label_attribute(input_file_path: str, output_file_path: str, label: str) -> bool:
    """
    Creates an attribute "label" from input label for each document.

    Args:
        input_file_path (str): Path to the input JSONL file (gzip compressed).
        output_file_path (str): Path to the output attribute JSONL file (gzip compressed).
        label (str): Quality classifier label.

    Returns:
        bool: True if the process is successful.
    """
    with (
        fsspec.open(input_file_path, "rt", compression="gzip") as f_in,
        fsspec.open(output_file_path, "wt", compression="gzip") as f_out,
    ):
        for line in f_in:
            json_obj = json.loads(line)
            attributes = {"label": label}
            f_out.write(
                json.dumps({"id": json_obj["id"], "source": json_obj["source"], "attributes": attributes}) + "\n"
            )

    return True


def get_example_from_input_line(input_line: str, label: str, file_format: DatasetFormat) -> Example:
    """Converts a single line of input from a jsonl file or fasttext formatted file to a training example

    Args:
        input_line (str): A single line of input.
        label (str): The label for the example.
        file_format (DatasetFormat): The format of the input line (e.g. DOLMA_FORMATTED_JSONL)

    Returns:
        dict: An example with the format {"text": <text>, "label": <label>}.
    """
    import re

    text = None
    if file_format == DatasetFormat.DOLMA_FORMATTED_JSONL:
        data = json.loads(input_line)
        text = data["text"]
    elif file_format == DatasetFormat.FASTTEXT:
        line = input_line.strip()
        match = re.match(r"__label__(\S+)\s+(.*)", line, re.DOTALL)
        if match:
            text = match.group(2).strip()
    else:
        raise ValueError(f"File format not supported: {file_format}")

    if not text:
        logging.warning(f"Document {data.get('id', '')} has no text field.")
        return None
    else:
        return Example(text=text, label=label)


@cached_or_construct_output(success_suffix="SUCCESS")
def write_examples(
    input_file_path: str,
    output_file_path: str,
    sampling_rate: float,
    seed: int,
    label: str,
    file_format: DatasetFormat,
) -> bool:
    """
    Writes training examples to an output file.
    Only a fraction of the examples, determined by the sampling rate, are written to the output file (eg, to control size
    of training dataset and/or weight different domains).

    Args:
        input_file_path (str): Path to the input JSONL file (gzip compressed).
        output_file_path (str): Path to the output file (gzip compressed).
        attr_file_path (str): Path to the attribute JSONL file (gzip compressed).
        sampling_rate (float): Fraction of lines to be written to the output file.
        seed (int): Seed for random number generator to ensure reproducibility.
        get_label (Callable[[dict,dict], str]): Function to extract label from documents and attributes.

    Returns:
        bool: True if the process is successful.
    """
    rng = np.random.default_rng(seed=seed)
    with (
        fsspec.open(input_file_path, "rt", compression="infer") as f_in,
        fsspec.open(output_file_path, "wt", compression="gzip") as f_out,
    ):
        for input_line in f_in:
            if rng.random() > sampling_rate:
                continue

            example = get_example_from_input_line(input_line, label, file_format)
            if example is not None:
                f_out.write(json.dumps({"text": example.text, "label": example.label}) + "\n")
    return True


def merge_shards_and_split(
    shard_paths: list[str],
    train_path: str,
    val_path: str,
    val_frac: float,
    seed: int,
    format_example: Callable[[dict], str],
) -> bool:
    """
    Merges multiple shard files into training and validation datasets.

    Args:
        shard_paths (List[str]): List of paths to shard files.
        train_path (str): Path to the output training dataset file.
        val_path (str): Path to the output validation dataset file.
        val_frac (float): Fraction of data to be used for validation.
        seed (int): Seed for random number generator to ensure reproducibility.
        format_example (Callable[[dict],str]): Function to format example into correct training
                                               format (e.g., BERT, fastText, etc.).

    Returns:
        bool: True if the process is successful.
    """
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(train_path, "wt") as f_train, fsspec.open(val_path, "wt") as f_val:
        for shard_path in shard_paths:
            with fsspec.open(shard_path, "rt", compression="gzip") as f_in:
                for input_line in f_in:
                    data = json.loads(input_line)
                    output_line = format_example(data)

                    if rng.random() < val_frac:
                        f_val.write(output_line + "\n")
                    else:
                        f_train.write(output_line + "\n")

    return True


def create_label_attribute(input_doc_path: str, output_attr_path: str, label: str) -> bool:
    """
    Create attribute for quality classifier label.

    Args:
        input_doc_path (str): Path to documents (i.e., gs://$BUCKET/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://$BUCKET/attributes/.../<experiment>).
        label (str): Quality classifier label to write as attribute.

    Returns:
        bool: True if the process is successful.
    """

    # curry write_label so that we can pass it to map_files_in_directory
    @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
    def processing_func(input_file_path, output_file_path):
        return write_label_attribute(input_file_path, output_file_path, label)

    responses = map_files_in_directory(processing_func.remote, input_doc_path, "**/*.jsonl.gz", output_attr_path)
    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing {input_doc_path}: {e}")

    return True


def get_output_path_for_input_doc_path(output_path: str, input_doc_path: str) -> str:
    """Converts an input document path to an output dataset path

    Examples:
        [A] (output_path, /documents/hello_world_fw/) ->
            [B] /output_path/data/documents/hello_world_fw.jsonl.gz
        [A] (output_path, /documents/hello_world_fw/file.jsonl.gz) ->
            [B] /output_path/data/documents/hello_world_fw/file.jsonl.gz
        [A] (output_path, /documents/hello_world_fw/file.txt) ->
            [B] /output_path/data/documents/hello_world_fw/file.jsonl.gz
    """
    _, doc_fs_path = fsspec.core.url_to_fs(input_doc_path)
    path = Path(doc_fs_path)

    # Remove all files extensions from the input path and replace with .jsonl.gz
    while path.suffix:
        path = path.with_suffix("")
    path = str(path) + ".jsonl.gz"

    return os.path.join(output_path, "data", path)


@cached_or_construct_output(success_suffix="SUCCESS")
@ray.remote(memory=16 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)
def reservoir_sample_and_write_examples(
    doc_path: str,
    output_dataset_path: str,
    sampling_rate: int,
    seed: int,
    label: str,
    file_format: DatasetFormat,
):
    """Sample a fixed number of examples K from any dataset of size N where K < N

    We use the reservoir sampling algorithm to sample K examples from the dataset of size N where
    each row in the dataset has a uniform probability of 1/N of being sampled.
    The dataset can be sharded across multiple files in the directory which we glob together and
    sample from.

    Args:
        doc_path (str): Path to the input dataset which can be a directory or a file.
        output_dataset_path (str): Path to the output dataset.
        sampling_rate (int): Number of examples to sample from the dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
        label (str): Label for the dataset.
        file_format (DatasetFormat): Format of the dataset.

    Returns:
        bool: True if the process is successful.
    """

    rng = np.random.default_rng(seed=seed)
    files = fsspec_glob(os.path.join(doc_path, "**/*.jsonl.gz"))
    reservoir = []
    reservoir_size = sampling_rate

    for input_file in files:
        with fsspec.open(input_file, "rt", compression="gzip") as f_in:
            for line in f_in:
                if len(reservoir) < reservoir_size:
                    reservoir.append(line)
                else:
                    reservoir[rng.integers(reservoir_size)] = line

    with fsspec.open(output_dataset_path, "wt", compression="gzip") as f_out:
        for line in reservoir:
            example = get_example_from_input_line(line, label, file_format)
            if example is not None:
                f_out.write(json.dumps({"text": example.text, "label": example.label}) + "\n")

    return True


def attributes_to_dataset(
    output_path: str,
    doc_path: str,
    sampling_rate: float,
    seed: int,
    label: str,
    file_format: DatasetFormat,
) -> bool:
    """
    Converts documents and attributes to quality classifier training data (text,label) pairs.

    Args:
        output_path (str): Path for output data (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        doc_path (str): Path to input documents (i.e., gs://$BUCKET/documents/reddit/v0/<doc_experiment>).
        sampling_rate (float): Fraction of documents from the dataset to add to fastText training dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
        label (str): Label for the dataset.
        file_format (DatasetFormat): Format of the dataset.

    Returns:
        bool: True if the process is successful.
    """
    logger = logging.getLogger("ray")

    # curry write_fasttext_lines so that we can pass it to map_files_in_directory
    @ray.remote(memory=1 * 1024 * 1024 * 1024, runtime_env={"pip": ["s3fs"]}, num_cpus=1)  # 1 GB
    def processing_func(input_file_path: str, output_file_path: str) -> bool:
        return write_examples(input_file_path, output_file_path, sampling_rate, seed, label, file_format)

    output_dataset_path = get_output_path_for_input_doc_path(output_path, doc_path)

    if fsspec_isdir(doc_path):
        if sampling_rate <= 1.0:
            responses = map_files_in_directory(processing_func.remote, doc_path, "**/*.jsonl.gz", output_dataset_path)
        else:
            responses = reservoir_sample_and_write_examples.remote(
                doc_path, output_dataset_path, sampling_rate, seed, label, file_format
            )
    else:
        responses = processing_func.remote(doc_path, output_dataset_path)

    try:
        ray.get(responses)
    except Exception as e:
        logger.exception(f"Error processing {doc_path}: {e}")
        raise

    return True


def shuffle(input_file_path: str, output_file_path: str, seed: int) -> bool:
    """
    Shuffles the lines of a file.

    Args:
        input_file_path (str): Path to the input file.
        output_file_path (str): Path to the output file.
        seed (int): Seed for random number generator to ensure reproducibility.

    Returns:
        bool: True if the process is successful.
    """
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(input_file_path, "rt", compression="infer") as f_in:
        lines = f_in.readlines()
    rng.shuffle(lines)
    with fsspec.open(output_file_path, "wt", compression="infer") as f_out:
        f_out.writelines(lines)

    return True
