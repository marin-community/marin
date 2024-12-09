"""
utils.py

Utility functions for building quality classifiers.
"""

import json
import logging
import os
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass

import fsspec
import numpy as np
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.utils import fsspec_glob, fsspec_rm, rebase_file_path

logger = logging.getLogger("ray")


def label_documents(
    input_doc_path: str,
    output_attr_path: str,
    attribute_func: Callable[[dict, list[dict] | None], dict],
    input_attr_paths: list[str] | None = None,
) -> None:
    """
    Create a new attribute by applying attribute_func to each document and its (optional) associated attributes.

    Args:
        input_doc_path (str): Path to documents (i.e., gs://$BUCKET/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://$BUCKET/attributes/...).
        attribute_func (Callable[[dict],dict]): Function to construct attributes from documents.
        input_attr_paths (list[str]): Path to attributes needed to determine new attribute.
    """

    # curry write_label so that we can pass it to map_files_in_directory
    @ray.remote(memory=1 * 1024 * 1024 * 1024, num_cpus=1)  # 1 GB
    def processing_func(input_file_path, output_file_path):
        attr_file_paths = (
            [rebase_file_path(input_doc_path, input_file_path, input_attr_path) for input_attr_path in input_attr_paths]
            if input_attr_paths is not None
            else []
        )
        return label_documents_shard(input_file_path, output_file_path, attribute_func, attr_file_paths)

    responses = map_files_in_directory(processing_func.remote, input_doc_path, "**/*.jsonl.gz", output_attr_path)
    try:
        ray.get(responses)
    except Exception as e:
        logger.exception(f"Error processing {input_doc_path}: {e}")
        raise


@cached_or_construct_output(success_suffix="SUCCESS")
def label_documents_shard(
    input_doc_path: str,
    output_file_path: str,
    attribute_func: Callable[[dict, list[dict] | None], dict],
    input_attr_paths: list[str] | None = None,
) -> None:
    """
    Writes new attribute file by applying attribute_func.

    Args:
        input_file_path (str): Path to the input JSONL file in Dolma format (gzip compressed).
        output_file_path (str): Path to the output attribute JSONL file (gzip compressed).
        attribute_func (Callable[[dict,list[dict] | None],dict]): Function to construct attributes from documents.
        input_attr_paths (list[str] | None): Path to attributes needed to determine new attribute.
    """
    with ExitStack() as stack:
        f_attrs = (
            [stack.enter_context(fsspec.open(attr_file, "rt")) for attr_file in input_attr_paths]
            if input_attr_paths is not None
            else []
        )
        with (
            fsspec.open(input_doc_path, "rt", compression="gzip") as f_doc,
            fsspec.open(output_file_path, "wt", compression="gzip") as f_out,
        ):
            for lines in zip(f_doc, *f_attrs, strict=False):
                doc_obj = json.loads(lines[0])
                attr_objs = [json.loads(line) for line in lines[1:]]
                attributes = attribute_func(doc_obj, attr_objs)
                f_out.write(
                    json.dumps({"id": doc_obj["id"], "source": doc_obj["source"], "attributes": attributes}) + "\n"
                )


def attribute_to_dataset(
    output_path: str,
    doc_path: str,
    attr_path: str,
    seed: int,
    get_label: Callable[[dict, dict], str] = lambda data, attribs: attribs["attributes"]["label"],
    sampling_rate: float = 1.0,
    max_sample_size: int | None = None,
) -> None:
    """
    Converts documents and specified attribute to quality classifier training data (text,label) pairs.

    Args:
        output_path (str): Path for output data (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        doc_path (str): Path to input documents (i.e., gs://$BUCKET/documents/reddit/v0/<doc_experiment>).
        attr_path (str): Path to input attributes (i.e., gs://$BUCKET/attributes/reddit/v0/<attr_experiment>).
        seed (int): Seed for random number generator to ensure reproducibility.
        get_label (Callable[[dict,dict], str]): Function to extract label from documents and attributes.
                                                Defaults to get_label.
        sampling_rate (float): Fraction of documents from the dataset to add to quality classifier training dataset.
        max_sample_size (Optional[int]): Maximum number of examples to include in the quality classifier
                                         training dataset. Defaults to None.
    """

    # curry write_fasttext_lines so that we can pass it to map_files_in_directory
    @ray.remote(memory=1 * 1024 * 1024 * 1024, num_cpus=1)  # 1 GB
    def processing_func(input_file_path: str, output_file_path: str) -> bool:
        attr_file_path = rebase_file_path(doc_path, input_file_path, attr_path)
        return attribute_to_dataset_shard(
            input_file_path, output_file_path, attr_file_path, sampling_rate, seed, get_label
        )

    _, doc_fs_path = fsspec.core.url_to_fs(doc_path)
    output_dataset_path = os.path.join(output_path, "data", doc_fs_path.lstrip("/"), "data.jsonl.gz")
    shard_path = os.path.join(output_path, "shards", doc_fs_path.lstrip("/"))

    responses = map_files_in_directory(processing_func.remote, doc_path, "**/*.jsonl.gz", shard_path)
    try:
        ray.get(responses)
    except Exception as e:
        logger.exception(f"Error processing {doc_path}: {e}")
        raise

    shard_paths = fsspec_glob(os.path.join(shard_path, "**/*.jsonl.gz"))
    if max_sample_size is None:
        merge_dataset_shards(shard_paths, output_dataset_path)
    else:
        with tempfile.NamedTemporaryFile() as tmpfile:
            merge_dataset_shards(shard_paths, tmpfile.name)
            reservoir_sample(tmpfile.name, output_dataset_path, max_sample_size, seed)

    fsspec_rm(shard_path)


@cached_or_construct_output(success_suffix="SUCCESS")
def attribute_to_dataset_shard(
    input_file_path: str,
    output_file_path: str,
    attr_file_path: str,
    sampling_rate: float,
    seed: int,
    get_label: Callable[[dict, dict], str],
) -> None:
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
    """
    rng = np.random.default_rng(seed=seed)
    with (
        fsspec.open(input_file_path, "rt", compression="gzip") as f_in,
        fsspec.open(attr_file_path, "rt", compression="gzip") as f_attr,
        fsspec.open(output_file_path, "wt", compression="gzip") as f_out,
    ):
        for input_line, attr_line in zip(f_in, f_attr, strict=False):
            if rng.random() > sampling_rate:
                continue

            data = json.loads(input_line)
            attribs = json.loads(attr_line)

            if "text" in data:
                example = {"text": data["text"], "label": get_label(data, attribs)}
                f_out.write(json.dumps(example) + "\n")
            else:
                logging.warning(f"Document {data['id']} has no text field.")


def merge_dataset_shards(
    shard_paths: list[str],
    output_path: str,
) -> None:
    """
    Merges multiple shard files into a single dataset file.

    Args:
        shard_paths (List[str]): List of paths to shard files.
        output_path (str): Path to the output dataset file.
    """
    with fsspec.open(output_path, "wt", compression="infer") as f_out:
        for shard_path in shard_paths:
            with fsspec.open(shard_path, "rt", compression="infer") as f_in:
                for line in f_in:
                    f_out.write(line)


def split_dataset(
    input_path: str,
    train_path: str,
    val_path: str,
    val_frac: float,
    seed: int,
) -> None:
    """
    Splits a dataset into training and validation datasets.

    Args:
        input_path str: Path to input dataset file.
        train_path (str): Path to the output training dataset file.
        val_path (str): Path to the output validation dataset file.
        val_frac (float): Fraction of data to be used for validation.
        seed (int): Seed for random number generator to ensure reproducibility.
    """
    rng = np.random.default_rng(seed=seed)
    with (
        fsspec.open(train_path, "wt", compression="infer") as f_train,
        fsspec.open(val_path, "wt", compression="infer") as f_val,
    ):
        with fsspec.open(input_path, "rt", compression="infer") as f_in:
            for line in f_in:
                if rng.random() < val_frac:
                    f_val.write(line)
                else:
                    f_train.write(line)


def format_dataset(
    input_path: str,
    format_example: Callable[[dict], str],
    output_path: str | None = None,
) -> None:
    """
    Formats a dataset using a custom function.

    Args:
        input_path (str): Path to the input dataset file.
        format_example (Callable[[dict], str]): Function to format examples.
        output_path (str): Path to the output dataset file. If None, the input file is overwritten.
    """
    if output_path is None:
        output_path = input_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "data.tmp")

        with fsspec.open(input_path, "rt", compression="infer") as f_in, fsspec.open(tmp_path, "wt") as f_tmp:
            for line in f_in:
                data = json.loads(line)
                f_tmp.write(format_example(data) + "\n")

        with fsspec.open(tmp_path, "rt") as f_tmp, fsspec.open(output_path, "wt", compression="infer") as f_out:
            for line in f_tmp:
                f_out.write(line)


def shuffle(input_file_path: str, output_file_path: str, seed: int) -> None:
    """
    Shuffles the lines of a file.

    Args:
        input_file_path (str): Path to the input file.
        output_file_path (str): Path to the output file.
        seed (int): Seed for random number generator to ensure reproducibility.
    """
    rng = np.random.default_rng(seed=seed)
    with fsspec.open(input_file_path, "rt", compression="infer") as f_in:
        lines = f_in.readlines()
    rng.shuffle(lines)
    with fsspec.open(output_file_path, "wt", compression="infer") as f_out:
        f_out.writelines(lines)


def reservoir_sample(
    input_dataset_path: str,
    output_dataset_path: str,
    sample_size: int,
    seed: int,
) -> None:
    """Sample a fixed number of examples K from any dataset of size N where K < N using reservoir sampling.

    Args:
        input_dataset_path (str): Path to the input dataset in a single file
            (e.g., output of attribute_to_dataset_shard, or after running merge_shards on attribute_to_dataset).
        output_dataset_path (str): Path to the output dataset.
        sample_size (int): Number of examples to sample from the dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
    """
    rng = np.random.default_rng(seed=seed)
    reservoir = []

    with fsspec.open(input_dataset_path, "rt", compression="infer") as f_in:
        for line in f_in:
            if len(reservoir) < sample_size:
                reservoir.append(line)
            else:
                reservoir[rng.integers(sample_size)] = line

    with fsspec.open(output_dataset_path, "wt", compression="infer") as f_out:
        for line in reservoir:
            f_out.write(line)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for curating a dataset for training a quality classfier

    Attributes:
        input_doc_path (str): Path to the input dataset directory (Dolma format).
        label (str): Label for the dataset. This should be in the format "<label>"
            where <label> is the label for the dataset. For example, "hq" or "lq", respectively.
        sampling_rate (Optional[float]): Subsampling fractioin to construct the dataset.
        max_sample_size (Optional[int]): Maximum number of examples to include in the dataset.
    """

    input_doc_path: str
    label: str
    sampling_rate: float = 1.0
    max_sample_size: int | None = None
