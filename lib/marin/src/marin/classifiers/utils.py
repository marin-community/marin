# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
utils.py

Utility functions for building quality classifiers.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import ExitStack
from dataclasses import dataclass, field

import fsspec
import numpy as np
from zephyr import Backend, Dataset

from marin.classifiers.types import Attribute, Document, LabeledExample
from marin.utils import fsspec_glob, fsspec_rm, rebase_file_path

logger = logging.getLogger("ray")


@dataclass(frozen=True)
class CreateDatasetConfig:
    """Configuration for creating a dataset such as ensembling attributes and randomly sampling examples.

    Attributes:
        input_doc_path (str): Path to the input dataset directory (Dolma format).
        output_dataset_path (str): Path to the output dataset directory.
        label_func (Callable[[Document, list[Attribute]], str]): Function to label the dataset. This function
            accepts a document and a list of attributes and returns a string label. This can be used to
            ensemble multiple attributes together or create new attributes based on the document (e.g. length).
        input_attr_paths (list[str] | None): Path to the attributes directory.
        seed (int): Seed for random number generator to ensure reproducibility.
        sampling_rate (float): Fraction of the dataset to sample.
        max_sample_size (int | None): Maximum number of examples to include in the dataset.
        filetype (str): Filetype of the input dataset.
        merge_dataset_shards (bool): Whether to merge dataset shards. If False, the dataset will
            be sharded across many files. If True, the dataset will be merged into a single file.
    """

    input_doc_path: str
    output_dataset_path: str
    label_func: Callable[[Document, list[Attribute]], str] | None = None
    input_attr_paths: list[str] | None = None
    seed: int = 0
    sampling_rate: float = 1.0
    max_sample_size: int | None = None
    filetype: str = "jsonl.gz"
    merge_dataset_shards: bool = True
    columns_to_keep: list[str] = field(default_factory=lambda: ["text"])


def label_documents(
    input_doc_path: str,
    output_attr_path: str,
    label_func: Callable[[Document, list[Attribute]], dict],
    input_attr_paths: list[str] | None = None,
) -> None:
    """
    Create a new attribute by applying label_func to each document and its (optional) associated attributes.

    Args:
        input_doc_path (str): Path to documents (i.e., gs://$BUCKET/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://$BUCKET/attributes/...).
        label_func (Callable[[Document, list[Attribute]], dict]): Generates attribute dict
            from document and other input attributes.
        input_attr_paths (list[str]): Path to attributes needed to determine new attribute.
    """

    logger.info(f"Creating custom attribute for documents in {input_doc_path}, writing to {output_attr_path}.")

    def processing_func(input_file_path):
        attr_file_paths = (
            [rebase_file_path(input_doc_path, input_file_path, input_attr_path) for input_attr_path in input_attr_paths]
            if input_attr_paths is not None
            else []
        )
        return label_documents_shard(input_file_path, label_func, attr_file_paths)

    pipeline = (
        Dataset.from_files(f"{input_doc_path}/**/*.jsonl.gz")
        .flat_map(processing_func)
        .write_jsonl(f"{output_attr_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    Backend.execute(pipeline, max_parallelism=1000)


def label_documents_shard(
    input_doc_file_path: str,
    label_func: Callable[[Document, list[Attribute]], dict],
    input_attr_file_paths: list[str] | None = None,
):
    """Process documents file and yield labeled attribute records.

    Args:
        input_doc_file_path: Path to the input JSONL file in Dolma format (gzip compressed).
        label_func: Generates attribute dict from document and other input attributes.
        input_attr_file_paths: Path to attributes needed to determine new attribute.

    Yields:
        Attribute records with labels
    """
    with ExitStack() as stack:
        f_attrs = (
            [
                stack.enter_context(fsspec.open(attr_file, "rt", compression="infer"))
                for attr_file in input_attr_file_paths
            ]
            if input_attr_file_paths is not None
            else []
        )
        with fsspec.open(input_doc_file_path, "rt", compression="infer") as f_doc:
            for lines in zip(f_doc, *f_attrs, strict=False):
                document: Document = json.loads(lines[0])
                input_attributes: list[Attribute] = [json.loads(line) for line in lines[1:]]

                output_attribute: Attribute = {
                    "id": document["id"],
                    "source": document["source"],
                    "attributes": label_func(document, input_attributes),
                }
                yield output_attribute


def create_dataset(
    config: CreateDatasetConfig,
) -> None:
    """
    Converts documents and specified attribute to quality classifier training data (text,label) pairs.

    Args:
        config (CreateDatasetConfig): Configuration object containing all parameters for dataset creation.
    """

    def processing_func(input_file_path: str) -> Generator:
        attr_file_paths = (
            [
                rebase_file_path(config.input_doc_path, input_file_path, input_attr_path)
                for input_attr_path in config.input_attr_paths
            ]
            if config.input_attr_paths is not None
            else []
        )
        return create_dataset_shard(
            input_file_path,
            config.label_func,
            attr_file_paths,
            config.sampling_rate,
            config.seed,
            config.columns_to_keep,
        )

    _, doc_fs_path = fsspec.core.url_to_fs(config.input_doc_path)
    dataset_file_path = os.path.join(
        config.output_dataset_path, "data", doc_fs_path.lstrip("/"), f"data.{config.filetype}"
    )
    shard_path = os.path.join(config.output_dataset_path, "shards", doc_fs_path.lstrip("/"))

    pipeline = (
        Dataset.from_files(f"{config.input_doc_path}/**/*.{config.filetype}")
        .flat_map(processing_func)
        .write_jsonl(f"{shard_path}/data-{{shard:05d}}-of-{{total:05d}}.{config.filetype}")
    )
    Backend.execute(pipeline, max_parallelism=1000)

    shard_paths = fsspec_glob(os.path.join(shard_path, f"**/*.{config.filetype}"))
    if config.max_sample_size is None:
        if config.merge_dataset_shards:
            merge_dataset_shards(shard_paths, dataset_file_path)
    else:
        if config.merge_dataset_shards:
            with tempfile.NamedTemporaryFile() as tmpfile:
                merge_dataset_shards(shard_paths, tmpfile.name)
                reservoir_sample(
                    [tmpfile.name],
                    dataset_file_path,
                    config.max_sample_size,
                    config.seed,
                )
        else:
            reservoir_sample(
                shard_paths,
                dataset_file_path,
                config.max_sample_size,
                config.seed,
            )

    fsspec_rm(shard_path)


def create_dataset_shard(
    input_doc_file_path: str,
    label_func: Callable[[Document, list[Attribute]], str] | None,
    input_attr_file_paths: list[str],
    sampling_rate: float,
    seed: int,
    columns_to_keep: list[str],
):
    """Process documents file and yield sampled training examples.

    Only a fraction of the examples, determined by the sampling rate, are yielded (eg, to control size
    of training dataset and/or weight different domains).

    Args:
        input_doc_file_path: Path to the input JSONL file (gzip compressed).
        label_func: Generates label from document and input attributes.
        input_attr_file_paths: Path to the attribute JSONL file (gzip compressed).
        sampling_rate: Fraction of lines to be yielded.
        seed: Seed for random number generator to ensure reproducibility.
        columns_to_keep: List of columns to keep in the output.

    Yields:
        Training examples with labels
    """

    def hash_fn(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest(), 16)

    # since we don't want the same seed for all shards
    rng = np.random.default_rng(seed=seed + hash_fn(input_doc_file_path))

    with ExitStack() as stack:
        f_attrs = (
            [
                stack.enter_context(fsspec.open(attr_file, "rt", compression="infer"))
                for attr_file in input_attr_file_paths
            ]
            if input_attr_file_paths is not None
            else []
        )
        with fsspec.open(input_doc_file_path, "rt", compression="infer") as f_doc:
            for lines in zip(f_doc, *f_attrs, strict=False):
                if rng.random() > sampling_rate:
                    continue

                doc_obj = json.loads(lines[0])
                attr_objs = [json.loads(line) for line in lines[1:]]

                if "text" in doc_obj:
                    example = {col: doc_obj[col] for col in columns_to_keep}
                    if label_func is not None:
                        example.update({"label": label_func(doc_obj, attr_objs)})
                    yield example
                else:
                    logging.warning(f"Document {doc_obj['id']} has no text field.")


def merge_dataset_shards(
    shard_file_paths: list[str],
    output_file_path: str,
) -> None:
    """
    Merges multiple shard files into a single dataset file.

    Args:
        shard_file_paths (List[str]): List of paths to shard files.
        output_file_path (str): Path to the output dataset file.
    """
    with fsspec.open(output_file_path, "wt", compression="infer") as f_out:
        for shard_path in shard_file_paths:
            with fsspec.open(shard_path, "rt", compression="infer") as f_in:
                for line in f_in:
                    f_out.write(line)


def split_dataset(
    input_file_path: str,
    train_file_path: str,
    val_file_path: str,
    val_frac: float,
    seed: int,
) -> None:
    """
    Splits a dataset into training and validation datasets.

    Args:
        input_file_path str: Path to input dataset file.
        train_file_path (str): Path to the output training dataset file.
        val_file_path (str): Path to the output validation dataset file.
        val_frac (float): Fraction of data to be used for validation.
        seed (int): Seed for random number generator to ensure reproducibility.
    """
    rng = np.random.default_rng(seed=seed)
    with (
        fsspec.open(train_file_path, "wt", compression="infer") as f_train,
        fsspec.open(val_file_path, "wt", compression="infer") as f_val,
    ):
        with fsspec.open(input_file_path, "rt", compression="infer") as f_in:
            for line in f_in:
                if rng.random() < val_frac:
                    f_val.write(line)
                else:
                    f_train.write(line)


def format_dataset(
    input_file_path: str,
    format_example: Callable[[LabeledExample], str],
    output_file_path: str | None = None,
) -> None:
    """
    Formats a dataset using a custom function.

    Args:
        input_file_path (str): Path to the input dataset file.
        format_example (Callable[[LabeledExample], str]): Function to format examples.
        output_file_path (str): Path to the output dataset file. If None, the input file is overwritten.
    """
    if output_file_path is None:
        output_file_path = input_file_path

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "data.tmp")

        with fsspec.open(input_file_path, "rt", compression="infer") as f_in, fsspec.open(tmp_path, "wt") as f_tmp:
            for line in f_in:
                data = json.loads(line)
                f_tmp.write(format_example(data) + "\n")

        with fsspec.open(tmp_path, "rt") as f_tmp, fsspec.open(output_file_path, "wt", compression="infer") as f_out:
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
    input_file_paths: list[str],
    output_file_path: str,
    sample_size: int,
    seed: int,
) -> None:
    """Sample a fixed number of examples K from any dataset of size N where K < N using reservoir sampling.

    Args:
        input_file_path (str): Path to the input dataset in a single file
            (e.g., output of attribute_to_dataset_shard, or after running merge_shards on attribute_to_dataset).
        output_file_path (str): Path to the output dataset.
        sample_size (int): Number of examples to sample from the dataset.
        seed (int): Seed for random number generator to ensure reproducibility.
    """
    rng = np.random.default_rng(seed=seed)
    reservoir = []

    for input_file_path in input_file_paths:
        with fsspec.open(input_file_path, "rt", compression="infer") as f_in:
            for line in f_in:
                if len(reservoir) < sample_size:
                    reservoir.append(line)
                else:
                    reservoir[rng.integers(sample_size)] = line

    with fsspec.open(output_file_path, "wt", compression="infer") as f_out:
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
