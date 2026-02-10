# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
train_fasttext.py

Training script for fastText quality classifiers.
"""

import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import datetime

import draccus
import fsspec
from fray.v1.cluster import Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig, current_cluster
from marin.processing.classification.dataset_utils import (
    Attribute,
    DatasetConfig,
    Document,
)
from marin.utils import fsspec_cpdir, fsspec_exists, fsspec_glob, fsspec_rm, rebase_file_path
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)


def format_example(data: dict) -> str:
    """
    Converts example to fastText training data format.
    """
    text = re.sub(r"[\n\r]", " ", data["text"])
    return f"__label__{data['label']}" + " " + text


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


def create_dataset(config: CreateDatasetConfig) -> None:
    """
    Create a dataset from documents by optionally labeling and sampling.

    Args:
        config (CreateDatasetConfig): Configuration for dataset creation
    """

    logger.info(f"Creating dataset from {config.input_doc_path}, writing to {config.output_dataset_path}.")

    def processing_func(input_file_path):
        attr_file_paths = (
            [
                rebase_file_path(config.input_doc_path, input_file_path, input_attr_path)
                for input_attr_path in config.input_attr_paths
            ]
            if config.input_attr_paths is not None
            else []
        )
        return create_dataset_shard(input_file_path, config, attr_file_paths)

    output_path = os.path.join(config.output_dataset_path, "data", "{{shard:05d}}.jsonl.gz")
    if config.merge_dataset_shards:
        output_path = os.path.join(config.output_dataset_path, "data", "data.jsonl.gz")

    with ZephyrContext(name="fasttext-prep") as ctx:
        ctx.execute(
            Dataset.from_files(f"{config.input_doc_path}/**/*.{config.filetype}")
            .flat_map(processing_func)
            .write_jsonl(output_path)
        )


def create_dataset_shard(
    input_file_path: str, config: CreateDatasetConfig, attr_file_paths: list[str]
) -> Generator[dict, None, None]:
    """
    Process a shard of documents to create dataset examples.

    Args:
        input_file_path (str): Path to the input document file.
        config (CreateDatasetConfig): Configuration for dataset creation.
        attr_file_paths (list[str]): Paths to attribute files.

    Yields:
        dict: Dataset example with text and label
    """
    from contextlib import ExitStack

    with ExitStack() as stack:
        doc_fs = fsspec.open(input_file_path, "r", compression="infer")
        doc_file = stack.enter_context(doc_fs)

        attr_files = []
        for attr_file_path in attr_file_paths:
            attr_fs = fsspec.open(attr_file_path, "r", compression="infer")
            attr_file = stack.enter_context(attr_fs)
            attr_files.append(attr_file)

        examples = []
        for doc_line in doc_file:
            doc = json.loads(doc_line)
            attrs = [json.loads(attr_file.readline()) for attr_file in attr_files]

            if config.label_func is not None:
                label = config.label_func(doc, attrs)
            else:
                label = None

            example = {"text": doc["text"], "label": label}
            for col in config.columns_to_keep:
                if col in doc and col not in example:
                    example[col] = doc[col]

            examples.append(example)

        if config.max_sample_size is not None:
            examples = reservoir_sample(examples, config.max_sample_size, config.seed)
        elif config.sampling_rate < 1.0:
            random.seed(config.seed)
            examples = [ex for ex in examples if random.random() < config.sampling_rate]

        for example in examples:
            yield example


def merge_dataset_shards(shard_paths: list[str], output_path: str) -> None:
    """
    Merge multiple dataset shards into a single file.

    Args:
        shard_paths (list[str]): Paths to the input shards.
        output_path (str): Path to write the merged output.
    """
    logger.info(f"Merging {len(shard_paths)} shards into {output_path}.")

    with open(output_path, "w") as output_file:
        for shard_path in shard_paths:
            with fsspec.open(shard_path, "r", compression="infer") as shard_file:
                for line in shard_file:
                    output_file.write(line)


def split_dataset(input_path: str, train_path: str, val_path: str, val_frac: float, seed: int) -> None:
    """
    Split a dataset into training and validation sets.

    Args:
        input_path (str): Path to the input dataset.
        train_path (str): Path to write the training set.
        val_path (str): Path to write the validation set.
        val_frac (float): Fraction of data to use for validation.
        seed (int): Random seed for reproducibility.
    """
    logger.info(f"Splitting dataset {input_path} into train ({1 - val_frac}) and val ({val_frac}).")

    random.seed(seed)

    with open(input_path, "r") as input_file, open(train_path, "w") as train_file, open(val_path, "w") as val_file:
        for line in input_file:
            if random.random() < val_frac:
                val_file.write(line)
            else:
                train_file.write(line)


def format_dataset(input_path: str, format_func: Callable[[dict], str]) -> None:
    """
    Format dataset examples in-place using format_func.

    Args:
        input_path (str): Path to the dataset file.
        format_func (Callable[[dict], str]): Function to format each example.
    """
    logger.info(f"Formatting dataset at {input_path}.")

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        temp_path = temp_file.name
        with open(input_path, "r") as input_file:
            for line in input_file:
                example = json.loads(line)
                formatted = format_func(example)
                temp_file.write(formatted + "\n")

    os.replace(temp_path, input_path)


def shuffle(input_file_path: str, output_file_path: str, seed: int) -> None:
    """
    Shuffle lines in a file.

    Args:
        input_file_path (str): Path to the input file.
        output_file_path (str): Path to write the shuffled output.
        seed (int): Random seed for reproducibility.
    """
    logger.info(f"Shuffling {input_file_path} to {output_file_path}.")

    random.seed(seed)

    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()

    random.shuffle(lines)

    with open(output_file_path, "w") as output_file:
        output_file.writelines(lines)


def reservoir_sample(examples: list, sample_size: int, seed: int) -> list:
    """
    Perform reservoir sampling to get a random sample.

    Args:
        examples (list): List of examples to sample from.
        sample_size (int): Number of examples to sample.
        seed (int): Random seed for reproducibility.

    Returns:
        list: Sampled examples
    """
    random.seed(seed)

    reservoir = []
    for i, example in enumerate(examples):
        if i < sample_size:
            reservoir.append(example)
        else:
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = example

    return reservoir


def train_model(
    input_path: str, output_path: str, seed: int, val_frac: float, memory_req: int, **fasttext_args: dict
) -> None:
    """
    Train a fastText model.

    Args:
        input_path (str): Path for input training data.
        output_path (str): Path to save the trained model (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_frac (float): Fraction of data to be used for validation.
        memory_req (int): Amount of memory allocated for remote training process (in GB).
        fasttext_args (dict): Arguments for the fastText training process
            (see fastText docs for the full list of options).

    Returns:
        None: No return value.
    """
    logger = logging.getLogger("ray")

    logger.info(f"Training fastText model for experiment {output_path}")
    datetime_start = datetime.utcnow()

    if fsspec_exists(os.path.join(output_path, "model.bin")):
        logger.info(f"Model already exists at {output_path}/model.bin. Skipping training.")
        return

    import floret as fasttext

    shard_paths = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))
    logger.info(f"Received input paths: {shard_paths}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        merge_path = os.path.join(tmp_dir, "data.full")
        train_path = os.path.join(tmp_dir, "data.train")
        val_path = os.path.join(tmp_dir, "data.val")
        model_path = os.path.join(tmp_dir, "model.bin")

        merge_dataset_shards(shard_paths, merge_path)
        format_dataset(merge_path, format_example)
        split_dataset(merge_path, train_path, val_path, val_frac, seed)
        shuffle(train_path, train_path, seed)

        model = fasttext.train_supervised(train_path, **fasttext_args)
        model.save_model(model_path)

        fsspec_rm(merge_path)
        fsspec_cpdir(tmp_dir, output_path)

    datetime_end = datetime.utcnow()
    logger.info(f"Training fastText for experiment {output_path} completed in {datetime_end - datetime_start}.")


@dataclass
class TrainFasttextClassifierConfig:
    """
    Configuration class for main process.

    Attributes:
        output_path (str): Path for output data (i.e., gs://$BUCKET/classifiers/$EXPERIMENT).
        datasets (list[DatasetConfig]): List of configurations for converting Dolma documents into
            labeled training datasets.
        fasttext_args (dict): Arguments for the fastText training process (see fastText docs for list of options).
        seed (int): Seed for random number generator to ensure reproducibility.
        val_frac (float): Fraction of data to be used for validation.
        memory (int): Amount of memory allocated for remote training process (in GB).
    """

    output_path: str | None = field(default=None)
    datasets: list[DatasetConfig] = field(default_factory=list)
    fasttext_args: dict = field(default_factory=dict)
    seed: int = 0
    val_frac: float = 0.1
    memory: int = 1


def train(cfg: TrainFasttextClassifierConfig):
    for dataset in cfg.datasets:
        create_dataset(
            config=CreateDatasetConfig(
                input_doc_path=dataset.input_doc_path,
                output_dataset_path=cfg.output_path,
                label_func=lambda doc, attrs, dataset=dataset: dataset.label,
                seed=cfg.seed,
                sampling_rate=dataset.sampling_rate,
                max_sample_size=dataset.max_sample_size,
            )
        )

    input_dataset_path = os.path.join(cfg.output_path, "data")

    job_request = JobRequest(
        name=f"train-fasttext-{cfg.output_path}",
        resources=ResourceConfig.with_cpu(ram=f"{cfg.memory}g", disk="10G", preemptible=True),
        entrypoint=Entrypoint.from_callable(
            train_model,
            kwargs={
                "input_path": input_dataset_path,
                "output_path": cfg.output_path,
                "seed": cfg.seed,
                "val_frac": cfg.val_frac,
                "memory_req": cfg.memory,
                **cfg.fasttext_args,
            },
        ),
        environment=EnvironmentConfig.create(),
    )
    cluster = current_cluster()
    job_id = cluster.launch(job_request)
    cluster.wait(job_id, raise_on_failure=True)


@draccus.wrap()
def main(cfg: TrainFasttextClassifierConfig):
    train(cfg)


if __name__ == "__main__":
    main()
