# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.pretraining_datasets import DATASETS
from experiments.pretraining_datasets.diagnostic_logs import ghalogs_dev, ghalogs_download


def test_ghalogs_dataset_registry_wires_train_and_dev_parquet_steps():
    dataset_info = DATASETS["ghalogs"]

    assert dataset_info["subsets"] == ["train"]
    assert dataset_info["download"].name == ghalogs_download.name

    tokenize_steps = dataset_info["tokenize_fn"]()

    assert list(tokenize_steps) == ["ghalogs/train"]
    tokenize_step = tokenize_steps["ghalogs/train"]
    assert tokenize_step.config.train_paths == [ghalogs_download.as_input_name() / "*.parquet"]
    assert tokenize_step.config.validation_paths == [ghalogs_dev.as_input_name() / "*.parquet"]
