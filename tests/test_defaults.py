# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

from levanter.data.text import LmDataConfig
from marin.execution.executor import ExecutorStep, unwrap_versioned_value
from marin.processing.tokenize.tokenize import TokenizeConfig

from experiments.defaults import DEFAULT_NEW_RUN_DATA_SHUFFLE, _prepare_data_config
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


def _tokenized_step() -> ExecutorStep[TokenizeConfig]:
    return ExecutorStep(
        name="tokenized/test-defaults",
        fn=lambda config: None,
        config=TokenizeConfig(
            train_paths=["gs://example-bucket/train.jsonl"],
            validation_paths=[],
            cache_path="gs://example-bucket/cache",
            tokenizer="gpt2",
        ),
    )


def test_prepare_data_config_uses_block_shuffle_for_new_default_train_runs():
    prepared = _prepare_data_config(_tokenized_step(), use_default_validation=False)

    assert unwrap_versioned_value(prepared.shuffle) == DEFAULT_NEW_RUN_DATA_SHUFFLE
    assert prepared.permutation_type == "feistel"


def test_prepare_data_config_preserves_explicit_existing_mixture_config():
    existing = LmDataConfig(tokenizer="gpt2", shuffle=True)

    prepared = _prepare_data_config(existing, use_default_validation=False)

    assert prepared is existing
    assert prepared.shuffle is True


def test_grug_launch_defaults_use_block_shuffle_for_new_runs():
    assert nemotron_mix_block_shuffle.shuffle == DEFAULT_NEW_RUN_DATA_SHUFFLE

    for module_name in (
        "experiments.grug.base.launch",
        "experiments.grug.moe.launch",
        "experiments.grug.modular_opt.launch",
    ):
        module = importlib.import_module(module_name)
        assert module.NEMOTRON_MIX_WITH_DEFAULT_VALIDATION.shuffle == DEFAULT_NEW_RUN_DATA_SHUFFLE
