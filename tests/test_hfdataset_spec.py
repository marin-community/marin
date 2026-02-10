# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.defaults import default_tokenize
from marin.processing.tokenize import HfDatasetSpec
from marin.processing.tokenize.tokenize import HfTokenizeConfig


def test_default_tokenize_with_dataset_name():
    step = default_tokenize(
        name="dummy",
        dataset=HfDatasetSpec(id="cnn_dailymail", name="3.0.0"),
        tokenizer="gpt2",
    )
    assert isinstance(step.config, HfTokenizeConfig)
    assert step.config.id == "cnn_dailymail"
    assert step.config.name == "3.0.0"
