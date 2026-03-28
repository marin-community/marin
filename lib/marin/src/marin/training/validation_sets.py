# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.defaults import default_validation_sets
from levanter.data.text.datasets import LmDataConfig

from marin.processing.tokenize import add_validation_sets_to_mixture


def config_with_default_validation_sets(data_config: LmDataConfig) -> LmDataConfig:
    """Add the standard default validation sets used by default_train."""
    return add_validation_sets_to_mixture(
        data_config,
        default_validation_sets(tokenizer=data_config.tokenizer),
    )
