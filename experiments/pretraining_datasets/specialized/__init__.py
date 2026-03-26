# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.pretraining_datasets.specialized.lean import lean_concat_config_llama3

DATASETS = {
    "lean": lean_concat_config_llama3,
}

__all__ = [
    "DATASETS",
    "lean_concat_config_llama3",
]
