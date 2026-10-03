# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from fray.types import ResourceConfig
from marin.execution.lazy import StepContext
from marin.experiment.data import tokenized

_TOKENIZER = "gpt2"
_V = "2026.06.28"


def test_tokenized_requires_exactly_one_raw_input():
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", tokenizer=_TOKENIZER, version=_V)
    with pytest.raises(ValueError, match="exactly one of source, paths, or raw"):
        tokenized("c", source="org/corpus", paths=["gs://b/x"], tokenizer=_TOKENIZER, version=_V)


def test_tokenized_configures_zephyr_worker_resources():
    workers = ResourceConfig(cpu=2, ram="64g", disk="5g")
    handle = tokenized(
        "c",
        paths=["gs://b/x"],
        tokenizer=_TOKENIZER,
        version=_V,
        worker_resources=workers,
    )

    config = handle.build_config(StepContext.for_fingerprint(handle.runtime_args, handle.deps))

    assert config.worker_resources == workers
