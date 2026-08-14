# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Datakit Testbed Grug-MoE training harness.

Covers the sample → tokenize ``StepSpec`` bridge and its integration with
``lm_mixture_data_config``. ``run_testbed_config`` itself loads the Llama-3
tokenizer (gated repo), so its end-to-end shape is validated by a live smoke
run rather than unit tests.
"""

import pytest
from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config

from experiments.datakit.testbed.sampler import build_testbed_steps
from experiments.datakit.testbed.settings import TESTBED_TOKENIZER
from experiments.datakit.testbed.train import _bucket_tokenize_config
from experiments.datakit.testbed.train import testbed_tokenize as _testbed_tokenize


@pytest.fixture
def smoke_source() -> DatakitSource:
    return all_sources()["nemotron_cc_code_v1/all"]


def test_tokenize_step_bridge_depends_on_sample_and_yields_tokenize_config(monkeypatch, smoke_source):
    """sample StepSpec -> tokenize StepSpec, wired to the sample as a dependency.

    The bridge must produce a step that depends on the sample shards and whose
    rebuilt config is a real ``TokenizeConfig`` reading the sample's main parquet.
    """
    monkeypatch.setenv("MARIN_PREFIX", "gs://example-bucket")

    steps = build_testbed_steps(sources=[smoke_source])
    sampled = next(s for s in steps if s.name.startswith("data/datakit/normalized/"))
    step = _testbed_tokenize(smoke_source.name, sampled)

    assert isinstance(step, StepSpec)
    assert step.deps == [sampled]

    config = _bucket_tokenize_config(step, TESTBED_TOKENIZER)
    assert isinstance(config, TokenizeConfig)
    assert config.tokenizer == TESTBED_TOKENIZER
    assert config.train_paths == [f"{sampled.output_path}/outputs/main/*.parquet"]


def test_bridge_steps_compose_into_lm_mixture(monkeypatch, smoke_source):
    """The bridged tokenize config composes into ``lm_mixture_data_config``."""
    monkeypatch.setenv("MARIN_PREFIX", "gs://example-bucket")

    steps = build_testbed_steps(sources=[smoke_source])
    sampled = next(s for s in steps if s.name.startswith("data/datakit/normalized/"))
    step = _testbed_tokenize(smoke_source.name, sampled)
    components = {smoke_source.name: _bucket_tokenize_config(step, TESTBED_TOKENIZER)}

    mixture = lm_mixture_data_config(components=components, weights={smoke_source.name: 1.0})
    assert smoke_source.name in mixture.train_weights
    assert mixture.train_weights[smoke_source.name] == 1.0
