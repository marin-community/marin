# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Datakit Testbed Grug-MoE training harness.

Covers the pure arithmetic helpers and the sample-to-TokenizerStep bridge.
``run_testbed_config`` itself loads the Llama-3 tokenizer (gated repo) via
``default_validation_sets``, so its end-to-end integration is validated by
a live smoke run, not unit tests.
"""

import pytest

from experiments.datakit_testbed.dag import build_testbed_steps
from marin.datakit.sources import all_sources
from experiments.datakit_testbed.train import (
    DEFAULT_TARGET_BUDGET_TOKENS,
    build_testbed_tokenize_steps,
    simulated_experiment_budget,
)

_SMOKE_SOURCE = all_sources()["nemotron_cc_code_v1/all"]


def test_default_target_budget_is_1t():
    assert DEFAULT_TARGET_BUDGET_TOKENS == 1_000_000_000_000


def test_simulated_experiment_budget_is_product_of_schedule():
    # 32 batch * 16384 steps * 4096 seq_len ~= 2.1B tokens
    assert simulated_experiment_budget(train_batch_size=32, num_train_steps=2**14, seq_len=4096) == 32 * (2**14) * 4096


@pytest.mark.parametrize(
    "batch, steps, seq_len",
    [
        (1, 1, 1),
        (256, 1000, 4096),
        (2048, 2**14, 8192),
    ],
)
def test_simulated_experiment_budget_returns_positive_int(batch: int, steps: int, seq_len: int):
    budget = simulated_experiment_budget(train_batch_size=batch, num_train_steps=steps, seq_len=seq_len)
    assert isinstance(budget, int)
    assert budget > 0
    assert budget == batch * steps * seq_len


def test_tokenize_step_bridge_produces_training_ready_steps():
    """sample StepSpec -> ExecutorStep[TokenizeConfig] via the bridge helper.

    This catches the StepSpec/ExecutorStep type gap that blocked training
    end-to-end before the dag.py refactor: the bridge must produce real
    ``ExecutorStep[TokenizeConfig]`` (= ``TokenizerStep``), not a generic
    ExecutorStep wrapping _StepSpecMigrationConfig.
    """
    from marin.execution.executor import ExecutorStep
    from marin.processing.tokenize import TokenizeConfig

    steps = build_testbed_steps("run0", sources=[_SMOKE_SOURCE])
    sampled = {_SMOKE_SOURCE.name: next(s for s in steps if s.name.startswith("datakit-testbed/sample/"))}
    tokenize_steps = build_testbed_tokenize_steps(sampled)

    assert set(tokenize_steps) == {_SMOKE_SOURCE.name}
    step = tokenize_steps[_SMOKE_SOURCE.name]
    assert isinstance(step, ExecutorStep)
    assert isinstance(step.config, TokenizeConfig)


def test_bridge_steps_compose_into_lm_mixture():
    """The refactor's real proof: bridged steps feed lm_mixture_data_config.

    Exercises step_to_lm_mixture_component + _verify_tokenizers_same, which
    silently failed on _StepSpecMigrationConfig before the bridge fix.
    """
    from experiments.datakit_testbed.mixture import build_testbed_mixture

    steps = build_testbed_steps("run0", sources=[_SMOKE_SOURCE])
    sampled = {_SMOKE_SOURCE.name: next(s for s in steps if s.name.startswith("datakit-testbed/sample/"))}
    tokenize_steps = build_testbed_tokenize_steps(sampled)

    mixture = build_testbed_mixture(tokenize_steps, weights={_SMOKE_SOURCE.name: 1.0})
    assert _SMOKE_SOURCE.name in mixture.train_weights
    assert mixture.train_weights[_SMOKE_SOURCE.name] == 1.0
