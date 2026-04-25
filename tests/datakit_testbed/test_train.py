# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Datakit Testbed Grug-MoE training harness.

Covers the sample → ``TokenizerStep`` bridge and its integration with
``lm_mixture_data_config``. ``run_testbed_config`` itself loads the
Llama-3 tokenizer (gated repo) via ``default_validation_sets``, so its
end-to-end shape is validated by a live smoke run rather than unit tests.
"""

from experiments.datakit_testbed.sampler import build_testbed_steps
from experiments.datakit_testbed.train import testbed_tokenize as _testbed_tokenize
from marin.datakit.sources import all_sources

_SMOKE_SOURCE = all_sources()["nemotron_cc_code_v1/all"]


def test_tokenize_step_bridge_produces_training_ready_steps():
    """sample StepSpec -> ExecutorStep[TokenizeConfig] via the bridge helper.

    Catches the StepSpec/ExecutorStep type gap that blocked training
    end-to-end before the dag.py refactor: the bridge must produce real
    ``ExecutorStep[TokenizeConfig]`` (= ``TokenizerStep``), not a generic
    ExecutorStep wrapping ``_StepSpecMigrationConfig``.
    """
    from marin.execution.executor import ExecutorStep
    from marin.processing.tokenize import TokenizeConfig

    steps = build_testbed_steps(sources=[_SMOKE_SOURCE])
    sampled = next(s for s in steps if s.name.startswith("data/datakit/"))
    step = _testbed_tokenize(_SMOKE_SOURCE.name, sampled)

    assert isinstance(step, ExecutorStep)
    assert isinstance(step.config, TokenizeConfig)


def test_bridge_steps_compose_into_lm_mixture():
    """Bridged steps feed ``lm_mixture_data_config``.

    Exercises ``step_to_lm_mixture_component`` + ``_verify_tokenizers_same``,
    which silently failed on ``_StepSpecMigrationConfig`` before the bridge fix.
    """
    from marin.processing.tokenize import lm_mixture_data_config

    steps = build_testbed_steps(sources=[_SMOKE_SOURCE])
    sampled = next(s for s in steps if s.name.startswith("data/datakit/"))
    tokenize_steps = {_SMOKE_SOURCE.name: _testbed_tokenize(_SMOKE_SOURCE.name, sampled)}

    mixture = lm_mixture_data_config(components=tokenize_steps, weights={_SMOKE_SOURCE.name: 1.0})
    assert _SMOKE_SOURCE.name in mixture.train_weights
    assert mixture.train_weights[_SMOKE_SOURCE.name] == 1.0
