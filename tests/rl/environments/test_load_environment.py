# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment loading from EnvConfig."""

import sys
from types import ModuleType

from marin.rl.environments import EnvConfig, load_environment_from_spec
from marin.rl.environments.math_env import MathEnv
from marin.rl.environments.mock_env import MockEnv


def test_load_mock_environment():
    """Test loading MockEnv via EnvConfig."""
    config = EnvConfig(env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "cats", "seed": 42})

    env = load_environment_from_spec(config)

    assert isinstance(env, MockEnv)
    assert env.task_type == "cats"
    assert len(env.train_examples) > 0
    assert len(env.eval_examples) > 0


def test_load_math_environment():
    """Test loading MathEnv via EnvConfig with inline data (no HF download)."""
    config = EnvConfig(
        env_class="marin.rl.environments.math_env.MathEnv",
        env_args={
            "seed": 42,
            "train_dataset": [{"problem": "What is 1+1?", "solution": "\\boxed{2}"}],
            "eval_dataset": [{"problem": "What is 2+2?", "solution": "\\boxed{4}"}],
        },
    )

    env = load_environment_from_spec(config)

    assert isinstance(env, MathEnv)
    assert len(env.train_examples) > 0
    assert len(env.eval_examples) > 0


def test_load_environment_from_spec_does_not_call_prepare():
    module_name = "test_prepare_env_module"
    test_module = ModuleType(module_name)

    class PreparingEnv:
        prepare_calls = 0

        def __init__(self):
            pass

        def prepare(self):
            type(self).prepare_calls += 1

    test_module.PreparingEnv = PreparingEnv
    sys.modules[module_name] = test_module

    try:
        env = load_environment_from_spec(EnvConfig(env_class=f"{module_name}.PreparingEnv", env_args={}))
    finally:
        sys.modules.pop(module_name, None)

    assert isinstance(env, PreparingEnv)
    assert PreparingEnv.prepare_calls == 0
