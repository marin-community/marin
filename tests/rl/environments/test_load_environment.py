# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment loading from EnvConfig."""

from marin.rl.environments import EnvConfig, load_environment_from_spec
from marin.rl.environments.mock_env import MockEnv
from marin.rl.environments.math_env import MathEnv


def test_load_mock_environment():
    """Test loading MockEnv via EnvConfig."""
    config = EnvConfig(env_class="marin.rl.environments.mock_env.MockEnv", env_args={"task_type": "cats", "seed": 42})

    env = load_environment_from_spec(config)

    assert isinstance(env, MockEnv)
    assert env.task_type == "cats"
    assert len(env.train_examples) > 0
    assert len(env.eval_examples) > 0


def test_load_math_environment():
    """Test loading MathEnv via EnvConfig."""
    config = EnvConfig(env_class="marin.rl.environments.math_env.MathEnv", env_args={"seed": 42})

    env = load_environment_from_spec(config)

    assert isinstance(env, MathEnv)
    assert len(env.train_examples) > 0
    assert len(env.eval_examples) > 0
