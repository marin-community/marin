# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment loading from EnvConfig."""

from tests.rl.environments.reasoning_gym_test_support import install_fake_reasoning_gym
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


def test_load_reasoning_gym_environment(monkeypatch):
    """Test loading ReasoningGymEnv via EnvConfig."""
    install_fake_reasoning_gym(monkeypatch)
    from marin.rl.environments.reasoning_gym_env import ReasoningGymEnv

    config = EnvConfig(
        env_class="marin.rl.environments.reasoning_gym_env.ReasoningGymEnv",
        env_args={
            "dataset_name": "leg_counting",
            "train_dataset_args": {"seed": 42, "size": 2},
            "eval_dataset_args": {"seed": 43, "size": 2},
        },
    )

    env = load_environment_from_spec(config)

    assert isinstance(env, ReasoningGymEnv)
