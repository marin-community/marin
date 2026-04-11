# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment loading from EnvConfig."""

from marin.rl.openreward import (
    OpenRewardPromptBlock,
    OpenRewardPromptBlockType,
    OpenRewardTaskManifest,
    OpenRewardTaskManifestEntry,
    OpenRewardToolSpec,
    save_openreward_task_manifest,
)
from marin.rl.environments import EnvConfig, load_environment_from_spec
from marin.rl.environments.math_env import MathEnv
from marin.rl.environments.mock_env import MockEnv
from marin.rl.environments.openreward_env import OpenRewardEnv


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


def test_load_openreward_environment(tmp_path):
    """Test loading OpenRewardEnv via EnvConfig with a local manifest."""
    manifest = OpenRewardTaskManifest(
        deployment_name="marin/openreward-math-agent",
        environment_name="math-agent-v1",
        split="train",
        tasks=[
            OpenRewardTaskManifestEntry(
                task_index=0,
                task_spec={"problem_id": "train-0"},
                prompt_blocks=[OpenRewardPromptBlock(type=OpenRewardPromptBlockType.TEXT, text="What is 2+2?")],
                tools=[
                    OpenRewardToolSpec(
                        name="submit_answer",
                        description="Submit the final answer.",
                        input_schema={"type": "object"},
                    )
                ],
            )
        ],
    )
    manifest_path = tmp_path / "train-openreward-manifest.json"
    save_openreward_task_manifest(manifest, str(manifest_path))
    config = EnvConfig(
        env_class="marin.rl.environments.openreward_env.OpenRewardEnv",
        env_args={"train_manifest_path": str(manifest_path)},
    )

    env = load_environment_from_spec(config)

    assert isinstance(env, OpenRewardEnv)
    assert env.deployment_name == "marin/openreward-math-agent"
    assert env.environment_name == "math-agent-v1"
