# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

import jax.random
import pytest

from marin.rl.environments.reasoning_gym_env import ReasoningGymEnv
from tests.rl.environments.reasoning_gym_test_support import DummyInferenceContext, install_fake_reasoning_gym


def test_reasoning_gym_env_sample_uses_scores_and_binary_correctness(monkeypatch):
    modules = install_fake_reasoning_gym(
        monkeypatch,
        datasets_by_seed={
            0: [
                {
                    "question": "How many legs?",
                    "answer": "4",
                    "metadata": {
                        "source_index": 3,
                        "score_map": {"4": 1.0, "5": 0.25},
                    },
                }
            ],
            1: [
                {
                    "question": "Eval legs?",
                    "answer": "6",
                    "metadata": {
                        "source_index": 8,
                        "score_map": {"6": 1.0},
                    },
                }
            ],
        },
    )
    env = ReasoningGymEnv(
        dataset_name="leg_counting",
        train_dataset_args={"seed": 0, "size": 1},
        eval_dataset_args={"seed": 1, "size": 1},
        prompt_template="Solve carefully:\n{question}",
    )
    inference_ctx = DummyInferenceContext({"Solve carefully:\nHow many legs?": ["4", "5"]})

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=2,
        temperature=0.7,
        prng_key=jax.random.PRNGKey(0),
        mode="train",
        system_prompt="Use concise answers.",
    )

    assert len(modules.create_calls) == 2
    assert len(rollout_groups) == 1
    assert len(rollout_groups[0].rollouts) == 2

    correct_rollout, partial_rollout = rollout_groups[0].rollouts
    assert correct_rollout.env_name == "reasoning_gym:leg_counting"
    assert correct_rollout.episode_reward == pytest.approx(1.0)
    assert correct_rollout.correctness_reward == pytest.approx(1.0)
    assert partial_rollout.episode_reward == pytest.approx(0.25)
    assert partial_rollout.correctness_reward == pytest.approx(0.0)
    assert correct_rollout.env_example_id.endswith(":0:3")

    assert inference_ctx.last_request is not None
    assert inference_ctx.last_request["system_prompt"] == "Use concise answers."
    assert metrics["reasoning_gym.leg_counting.train_mean_reward"] == pytest.approx(0.625)
    assert metrics["reasoning_gym.leg_counting.train_solve_rate"] == pytest.approx(0.5)
    assert metrics["reasoning_gym.leg_counting.train_sampled_examples"] == pytest.approx(1.0)


def test_reasoning_gym_env_normalizes_composite_dataset_specs(monkeypatch):
    modules = install_fake_reasoning_gym(
        monkeypatch,
        datasets_by_seed={
            7: [
                {
                    "question": "Composite question",
                    "answer": "ABC",
                    "metadata": {
                        "source_dataset": "tower_of_hanoi",
                        "source_index": 11,
                        "score_map": {"ABC": 1.0},
                    },
                }
            ],
            8: [
                {
                    "question": "Composite eval question",
                    "answer": "XYZ",
                    "metadata": {
                        "source_dataset": "leg_counting",
                        "source_index": 5,
                        "score_map": {"XYZ": 1.0},
                    },
                }
            ],
        },
    )
    env = ReasoningGymEnv(
        dataset_name="composite",
        train_dataset_args={
            "seed": 7,
            "size": 1,
            "datasets": [
                {"name": "tower_of_hanoi", "weight": 1.0, "config": {"min_disks": 3, "max_disks": 4}},
                {"name": "leg_counting", "weight": 1.0, "config": {"min_animals": 2, "max_animals": 3}},
            ],
        },
        eval_dataset_args={
            "seed": 8,
            "size": 1,
            "datasets": [
                {"name": "leg_counting", "weight": 1.0, "config": {"min_animals": 2, "max_animals": 3}},
            ],
        },
    )
    inference_ctx = DummyInferenceContext({"Composite question": ["ABC"]})

    rollout_groups, metrics = env.sample(
        inference_ctx=inference_ctx,
        n_examples=1,
        n_generations=1,
        temperature=1.0,
        prng_key=jax.random.PRNGKey(1),
        mode="train",
    )

    train_call = modules.create_calls[0]
    assert train_call["name"] == "composite"
    assert all(isinstance(spec, modules.dataset_spec_cls) for spec in train_call["kwargs"]["datasets"])
    assert len(rollout_groups) == 1
    rollout = rollout_groups[0].rollouts[0]
    assert rollout.env_name == "reasoning_gym:composite"
    assert "tower_of_hanoi" in rollout.env_example_id
    assert metrics["reasoning_gym.composite.train_source_tower_of_hanoi_count"] == pytest.approx(1.0)


def test_reasoning_gym_env_sampling_is_deterministic_for_fixed_prng_key(monkeypatch):
    install_fake_reasoning_gym(
        monkeypatch,
        datasets_by_seed={
            10: [
                {"question": "Q0", "answer": "A0", "metadata": {"source_index": 0}},
                {"question": "Q1", "answer": "A1", "metadata": {"source_index": 1}},
                {"question": "Q2", "answer": "A2", "metadata": {"source_index": 2}},
            ],
            11: [{"question": "Eval", "answer": "A", "metadata": {"source_index": 0}}],
        },
    )
    env = ReasoningGymEnv(
        dataset_name="leg_counting",
        train_dataset_args={"seed": 10, "size": 3},
        eval_dataset_args={"seed": 11, "size": 1},
    )
    inference_ctx = DummyInferenceContext({"Q0": ["A0"], "Q1": ["A1"], "Q2": ["A2"]})
    prng_key = jax.random.PRNGKey(123)

    first_groups, _ = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=1,
        temperature=1.0,
        prng_key=prng_key,
        mode="train",
    )
    second_groups, _ = env.sample(
        inference_ctx=inference_ctx,
        n_examples=2,
        n_generations=1,
        temperature=1.0,
        prng_key=prng_key,
        mode="train",
    )

    first_ids = [group.rollouts[0].env_example_id for group in first_groups]
    second_ids = [group.rollouts[0].env_example_id for group in second_groups]
    assert first_ids == second_ids


def test_reasoning_gym_env_requires_explicit_seed_and_size(monkeypatch):
    install_fake_reasoning_gym(monkeypatch)

    with pytest.raises(ValueError, match="train_dataset_args must include explicit size"):
        ReasoningGymEnv(
            dataset_name="leg_counting",
            train_dataset_args={"seed": 0},
            eval_dataset_args={"seed": 1, "size": 1},
        )


def test_reasoning_gym_env_raises_informative_import_error(monkeypatch):
    module = importlib.import_module("marin.rl.environments.reasoning_gym_env")
    real_import_module = module.importlib.import_module

    def fake_import_module(name: str):
        if name == "reasoning_gym":
            error = ModuleNotFoundError("No module named 'reasoning_gym'")
            error.name = "reasoning_gym"
            raise error
        return real_import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="uv sync --extra reasoning-gym"):
        ReasoningGymEnv(
            dataset_name="leg_counting",
            train_dataset_args={"seed": 0, "size": 1},
            eval_dataset_args={"seed": 1, "size": 1},
        )
