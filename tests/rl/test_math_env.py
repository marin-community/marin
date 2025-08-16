"""Tests MathEnv rollout generation with mocked OpenAI client and dataset."""

import asyncio
from collections import deque

import pytest

try:
    import openai_responses  # type: ignore
except ImportError:  # pragma: no cover
    openai_responses = None  # type: ignore

import datasets

from marin.rl.envs.math_env import MathEnv
from marin.rl.datatypes import InferenceEndpoint, RolloutGroup


@pytest.mark.skipif(openai_responses is None, reason="openai_responses not installed")
@openai_responses.mock()  # type: ignore[arg-type]
def test_math_env_rollout(openai_mock, monkeypatch):  # type: ignore[valid-type]
    """Ensure MathEnv produces a correctly graded rollout group."""

    # ------------------------------------------------------------------
    # Prepare mock dataset (train & test identical for simplicity)
    # ------------------------------------------------------------------
    example = {
        "problem": (
            "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. "
            "The probability of the spinner landing on $A$ is \\frac{1}{3} and the probability "
            "of the spinner landing on $B$ is \\frac{5}{12}.  What is the probability of the "
            "spinner landing on $C$? Express your answer as a common fraction."
        ),
        "level": "Level 1",
        "type": "Counting & Probability",
        "solution": (
            "The spinner is guaranteed to land on exactly one of the three regions, so we know that "
            "the sum of the probabilities of it landing in each region will be 1. If we let the probability "
            "of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, "
            "from which we have $x=\\boxed{\\frac{1}{4}}$."
        ),
    }
    fake_dataset = {"train": [example], "test": [example]}

    # Monkeypatch datasets.load_dataset to return our fake dataset.
    def _fake_load_dataset(name, *_, **__):
        assert name == "mock"
        return fake_dataset

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    # ------------------------------------------------------------------
    # Prepare mocked OpenAI response (correct answer inside <answer> tags)
    # ------------------------------------------------------------------
    openai_mock.chat.completions.create.response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "content": "Sure! <think>some reasoning</think> <answer>\\frac{1}{4}</answer>",
                    "role": "assistant",
                },
            }
        ],
    }

    # ------------------------------------------------------------------
    # Collect rollouts emitted by the environment
    # ------------------------------------------------------------------
    collected: deque[RolloutGroup] = deque()

    def sink(groups):  # type: ignore[override]
        collected.extend(groups)

    env = MathEnv(
        inference=InferenceEndpoint("https://api.openai.com/v1"),
        rollout_sink=sink,  # type: ignore[arg-type]
        data_source="mock",
        split="train",
        max_iters=1,
        api_key="sk-fake",
        seed=123,
    )

    asyncio.run(env.run())

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert len(collected) == 1
    group = collected.pop()
    assert group.metadata["correct"] is True
    assert group.rollouts[0].turns[1].reward == 1.0

    # The mocked endpoint should have been called exactly once
    assert openai_mock.chat.completions.create.route.call_count == 1
