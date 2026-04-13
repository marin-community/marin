# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from marin.rl.types import Rollout


class TestTokenizer:
    """Tiny deterministic tokenizer for environment tests."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, token_ids) -> str:
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        del tokenize, add_generation_prompt
        return [ord(char) for char in messages[0]["content"]]


def create_test_chat_completion(prompt: str, responses: list[str]) -> ChatCompletion:
    """Create a minimal chat completion with one choice per response."""
    choices = [
        Choice(
            finish_reason="stop",
            index=index,
            message=ChatCompletionMessage(role="assistant", content=response_text),
            logprobs=None,
        )
        for index, response_text in enumerate(responses)
    ]
    completion_tokens = sum(len(response_text) for response_text in responses)
    return ChatCompletion(
        id=f"chatcmpl-test-{hash(prompt)}",
        choices=choices,
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=len(prompt),
            total_tokens=len(prompt) + completion_tokens,
        ),
    )


class DummyInferenceContext:
    """Inference context with deterministic prompt -> response mapping."""

    def __init__(self, responses_by_prompt: dict[str, list[str]], default_responses: list[str] | None = None):
        self.tokenizer = TestTokenizer()
        self.responses_by_prompt = responses_by_prompt
        self.default_responses = default_responses or ["fallback"]
        self.last_request: dict[str, Any] | None = None

    def batch_completions(
        self,
        prompts,
        temperature,
        n,
        max_tokens=None,
        top_k=None,
        stop=None,
        system_prompt=None,
    ):
        self.last_request = {
            "prompts": prompts,
            "temperature": temperature,
            "n": n,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "stop": stop,
            "system_prompt": system_prompt,
        }
        completions = []
        for prompt in prompts:
            responses = list(self.responses_by_prompt.get(prompt, self.default_responses))
            if len(responses) < n:
                responses.extend([responses[-1]] * (n - len(responses)))
            completions.append(create_test_chat_completion(prompt, responses[:n]))
        return completions

    def create_rollout_from_choice(
        self,
        prompt,
        choice,
        env_name,
        env_example_id,
        reward,
        temperature,
        top_k=None,
        system_prompt=None,
        correctness_reward=None,
    ):
        del system_prompt
        prompt_tokens = np.array([ord(char) for char in prompt], dtype=np.int32)
        response_text = choice.message.content or ""
        response_tokens = np.array([ord(char) for char in response_text], dtype=np.int32)
        response_logprobs = np.full(len(response_tokens), -1.0, dtype=np.float32)
        token_rewards = np.full(len(response_tokens), reward, dtype=np.float32)

        return Rollout(
            env_name=env_name,
            env_example_id=env_example_id,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            response_logprobs=response_logprobs,
            token_rewards=token_rewards,
            episode_reward=float(reward),
            correctness_reward=correctness_reward,
            temperature=temperature,
            top_k=top_k,
            is_truncated=choice.finish_reason == "length",
        )


class FakeReasoningGymDataset:
    """Minimal procedural dataset for Reasoning Gym adapter tests."""

    def __init__(self, entries: list[dict[str, Any]], *, seed: int, size: int):
        self._entries = [copy.deepcopy(entry) for entry in entries]
        self.seed = seed
        self.size = size

    def __len__(self) -> int:
        return min(self.size, len(self._entries))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return copy.deepcopy(self._entries[idx])

    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> float:
        metadata = entry.get("metadata", {})
        score_map = metadata.get("score_map")
        if isinstance(score_map, dict):
            return float(score_map.get(answer, 0.0))
        return 1.0 if answer == entry.get("answer") else 0.0


@dataclass
class FakeReasoningGymModules:
    dataset_spec_cls: type
    create_calls: list[dict[str, Any]]


def install_fake_reasoning_gym(
    monkeypatch,
    *,
    datasets_by_seed: dict[int, list[dict[str, Any]]] | None = None,
) -> FakeReasoningGymModules:
    """Install fake reasoning_gym modules into sys.modules for testing."""

    class DatasetSpec:
        def __init__(self, name: str, weight: float, config: dict[str, Any]):
            self.name = name
            self.weight = weight
            self.config = config

        def validate(self) -> None:
            if not self.name:
                raise ValueError("Dataset name cannot be empty")
            if self.weight <= 0:
                raise ValueError("Weight must be positive")

    create_calls: list[dict[str, Any]] = []
    seeded_entries = datasets_by_seed or {}

    def create_dataset(name: str, **kwargs):
        create_calls.append({"name": name, "kwargs": copy.deepcopy(kwargs)})
        seed = kwargs["seed"]
        size = kwargs["size"]
        if seed in seeded_entries:
            entries = seeded_entries[seed]
        else:
            entries = [
                {
                    "question": f"{name} question {seed}",
                    "answer": "ok",
                    "metadata": {"source_index": 0, "source_dataset": name},
                }
            ]
        return FakeReasoningGymDataset(entries, seed=seed, size=size)

    reasoning_gym_module = ModuleType("reasoning_gym")
    reasoning_gym_module.create_dataset = create_dataset
    composite_module = ModuleType("reasoning_gym.composite")
    composite_module.DatasetSpec = DatasetSpec

    monkeypatch.setitem(sys.modules, "reasoning_gym", reasoning_gym_module)
    monkeypatch.setitem(sys.modules, "reasoning_gym.composite", composite_module)

    return FakeReasoningGymModules(dataset_spec_cls=DatasetSpec, create_calls=create_calls)
