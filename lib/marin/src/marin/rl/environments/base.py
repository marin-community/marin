# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from marin.rl.environments.inference_ctx.base import BaseInferenceContext
from marin.rl.types import RolloutGroup

logger = logging.getLogger(__name__)


class MarinEnv(ABC):
    """Abstract base class for RL environments.

    Environments manage datasets, generate responses, and evaluate them.
    Subclasses must implement sample() method.
    """

    @abstractmethod
    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | list[int] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Sample examples, generate responses, and create rollouts.

        Args:
            inference_ctx: Context for generating responses from the model
            n_examples: Number of examples to sample
            n_generations: Number of generations per example
            temperature: Sampling temperature for generation
            prng_key: JAX random key for sampling
            mode: "train" or "eval" - which dataset to sample from
            max_tokens: Maximum number of tokens to generate
            top_k: Top-k sampling parameter
            stop: Stop tokens to use for generation
            system_prompt: Optional system prompt to use for generation

        Returns:
            Tuple of (rollout_groups, metrics)
        """
        ...


class FiniteDatasetEnv(MarinEnv, ABC):
    """Base class for finite train/eval dataset RL environments.

    Subclasses own dataset loading, prompt rendering, and answer scoring. This
    base class owns explicit-index sampling so rollout and eval scheduling can
    reason about finite split order without duplicating inference plumbing.
    """

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Stable name used in rollout records and metrics."""
        ...

    @abstractmethod
    def train_len(self) -> int:
        """Number of examples in the training split."""
        ...

    @abstractmethod
    def eval_len(self) -> int:
        """Number of examples in the evaluation split."""
        ...

    @abstractmethod
    def train_examples_by_indices(self, indices: Sequence[int]) -> list[Any]:
        """Return train examples in the same order as ``indices``."""
        ...

    @abstractmethod
    def eval_examples_by_indices(self, indices: Sequence[int]) -> list[Any]:
        """Return eval examples in the same order as ``indices``."""
        ...

    @abstractmethod
    def inference_prompt_for_example(self, example: Any) -> str | list[dict[str, str]]:
        """Prompt/messages sent to the inference engine for one example."""
        ...

    @abstractmethod
    def rollout_prompt_for_example(self, example: Any) -> str:
        """Prompt text stored in the rollout for trainer tokenization/logging."""
        ...

    @abstractmethod
    def example_id(self, example: Any) -> str:
        """Stable example id for one dataset example."""
        ...

    @abstractmethod
    def score_choice(
        self, example: Any, response_text: str, finish_reason: str, tokenizer
    ) -> tuple[float, float, float]:
        """Return ``(reward, format_score, correctness_score)`` for one generated choice."""
        ...

    def split_len(self, mode: str) -> int:
        """Return split length for ``mode``."""
        if mode == "train":
            return self.train_len()
        if mode == "eval":
            return self.eval_len()
        raise ValueError(f"Unsupported mode: {mode}")

    def examples_by_indices(self, mode: str, indices: Sequence[int]) -> list[Any]:
        """Return split examples in requested order."""
        if mode == "train":
            return self.train_examples_by_indices(indices)
        if mode == "eval":
            return self.eval_examples_by_indices(indices)
        raise ValueError(f"Unsupported mode: {mode}")

    def _validate_indices(self, mode: str, indices: Sequence[int]) -> list[int]:
        split_len = self.split_len(mode)
        normalized = [int(idx) for idx in indices]
        for idx in normalized:
            if idx < 0 or idx >= split_len:
                raise IndexError(f"{mode} index {idx} is out of bounds for length {split_len}")
        return normalized

    def sample_by_indices(
        self,
        inference_ctx: BaseInferenceContext,
        indices: Sequence[int],
        n_generations: int,
        temperature: float,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | list[int] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Generate and score examples selected by explicit split indices."""
        normalized_indices = self._validate_indices(mode, indices)
        examples = self.examples_by_indices(mode, normalized_indices)
        prompts = [self.inference_prompt_for_example(example) for example in examples]

        completions = inference_ctx.batch_completions(
            prompts=prompts,
            temperature=temperature,
            n=n_generations,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            system_prompt=system_prompt,
        )

        rollout_groups: list[RolloutGroup] = []
        total_choices = 0
        reward_sum = 0.0
        format_sum = 0.0
        correct_sum = 0.0
        response_token_count = 0
        truncated_count = 0

        for example, completion in zip(examples, completions, strict=True):
            group_rollouts = []
            for choice in completion.choices:
                response_text = choice.message.content or ""
                reward, format_score, correct_score = self.score_choice(
                    example=example,
                    response_text=response_text,
                    finish_reason=choice.finish_reason,
                    tokenizer=inference_ctx.tokenizer,
                )
                rollout = inference_ctx.create_rollout_from_choice(
                    prompt=self.rollout_prompt_for_example(example),
                    choice=choice,
                    env_name=self.env_name,
                    env_example_id=self.example_id(example),
                    reward=reward,
                    correctness_reward=correct_score,
                    temperature=temperature,
                    top_k=top_k,
                    system_prompt=system_prompt,
                )
                group_rollouts.append(rollout)
                total_choices += 1
                reward_sum += reward
                format_sum += format_score
                correct_sum += correct_score
                response_token_count += rollout.response_tokens.size

                if choice.finish_reason == "length":
                    truncated_count += 1

            if group_rollouts:
                rollout_groups.append(RolloutGroup(rollouts=group_rollouts))

        if total_choices == 0:
            raise RuntimeError("Inference context returned no choices; cannot compute metrics")

        prefix = f"{self.env_name}.{mode}"
        metrics = {
            f"{prefix}_mean_reward": reward_sum / total_choices,
            f"{prefix}_format_accuracy": format_sum / total_choices,
            f"{prefix}_correct_accuracy": correct_sum / total_choices,
            f"{prefix}_mean_response_tokens": response_token_count / total_choices,
            f"{prefix}_total_responses": float(total_choices),
            f"{prefix}_sampled_examples": float(len(examples)),
            f"{prefix}_truncated_percentage": float(truncated_count) / total_choices,
        }
        return rollout_groups, metrics

    def sample(
        self,
        inference_ctx: BaseInferenceContext,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | list[int] | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutGroup], dict[str, float]]:
        """Compatibility wrapper that samples finite split indices with a PRNG."""
        split_len = self.split_len(mode)
        if split_len == 0:
            raise ValueError(f"No examples available for mode '{mode}'")

        n_to_sample = min(n_examples, split_len)
        rng = np.random.default_rng(extract_seed(prng_key))
        indices = rng.choice(split_len, size=n_to_sample, replace=False)
        return self.sample_by_indices(
            inference_ctx=inference_ctx,
            indices=[int(idx) for idx in indices],
            n_generations=n_generations,
            temperature=temperature,
            mode=mode,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            system_prompt=system_prompt,
        )


@dataclass
class EnvConfig:
    """Configuration for an environment."""

    env_class: str
    """Fully qualified class name of the environment, e.g. 'marin.rl.environments.math.MathEnvironment'."""

    env_args: dict
    """Arguments to pass to the environment constructor."""


def load_environment_from_spec(config: EnvConfig) -> MarinEnv:
    """Load an environment from the given configuration."""
    env_class = config.env_class
    env_args = config.env_args
    # Dynamically import the environment class
    module_name, class_name = env_class.rsplit(".", 1)
    env_module = __import__(module_name, fromlist=[class_name])
    env_class = getattr(env_module, class_name)

    # TODO(power) - thread random seed from the rollout worker.
    return env_class(**env_args)


def extract_seed(prng_key) -> int:
    """Extract an integer seed from either a JAX PRNG key or an integer."""
    if isinstance(prng_key, int):
        return prng_key
    # It's a JAX key - extract seed using JAX
    return jax.random.randint(prng_key, (), 0, 1_000_000).item()
