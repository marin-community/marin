# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared decoding configuration for RL rollout generation."""

import logging
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal

logger = logging.getLogger(__name__)


class DecodingStrategy(StrEnum):
    """High-level decoding strategies used by RL rollouts."""

    SAMPLE = "sample"
    GREEDY = "greedy"


@dataclass(frozen=True)
class RolloutDecodingTrace:
    """Concrete decoding configuration attached to a generated rollout."""

    strategy: str
    temperature: float
    top_k: int | None
    top_p: float | None
    min_p: float | None
    repetition_penalty: float | None
    presence_penalty: float | None
    frequency_penalty: float | None
    max_output_tokens: int
    min_output_tokens: int | None
    stop_strings: tuple[str, ...] | None
    stop_token_ids: tuple[int, ...] | None
    ignore_eos: bool
    seed: int | None


@dataclass(frozen=True)
class DecodingConfig:
    """How one completion should be generated for an RL rollout."""

    strategy: DecodingStrategy = DecodingStrategy.SAMPLE
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_output_tokens: int = 512
    min_output_tokens: int | None = None
    stop_strings: list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self):
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.min_output_tokens is not None and self.min_output_tokens < 0:
            raise ValueError("min_output_tokens must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when set")
        if self.stop_strings is not None and self.stop_token_ids is not None:
            raise ValueError("Specify at most one of stop_strings or stop_token_ids")

    def with_temperature(self, temperature: float) -> "DecodingConfig":
        """Return a copy with a different temperature."""
        return replace(self, temperature=temperature)

    def as_trace(self) -> RolloutDecodingTrace:
        """Convert to a rollout-stable decoding trace."""
        return RolloutDecodingTrace(
            strategy=self.strategy.value,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            max_output_tokens=self.max_output_tokens,
            min_output_tokens=self.min_output_tokens,
            stop_strings=tuple(self.stop_strings) if self.stop_strings is not None else None,
            stop_token_ids=tuple(self.stop_token_ids) if self.stop_token_ids is not None else None,
            ignore_eos=self.ignore_eos,
            seed=self.seed,
        )


@dataclass
class SamplingParams:
    """Parameters for sampling rollout batches from an environment."""

    n_prompts: int = 8
    n_generations_per_prompt: int = 4
    train_decoding: DecodingConfig = field(default_factory=DecodingConfig)
    eval_decoding: DecodingConfig | None = None

    def __post_init__(self):
        train_decoding = self.train_decoding
        if train_decoding.strategy == DecodingStrategy.GREEDY:
            logger.warning("SamplingParams.train_decoding is greedy. Greedy decoding is generally not useful for RL.")
        if train_decoding.temperature < 1e-4:
            logger.warning(
                "SamplingParams.train_decoding.temperature is very low (%f). Low-temperature decoding is generally "
                "not useful for RL training as it limits exploration.",
                train_decoding.temperature,
            )
        if train_decoding.top_k == 1:
            logger.warning("SamplingParams.train_decoding.top_k is 1. Greedy decoding is generally not useful for RL.")

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens across train and eval decoding."""
        if self.eval_decoding is None:
            return self.train_decoding.max_output_tokens
        return max(self.train_decoding.max_output_tokens, self.eval_decoding.max_output_tokens)


def default_eval_decoding(train_decoding: DecodingConfig) -> DecodingConfig:
    """Derive a deterministic default eval decoding config from train decoding."""
    return replace(
        train_decoding,
        strategy=DecodingStrategy.GREEDY,
        temperature=0.0,
        top_k=None,
        top_p=None,
        min_p=None,
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        seed=None,
    )


def resolve_decoding_for_mode(sampling_params: SamplingParams, mode: Literal["train", "eval"]) -> DecodingConfig:
    """Resolve the decoding config used for a particular rollout mode."""
    if mode == "train":
        return sampling_params.train_decoding
    return sampling_params.eval_decoding or default_eval_decoding(sampling_params.train_decoding)


def stop_strings_for_decoding(decoding: DecodingConfig, tokenizer) -> list[str] | None:
    """Return stop strings for backends that only accept string stop sequences."""
    if decoding.stop_strings is not None:
        return decoding.stop_strings
    if decoding.stop_token_ids is None:
        return None
    return [tokenizer.decode([token_id]) for token_id in decoding.stop_token_ids]
