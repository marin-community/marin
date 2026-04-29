# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ScoringMode(StrEnum):
    """Supported prompt scoring modes."""

    UNSCORED = "unscored"
    MATH_BOXED = "math_boxed"


class SelectorName(StrEnum):
    """Built-in selector names for replayable sample-only experiments."""

    FIRST_SAMPLE = "first_sample"
    MAJORITY_VOTE = "majority_vote"
    NORMALIZED_LOGPROB = "normalized_logprob"


DEFAULT_REASONING_SELECTORS: tuple[SelectorName, ...] = (
    SelectorName.FIRST_SAMPLE,
    SelectorName.MAJORITY_VOTE,
    SelectorName.NORMALIZED_LOGPROB,
)


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Sampling configuration for candidate-pool generation."""

    num_candidates: int
    temperature: float
    top_p: float = 1.0
    max_gen_toks: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.max_gen_toks is not None and self.max_gen_toks <= 0:
            raise ValueError("max_gen_toks must be positive when set")


@dataclass(frozen=True)
class TestTimeScalingConfig:
    """Top-level sample-only run configuration for PR 1."""

    generation: CandidateGenerationConfig
    selectors: tuple[SelectorName, ...] = DEFAULT_REASONING_SELECTORS

    def __post_init__(self) -> None:
        if not self.selectors:
            raise ValueError("selectors must be non-empty")
