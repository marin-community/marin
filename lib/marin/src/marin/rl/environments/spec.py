# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment contracts for trace-aware rollout generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from marin.rl.traces import EpisodeTrace
from marin.rl.types import RolloutGroup


@dataclass(frozen=True)
class EnvironmentIdentity:
    """Stable task and verifier identity for an environment."""

    task_name: str
    task_version: str = "v1"
    verifier_name: str | None = None
    verifier_version: str | None = None


@dataclass(frozen=True)
class EnvironmentSample:
    """Rollout output plus cold-path trace metadata."""

    rollout_groups: list[RolloutGroup]
    metrics: dict[str, float | int] = field(default_factory=dict)
    traces: list[EpisodeTrace] = field(default_factory=list)
    identity: EnvironmentIdentity = field(default_factory=lambda: EnvironmentIdentity(task_name=""))


class EnvironmentSpec(Protocol):
    """Protocol for trace-aware Marin environments."""

    def environment_identity(self) -> EnvironmentIdentity:
        """Return stable task and verifier identity."""
        ...

    def sample(
        self,
        inference_ctx,
        n_examples: int,
        n_generations: int,
        temperature: float,
        prng_key,
        mode: str = "train",
        max_tokens: int | None = None,
        top_k: int | None = None,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
    ) -> EnvironmentSample:
        """Sample prompts, run generation, and return rollout groups plus traces."""
        ...


def environment_metrics(sample: EnvironmentSample) -> Mapping[str, float | int]:
    """Return environment metrics from a sample."""

    return sample.metrics
