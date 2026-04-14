# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verifier contracts for RL/post-training environments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from marin.rl.traces import EpisodeTrace


@dataclass(frozen=True)
class VerifierResult:
    """Structured verifier output for one trace."""

    reward: float
    token_rewards: np.ndarray | None = None
    correctness_reward: float | None = None
    passed: bool | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class VerifierSpec(Protocol):
    """Protocol for reusable verifier implementations."""

    @property
    def name(self) -> str:
        """Stable verifier identifier."""
        ...

    @property
    def version(self) -> str:
        """Stable verifier version."""
        ...

    def verify(self, trace: EpisodeTrace) -> VerifierResult:
        """Verify a prompt-level episode trace."""
        ...


@dataclass(frozen=True)
class VerifierMetadata:
    """Stable verifier identity for provenance."""

    name: str
    version: str = "v1"


def verifier_metrics(result: VerifierResult) -> Mapping[str, float]:
    """Return verifier metrics for logging or sidecar storage."""

    return result.metrics
