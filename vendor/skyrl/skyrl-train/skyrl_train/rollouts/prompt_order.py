# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prompt selection and learning hooks used by the rollout loop."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence, Set
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


_SCORE_SPREAD = 0.05


@dataclass(frozen=True)
class JudgedGroup:
    """A prompt group that batch selection judged in one step, kept or discarded, with its per-sample rewards.

    ``uid`` is the filtered dataset row. ``rewards`` carries the optimization signal.
    Missing verifier grades stay absent from ``verified_scores`` even when training assigns a penalty.
    """

    uid: str
    rewards: tuple[float, ...]
    verified_scores: tuple[float | None, ...] | None


@dataclass(frozen=True)
class PromptFeedback:
    """The outcome of one dispatched training group, including work spent on failures."""

    lease_id: str
    uid: str
    dispatch_step: int
    sample_count: int
    verified_scores: tuple[float | None, ...] | None
    generated_tokens: int
    nonzero_advantage_tokens: int | None

    @property
    def fully_verified(self) -> bool:
        scores = self.verified_scores
        return (
            scores is not None
            and len(scores) == self.sample_count >= 2
            and all(score is not None and math.isfinite(score) and 0 <= score <= 1 for score in scores)
        )

    @property
    def useful(self) -> bool:
        """Check for full verification, score spread of at least 0.05, and nonzero training advantage."""
        if not self.fully_verified or not self.nonzero_advantage_tokens:
            return False
        scores = [score for score in self.verified_scores or () if score is not None]
        return max(scores) - min(scores) >= _SCORE_SPREAD


class SamplingFeedbackPhase(StrEnum):
    """When the rollout loop records feedback for tagged sampling receipts."""

    NONE = "none"
    ADMISSION = "admission"
    OPTIMIZER = "optimizer"


@dataclass(frozen=True)
class PromptSelection:
    """The eligible rows of the full dataset or a leased family."""

    family: str | None
    rows: Sequence[int]
    live: Collection[str]
    quarantined: Set[str]
    eligible: Callable[[int], bool]
    draw: Callable[[Callable[[], int]], int | None]


class PromptOrder(ABC):
    """Select prompts and receive feedback at the phases relevant to the policy."""

    feedback_phase = SamplingFeedbackPhase.NONE

    @property
    def retry_stale(self) -> bool:
        return True

    @property
    def supports_family_leases(self) -> bool:
        return False

    @abstractmethod
    def next_index(self) -> int: ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...

    def select_index(self, selection: PromptSelection) -> int | None:
        return selection.draw(self.next_index)

    def for_pool(self, num_rows: int) -> PromptOrder | None:
        """Return an independent order when a family needs its own row sequence."""
        return None

    def assign(self, uid: str) -> str | None:
        """Tag the dispatch with its source when the order emits sampling receipts."""
        return None

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Observe the groups judged at admission."""
        return {}

    def finish_step(self, step: int, groups: Sequence[PromptFeedback]) -> dict[str, float]:
        """Observe optimizer participation and outcomes after the step."""
        return {}
