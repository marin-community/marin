# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A decision client for tests: each candidate text reads at a fixed level, and every request is kept."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from taskforge.spec.decision import LEVELS, ScoreQuestion


def level_probabilities(level: float) -> tuple[float, ...]:
    """Probabilities over ``LEVELS`` whose expected level is ``level``, split between its two neighbours."""
    low = min(int(level), len(LEVELS) - 1)
    probabilities = [0.0] * len(LEVELS)
    probabilities[low] = 1 - (level - low)
    if level > low:
        probabilities[low + 1] = level - low
    return tuple(probabilities)


@dataclass
class FakeDecisionClient:
    """Reads ``levels[candidate]`` for a known candidate and ``default`` for any other.

    The candidate is the state's ``candidate_key`` value, the default template's solver output.
    """

    levels: Mapping[str, float]
    default: float = 0.0
    candidate_key: str = "z_candidate_answer"
    requests: list[tuple[Mapping[str, JsonValue], Mapping[str, ScoreQuestion]]] = field(default_factory=list)

    async def decide(
        self, state: Mapping[str, JsonValue], questions: Mapping[str, ScoreQuestion]
    ) -> Mapping[str, tuple[float, ...]]:
        self.requests.append((state, questions))
        candidate = state[self.candidate_key]
        assert isinstance(candidate, str)
        level = self.levels.get(candidate, self.default)
        return {name: level_probabilities(level) for name in questions}
