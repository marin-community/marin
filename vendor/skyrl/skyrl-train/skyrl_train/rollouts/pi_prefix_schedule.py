"""Deterministic student-prefix sampling for the teacher-completion Pi scenario.

The scheduled upper bound changes with the leased policy step. A prefix is
sampled once for each physical trial; a retry of the same lease uses the same
prefix. Executor labels are derived before Pi starts and never depend on a
student action, grader result, or a private teacher label.
"""

import hashlib
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictInt, model_validator


class LinearPrefixBound(BaseModel):
    """Linear, clamped upper bound indexed by the leased policy step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scheduler: Literal["linear"]
    start: StrictInt
    end: StrictInt
    duration_iterations: StrictInt

    @model_validator(mode="after")
    def valid_range(self) -> "LinearPrefixBound":
        if self.start < 0 or self.end < self.start or self.duration_iterations <= 0:
            raise ValueError("Pi prefix schedule requires nonnegative, increasing bounds and positive duration")
        return self

    def value(self, policy_step: int) -> int:
        """Return the upper bound, clamped after the last scheduled step."""
        if type(policy_step) is not int or policy_step < 0:
            raise ValueError("Pi prefix schedule needs a nonnegative leased policy step")
        step = min(policy_step, self.duration_iterations)
        return self.start + (self.end - self.start) * step // self.duration_iterations


class PiPrefixSchedule(BaseModel):
    """Uniform integer prefix length over a scheduled inclusive support."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution: Literal["uniform_int"]
    min: Literal[0]
    max: LinearPrefixBound

    def sample(self, *, trial_id: str, policy_step: int, max_turns: int) -> int:
        """Sample one reproducible student-prefix length for a physical trial."""
        if not trial_id or type(max_turns) is not int or max_turns <= 0 or self.max.end > max_turns:
            raise ValueError("Pi prefix schedule must fit the trial's positive turn limit")
        upper = self.max.value(policy_step)
        seed = hashlib.sha256(f"pi-prefix-v1:{trial_id}:{policy_step}".encode()).digest()
        return random.Random(int.from_bytes(seed)).randint(0, upper)

    def executors(self, *, trial_id: str, policy_step: int, max_turns: int) -> tuple[str, ...]:
        """Plan the full trial before either model observes an action."""
        prefix = self.sample(trial_id=trial_id, policy_step=policy_step, max_turns=max_turns)
        return ("student",) * prefix + ("teacher",) * (max_turns - prefix)
