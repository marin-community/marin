"""Deterministic, independently sampled per-turn Pi execution decisions."""

import hashlib
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class LinearTeacherProbability(BaseModel):
    """Teacher execution probability at a leased rollout iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scheduler: Literal["linear"]
    start: float
    end: float
    duration_iterations: int

    @model_validator(mode="after")
    def valid_range(self) -> "LinearTeacherProbability":
        if (
            not (0 <= self.end <= self.start <= 1)
            or type(self.duration_iterations) is not int
            or self.duration_iterations <= 0
        ):
            raise ValueError("Pi turn mixture requires decreasing probabilities in [0, 1] and positive duration")
        return self

    def value(self, policy_step: int) -> float:
        """Interpolate by rollout iteration, clamping after the scheduled duration."""
        if type(policy_step) is not int or policy_step < 0:
            raise ValueError("Pi turn mixture needs a nonnegative leased policy step")
        return (
            self.start + (self.end - self.start) * min(policy_step, self.duration_iterations) / self.duration_iterations
        )


class PiTurnMixtureSchedule(BaseModel):
    """Plan independently drawn executors without observing student actions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_execution_probability: LinearTeacherProbability

    def executors(self, *, trial_id: str, policy_step: int, max_turns: int) -> tuple[str, ...]:
        """Reproduce per-turn Bernoulli choices for one leased physical trial."""
        if not trial_id or type(max_turns) is not int or max_turns <= 0:
            raise ValueError("Pi turn mixture requires a trial identity and positive turn limit")
        probability = self.teacher_execution_probability.value(policy_step)
        seed = hashlib.sha256(f"pi-turn-mixture-v1:{trial_id}:{policy_step}".encode()).digest()
        rng = random.Random(int.from_bytes(seed))
        return tuple("teacher" if rng.random() < probability else "student" for _ in range(max_turns))
