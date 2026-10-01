# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed boundaries for the bounded Baby RSI feedback loop."""

from collections.abc import Callable, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict
from taskcompendium.models import TaskSpec


class TaskPrompt(BaseModel):
    """An abstract error description that can seed new tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    capability_id: str
    description: str


class PolicyState(BaseModel):
    """A model reference used by the dummy implementation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    learned_capabilities: tuple[str, ...]


class RolloutRecord(BaseModel):
    """The serializable fields that the feedback loop uses from one rollout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    capability_id: str
    response: str
    reward: float


class RolloutBuffer(BaseModel):
    """The graded responses consumed by a trainer or an analysis engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str
    rollouts: tuple[RolloutRecord, ...]


class TaskPromptSet(BaseModel):
    """Error descriptions returned by the analysis engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompts: tuple[TaskPrompt, ...]


class BabyRsiRound(BaseModel):
    """The observable state transition for one feedback round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_before: RolloutBuffer
    prompts: TaskPromptSet
    training_tasks: tuple[TaskSpec, ...]
    training_rollouts: RolloutBuffer
    policy_after: PolicyState
    evaluation_after: RolloutBuffer


class BabyRsiRun(BaseModel):
    """A finite sequence of feedback rounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial_evaluation: RolloutBuffer
    rounds: tuple[BabyRsiRound, ...]
    final_evaluation: RolloutBuffer


class EvaluationSummary(BaseModel):
    """Pass counts for one evaluation buffer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: int
    total: int


class RoundSummary(BaseModel):
    """The capabilities selected during one feedback round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int
    targeted_capabilities: tuple[str, ...]


class BabyRsiReport(BaseModel):
    """The final comparison written by the dummy artifact graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    initial: EvaluationSummary
    final: EvaluationSummary
    rounds: tuple[RoundSummary, ...]


TaskFilter = Callable[[TaskSpec], bool]


class ProblemGenerator(Protocol):
    """Generate TaskSpecs from error descriptions and an inference policy."""

    def generate_tasks(
        self,
        task_prompts: Sequence[TaskPrompt],
        inference_policy: PolicyState,
        task_filter: TaskFilter,
    ) -> tuple[TaskSpec, ...]: ...


class Trainer(Protocol):
    """Update a policy from a graded rollout buffer."""

    def train(self, buffer: RolloutBuffer) -> PolicyState: ...


class AnalysisEngine(Protocol):
    """Convert failed rollouts into abstract task prompts."""

    def analyze_failed_runs(self, rollouts: RolloutBuffer) -> TaskPromptSet: ...
