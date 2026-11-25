# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate gate results and retained progress records."""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator

SHARED_TASKS_INVARIANT = "shared_tasks"
SHARED_TASK_FRACTION_INVARIANT = "shared_task_fraction"
EVALUATION_PROTOCOL_DIFFERS = "evaluation protocol differs"
GROUP_COMPARISON_INVARIANTS = frozenset({SHARED_TASKS_INVARIANT, SHARED_TASK_FRACTION_INVARIANT})

GRADER_REPORTS_DIR = "grader_reports"
GRADER_REPORT_FILENAME = "grader_report.json"


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


class Effect(StrEnum):
    IMPROVES = "improves"
    TRADE_OFF = "trade-off"
    NO_CHANGE = "no change"
    REGRESSION = "regression"


class Change(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    SAME = "same"
    UNAVAILABLE = "unavailable"
    CHANGED = "changed"


class ComparisonRole(StrEnum):
    OBJECTIVE = "objective"
    GUARD = "guard"
    REQUIREMENT = "requirement"
    REPORTED = "reported"
    SLICE = "slice"


class Invariant(BaseModel):
    name: str
    value: bool | int | float
    limit: bool | int | float
    comparator: Literal["<=", ">=", "=="] | None = None
    holds: bool
    replicate: int = 0


class EvaluationScore(BaseModel):
    profile: str
    step: int
    statistic: str
    final: bool
    comparability: str | None = None


class Comparison(BaseModel):
    metric: str
    role: ComparisonRole
    higher_is_better: bool | None = None
    run: float | None = None
    control: float | None = None
    delta: float | None = None
    delta_percent: float | None = None
    noise_percent: float | None = None
    tolerance: float | None = None
    sd: float | None = None
    df: int | None = None
    n_run: int
    n_control: int
    passes: bool | None = None
    change: Change | None = None
    tasks: int | None = None
    control_min: float | None = None
    control_max: float | None = None
    evaluation_score: EvaluationScore | None = Field(default=None, exclude_if=lambda value: value is None)


Count = Annotated[StrictInt, Field(ge=0)]
Identity = Annotated[str, Field(min_length=1)]


class GateResult(BaseModel):
    model_config = ConfigDict(strict=True)
    comparisons: list[Comparison]
    invariants: list[Invariant]
    verdict: Verdict
    effect: Effect | None
    reason: str | None
    attempt_ids: list[Identity]
    replicates: Annotated[StrictInt, Field(gt=0)]
    control_commit: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    control_version: Identity | None = Field(default=None, exclude_if=lambda value: value is None)


class GradingLatency(TypedDict):
    count: Count
    mean: FiniteFloat | None
    p95: FiniteFloat | None
    max: FiniteFloat | None


class SuccessSummary(TypedDict):
    attempted: Count
    verified: Count
    passed: Count
    verification_coverage: FiniteFloat
    success_rate: FiniteFloat | None


class SubsetSummary(SuccessSummary):
    subset_id: str
    data_sources: list[str]
    verification_statuses: dict[str, Count]
    grading_failures: dict[str, Count]
    rollout_failures: dict[str, Count]
    grading_latency_seconds: GradingLatency


class TaskSummary(SuccessSummary):
    task_id: str
    subset_ids: list[str]


class GraderReport(BaseModel):
    """Stored training and evaluation counts by subset and task."""

    model_config = ConfigDict(strict=True, extra="ignore")
    training: dict[str, list[SubsetSummary]] = Field(default_factory=dict)
    training_tasks: dict[str, list[TaskSummary]] = Field(default_factory=dict)
    evaluation: dict[str, list[SubsetSummary]] = Field(default_factory=dict)
    evaluation_tasks: dict[str, list[TaskSummary]] = Field(default_factory=dict)
    evaluation_duration_seconds: dict[str, FiniteFloat] = Field(default_factory=dict, exclude_if=lambda value: not value)


class ControlReplicate(BaseModel, frozen=True):
    commit: Identity
    version: Identity
    replicate: Count


class AttemptSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    run_id: Identity
    native_run_id: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    attempt_id: Identity
    label: Identity
    series: str | None = None
    model: Identity | None = None
    preset_version: Identity | None = None
    base_commit: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    base_preset_version: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    control: ControlReplicate | None = None
    control_preset: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    evaluation_data_identity: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    evaluation_task_set: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    evaluation_protocol_identity: Identity | None = Field(default=None, exclude_if=lambda value: value is None)
    is_control: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    evaldash_run_ids: tuple[Identity, ...] | None = Field(default=None, exclude=True)
    user: Identity | None = None
    branch: str | None = None
    commit: Identity | None = None
    settings: list[str] = Field(default_factory=list)
    launch_config: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    dirty: bool | None = None
    pr_number: Annotated[StrictInt, Field(gt=0)] | None = None
    cluster: Identity | None = None
    issue: StrictInt | None = None
    iris_job_id: str | None = None
    wandb_url: str | None = None
    grafana_url: str | None = None
    evaluations: list[Comparison] = Field(default_factory=list)
    grader_report: GraderReport | None = None
    engine: dict[str, Any] | None = None
    gate_result: GateResult | None = None
    gate: str | None = None
    failed_facts: list[str] = Field(default_factory=list)
    run_date: str | None
    started_at: str | None
    completed_at: str | None
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)] | None
    elapsed_excluding_eval_seconds: Annotated[FiniteFloat, Field(ge=0)] | None = None
    timing_scope: Identity
    status: Literal["running", "succeeded", "failed"]
    optimizer_steps: Count | None
    planned_updates: Count | None = None
    measured_updates: list[Count] | None = None
    source_revision: Identity
    data_identity: Identity | None
    initializer_identity: Identity
    artifact_uri: str | None
    evidence: list[str]

    @field_validator("run_date", "started_at", "completed_at")
    @classmethod
    def utc_timestamp(cls, value):
        if value is not None:
            parsed = datetime.fromisoformat(value)
            if parsed.utcoffset() != timedelta(0):
                raise ValueError("Run timestamps must explicitly use UTC")
        return value

    @model_validator(mode="after")
    def valid_outcome(self):
        if self.elapsed_seconds is not None and self.completed_at is None:
            raise ValueError("Terminal attempts require a completion timestamp")
        gate = self.gate_result
        if gate is not None and len(gate.attempt_ids) == gate.replicates and self.attempt_id in gate.attempt_ids:
            replicate = gate.attempt_ids.index(self.attempt_id)
            self.failed_facts = list(
                dict.fromkeys(
                    [
                        *self.failed_facts,
                        *(
                            fact.name
                            for fact in gate.invariants
                            if not fact.holds
                            and fact.replicate == replicate
                            and fact.name.rpartition("/")[2] not in GROUP_COMPARISON_INVARIANTS
                        ),
                    ]
                )
            )
        if self.failed_facts:
            self.status = "failed"
        return self
