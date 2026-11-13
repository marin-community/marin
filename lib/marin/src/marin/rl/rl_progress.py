# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate gate results and retained progress records."""

from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


class Label(StrEnum):
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
    CONSTRAINT = "constraint"
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


Count = Annotated[StrictInt, Field(ge=0)]
Identity = Annotated[str, Field(min_length=1)]


class GateResult(BaseModel):
    model_config = ConfigDict(strict=True)
    comparisons: list[Comparison]
    invariants: list[Invariant]
    verdict: Verdict
    label: Label | None
    reason: str | None
    attempt_ids: list[Identity]
    replicates: Annotated[StrictInt, Field(gt=0)]


class RunSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    run_id: Identity
    attempt_id: Identity
    label: Identity
    series: str | None = None
    issue: StrictInt | None = None
    iris_job_id: str | None = None
    wandb_url: str | None = None
    engine: dict[str, Any] | None = None
    gate_result: GateResult | None = None
    gate: str | None = None
    failed_facts: list[str] = Field(default_factory=list)
    run_date: str | None
    started_at: str | None
    completed_at: str | None
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)] | None
    timing_scope: Identity
    status: Literal["succeeded", "failed"]
    optimizer_steps: Count | None
    source_revision: Identity
    data_identity: Identity
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
        return self
