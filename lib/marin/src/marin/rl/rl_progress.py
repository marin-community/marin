# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate retained RL progress records and evaluation identities."""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, field_validator, model_validator

Count = Annotated[StrictInt, Field(ge=0)]
Identity = Annotated[str, Field(min_length=1)]


def membership_sha256(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def protocol_sha256(protocol: dict[str, Any]) -> str:
    encoded = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class EvaluationResult(BaseModel):
    model_config = ConfigDict(strict=True)
    reward: FiniteFloat
    attempted: Count
    scored: Count

    @model_validator(mode="after")
    def valid_coverage(self):
        if self.scored > self.attempted:
            raise ValueError("Scored count exceeds attempted count")
        return self


class EvaluationIdentity(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    name: Identity
    version: Identity
    task_count: Annotated[StrictInt, Field(gt=0)]
    task_set_sha256: Identity | None
    protocol_sha256: Identity | None
    membership_sha256: str | None = None
    protocol: dict[str, Any] | None = None

    @model_validator(mode="after")
    def matching_evaluation_identity(self):
        if self.protocol is not None:
            if protocol_sha256(self.protocol) != self.protocol_sha256:
                raise ValueError("Evaluation protocol does not match its recorded identity")
        return self


class RunSummary(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow")
    run_id: Identity
    attempt_id: Identity
    label: Identity
    series: str | None = None
    issue: StrictInt | None = None
    iris_job_id: str | None = None
    wandb_url: str | None = None
    engine: dict[str, Any] | None = None
    resume_from: str | None = None
    verdict: Literal["pass", "fail", "invalid"] | None = None
    gate: str | None = None
    failed_facts: list[str] = Field(default_factory=list)
    min_scored_fraction: Annotated[FiniteFloat, Field(gt=0, le=1)] = 1.0
    run_date: str | None
    date_basis: Literal["submission_acknowledged", "launch_started"]
    started_at: str | None
    completed_at: str | None
    elapsed_seconds: Annotated[FiniteFloat, Field(ge=0)] | None
    timing_scope: Identity
    status: Literal["succeeded", "failed"]
    optimizer_steps: Count
    source_revision: Identity
    data_identity: Identity
    initializer_identity: Identity
    evaluation: EvaluationIdentity
    baseline: EvaluationResult | None
    final: EvaluationResult | None
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
        if self.final is not None and (
            self.baseline is None or (self.elapsed_seconds is not None and self.elapsed_seconds <= 0)
        ):
            raise ValueError("A final measurement requires a baseline and positive elapsed time when known")
        for phase in (self.baseline, self.final):
            if phase is not None and phase.attempted > self.evaluation.task_count:
                raise ValueError("Attempted count exceeds the fixed evaluation cohort")
        return self
