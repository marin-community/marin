# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist control trials and their verified task backends."""

from pydantic import BaseModel, ConfigDict, Field
from shellbox.machine import Backend

from taskcompendium.models import Source
from taskcompendium.pipeline.models import CheckResult, CheckStatus, TierVerdict
from taskcompendium.runtime.models import RolloutRecord


class VerificationTrial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    attempt: int
    status: CheckStatus
    checks: list[CheckResult]
    rollouts: tuple[RolloutRecord, ...]
    verdicts: tuple[TierVerdict, ...] = ()
    """Every tier the controls tried, in order, with each tier's outcome."""
    route_backends: tuple[Backend, ...] = ()
    """The passing task backend and higher admitted candidates, in ingest order."""


class TrialEvidence(BaseModel):
    """Identity and original execution provenance for one independent trial."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    task_sha256: str
    verification_identity: str
    attempt: int
    execution_id: str
    original_report: str
    reused_from: str | None = None


class HistoricalTrial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    trial: VerificationTrial
    evidence: TrialEvidence


class SampleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    source: Source
    trials: list[VerificationTrial]
    previous_trials: list[HistoricalTrial] = Field(default_factory=list)


def sample_result_routes(result: SampleResult) -> tuple[Backend, ...]:
    """The passing trial's task backend and higher admitted candidates."""
    for trial in (*result.trials, *(previous.trial for previous in result.previous_trials)):
        if trial.status is CheckStatus.PASS and trial.route_backends:
            return trial.route_backends
    return ()
