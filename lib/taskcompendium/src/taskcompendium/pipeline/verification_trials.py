# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist control trials and their verified routing evidence."""

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import Source
from taskcompendium.pipeline.machine_routes import RouteEvidence, route_evidence
from taskcompendium.pipeline.models import CheckStatus, VerificationReport


class VerificationTrial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    attempt: int
    status: CheckStatus
    report: VerificationReport
    """The trial's control report; each check carries the backend verdicts it produced."""


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


def result_route_evidence(result: SampleResult) -> RouteEvidence | None:
    """The passing trial's verdicts as routing evidence; None when no trial passed with verdicts."""
    for trial in (*result.trials, *(previous.trial for previous in result.previous_trials)):
        verdicts = [verdict for check in trial.report.checks for verdict in check.verdicts]
        if trial.status is CheckStatus.PASS and verdicts:
            return route_evidence(verdicts)
    return None
