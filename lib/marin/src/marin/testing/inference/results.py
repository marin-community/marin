# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical results and caller-controlled persistence for the four checks."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from marin.execution.artifact import write_artifact
from marin.testing.inference.endpoint import HttpObservation
from marin.testing.inference.goldens import ErrorLimits, Record


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class ProbabilityErrors(Record):
    """Per-position probability-error bounds and the fraction strictly above 0.075."""

    maximum: float
    p99: float
    p90: float
    fraction_above_0075: float


class ComparisonResult(Record):
    status: CheckStatus
    positions: int
    limits: ErrorLimits
    statistics: ProbabilityErrors | None = None
    worst: tuple[dict[str, JsonValue], ...] = ()


class CoverageResult(Record):
    status: CheckStatus | Literal["not_required"]
    observations: JsonValue = None


class CheckResult(Record):
    """A check's measurements, required coverage and retained HTTP observations."""

    status: CheckStatus
    test_spec: dict[str, JsonValue]
    run_metadata: dict[str, JsonValue]
    differences: dict[str, ComparisonResult] = Field(default_factory=dict)
    coverage: dict[str, CoverageResult] = Field(default_factory=dict)
    observations: JsonValue = None
    details: dict[str, JsonValue] = Field(default_factory=dict)
    requests: tuple[HttpObservation, ...] = ()
    seconds: float = 0


class CheckReport(Record):
    """Version 2 report; only four passed checks produce an overall pass.

    Each result carries its specification and observed run metadata, so results
    returned by individual checks retain the same context as ``TestSpec.run``.
    Saving is the only operation here that writes files.
    """

    schema_version: Literal[2] = 2
    reference: CheckResult
    batch: CheckResult
    cache: CheckResult
    decode: CheckResult
    status: CheckStatus = CheckStatus.INCOMPLETE

    @model_validator(mode="after")
    def derive_status(self) -> Self:
        statuses = (self.reference.status, self.batch.status, self.cache.status, self.decode.status)
        if CheckStatus.FAILED in statuses:
            status = CheckStatus.FAILED
        elif all(value == CheckStatus.PASSED for value in statuses):
            status = CheckStatus.PASSED
        else:
            status = CheckStatus.INCOMPLETE
        object.__setattr__(self, "status", status)
        return self

    def save(self, output_path: str) -> None:
        """Write the typed report to ``output_path/.artifact.json``."""
        write_artifact(self, output_path)
