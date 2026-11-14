# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed progress receipts and the portable preflight input contract."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel


@dataclass(frozen=True)
class ReleaseInputs:
    source_counts_sha256: str
    source_counts_json: str
    oracle_inventory_sha256: str | None = None


@dataclass(frozen=True)
class CommandResult:
    gate: str
    command: list[str]
    exit_code: int
    log: str


@dataclass(frozen=True)
class ReleaseTarget:
    id: str
    origin: str
    service_url: str


@dataclass(frozen=True)
class ReleaseFailure:
    gate: str
    exception_type: str
    message: str


@dataclass
class ReleaseReceipt:
    phase: Literal["preflight", "hosted"]
    revision: int | None
    started_at: str
    target: ReleaseTarget
    production_id: str
    version: int = 1
    passed: bool = False
    gate: str = "arguments"
    source_revision: str | None = None
    finished_at: str | None = None
    inputs: ReleaseInputs | None = None
    production_version: int | None = None
    preview_base_version: int | None = None
    commands: list[CommandResult] = field(default_factory=list)
    package: dict[str, Any] | None = None
    evidence_before: dict[str, Any] | None = None
    evidence_after: dict[str, Any] | None = None
    expected_sources: dict[str, Any] | None = None
    browser: dict[str, Any] | None = None
    error: ReleaseFailure | None = None


class PreflightBaseline(BaseModel):
    """Required preflight fields for a later immutable hosted-revision check."""

    version: Literal[1]
    phase: Literal["preflight"]
    passed: Literal[True]
    source_revision: str
    inputs: ReleaseInputs
    production_version: int
    preview_base_version: int
    package: dict[str, Any]
    evidence_before: dict[str, Any]
    expected_sources: dict[str, Any]
    browser: dict[str, Any]
