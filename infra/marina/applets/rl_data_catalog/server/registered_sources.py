# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checked-in release registrations, independent of training integrations."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

REGISTERED_ORIGIN = "Registered releases"


class ReleaseHost(StrEnum):
    HUGGING_FACE = "Hugging Face"
    HARBOR_HUB = "Harbor Hub"
    GIT = "Git"


@dataclass(frozen=True)
class ReleaseReference:
    host: ReleaseHost
    repository: str
    revision: str


@dataclass(frozen=True)
class ExecutionContract:
    environment: str
    type: Literal["RLVR", "Alignment", "Agentic"]
    turns: Literal["Single-turn", "Multi-turn"]
    tools: Literal["Disabled", "Allowed"]
    agent: str
    scoring: str


@dataclass(frozen=True)
class RegisteredSource:
    name: str
    release: ReleaseReference
    version: str
    revised_at: str
    url: str
    counts: tuple[tuple[str, int], ...]
    count_basis: str
    count_url: str
    execution: ExecutionContract
    classification_basis: str
    verifier_revision: str
    verifier_url: str
    family: str
    family_url: str
    is_benchmark: bool
    benchmark_basis: str
    license: str
    notes: str
    usage_url: str = ""
    validation_url: str = ""
    upstream_url: str = ""
    license_url: str = ""
    paper_url: str = ""


REGISTERED_SOURCES: tuple[RegisteredSource, ...] = ()
