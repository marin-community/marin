# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval policy releases and run compliance files, as stored beside eval records.

A release pins, for each baseline and evaluation of a policy version, the exact run that represents it. Its
manifest is written once, at ``{CW_RECORDS_PREFIX}/policy-releases/{version}.json``; the git tag
``eval-policy-{version}`` at the manifest's commit makes it the release. A compliance file, written beside a
run's record by ``policy verify``, says whether that run complies with its policy.
"""

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import prefix_join

from evalrecords.records import CW_RECORDS_PREFIX

RELEASES_PREFIX = prefix_join(CW_RECORDS_PREFIX, "policy-releases")
RELEASE_TAG_PREFIX = "eval-policy-"
COMPLIANCE_FILE = "policy-compliance.json"


class ReleasedRun(BaseModel):
    """One run a release pins: which model and evaluation it stands for, where it is, and what it scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    evaluation: str
    run_id: str
    records_prefix: str
    commit: str
    """The full commit the run was launched from."""
    record_sha256: str
    """The digest of the run's ``record.json`` bytes when it was released."""
    archive_version: str
    """The run archive's HEAD version when it was released."""
    scores: dict[str, dict[str, float]]
    """The record's metrics, by task."""
    carried_from: str | None = None
    """The release this run was carried over from unchanged, for a patch."""


class PolicyRelease(BaseModel):
    """A released policy version: its commit, the release it patches if any, and its runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    commit: str
    base: str | None
    runs: tuple[ReleasedRun, ...]

    def content(self) -> bytes:
        return self.model_dump_json(indent=2).encode()


class Verdict(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class RunCompliance(BaseModel):
    """What ``policy verify`` found for one run, from a clean checkout of the run's commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    run_id: str
    commit: str
    full: bool
    """False for a smoke run, which caps items per task."""
    record_sha256: str
    verdict: Verdict
    problems: tuple[str, ...]
    warnings: tuple[str, ...]


def release_tag(version: str) -> str:
    return f"{RELEASE_TAG_PREFIX}{version}"


def release_path(version: str) -> str:
    return prefix_join(RELEASES_PREFIX, f"{version}.json")


def compliance_path(records_prefix: str, run_id: str) -> str:
    return prefix_join(prefix_join(records_prefix, run_id), COMPLIANCE_FILE)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
