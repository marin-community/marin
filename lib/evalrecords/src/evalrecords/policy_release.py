# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval policy releases and run compliance files, as stored beside eval records.

A release pins, for each baseline and evaluation of a policy version, the exact run that represents it. Its
manifest is written once, at ``{CW_RECORDS_PREFIX}/policy-releases/{version}.json``; the git tag
``eval-policy-{version}`` at the manifest's commit makes it the release. A compliance file, written beside a
run's record by ``policy verify``, says whether that run complies with its policy.
"""

import hashlib
from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import prefix_join

from evalrecords.eval_policy import record_policy_violations
from evalrecords.records import CW_RECORDS_PREFIX, EvalRunRecord, record_path

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


def record_sha256(records_prefix: str, run_id: str) -> str:
    """The digest of a run's ``record.json`` bytes, as a release or compliance file pins it."""
    with open_url(record_path(records_prefix, run_id), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class ReleaseAdmission(BaseModel):
    """Which runs of released policy versions a published cohort admits.

    A released version's cohort holds the runs its manifest pins, scored as the manifest records them, and the
    version's other runs that ``policy verify`` found valid and full at the release commit, such as a training
    checkpoint evaluated after the release. Other versions keep their own admission rules.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    releases: dict[str, PolicyRelease] = {}
    """By version."""
    compliance: dict[str, RunCompliance] = {}
    """By run ID."""

    def violations(self, record: EvalRunRecord) -> tuple[str, ...]:
        """Why a record is not admitted to its cohort; empty when it is."""
        release = self.releases.get(record.version) if record.version is not None else None
        if release is None:
            return record_policy_violations(record)
        if any(run.run_id == record.run_id for run in release.runs):
            return ()
        compliance = self.compliance.get(record.run_id)
        if compliance is None or compliance.version != record.version:
            return ("not pinned by the release and not verified",)
        reasons = [f"verify: {problem}" for problem in compliance.problems]
        if compliance.verdict is not Verdict.VALID and not reasons:
            reasons.append("verify found it invalid")
        if not compliance.full:
            reasons.append("a smoke run")
        if record.provenance.git_sha != release.commit or compliance.commit != release.commit:
            reasons.append(f"ran at {record.provenance.git_sha[:12]}, not the release commit {release.commit[:12]}")
        return tuple(reasons)

    def view(self, records: Sequence[EvalRunRecord], version: str | None) -> list[EvalRunRecord]:
        """The records as a cohort of ``version`` sees them: a release's pinned runs, including runs a patch carries
        from its base, carry its version and the scores its manifest records."""
        release = self.releases.get(version) if version is not None else None
        if release is None:
            return list(records)
        pinned = {run.run_id: run for run in release.runs}
        return [
            (
                record.model_copy(
                    update={
                        "version": version,
                        "metrics": pinned[record.run_id].scores,
                        "canonical_metrics": pinned[record.run_id].scores,
                    }
                )
                if record.run_id in pinned
                else record
            )
            for record in records
        ]


NO_RELEASES = ReleaseAdmission()
