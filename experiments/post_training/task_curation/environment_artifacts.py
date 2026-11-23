# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declared environment artifacts: the scripted in-region rebuild and the build-stage check.

Every TaskSpec declares the environment artifacts its execution needs, and the grading
census records each declared URI in ``GradingEnvironmentEvidence.declared_artifact_counts``.
The Atlas build stage that assembles declarations before publication runs
``verify_declared_environment_artifacts`` over the bound census receipts: a declared
artifact that exists in no region first triggers the scripted in-region rebuild through
the standard image builder, and the stage fails with the typed
``ExclusionReason.ENVIRONMENT_ARTIFACT_MISSING`` reason only when the rebuild fails.
"""

import re
from dataclasses import dataclass

from finestore.rl.status_taxonomy import ExclusionReason
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.difficulty.planner import RebuildEnvironmentArtifact
from experiments.post_training.task_curation.difficulty.state import REPAIR_BLOCKED, GapReason
from experiments.post_training.task_curation.grading_binding import GradingCensusReceipt
from experiments.post_training.task_curation.images.__main__ import declared_environments
from experiments.post_training.task_curation.images.build import (
    DEFAULT_REPOSITORY,
    EnvironmentBuild,
    build_environment,
)


# Unknown recipes and judge routes require evidence or a ruling. They must not
# be replaced by an environment or model that merely resembles the original.
def rebuild_artifact(action: RebuildEnvironmentArtifact) -> dict:
    path = StoragePath(action.uri)
    if path.exists():
        return {"status": "ready", "artifact_uri": str(path), "sha256": file_sha256(path)}
    match = re.search(r"/images/env-([0-9a-f]{16})/[^/]+/requirements.lock$", action.uri)
    if match is None:
        return {
            "status": REPAIR_BLOCKED,
            "reason": GapReason.ENVIRONMENT_ARTIFACT_MISSING,
            "detail": "Original artifact recipe is not registered",
            "artifact_uri": action.uri,
        }
    matches = [
        (identity, environment)
        for identity, environment in declared_environments().items()
        if identity.startswith(match.group(1))
    ]
    if len(matches) != 1 or path.parent.exists():
        return {
            "status": REPAIR_BLOCKED,
            "reason": GapReason.ENVIRONMENT_ARTIFACT_MISSING,
            "detail": "Original environment identity needs a recovered recipe or a new qualified release",
            "artifact_uri": action.uri,
        }
    identity, environment = matches[0]
    artifact = build_environment(environment, EnvironmentBuild(identity, DEFAULT_REPOSITORY, str(path.parent)))
    return {"status": "ready", "artifact_uri": artifact.lock_url, "sha256": file_sha256(path)}


@dataclass(frozen=True)
class EnvironmentArtifactFinding:
    """One declared artifact the build-stage check measured, with its rebuild outcome.

    ``reason`` carries the typed ``ExclusionReason`` value only when the rebuild failed;
    the finding then blocks the stage.
    """

    source_id: str
    artifact_uri: str
    task_count: int
    state: str
    detail: str | None
    reason: str | None = None


def _rebuild_finding(source_id: str, uri: str, task_count: int, attempt: dict) -> EnvironmentArtifactFinding:
    blocked = attempt["status"] == REPAIR_BLOCKED
    return EnvironmentArtifactFinding(
        source_id=source_id,
        artifact_uri=uri,
        task_count=task_count,
        state=attempt["status"],
        detail=attempt.get("detail"),
        reason=ExclusionReason.ENVIRONMENT_ARTIFACT_MISSING.value if blocked else None,
    )


def verify_declared_environment_artifacts(
    receipts: dict[str, GradingCensusReceipt],
) -> tuple[list[EnvironmentArtifactFinding], list[str]]:
    """Verify every declared environment artifact in-region, rebuilding what is missing.

    A missing artifact first triggers the scripted in-region rebuild. Returns the
    blocking findings — one per artifact whose rebuild failed, each carrying the typed
    ``ExclusionReason.ENVIRONMENT_ARTIFACT_MISSING`` reason — alongside notes for
    artifacts the rebuild restored and sources whose census predates the declared set.
    """
    blocking: list[EnvironmentArtifactFinding] = []
    notes: list[str] = []
    configure_coreweave_s3()
    for source_id, receipt in sorted(receipts.items()):
        evidence = receipt.evidence
        if evidence is None or not evidence.declared_artifact_counts:
            notes.append(f"{source_id}: grading census predates the declared artifact set")
            continue
        for uri, task_count in sorted(evidence.declared_artifact_counts.items()):
            if StoragePath(uri).exists():
                continue
            finding = _rebuild_finding(source_id, uri, task_count, rebuild_artifact(
                RebuildEnvironmentArtifact(uri=uri, cells=(), evidence_ids=())
            ))
            if finding.reason is not None or not StoragePath(uri).exists():
                blocking.append(finding)
            else:
                notes.append(f"{finding.source_id}: rebuilt {uri} ({finding.task_count} tasks declare it)")
    return blocking, notes
