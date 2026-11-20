# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete repairs reuse the source's original environment and judge identities."""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.difficulty.legacy import legacy_gaps
from experiments.post_training.task_curation.difficulty.planner import (
    Action,
    BuildSnapshot,
    BuildTaskSpecFactory,
    QualifyJudgeRoute,
    RebuildEnvironmentArtifact,
    RecheckExecutabilityAdmission,
    action_payload,
)
from experiments.post_training.task_curation.difficulty.selection import selected_tasks
from experiments.post_training.task_curation.difficulty.state import GapReason
from experiments.post_training.task_curation.difficulty_preparation import prepare_source
from experiments.post_training.task_curation.difficulty_samples import load_queue_sources
from experiments.post_training.task_curation.images.__main__ import declared_environments
from experiments.post_training.task_curation.images.build import DEFAULT_REPOSITORY, EnvironmentBuild, build_environment


# Unknown recipes and judge routes require evidence or a ruling. They must not
# be replaced by an environment or model that merely resembles the original.
def rebuild_artifact(action: RebuildEnvironmentArtifact) -> dict:
    path = StoragePath(action.uri)
    if path.exists():
        return {"status": "ready", "artifact_uri": str(path), "sha256": file_sha256(path)}
    match = re.search(r"/images/env-([0-9a-f]{16})/[^/]+/requirements.lock$", action.uri)
    if match is None:
        return {
            "status": "blocked",
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
            "status": "blocked",
            "reason": GapReason.ENVIRONMENT_ARTIFACT_MISSING,
            "detail": "Original environment identity needs a recovered recipe or a new qualified release",
            "artifact_uri": action.uri,
        }
    identity, environment = matches[0]
    artifact = build_environment(environment, EnvironmentBuild(identity, DEFAULT_REPOSITORY, str(path.parent)))
    return {"status": "ready", "artifact_uri": artifact.lock_url, "sha256": file_sha256(path)}


def build_factory(action: BuildTaskSpecFactory, local: Path) -> dict:
    if not action.source.queue_uri:
        return {
            "status": "blocked",
            "reason": GapReason.TASK_SPEC_FACTORY_PENDING,
            "detail": "Current catalog pin needs a new frozen SAMPLE panel",
        }
    sources = load_queue_sources(action.source.queue_uri, action.source.queue_sha256, (action.source.source_id,))
    prepared = prepare_source(sources[0], local / "prepared")
    reasons = sorted({task.reason for task in prepared.tasks if task.reason})
    if reasons:
        return {
            "status": "blocked",
            "reason": GapReason.TASK_SPEC_FACTORY_PENDING,
            "detail": "Original task factory still has typed preflight gaps",
            "legacy_reasons": reasons,
        }
    return {
        "status": "ready",
        "source_id": action.source.source_id,
        "registry_revision": action.source.registry_revision,
        "tasks": len(prepared.tasks),
        "sample_receipt": prepared.sample_receipt.model_dump(mode="json"),
    }


@dataclass(frozen=True)
class AdmissionBlocker:
    task_id: str
    task_sha256: str
    reason: str


def recheck_admission(action: RecheckExecutabilityAdmission, local: Path) -> dict:
    """Re-run frozen executability checks; retain blocked contracts as evidence."""
    sources = load_queue_sources(action.source.queue_uri, action.source.queue_sha256, (action.source.source_id,))
    (source,) = sources
    prepared = prepare_source(source, local / "admission")
    tasks = selected_tasks(prepared, action.source)
    if len(tasks) != len(prepared.sample.tasks):
        raise ValueError("Admission recheck does not cover the complete frozen sample")
    hashes = {task.task_id: task.task_sha256 for task in source.sample.tasks}
    blockers = [AdmissionBlocker(task.task_id, hashes[task.task_id], task.reason) for task in tasks if task.reason]
    result = {
        "source_id": action.source.source_id,
        "registry_revision": action.source.registry_revision,
        "checked_tasks": len(prepared.tasks),
        "sample_receipt": prepared.sample_receipt.model_dump(mode="json"),
        "blockers": [asdict(blocker) for blocker in blockers],
        "population_changed": False,
    }
    if not blockers:
        return {**result, "status": "ready"}
    gap = legacy_gaps(blockers[0].reason)[0]
    return {
        **result,
        "status": "blocked",
        "reason": gap.reason,
        "detail": (
            "Frozen executability admission recheck still requires: "
            + "; ".join(sorted({blocker.reason for blocker in blockers}))
            + ". Resolve the executor/provider contract or qualify a new admission population. "
            "Frozen tasks were retained."
        ),
    }


def repair_action(action: Action, local: Path) -> dict:
    if isinstance(action, RebuildEnvironmentArtifact):
        return rebuild_artifact(action)
    if isinstance(action, BuildTaskSpecFactory):
        return build_factory(action, local)
    if isinstance(action, RecheckExecutabilityAdmission):
        return recheck_admission(action, local)
    if isinstance(action, BuildSnapshot):
        manifest = local / "snapshot-action.json"
        manifest.write_text(json.dumps(action_payload(action)))
        result = local / "snapshot-result.json"
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                "vendor/harbor",
                "--frozen",
                "--extra",
                "runtime",
                "python",
                "-m",
                "experiments.post_training.task_curation.difficulty.snapshot_repair",
                "--manifest",
                str(manifest),
                "--result",
                str(result),
                "--local-root",
                str(local / "snapshots"),
            ],
            check=True,
        )
        return json.loads(result.read_text())
    if isinstance(action, QualifyJudgeRoute):
        command = [
            "python",
            "-m",
            "experiments.post_training.task_curation.difficulty.judge_repair",
            "--model",
            action.model,
            "--result",
            str(local / "judge-result.json"),
        ]
        subprocess.run(command, check=True)
        result = json.loads((local / "judge-result.json").read_text())
        result["requested_temperature"] = action.requested_temperature
        result["effective_temperature"] = PROTOCOLS[PROTOCOL_ID].temperature
        return result
    raise ValueError(f"Action is not an automatic repair: {type(action).__name__}")
