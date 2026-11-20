# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete repairs reuse the source's original environment and judge identities."""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from finestore.rl.curation import QualifiedSourceDeclaration
from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS, DifficultyGap
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.admission_inventory import inventory_source
from experiments.post_training.task_curation.difficulty.importer import (
    declaration_population_revision,
    population_revision,
)
from experiments.post_training.task_curation.difficulty.ledger import LEDGER_VERSION, PANELS_DIRECTORY, read_rows
from experiments.post_training.task_curation.difficulty.legacy import legacy_gaps
from experiments.post_training.task_curation.difficulty.planner import (
    Action,
    BuildSnapshot,
    BuildTaskSpecFactory,
    FreezeSamplePanel,
    QualifyJudgeRoute,
    RebuildEnvironmentArtifact,
    RecheckExecutabilityAdmission,
    action_id,
    action_payload,
)
from experiments.post_training.task_curation.difficulty.selection import selected_tasks
from experiments.post_training.task_curation.difficulty.state import (
    CatalogSource,
    CoveragePolicy,
    GapReason,
    PanelReference,
    cell_state,
)
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence
from experiments.post_training.task_curation.difficulty_preparation import prepare_source
from experiments.post_training.task_curation.difficulty_samples import (
    load_queue_sources,
    prepare_sample,
    write_source_queue,
)
from experiments.post_training.task_curation.images.__main__ import declared_environments
from experiments.post_training.task_curation.images.build import DEFAULT_REPOSITORY, EnvironmentBuild, build_environment

REPAIR_BLOCKED = "blocked"
STATIC_ADMISSION_CAUSES = frozenset(
    {
        DifficultyGap.EXECUTABILITY_ADMISSION_PENDING,
        DifficultyGap.FILTRATION_FINDINGS_ADMISSION_PENDING,
        DifficultyGap.NO_EXECUTABLE_GRADER,
        DifficultyGap.STATE_CAPTURE_REQUIRED,
        DifficultyGap.TOOL_PROVIDER_REQUIRED,
        DifficultyGap.SHARED_EXECUTOR_REQUIRED,
        DifficultyGap.PROVIDER_CONTRACT_INVALID,
    }
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


def build_factory(action: BuildTaskSpecFactory, local: Path) -> dict:
    if not action.source.queue_uri:
        raise ValueError("Freeze the missing sample panel before building its task factory")
    sources = load_queue_sources(action.source.queue_uri, action.source.queue_sha256, (action.source.source_id,))
    prepared = prepare_source(sources[0], local / "prepared")
    reasons = sorted({task.reason for task in prepared.tasks if task.reason})
    if reasons:
        return {
            "status": REPAIR_BLOCKED,
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


def unverified_admission_capabilities(action: RecheckExecutabilityAdmission, ledger: str) -> list[str]:
    """Retain active causes that frozen contract preparation cannot verify."""
    evidence = set(action.source.evidence_ids)
    rows = [row for row in read_rows(ledger) if row.attempt_id in evidence]
    if not rows:
        raise ValueError("Admission recheck lacks its planned ledger evidence")
    source = CatalogSource(
        action.source.source_id,
        action.source.registry_revision,
        tuple((task_id, sha) for task_id, sha, _ in action.source.tasks),
        action.source.queue_uri,
        action.source.queue_sha256,
    )
    return sorted(
        {
            gap.capability
            for _, solver in action.cells
            for gap in cell_state(rows, source, solver, CoveragePolicy()).gaps
            if gap.reason == GapReason.EXECUTABILITY_ADMISSION_PENDING
            and gap.capability
            and gap.capability.partition(":")[0] not in STATIC_ADMISSION_CAUSES
        }
    )


def recheck_admission(action: RecheckExecutabilityAdmission, local: Path, ledger: str) -> dict:
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
        unverified = unverified_admission_capabilities(action, ledger)
        if unverified:
            return {
                **result,
                "status": REPAIR_BLOCKED,
                "reason": GapReason.EXECUTABILITY_ADMISSION_PENDING,
                "unverified_capabilities": unverified,
                "detail": "Frozen contract checks cannot verify these admission prerequisites: " + "; ".join(unverified),
            }
        return {**result, "status": "ready"}
    gap = legacy_gaps(blockers[0].reason)[0]
    return {
        **result,
        "status": REPAIR_BLOCKED,
        "reason": gap.reason,
        "detail": (
            "Frozen executability admission recheck still requires: "
            + "; ".join(sorted({blocker.reason for blocker in blockers}))
            + ". Resolve the executor/provider contract or qualify a new admission population. "
            "Frozen tasks were retained."
        ),
    }


def freeze_panel(action: FreezeSamplePanel, ledger: str) -> dict:
    """Freeze a seed-zero sample of the current population and publish its verified queue."""
    declaration = QualifiedSourceDeclaration.model_validate_json(action.declaration_json)
    binding = declaration.actor_census_input
    if declaration.id != action.source.source_id:
        raise ValueError("Panel repair belongs to another source")
    if binding is None:
        return {
            "status": REPAIR_BLOCKED,
            "reason": GapReason.TASK_SPEC_FACTORY_PENDING,
            "detail": "Current served declaration lacks its exact input census; qualify the population first",
        }
    revision = declaration_population_revision(declaration.model_dump(mode="json"))
    if revision != action.source.registry_revision:
        raise ValueError("Panel repair declaration differs from the planned population")
    root = StoragePath(ledger) / LEDGER_VERSION / "panel-builds" / action_id(action)
    inventory = inventory_source(declaration, str(root / "inventory"), None)
    source = prepare_sample(declaration, inventory, str(root / "samples"), None)
    if population_revision(source.sample.model_dump(mode="json")) != revision:
        raise ValueError("Frozen sample differs from the planned population")
    queue = write_source_queue(root / "sources.json", [source])
    reference = PanelReference(declaration.id, revision, queue.uri, queue.sha256)
    index = StoragePath(ledger) / LEDGER_VERSION / PANELS_DIRECTORY / (action_id(action) + ".json")
    (verified,) = load_queue_sources(reference.queue_uri, reference.queue_sha256, (declaration.id,))
    if verified != source:
        raise ValueError("Published panel queue changed on readback")
    index.parent.mkdirs(exist_ok=True)
    write_json_evidence(index, asdict(reference))
    return {"status": "ready", **asdict(reference)}


def repair_action(action: Action, local: Path, ledger: str) -> dict:
    if isinstance(action, FreezeSamplePanel):
        return freeze_panel(action, ledger)
    if isinstance(action, RebuildEnvironmentArtifact):
        return rebuild_artifact(action)
    if isinstance(action, BuildTaskSpecFactory):
        return build_factory(action, local)
    if isinstance(action, RecheckExecutabilityAdmission):
        return recheck_admission(action, local, ledger)
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
