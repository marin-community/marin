# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete repairs reuse the source's original environment and judge identities."""

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from finestore.rl.curation import QualifiedSourceDeclaration, normalize_snapshot_declaration
from finestore.rl.difficulty import PROTOCOL_ID, PROTOCOLS, DifficultyGap
from marin.execution.artifact import RECORD_FILENAME
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret

from experiments.post_training.task_curation.admission_inventory import inventory_source
from experiments.post_training.task_curation.difficulty.campaign import JUDGE_VERSION, campaign_record
from experiments.post_training.task_curation.difficulty.importer import (
    declaration_population_revision,
    population_revision,
)
from experiments.post_training.task_curation.difficulty.ledger import LEDGER_VERSION, PANELS_DIRECTORY, read_rows
from experiments.post_training.task_curation.difficulty.legacy import admission_cause, legacy_gaps
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
from experiments.post_training.task_curation.difficulty.regional import regional_worker
from experiments.post_training.task_curation.difficulty.selection import selected_tasks
from experiments.post_training.task_curation.difficulty.state import (
    REPAIR_BLOCKED,
    AdmissionBlocker,
    CatalogSource,
    CoveragePolicy,
    GapReason,
    PanelReference,
    cell_state,
)
from experiments.post_training.task_curation.difficulty_config import (
    SUBSTITUTE_JUDGE_MODEL,
    SUBSTITUTE_JUDGE_PIN,
    TOGETHER_API_KEY_ENV,
)
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence
from experiments.post_training.task_curation.difficulty_preparation import prepare_source
from experiments.post_training.task_curation.difficulty_samples import (
    load_queue_sources,
    prepare_sample,
    write_source_queue,
)
from experiments.post_training.task_curation.difficulty_serving_probe import probe_declared_judge_endpoint
from experiments.post_training.task_curation.environment_artifacts import rebuild_artifact
from experiments.post_training.task_curation.lexical_artifacts import population_for_census

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
            and admission_cause(gap.capability)[0] not in STATIC_ADMISSION_CAUSES
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
    document = json.loads(action.declaration_json)
    if document["id"] != action.source.source_id:
        raise ValueError("Panel repair belongs to another source")
    revision = declaration_population_revision(document)
    if revision != action.source.registry_revision:
        raise ValueError("Panel repair declaration differs from the planned population")
    if document.get("actor_census_input") is None:
        return {
            "status": REPAIR_BLOCKED,
            "reason": GapReason.TASK_SPEC_FACTORY_PENDING,
            "detail": "Current served declaration lacks its exact input census; qualify the population first",
        }
    try:
        declaration = QualifiedSourceDeclaration.model_validate(normalize_snapshot_declaration(document))
    except ValueError as error:
        return {"status": REPAIR_BLOCKED, "reason": GapReason.SOURCE_QUEUE_VALIDATION_FAILED, "detail": str(error)}
    if declaration_population_revision(declaration.model_dump(mode="json")) != revision:
        raise ValueError("Snapshot normalization changed the planned population")
    if campaign_record(ledger) is not None:
        regional_worker()
        binding = declaration.actor_census_input
        assert binding is not None
        population = population_for_census(
            str(StoragePath(binding.evidence_url).parent / RECORD_FILENAME), declaration.id
        )
        foreign = [
            uri
            for uri in (population.main_output_dir, population.specs_output_dir)
            if uri.startswith("gs://") and not uri.startswith("gs://marin-us-central2/")
        ]
        if foreign:
            return {
                "status": REPAIR_BLOCKED,
                "reason": GapReason.EXECUTABILITY_ADMISSION_PENDING,
                "detail": "Population requires regional staging before panel freezing: " + ", ".join(foreign),
            }
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
        if campaign_record(ledger) is not None:
            if action.model != SUBSTITUTE_JUDGE_MODEL:
                return {
                    "status": "blocked",
                    "reason": GapReason.JUDGE_ROUTE_UNQUALIFIED,
                    "detail": (
                        "The versioned TPU campaign serves only Together Flash; "
                        "this source needs an explicit judge-contract repair"
                    ),
                }
            probe = probe_declared_judge_endpoint(
                "https://api.together.xyz/v1",
                SUBSTITUTE_JUDGE_MODEL,
                require_secret(TOGETHER_API_KEY_ENV),
                PROTOCOLS[PROTOCOL_ID].temperature,
                SUBSTITUTE_JUDGE_PIN,
            )
            return {
                "status": "ready",
                "model": action.model,
                "served_model": SUBSTITUTE_JUDGE_MODEL,
                "judge_version": JUDGE_VERSION,
                "probe": probe,
                "requested_temperature": action.requested_temperature,
                "effective_temperature": PROTOCOLS[PROTOCOL_ID].temperature,
            }
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
