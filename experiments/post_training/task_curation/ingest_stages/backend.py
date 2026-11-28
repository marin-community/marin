# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind production backend survey and positive/negative grader controls to sealed tasks."""

import hashlib
import json
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from uuid import uuid4

from rigging.filesystem.storage_path import StoragePath, prefix_join
from shellbox.machine import NetworkPolicy
from shellbox.survey import production_checkers
from taskcompendium.backend_requirements import RuntimeProfile
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.models import CheckStatus, Controls
from taskcompendium.pipeline.sampling import seeded_sample

from experiments.post_training.task_curation.backend_capabilities import build_survey
from experiments.post_training.task_curation.backend_census_manifest import (
    MeasurementRow,
    compose_manifest,
    seal_manifest,
)
from experiments.post_training.task_curation.ingest_stages.converter_binding import load_converter
from experiments.post_training.task_curation.ingest_stages.graph import StageJob, parent_receipt
from experiments.post_training.task_curation.ingest_stages.placement import regional_worker
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.pipeline import CampaignMachines
from experiments.post_training.task_curation.published_census import population_rows
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

CONTROL_SAMPLE = 12
CONTROL_SEED = 0


def runtime_profiles() -> dict[str, dict]:
    """Use declared task resources without inventing absent deployment values."""
    profile = RuntimeProfile(
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=None,
        cleanup_timeout=None,
    )
    return {role: profile.model_dump(mode="json") for role in ("agent", "grader")}


def sealed_backend_manifest(job: StageJob) -> dict:
    declaration = read_ref(job.inputs.artifacts["fingerprint-overlap/declaration-input"])
    survey = read_ref(job.inputs.artifacts["fingerprint-overlap/survey"])
    fingerprint = survey["fingerprints"][job.execution.policy.source_id]
    row = MeasurementRow(
        source_id=job.execution.policy.source_id,
        actor_artifact=job.inputs.artifacts["census/actor-record"].model_dump(),
        declaration=declaration,
        input=declaration["actor_census_input"],
        inputs=fingerprint["inputs"],
    )
    return compose_manifest(
        [row],
        snapshot_id=canonical_sha256(job.execution.identity()),
        catalog_root=job.execution.artifact_root,
        source_revision=job.execution.source_revision,
        runtime_defaults=runtime_profiles(),
    )


def backend_oracle_stage(job: StageJob) -> dict[str, ArtifactRef]:
    """Measure the whole population, then require reference and noop controls."""
    regional_worker(job.execution.policy.cluster, job.execution.artifact_root, job.output_root)
    parent = parent_receipt(
        job.execution, IngestStage.FINGERPRINT_OVERLAP, job.inputs.dependencies[IngestStage.FINGERPRINT_OVERLAP]
    )
    if parent.outputs["survey"] != job.inputs.artifacts["fingerprint-overlap/survey"]:
        raise ValueError("backend population differs from its fingerprint dependency")
    manifest = sealed_backend_manifest(job)
    root = prefix_join(job.output_root, f"attempts/{uuid4().hex}")
    with TemporaryDirectory(prefix="atlas-backend-") as temporary:
        scratch = Path(temporary)
        local_manifest = {**manifest, "inputs": []}
        inputs = read_ref(job.inputs.artifacts["fingerprint-overlap/survey"])["fingerprints"][
            job.execution.policy.source_id
        ]["inputs"]
        for entry in manifest["inputs"]:
            local_paths = []
            for index, uri in enumerate(entry["paths"]):
                target = scratch / f"specs-{index}.parquet"
                with StoragePath(uri).open("rb") as incoming, target.open("wb") as outgoing:
                    copyfileobj(incoming, outgoing)
                with target.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != inputs[uri]:
                        raise ValueError("backend survey shard differs from its sealed input")
                local_paths.append(str(target))
            local_manifest["inputs"].append({**entry, "paths": local_paths})
        survey = build_survey(local_manifest, production_checkers(), job.execution.source_revision)
    population = SourceTextPopulation.model_validate(read_ref(job.inputs.artifacts["dedup/population"]))
    count, selected = seeded_sample(
        population_rows(population),
        size=CONTROL_SAMPLE,
        key=lambda row: canonical_sha256({"seed": CONTROL_SEED, "task": json.loads(row["task_spec"])["id"]}),
    )
    if count != population.task_count:
        raise ValueError("oracle sample input count differs from the declared population")
    controls = load_converter(job.execution.policy.converter_module).controls
    if controls.golden is None:
        raise ValueError("reference-solution controls are required before backend/oracle qualification")
    machines = CampaignMachines()
    outcomes = []
    for row in selected:
        task = TaskSpec.model_validate_json(row["task_spec"])
        if controls.golden(task) is None:
            raise ValueError("reference control is missing for a sampled task")
        positive = run_controls(task, controls=controls, machines=machines)
        negative = run_controls(task, controls=Controls(memory_mb=controls.memory_mb), machines=machines)
        checks = [*positive.checks, *negative.checks]
        outcomes.append({"task_id": task.id, "checks": [check.model_dump(mode="json") for check in checks]})
        if any(check.status is not CheckStatus.PASS for check in checks):
            seal_manifest({"outcomes": outcomes}, prefix_join(root, "refused-controls.json"))
            raise ValueError("reference or noop grading control failed; source cannot be admitted")
    return {
        "manifest": seal_manifest(manifest, prefix_join(root, "manifest.json")),
        "survey": seal_manifest(survey.model_dump(mode="json"), prefix_join(root, "survey.json")),
        "oracle": seal_manifest(
            {"population_count": count, "sample_count": len(selected), "seed": CONTROL_SEED, "outcomes": outcomes},
            prefix_join(root, "oracle.json"),
        ),
    }
