# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The build-stage environment-artifact check and the census's declared-artifact set."""

import hashlib
import json

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TaskSpec

from experiments.post_training.task_curation.difficulty.planner import RebuildEnvironmentArtifact
from experiments.post_training.task_curation.difficulty.state import REPAIR_BLOCKED, GapReason
from experiments.post_training.task_curation.environment_artifacts import (
    rebuild_artifact,
    verify_declared_environment_artifacts,
)
from experiments.post_training.task_curation.grading_binding import GradingCensusReceipt
from experiments.post_training.task_curation.grading_census import GradingCensusInput, census_source
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from experiments.post_training.task_curation.tests.test_census_binding import census_receipt

AVAILABLE_LOCK = "s3://marin-us-east-02a/marin/rl-data/atlas/images/env-aaaa000000000000/2026.10.08/requirements.lock"
MISSING_LOCK = "s3://marin-us-east-02a/marin/rl-data/atlas/images/env-bbbb000000000000/2026.10.08/requirements.lock"
UNREGISTERED_LOCK = "s3://marin-us-east-02a/marin/images/not-an-environment/requirements.lock"


def lock_declaring_task(task_id: str, lock: str):
    document = json.loads(prebuilt_task("separate")["lowered_task_spec"])
    task = document["task"]
    grader = task["grader"]
    grader["environment"] = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock=lock
    ).model_dump(mode="json")
    task["grader"] = grader
    task["id"] = task_id
    return TaskSpec.model_validate(task)


def write_census_population(tmp_path: Path, encoded: list[str], source) -> tuple[SourceTextPopulation, object]:
    specs = tmp_path / "specs"
    specs.mkdir()
    pq.write_table(pa.table({"task_spec": encoded}), specs / "part-000000.parquet")
    digest = hashlib.sha256("".join(encoded).encode()).hexdigest()
    binding = (
        census_receipt(source)
        .input.model_copy(update={"task_specs_sha256": digest, "measured_tasks": len(encoded)})
    )
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id=source.info.id,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
        source_version=binding.converter_version,
        converter_revision=binding.converter_revision,
        text_renderer_version="fixture",
        task_count=len(encoded),
        text_sha256="c" * 64,
        task_specs_sha256=digest,
        main_output_dir=str(tmp_path / "text"),
        specs_output_dir=str(specs),
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=population.path,
            result_type=result_type_name(SourceTextPopulation),
            result=population.result_payload(),
        )
    )
    return population, binding


def test_census_records_every_declared_lock_and_the_missing_subset(tmp_path, monkeypatch):
    source = number_source(tmp_path / "input")
    monkeypatch.setattr(
        "experiments.post_training.task_curation.grading_census.packages_lock_available",
        lambda path: path != MISSING_LOCK,
    )
    encoded = [
        lock_declaring_task("locked-available", AVAILABLE_LOCK).model_dump_json(),
        lock_declaring_task("locked-missing", MISSING_LOCK).model_dump_json(),
    ]
    population, binding = write_census_population(tmp_path, encoded, source)
    receipt = census_source(GradingCensusInput(population=population, input=binding), str(tmp_path / "census"))
    assert receipt.evidence.declared_artifact_counts == {AVAILABLE_LOCK: 1, MISSING_LOCK: 1}
    assert receipt.evidence.missing_artifact_counts == {MISSING_LOCK: 1}


class _Existing:
    """Stands in for StoragePath(uri); answers exists() from a shared table."""

    def __init__(self, existing: set[str], uri: str):
        self._existing = existing
        self._uri = uri

    def exists(self) -> bool:
        return self._uri in self._existing


def verify_receipt(declared: dict[str, int]) -> GradingCensusReceipt:
    from finestore.rl.curation import ActorCensusInput, GradingEnvironmentEvidence

    assessed = sum(declared.values())
    return GradingCensusReceipt(
        source_id="Evalchemy:fixture",
        input=ActorCensusInput(
            dataset_id="fixture/dataset",
            dataset_revision="r0",
            converter_version="v1",
            converter_revision="c0",
            measured_tasks=assessed,
            task_specs_sha256="a" * 64,
            evidence_url="s3://fixture/actor",
        ),
        population_path="s3://fixture/population",
        scanned_task_specs_sha256="a" * 64,
        evidence=GradingEnvironmentEvidence(
            assessed_tasks=assessed,
            required_counts={"separate": assessed},
            declared_counts={"separate": assessed},
            status_counts={"matched": assessed},
            reason_counts={"The typed grader reads the declared submitted answer or captured files in its own environment": assessed},
            evidence_url="s3://fixture/receipt.json",
            missing_artifact_counts={uri: 1 for uri in declared},
            declared_artifact_counts=dict(declared),
        ),
        task_evidence_uri="s3://fixture/tasks.parquet",
        unassessable_reasons={},
    )


def test_verify_passes_when_every_declared_artifact_exists(monkeypatch):
    monkeypatch.setattr(
        "experiments.post_training.task_curation.environment_artifacts.StoragePath",
        lambda uri: _Existing({AVAILABLE_LOCK}, uri),
    )
    blocking, notes = verify_declared_environment_artifacts({"Evalchemy:fixture": verify_receipt({AVAILABLE_LOCK: 4})})
    assert blocking == []
    assert notes == []


def test_verify_rebuilds_a_missing_artifact_and_only_blocks_when_the_rebuild_fails(monkeypatch):
    calls = []
    rebuilt: set[str] = set()

    def fake_rebuild(action: RebuildEnvironmentArtifact) -> dict:
        calls.append(action.uri)
        if action.uri == MISSING_LOCK:
            rebuilt.add(action.uri)
            return {"status": "ready", "artifact_uri": action.uri, "sha256": "b" * 64}
        return {
            "status": REPAIR_BLOCKED,
            "reason": GapReason.ENVIRONMENT_ARTIFACT_MISSING,
            "detail": "Original environment identity needs a recovered recipe or a new qualified release",
            "artifact_uri": action.uri,
        }

    monkeypatch.setattr("experiments.post_training.task_curation.environment_artifacts.rebuild_artifact", fake_rebuild)
    monkeypatch.setattr(
        "experiments.post_training.task_curation.environment_artifacts.StoragePath",
        lambda uri: _Existing(rebuilt, uri),
    )
    receipts = {
        "Evalchemy:blocked": verify_receipt({AVAILABLE_LOCK: 5}),
        "Evalchemy:rebuilt": verify_receipt({MISSING_LOCK: 30}),
    }
    blocking, notes = verify_declared_environment_artifacts(receipts)
    # Sources iterate in sorted order: the blocked source first, then the rebuilt one.
    assert calls == [AVAILABLE_LOCK, MISSING_LOCK]
    assert [(finding.source_id, finding.task_count) for finding in blocking] == [("Evalchemy:blocked", 5)]
    assert blocking[0].reason == "environment-artifact-missing"
    assert blocking[0].detail == "Original environment identity needs a recovered recipe or a new qualified release"
    assert blocking[0].state == REPAIR_BLOCKED
    assert notes == ["Evalchemy:rebuilt: rebuilt " + MISSING_LOCK + " (30 tasks declare it)"]


def test_verify_reports_sources_whose_census_predates_the_declared_set(monkeypatch):
    base = verify_receipt({AVAILABLE_LOCK: 1})
    receipt = base.model_copy(update={"evidence": base.evidence.model_copy(update={"declared_artifact_counts": {}})})
    monkeypatch.setattr(
        "experiments.post_training.task_curation.environment_artifacts.StoragePath",
        lambda uri: _Existing(set(), uri),
    )
    blocking, notes = verify_declared_environment_artifacts({"Evalchemy:old": receipt})
    assert blocking == []
    assert notes == ["Evalchemy:old: grading census predates the declared artifact set"]


def test_rebuild_artifact_blocks_an_unregistered_recipe(monkeypatch):
    monkeypatch.setattr(
        "experiments.post_training.task_curation.environment_artifacts.StoragePath",
        lambda uri: _Existing(set(), uri),
    )
    monkeypatch.setattr(
        "experiments.post_training.task_curation.environment_artifacts.declared_environments", lambda: {}
    )
    attempt = rebuild_artifact(RebuildEnvironmentArtifact(uri=MISSING_LOCK, cells=(), evidence_ids=()))
    assert attempt["status"] == REPAIR_BLOCKED
    assert attempt["reason"] == GapReason.ENVIRONMENT_ARTIFACT_MISSING
    assert attempt["detail"] == "Original environment identity needs a recovered recipe or a new qualified release"
    outside = rebuild_artifact(RebuildEnvironmentArtifact(uri=UNREGISTERED_LOCK, cells=(), evidence_ids=()))
    assert outside["status"] == REPAIR_BLOCKED
    assert outside["detail"] == "Original artifact recipe is not registered"
