# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measured original bindings and frozen QC retain the sealed population's identity."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ConversationInput, NoGrader, ResourceGroups, TaskSpec, TextMessage
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.source_fingerprints import SourceIdentitySurvey

from experiments.post_training.task_curation.atlas_deployment import region_deployment
from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages import qc, submission
from experiments.post_training.task_curation.ingest_stages.conversion import convert_stage
from experiments.post_training.task_curation.ingest_stages.licensing import LicenseAssessment
from experiments.post_training.task_curation.ingest_stages.normalized_population import census_stage, dedup_stage
from experiments.post_training.task_curation.ingest_stages.overlap import measured_declaration, review_population
from experiments.post_training.task_curation.ingest_stages.population import fingerprint_population
from experiments.post_training.task_curation.ingest_stages.publication import retention_plan, source_conversion
from experiments.post_training.task_curation.ingest_stages.qc import (
    QCRequest,
    prepare_sample,
    sample_qc_stage,
)
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage, StageReceipt, seal_receipt
from experiments.post_training.task_curation.lexical_artifacts import (
    ActorCensus,
    LexicalMeasurements,
    SourceTextPopulation,
)
from experiments.post_training.task_curation.license_check import CHECKER_VERSION
from experiments.post_training.task_curation.release_stages.manifest import read_ref
from experiments.post_training.task_curation.source_fingerprints import fingerprint_source
from experiments.post_training.task_curation.source_population_cache import PopulationLocations
from experiments.post_training.task_curation.tests.ingest_fixtures import next_job
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence


@pytest.fixture
def measured_job(staged_job, tmp_path):
    policy = staged_job.execution.policy
    assessment = LicenseAssessment(
        input_sha256="a" * 64,
        checker_version=CHECKER_VERSION,
        source_id=policy.source_id,
        dataset_id=policy.upstream.repository,
        revision=policy.upstream.revision,
        inherited=False,
        disposition="permit",
        receipt=fixture_license(policy.source_id, policy.upstream.repository, policy.upstream.revision),
        checks={},
        refusals=(),
    )
    artifacts = {
        **staged_job.inputs.artifacts,
        "license/assessment": seal_manifest(assessment.model_dump(mode="json"), str(tmp_path / "license.json")),
        "catalog-declarations": seal_manifest({}, str(tmp_path / "catalog.json")),
        "starting-declarations": seal_manifest({}, str(tmp_path / "starting.json")),
    }
    job = replace(staged_job, inputs=staged_job.inputs.model_copy(update={"artifacts": artifacts}))
    job, _ = next_job(job, convert_stage, IngestStage.DEDUP)
    job, _ = next_job(job, dedup_stage, IngestStage.CENSUS)
    job, _ = next_job(job, census_stage, IngestStage.FINGERPRINT_OVERLAP)
    census = ActorCensus.model_validate(read_ref(job.inputs.artifacts["census/actor-integrity"]))
    declaration = seal_manifest(measured_declaration(job, census), str(tmp_path / "declaration-input.json"))
    return replace(
        job,
        inputs=job.inputs.model_copy(
            update={"artifacts": {**job.inputs.artifacts, "fingerprint-overlap/declaration-input": declaration}}
        ),
    )


def test_original_source_binding_uses_measured_unique_count_and_preserves_conversion_drops(measured_job):
    source = source_conversion(measured_job)
    declaration = source.declaration
    assert declaration.task_count == 1
    assert source.info.converted_examples.input_rows == 2
    assert source.info.within_source_dedup.within_source_duplicates == 1
    assert (
        source.info.actor_census_input.task_specs_sha256
        == read_ref(measured_job.inputs.artifacts["dedup/population"])["task_specs_sha256"]
    )
    assert source.info.license_receipt.source_id == source.info.id


def test_failed_qc_is_a_sealed_informational_outcome_with_no_invented_verdict(measured_job, tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "regional_worker", lambda *args: "fixture")
    inputs = measured_job.inputs.model_copy(update={"stage": IngestStage.LEXICAL, "dependencies": {}})
    report = seal_manifest({"complete": True}, str(tmp_path / "lexical.json"))
    parent = seal_receipt(str(tmp_path / "lexical"), StageReceipt(inputs=inputs, outputs={"report": report}))
    inputs = inputs.model_copy(update={"stage": IngestStage.SAMPLE_QC, "dependencies": {IngestStage.LEXICAL: parent}})
    job = replace(measured_job, inputs=inputs, output_root=str(tmp_path / "sample"))
    reference = prepare_sample(job)
    request = QCRequest.model_validate(read_ref(reference))
    panel = read_ref(request.queue)["panels"][0]
    original = Path(panel["evidence_url"]).read_bytes()

    def unavailable(argv, *, check=False, **kwargs):
        if check:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(submission.subprocess, "run", unavailable)
    prepared = seal_manifest(
        {
            "sample_request": reference.model_dump(mode="json"),
            "qc_outcome_uri": str(Path(job.output_root) / "outcome.json"),
        },
        str(tmp_path / "prepared.json"),
    )
    outcome_ref = submission.review_sample(prepared, job.execution, tmp_path)
    outcome = qc.QCOutcome.model_validate(read_ref(outcome_ref))
    assert outcome.sample_count == outcome.no_verdict_count == 1
    assert not outcome.sample_passed and outcome.review is None
    persisted = sample_qc_stage(job)
    assert read_ref(persisted["outcome"])["state"] == "unavailable"
    assert prepare_sample(job) == reference
    assert Path(panel["evidence_url"]).read_bytes() == original


def test_overlap_adapter_persists_only_sha_bound_regional_population_caches(measured_job, remote, tmp_path):
    population = SourceTextPopulation.model_validate(read_ref(measured_job.inputs.artifacts["dedup/population"]))
    root = region_deployment(measured_job.execution.policy.cluster).region_prefix + f"tests/{tmp_path.name}"
    specs = StoragePath(root) / "candidate/specs/part-000000.parquet"
    specs.write_bytes(Path(population.specs_output_dir, "part-000000.parquet").read_bytes(), auto_mkdir=True)
    text = StoragePath(root) / "candidate/text/part-000000.parquet"
    text.write_bytes(next(Path(population.main_output_dir).glob("*.parquet")).read_bytes(), auto_mkdir=True)
    candidate = population.model_copy(
        update={"path": root + "/candidate", "specs_output_dir": str(specs.parent), "main_output_dir": str(text.parent)}
    )
    reference = seal_manifest(candidate.model_dump(mode="json"), root + "/candidate.json")
    actor = source_conversion(measured_job).info.actor_census_input
    measured = fingerprint_population(candidate.source_id, actor, reference, region_prefix=root)
    with specs.open("rb") as stream:
        row = pq.read_table(stream).to_pylist()[0]
    task = TaskSpec.model_validate_json(row["task_spec"]).model_copy(
        update={
            "id": "unrelated-reference",
            "context": ConversationInput(
                events=(TextMessage(role="user", content=" ".join(f"UnrelatedWord{n}" for n in range(40))),)
            ),
            "grader": NoGrader(reason="The reference fixture is never graded"),
            "resources": ResourceGroups(),
        }
    )
    baseline_path = StoragePath(root) / "baseline/specs.parquet"
    with baseline_path.open("wb", auto_mkdir=True) as stream:
        pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json(), "runtime": "{}"}]), stream)
    digest = file_sha256(baseline_path)
    baseline = fingerprint_source([task.model_dump(mode="json")], {str(baseline_path): digest})
    survey = SourceIdentitySurvey(
        catalog_root=root,
        snapshot_id="fixture",
        generated_at="2026-11-28T00:00:00Z",
        fingerprints={"fixture:reference": baseline},
        reviews=[],
        groups=[],
        judge=None,
    )
    job = replace(measured_job, execution=measured_job.execution.model_copy(update={"artifact_root": root}))
    outputs = review_population(
        job, measured, survey, PopulationLocations(files={digest: str(baseline_path)}), root + "/overlap"
    )
    reviewed = SourceIdentitySurvey.model_validate(read_ref(outputs["survey"]))
    locations = PopulationLocations.model_validate(read_ref(outputs["locators"]))
    prepared = locations.prepared[candidate.source_id]
    assert prepared.task_count == 1 and prepared.inputs_sha256 == canonical_sha256(measured.fingerprint.inputs)
    assert prepared.sha256 == file_sha256(StoragePath(prepared.uri))
    assert prepared.uri.startswith(root + "/overlap/")
    assert reviewed.reviews and all(row.verdict.label.value == "distinct" for row in reviewed.reviews)
    assert locations.files == measured.locations.files
    assert all(StoragePath(uri).bucket == StoragePath(root).bucket for uri in locations.files.values())


@pytest.mark.parametrize(("use", "flagged", "retained"), [("train", False, 1), ("train", True, 0), ("eval", True, 1)])
def test_retention_adapter_drops_only_training_overlap_after_quick_dedup(measured_job, tmp_path, use, flagged, retained):
    population = SourceTextPopulation.model_validate(read_ref(measured_job.inputs.artifacts["dedup/population"]))
    spec = next(Path(population.specs_output_dir).glob("*.parquet"))
    task_id = pq.read_table(spec).to_pylist()[0]["id"]
    evidence = tmp_path / "lexical"
    evidence.mkdir()
    pq.write_table(pa.table({"id": [task_id] if flagged else []}), evidence / "drop_manifest.parquet")
    measurement = decontamination_evidence(population.dataset_id, population.dataset_revision, population.task_count)
    measurement = measurement.model_copy(
        update={
            "receipt": None,
            "converter_revision": population.converter_revision,
            "text_renderer_version": population.text_renderer_version,
            "contaminated_tasks": int(flagged),
            "contaminated_fraction": float(flagged),
            "evidence_url": str(evidence / "measurement.json"),
        }
    )
    report = LexicalMeasurements(
        path=str(evidence),
        population_set="fixture",
        source_ids=[population.source_id],
        measurements={population.source_id: measurement.model_dump(mode="json")},
        drop_manifests={population.source_id: str(evidence / "drop_manifest.parquet")},
    )
    policy = measured_job.execution.policy.model_copy(update={"intended_use": (IntendedUse(use),)})
    job = replace(
        measured_job,
        execution=measured_job.execution.model_copy(update={"policy": policy}),
        inputs=measured_job.inputs.model_copy(
            update={
                "artifacts": {
                    **measured_job.inputs.artifacts,
                    "lexical/report": seal_manifest(report.model_dump(mode="json"), str(evidence / "report.json")),
                }
            }
        ),
    )
    plan, bound = retention_plan(job, population, str(tmp_path / "retention"))
    assert plan.raw_count == plan.unique_count == 1 and plan.within_source_duplicates == 0
    assert plan.retained_count == retained and plan.combined_drop_count == 1 - retained
    dropped = pq.read_table(plan.combined_drop_manifest_uri).to_pylist()
    assert [row["task_id"] for row in dropped] == ([] if retained else [task_id])
    assert bound.input.task_specs_sha256 == population.task_specs_sha256
    assert bound.use.value == use and bound.measurement.contaminated_tasks == int(flagged)
    assert retention_plan(job, population, str(tmp_path / "retention")) == (plan, bound)
