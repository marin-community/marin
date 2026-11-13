# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    DaytonaMeasurement,
    GradingEnvironmentEvidence,
    LexicalContaminationEvidence,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    SourceOrigin,
    SourceQualityResult,
    WithinSourceDedupEvidence,
)
from marin.execution.lazy import StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import EnvironmentRequirements, NoGrader, TaskSpec
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation import calibforge_publish, retained_release
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.decontamination import require_private_release
from experiments.post_training.task_curation.grading_binding import GradingCensusReceipt, bind_grading_census
from experiments.post_training.task_curation.grading_census import GradingCensusInput, census_source
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_policy import source_lexical_use
from experiments.post_training.task_curation.remediation_binding import (
    PublicationReceipt,
    RetainedReleaseReceipt,
    bind_retained_release,
    release_lexical_evidence,
)
from experiments.post_training.task_curation.remediation_publish import (
    ReleasePlan,
    filter_population,
    verify_retained,
)
from experiments.post_training.task_curation.retained_census import census_release
from experiments.post_training.task_curation.retained_grading import RetainedGradingReceipt, grading_receipt
from experiments.post_training.task_curation.retained_release import (
    RETAINED_IMPORT_VERSION,
    RetainedCohort,
    RetainedImport,
    run_retained_import,
)
from experiments.post_training.task_curation.source import SourceInfo, SourceReference
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_source, prebuilt_task
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence


@pytest.fixture
def frozen_population(tmp_path):
    specs = tmp_path / "specs"
    specs.mkdir()
    rows = [
        {"id": name, "task_spec": json.dumps({"id": name, "answer": name}), "runtime": '{ "cpus": 2 }'}
        for name in ("retained", "duplicate", "lexical")
    ]
    pq.write_table(pa.Table.from_pylist(rows), specs / "part-000000.parquet")
    drops = tmp_path / "drops.parquet"
    pq.write_table(pa.table({"task_id": ["duplicate", "lexical"]}), drops)
    plan = ReleasePlan(
        source_id="test:fixture",
        source_dataset_id="upstream/fixture",
        source_revision="a" * 40,
        converter_revision="b" * 40,
        converter_version="1",
        input_artifact=str(tmp_path),
        task_specs_sha256="c" * 64,
        raw_count=3,
        unique_count=2,
        within_source_duplicates=1,
        raw_lexical_count=1,
        unique_lexical_count=1,
        combined_drop_count=2,
        retained_count=1,
        multiplicities={"1": 1, "2": 1},
        drop_manifest_uri=str(drops),
        combined_drop_manifest_uri=str(drops),
        evidence_url="https://example.test/frozen-plan",
        lexical_use="train",
        census_evidence_url="https://example.test/census",
    )
    return plan, specs, rows


def test_combined_selection_preserves_task_and_runtime_bytes_and_archives_original(tmp_path, frozen_population):
    plan, specs, rows = frozen_population
    retained, original = tmp_path / "retained.parquet", tmp_path / "original.parquet"
    result = filter_population(plan, str(specs), retained, original)
    selected = pq.read_table(retained).to_pylist()
    assert [(row["id"], row["task_spec"], row["runtime"]) for row in selected] == [
        (rows[0]["id"], rows[0]["task_spec"], rows[0]["runtime"])
    ]
    assert pq.read_table(original).num_rows == 3
    assert verify_retained(retained, result["hashes"])


@pytest.mark.parametrize("change", ["task", "runtime", "missing", "duplicate"])
def test_readback_rejects_changed_or_incomplete_retained_population(tmp_path, frozen_population, change):
    plan, specs, _ = frozen_population
    tasks = tmp_path / "retained.parquet"
    result = filter_population(plan, str(specs), tasks, None)
    table = pq.read_table(tasks)
    rows = table.to_pylist()
    if change == "task":
        rows[0]["task_spec"] = json.dumps({"id": "retained", "answer": "changed"})
    elif change == "runtime":
        rows[0]["runtime"] = '{"cpus": 99}'
    elif change == "missing":
        rows.clear()
    else:
        rows += rows
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), tasks)
    with pytest.raises(ValueError, match="differs"):
        verify_retained(tasks, result["hashes"])


def test_unknown_manifest_task_prevents_publication(tmp_path, frozen_population):
    plan, specs, _ = frozen_population
    pq.write_table(pa.table({"task_id": ["duplicate", "unknown"]}), plan.combined_drop_manifest_uri)
    with pytest.raises(ValueError, match="does not match plan"):
        filter_population(plan, str(specs), tmp_path / "retained.parquet", None)


def test_existing_hf_destination_is_never_uploaded(tmp_path, monkeypatch):
    tasks, card = tmp_path / "tasks.parquet", tmp_path / "README.md"
    tasks.write_bytes(b"fixture")
    card.write_text("fixture")
    uploaded = []

    class ExistingRepo:
        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            if not exist_ok:
                raise FileExistsError("Repository already exists")

        def upload_folder(self, **kwargs):
            uploaded.append(kwargs["repo_id"])

    monkeypatch.setattr(calibforge_publish, "HfApi", ExistingRepo)
    with pytest.raises(FileExistsError, match="already exists"):
        calibforge_publish.publish(tasks, card, "open-athena/existing")
    assert uploaded == []


@pytest.fixture
def retained_receipt(frozen_population):
    plan, _, _ = frozen_population
    source = prebuilt_source(
        SourceInfo(
            origin=SourceOrigin.MARIN,
            upstream_dataset="fixture",
            upstream_name="Fixture",
            build_path="Fixture",
            count=3,
            oversight=OversightEvidence(
                level=OversightLevel.SAMPLE,
                campaign="fixture",
                verification="Original admission receipts",
                reviewed=3,
                admitted=3,
            ),
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
            quality_result=SourceQualityResult(
                reviewed=3, issue_count=0, issues=[], repair_recommendation="None needed"
            ),
        )
    )
    plan = plan.model_copy(
        update={
            "source_id": source.info.id,
            "source_dataset_id": source.dataset.name,
            "source_revision": source.dataset.revision,
            "converter_version": source.version,
        }
    )

    def actor(repo, revision, version, count, digest):
        return ActorCensusReceipt(
            source_id=source.info.id,
            input=ActorCensusInput(
                dataset_id=repo,
                dataset_revision=revision,
                converter_version=version,
                converter_revision=plan.converter_revision,
                measured_tasks=count,
                task_specs_sha256=digest,
                evidence_url="https://example.test/census",
            ),
            measurement=DaytonaMeasurement(
                dataset_id=repo,
                dataset_revision=revision,
                daytona_snapshot_count=0,
                daytona_machine_slots=0,
                daytona_execution_failures=0,
                daytona_count_basis="All tasks measured",
                daytona_unsafe_reasons=[],
                daytona_unsafe_details={},
                daytona_unsafe_counts={},
                daytona_roles={"agent": {}, "grader": {}},
            ),
            integrity_contracts=None,
        )

    source = bind_actor_census(
        source, actor(source.dataset.name, source.dataset.revision, source.version, 3, plan.task_specs_sha256)
    )
    raw = decontamination_evidence(source.dataset.name, source.dataset.revision, 3).model_dump(mode="json")
    raw.update(
        receipt=None, converter_revision=plan.converter_revision, contaminated_tasks=1, contaminated_fraction=1 / 3
    )
    publication = PublicationReceipt(
        source_id=source.info.id,
        plan=plan,
        repo="open-athena/retained-fixture",
        revision="d" * 40,
        url="https://huggingface.co/datasets/open-athena/retained-fixture/tree/" + "d" * 40,
        private=True,
        format="taskcompendium",
        original_private_pin={"repo": source.dataset.name, "revision": source.dataset.revision},
        tasks_parquet_sha256="e" * 64,
        retained_content_sha256="f" * 64,
        verified_retained_tasks=1,
        specs_output_dir="https://example.test/specs",
        evidence_url="https://example.test/publication",
        lexical=raw,
    )
    dedup = WithinSourceDedupEvidence(
        raw_count=1,
        unique_count=1,
        within_source_duplicates=0,
        source_dataset_id=publication.repo,
        source_revision=publication.revision,
        converter_version=RETAINED_IMPORT_VERSION,
        converter_revision="retained-importer",
        drop_manifest_uri="https://example.test/empty-drops",
        evidence_url=publication.evidence_url,
    )
    retained_actor = actor(publication.repo, publication.revision, RETAINED_IMPORT_VERSION, 1, "1" * 64)
    receipt = RetainedReleaseReceipt(
        source_id=source.info.id,
        publication=publication,
        actor=retained_actor,
        grading=RetainedGradingReceipt(
            source_id=source.info.id,
            input=retained_actor.input,
            normalized_path="https://example.test/normalized",
            evidence={
                "assessed_tasks": 1,
                "required_counts": {"separate": 1},
                "declared_counts": {"separate": 1},
                "status_counts": {"matched": 1},
                "reason_counts": {"fixture": 1},
                "evidence_url": "https://example.test/grading",
            },
        ),
        original_pipeline=source.pipeline_reference,
        original_info=source.info,
        retained_dedup=dedup,
    )
    return source, receipt


def test_binding_selects_private_retained_pin_and_uses_new_population_evidence(retained_receipt):
    source, receipt = retained_receipt
    bound = bind_retained_release(source, receipt)
    assert bound.current_release == receipt.publication.release
    assert bound.declaration.url == source.declaration.url
    assert bound.declaration.task_count == 1
    assert bound.info.within_source_dedup.raw_count == bound.info.within_source_dedup.unique_count == 1
    assert bound.info.actor_census_input.dataset_id == receipt.publication.repo
    assert bound.info.grading_environment.assessed_tasks == 1
    assert bound.info.actor_census_input.task_specs_sha256 != source.info.actor_census_input.task_specs_sha256
    assert bound.info.lexical_contamination.input_dataset_id == source.dataset.name
    assert bound.info.lexical_contamination.contaminated_tasks == 1
    assert bound.info.oversight == source.info.oversight
    assert bound.info.oracle_validation == source.info.oracle_validation
    assert bound.info.quality_result == source.info.quality_result
    assert bound.info.oversight.level is OversightLevel.SAMPLE and bound.info.oversight.admitted == 3
    assert bound.info.inherited_admission == source.current_release
    assert "remediated: dedup/decon" in bound.info.notes
    assert source_lexical_use(bound).value == "train"
    require_private_release(bound.info.lexical_contamination, bound.dataset.name, bound.dataset.revision)
    with pytest.raises(ValueError, match="different dataset pin"):
        require_private_release(bound.info.lexical_contamination, source.dataset.name, source.dataset.revision)
    changed = replace(source, config=replace(source.config, revision="2" * 40))
    with pytest.raises(ValueError, match="different conversion input"):
        bind_retained_release(changed, receipt)


def test_duplicate_first_receipt_preserves_overlapping_raw_lexical_numerator(retained_receipt):
    _, receipt = retained_receipt
    publication = receipt.publication
    plan = publication.plan.model_copy(
        update={
            "raw_count": 5,
            "unique_count": 3,
            "within_source_duplicates": 2,
            "raw_lexical_count": 2,
            "unique_lexical_count": 1,
            "retained_count": 2,
            "combined_drop_count": 3,
        }
    )
    raw = publication.lexical.model_copy(
        update={"measured_tasks": 5, "contaminated_tasks": 2, "contaminated_fraction": 0.4}
    )
    publication = publication.model_copy(update={"plan": plan, "lexical": raw, "verified_retained_tasks": 2})
    evidence = release_lexical_evidence(publication)
    assert evidence.contaminated_tasks == 2
    assert (
        evidence.receipt.retained_tasks,
        evidence.receipt.duplicate_dropped_tasks,
        evidence.receipt.dropped_tasks,
    ) == (2, 2, 1)
    require_private_release(evidence, publication.repo, publication.revision)


def test_duplicate_only_clean_release_requires_general_original_pin_and_keeps_private_archive_absent(retained_receipt):
    _, receipt = retained_receipt
    publication = receipt.publication
    plan = publication.plan.model_copy(
        update={
            "raw_lexical_count": 0,
            "unique_lexical_count": 0,
            "retained_count": 2,
            "combined_drop_count": 1,
        }
    )
    raw = publication.lexical.model_copy(update={"contaminated_tasks": 0, "contaminated_fraction": 0.0})
    publication = publication.model_copy(update={"plan": plan, "lexical": raw, "verified_retained_tasks": 2})
    evidence = release_lexical_evidence(publication)
    require_private_release(evidence, publication.repo, publication.revision)
    assert evidence.receipt.status.value == "clean"
    assert evidence.receipt.release.original_dataset_id == plan.source_dataset_id
    assert evidence.receipt.release.original_private_dataset_id is None
    document = evidence.model_dump(mode="json")
    document["receipt"]["release"]["original_dataset_id"] = None
    with pytest.raises(ValueError, match="original pin"):
        LexicalContaminationEvidence.model_validate(document)
    document["receipt"]["release"]["original_dataset_id"] = "upstream/another-population"
    with pytest.raises(ValueError, match="different measured input"):
        LexicalContaminationEvidence.model_validate(document)
    document["receipt"]["release"]["original_dataset_id"] = plan.source_dataset_id
    document["receipt"]["release"]["original_dataset_revision"] = "main"
    with pytest.raises(ValueError):
        LexicalContaminationEvidence.model_validate(document)


def test_retained_import_keeps_original_serialization_runtime_and_benchmark_use(tmp_path, retained_receipt, monkeypatch):
    _, receipt = retained_receipt
    spec = json.loads(prebuilt_task("retained")["lowered_task_spec"])["task"]
    encoded = json.dumps(spec, indent=2)
    local = tmp_path / "tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_spec": [encoded],
                "task_sha256": [hashlib.sha256(encoded.encode()).hexdigest()],
                "runtime": ['{ "task_machine": {"cpus": 2} }'],
            }
        ),
        local,
    )
    monkeypatch.setattr(retained_release, "hf_hub_download", lambda *args, **kwargs: str(local))
    cohort = RetainedCohort(
        name="fixture",
        version=RETAINED_IMPORT_VERSION,
        release=receipt.publication.release,
        original_dataset=SourceReference("upstream/fixture", "a" * 40, "https://example.test/upstream"),
        expected_rows=1,
        tasks_parquet_sha256=file_sha256(StoragePath(str(local))),
        intended_use=IntendedUse.EVAL,
        evidence_url=receipt.publication.evidence_url,
    )
    result = run_retained_import(
        RetainedImport(cohort, str(tmp_path / "imported"), SourceProcessingMode.QUICK, None, None, None)
    )
    normalized = pq.read_table(tmp_path / "imported/normalize/part-000000.parquet").to_pylist()
    assert normalized[0]["task_json"] == encoded
    assert normalized[0]["runtime"] == '{ "task_machine": {"cpus": 2} }'
    manifest = json.loads(StoragePath(result.result.evidence["manifest"]).read_text())
    assert manifest["intended_use"] == "eval" and manifest["admitted_rows"] == 0


def test_retained_census_measures_new_pin_and_normalizes_unchanged_json(tmp_path, retained_receipt, monkeypatch):
    _, receipt = retained_receipt
    spec = TaskSpec.model_validate(json.loads(prebuilt_task("retained")["lowered_task_spec"])["task"])
    encoded = spec.model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(),
            "grader": NoGrader(reason="Inspection fixture has no grader"),
        }
    ).model_dump_json()
    specs = tmp_path / "published/specs"
    specs.mkdir(parents=True)
    tasks = specs / "part-000000.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["retained"],
                "task_spec": [encoded],
                "runtime": ["{}"],
                "task_sha256": [hashlib.sha256(encoded.encode()).hexdigest()],
            }
        ),
        tasks,
    )
    publication = receipt.publication.model_copy(
        update={
            "specs_output_dir": str(specs),
            "tasks_parquet_sha256": file_sha256(StoragePath(str(tasks))),
        }
    )
    monkeypatch.setattr(retained_release, "hf_hub_download", lambda *args, **kwargs: str(tasks))
    result = census_release(publication, str(tmp_path / "census"), "retained-code-revision")
    assert result["actor"]["input"]["dataset_id"] == publication.repo
    assert result["actor"]["input"]["dataset_revision"] == publication.revision
    assert result["actor"]["measurement"]["daytona_machine_slots"] == 0
    assert result["dedup"]["raw_count"] == result["dedup"]["unique_count"] == 1
    assert result["population"]["task_specs_sha256"] == hashlib.sha256(encoded.encode()).hexdigest()
    normalized = pq.read_table(result["normalized_path"] + "/normalize/part-000000.parquet").to_pylist()
    assert normalized[0]["task_json"] == encoded and normalized[0]["runtime"] == "{}"
    grading = grading_receipt(result, str(tmp_path / "grading"))
    assert grading.input == ActorCensusInput.model_validate(result["actor"]["input"])
    assert grading.evidence.assessed_tasks == 1
    assert grading.evidence.required_counts == {"separate": 1}
    assert grading.evidence.status_counts == {"unavailable": 1}


def test_final_composition_binds_grading_evidence_measured_on_the_retained_census(
    tmp_path, retained_receipt, monkeypatch
):
    """The final all_sources chain keeps grading evidence from the retained census input.

    A grading census measured on an earlier population input must be rejected at the
    final composition, and one measured on the retained census must bind through the
    same composition the catalog uses.
    """
    source, receipt = retained_receipt
    original_input = source.info.actor_census_input
    spec = TaskSpec.model_validate(json.loads(prebuilt_task("retained")["lowered_task_spec"])["task"])
    encoded = spec.model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(),
            "grader": NoGrader(reason="Inspection fixture has no grader"),
        }
    ).model_dump_json()
    specs = tmp_path / "published/specs"
    specs.mkdir(parents=True)
    tasks = specs / "part-000000.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["retained"],
                "task_spec": [encoded],
                "runtime": ["{}"],
                "task_sha256": [hashlib.sha256(encoded.encode()).hexdigest()],
            }
        ),
        tasks,
    )
    publication = receipt.publication.model_copy(
        update={
            "specs_output_dir": str(specs),
            "tasks_parquet_sha256": file_sha256(StoragePath(str(tasks))),
        }
    )
    monkeypatch.setattr(retained_release, "hf_hub_download", lambda *args, **kwargs: str(tasks))
    result = census_release(publication, str(tmp_path / "census"), "retained-code-revision")
    measured_input = ActorCensusInput.model_validate(result["actor"]["input"])
    receipt = receipt.model_copy(
        update={
            "actor": receipt.actor.model_copy(update={"input": measured_input}),
            "grading": receipt.grading.model_copy(update={"input": measured_input}),
        }
    )
    bound = bind_retained_release(source, receipt)
    stale = GradingCensusReceipt(
        source_id=source.info.id,
        input=original_input,
        population_path="https://example.test/population",
        scanned_task_specs_sha256=original_input.task_specs_sha256,
        evidence=GradingEnvironmentEvidence(
            assessed_tasks=3,
            required_counts={"separate": 3},
            declared_counts={"separate": 3},
            status_counts={"matched": 3},
            reason_counts={"fixture": 3},
            evidence_url="https://example.test/stale-grading",
        ),
        task_evidence_uri="https://example.test/stale-tasks.parquet",
        unassessable_reasons={},
    )
    assert bind_grading_census(bound, stale).info.grading_environment is None
    census = census_source(
        GradingCensusInput(
            population=SourceTextPopulation.model_validate(result["population"]),
            input=ActorCensusInput.model_validate(result["actor"]["input"]),
        ),
        str(tmp_path / "retained-census"),
    )
    final = bind_grading_census(bound, census)
    assert final.info.grading_environment == census.evidence


@pytest.mark.parametrize("mode", [SourceProcessingMode.SAMPLE, SourceProcessingMode.FULL])
def test_qualified_retained_import_admits_exact_bytes_with_original_sample_provenance(
    tmp_path, retained_receipt, monkeypatch, mode
):
    original, receipt = retained_receipt
    encoded = json.dumps(json.loads(prebuilt_task("retained")["lowered_task_spec"])["task"], indent=2)
    runtime = '{ "task_machine": {"cpus": 2} }'
    local = tmp_path / "tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_spec": [encoded],
                "task_sha256": [hashlib.sha256(encoded.encode()).hexdigest()],
                "runtime": [runtime],
            }
        ),
        local,
    )
    publication = receipt.publication.model_copy(update={"tasks_parquet_sha256": file_sha256(StoragePath(str(local)))})
    source = bind_retained_release(original, receipt.model_copy(update={"publication": publication}))
    monkeypatch.setattr(retained_release, "hf_hub_download", lambda *args, **kwargs: str(local))
    step = source.pipeline(source, PipelineOptions(mode, CampaignRuntime()))
    destination = tmp_path / "admitted"
    artifact = step.run(step.build_config(StepContext.for_run(str(destination), str(tmp_path))))
    final = pq.read_table(destination / "final/part-000000.parquet").to_pylist()
    assert [(row["task_json"], row["runtime"]) for row in final] == [(encoded, runtime)]
    assert (destination / "tasks.parquet").read_bytes() == local.read_bytes()
    manifest = json.loads(StoragePath(artifact.result.evidence["manifest"]).read_text())
    assert manifest["mode"] == mode.value and manifest["admitted_rows"] == 1
    assert manifest["qualification_newly_executed"] is False
    assert manifest["inherited_oversight"]["level"] == "sample"
    assert manifest["original_dataset"]["repo"] == original.dataset.name
    assert manifest["original_dataset"]["revision"] == original.dataset.revision
    assert manifest["repo"] == publication.repo and manifest["revision"] == publication.revision
    assert manifest["retention_evidence"] == publication.evidence_url
    assert artifact.result.outputs["final"] == str(destination / "final")
    # Changing published bytes after the readback receipt invalidates admission.
    pq.write_table(pa.table({"task_spec": [encoded], "task_sha256": ["0" * 64], "runtime": [runtime]}), local)
    with pytest.raises(ValueError, match="publication readback"):
        step.run(step.build_config(StepContext.for_run(str(tmp_path / "changed"), str(tmp_path))))


@pytest.mark.parametrize("missing", ["oversight", "actor", "private_receipt", "excluded"])
def test_unqualified_retained_release_cannot_enter_admitted_import(retained_receipt, missing):
    original, receipt = retained_receipt
    source = bind_retained_release(original, receipt)
    changes = {
        "oversight": {"oversight": None},
        "actor": {"daytona_snapshot_count": None},
        "private_receipt": {"lexical_contamination": None},
        "excluded": {"tags": (*source.info.tags, "excluded")},
    }[missing]
    source = replace(source, info=replace(source.info, **changes))
    with pytest.raises(ValueError, match=r"cannot admit|excluded"):
        source.pipeline(source, PipelineOptions(SourceProcessingMode.FULL, CampaignRuntime()))
