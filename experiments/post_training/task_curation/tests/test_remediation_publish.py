# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    ContentEvidenceReference,
    DaytonaMeasurement,
    LexicalContaminationEvidence,
    NoOracleExecutabilityPolicy,
    NoOracleStaticRetentionEvidence,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    SourceOrigin,
    SourceQualityResult,
    WithinSourceDedupEvidence,
)
from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseReceipt
from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub import DatasetCard
from huggingface_hub.hf_api import RepoFile
from marin.execution.lazy import StepContext
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import EnvironmentRequirements, NoGrader, TaskSpec
from taskcompendium.pipeline.models import IntendedUse
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.post_training.task_curation import (
    license_publication,
    remediation_publish,
    retained_release,
)
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.census_binding import ActorCensusReceipt, bind_actor_census
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.lexical_binding import LexicalMeasurementReceipt
from experiments.post_training.task_curation.lexical_policy import source_lexical_use
from experiments.post_training.task_curation.license_resolution import license_evidence
from experiments.post_training.task_curation.remediation_binding import (
    ArchivedRetainedBinding,
    PublicationReceipt,
    RetainedReleaseReceipt,
    StaticRetentionBinding,
    bind_archived_retained_release,
    bind_retained_release,
    measured_intake_source,
    release_lexical_evidence,
    retained_lineage,
)
from experiments.post_training.task_curation.remediation_publish import (
    RawSourceArchive,
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
from experiments.post_training.task_curation.tests.admission_fixtures import with_consumer_receipt
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license, with_license
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_source, prebuilt_task
from infra.marina.apps.rl_data_catalog.server.admission import require_private_release
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
        data_restriction=DataRestriction.OPEN,
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


def test_retained_population_writes_the_retained_schema(tmp_path, frozen_population):
    """The retained census pins the retained tasks.parquet schema: every row carries its
    path and a task_sha256 verified byte-exact against the published task_spec payload."""
    plan, specs, _ = frozen_population
    retained, original = tmp_path / "retained.parquet", tmp_path / "original.parquet"
    filter_population(plan, str(specs), retained, original)
    expected_columns = ["id", "path", "task_spec", "task_sha256", "runtime"]
    assert pq.read_table(retained).column_names == expected_columns
    assert pq.read_table(original).column_names == expected_columns
    for row in pq.read_table(retained).to_pylist():
        assert row["path"] == row["id"]
        assert hashlib.sha256(row["task_spec"].encode()).hexdigest() == row["task_sha256"]


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


@pytest.fixture
def retained_receipt(frozen_population):
    plan, _, _ = frozen_population
    source = prebuilt_source(
        SourceInfo(
            data_restriction=DataRestriction.OPEN,
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
    return with_consumer_receipt(source), receipt


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


def static_binding(source_id, release, actor):
    policy = NoOracleExecutabilityPolicy(
        version=1,
        executability_proof_required_for_no_oracle=False,
        decision=ContentEvidenceReference(uri="s3://fixtures/no-oracle-ruling.md", sha256="a" * 64),
    )
    return StaticRetentionBinding(
        policy=policy,
        evidence=NoOracleStaticRetentionEvidence(
            source_id=source_id,
            source_release=release,
            current_input=actor,
            intended_use="train",
            retained_cohort=ContentEvidenceReference(uri="s3://fixtures/current-cohort.parquet", sha256="b" * 64),
            retained_count=actor.measured_tasks,
            static_retained_count=actor.measured_tasks,
            exclusion_counts={},
            oracle_filter=ContentEvidenceReference(uri="s3://fixtures/static-filter.json", sha256="c" * 64),
            policy_sha256=policy.sha256,
        ),
    )


def test_retained_static_evidence_survives_archive_roundtrip_and_binds_only_to_selected_pin(retained_receipt):
    source, receipt = retained_receipt
    static = static_binding(receipt.source_id, receipt.publication.release, receipt.actor.input)
    archived = RetainedReleaseReceipt.model_validate({**receipt.model_dump(), "static_retention": static})
    restored = RetainedReleaseReceipt.model_validate_json(archived.model_dump_json())
    bound = bind_retained_release(source, restored)
    assert bound.declaration.no_oracle_static_retention == static.evidence
    assert bound.declaration.no_oracle_executability_policy == static.policy
    assert bound.declaration.dataset_revision == static.evidence.source_release.revision
    assert source.info.no_oracle_static_retention is None
    assert restored.original_info == receipt.original_info


def test_old_upstream_static_evidence_cannot_certify_the_retained_pin(retained_receipt):
    source, receipt = retained_receipt
    old = static_binding(source.info.id, source.current_release, source.info.actor_census_input)
    with pytest.raises(ValueError, match="another retained population"):
        RetainedReleaseReceipt.model_validate({**receipt.model_dump(), "static_retention": old})


@pytest.mark.parametrize("stale_actor", [False, True])
def test_pending_retention_requires_original_measurement_without_inheriting_admission(retained_receipt, stale_actor):
    source, receipt = retained_receipt
    original = receipt.actor.model_copy(
        update={
            "input": source.info.actor_census_input,
            "measurement": receipt.actor.measurement.model_copy(
                update={"dataset_id": source.dataset.name, "dataset_revision": source.dataset.revision}
            ),
        }
    )
    pending = replace(
        source,
        info=replace(source.info, actor_census_input=None, within_source_dedup=None, oversight=None),
    )
    if stale_actor:
        original = original.model_copy(update={"input": original.input.model_copy(update={"measured_tasks": 4})})
        with pytest.raises(ValueError, match="complete original actor census"):
            measured_intake_source(pending, receipt.publication, original)
        return
    measured = measured_intake_source(pending, receipt.publication, original)
    bound = bind_retained_release(measured, receipt)
    assert bound.current_release == receipt.publication.release
    assert bound.info.oversight is None and bound.info.inherited_admission is None
    assert bound.declaration.status != "Available"
    assert pending.info.actor_census_input is None and pending.info.within_source_dedup is None


def test_resumed_publication_rechecks_license_and_remains_readable_by_retained_binding(
    tmp_path, monkeypatch, retained_receipt
):
    _, receipt = retained_receipt
    tasks = tmp_path / "cached-tasks.parquet"
    tasks.write_bytes(b"frozen publication fixture")
    publication = receipt.publication.model_copy(update={"tasks_parquet_sha256": file_sha256(StoragePath(str(tasks)))})
    evidence = tmp_path / "evidence"
    output = evidence / hashlib.sha256(publication.source_id.encode()).hexdigest()[:20]
    output.mkdir(parents=True)
    (output / "result.json").write_text(publication.model_dump_json())
    monkeypatch.setattr(remediation_publish, "hf_hub_download", lambda *args, **kwargs: str(tasks))
    monkeypatch.setattr(
        remediation_publish,
        "HfApi",
        lambda: SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(private=True)),
    )
    checked = []

    def resolve(target):
        checked.append((target.dataset_id, target.revision))
        return fixture_license(target.source_id, target.dataset_id, target.revision)

    monkeypatch.setattr(license_publication, "resolve_license", resolve)
    plan = publication.plan
    result = remediation_publish.publish_plan(
        plan,
        "fixture",
        str(evidence),
        fixture_license(plan.source_id, plan.source_dataset_id, plan.source_revision),
        None,
    )
    assert PublicationReceipt.model_validate(result).release == publication.release
    assert checked == [(publication.repo, publication.revision)]


def test_cli_publishes_clean_population_with_unchanged_tasks(tmp_path, monkeypatch, frozen_population):
    original, specs, rows = frozen_population
    drops = tmp_path / "empty-drops.parquet"
    pq.write_table(pa.table({"task_id": pa.array([], type=pa.string())}), drops)
    plan = original.model_copy(
        update={
            "unique_count": 3,
            "within_source_duplicates": 0,
            "raw_lexical_count": 0,
            "unique_lexical_count": 0,
            "combined_drop_count": 0,
            "retained_count": 3,
            "multiplicities": {"1": 3},
            "drop_manifest_uri": str(drops),
            "combined_drop_manifest_uri": str(drops),
        }
    )
    tasks = tmp_path / "published-tasks.parquet"
    selected = filter_population(plan, str(specs), tasks, None)
    lexical = decontamination_evidence(plan.source_dataset_id, plan.source_revision, 3).model_dump(mode="json")
    lexical.update(receipt=None, converter_revision=plan.converter_revision)
    publication = PublicationReceipt(
        source_id=plan.source_id,
        plan=plan,
        repo="open-athena/clean-fixture",
        revision="d" * 40,
        url="https://huggingface.co/datasets/open-athena/clean-fixture/tree/" + "d" * 40,
        private=True,
        format="taskcompendium",
        original_private_pin={"repo": plan.source_dataset_id, "revision": plan.source_revision},
        tasks_parquet_sha256=file_sha256(StoragePath(str(tasks))),
        retained_content_sha256=verify_retained(tasks, selected["hashes"]),
        verified_retained_tasks=3,
        specs_output_dir=str(specs),
        evidence_url="https://example.test/clean-publication",
        lexical=lexical,
    )
    evidence = tmp_path / "evidence"
    output = evidence / hashlib.sha256(plan.source_id.encode()).hexdigest()[:20]
    output.mkdir(parents=True)
    (output / "result.json").write_text(publication.model_dump_json())
    plans, licenses = tmp_path / "plans.json", tmp_path / "licenses"
    licenses.mkdir()
    plans.write_text(json.dumps([plan.model_dump(mode="json")]))
    (licenses / "part-000.json").write_text(
        json.dumps(
            {
                plan.source_id: (
                    fixture_license(plan.source_id, plan.source_dataset_id, plan.source_revision).model_dump(mode="json")
                )
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remediation_publish",
            "--plans",
            str(plans),
            "--license-receipts",
            str(licenses),
            "--evidence-root",
            str(evidence),
            "--version",
            "fixture",
            "--workers",
            "1",
        ],
    )
    # Keep the Hub I/O fakes in process while exercising the real publication path.
    monkeypatch.setattr(remediation_publish, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(remediation_publish, "hf_hub_download", lambda *args, **kwargs: str(tasks))
    monkeypatch.setattr(
        remediation_publish,
        "HfApi",
        lambda: SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(private=True)),
    )
    monkeypatch.setattr(
        license_publication,
        "resolve_license",
        lambda target: fixture_license(target.source_id, target.dataset_id, target.revision),
    )
    remediation_publish.main()
    (result,) = [PublicationReceipt.model_validate(row) for row in json.loads((evidence / "releases.json").read_text())]
    assert result.verified_retained_tasks == 3
    assert release_lexical_evidence(result).receipt.status.value == "clean"
    assert [(row["id"], row["task_spec"], row["runtime"]) for row in pq.read_table(tasks).to_pylist()] == [
        (row["id"], row["task_spec"], row["runtime"]) for row in rows
    ]


def test_flagged_external_input_requires_its_raw_snapshot_before_task_read(tmp_path, frozen_population):
    plan, _, _ = frozen_population
    with pytest.raises(ValueError, match="requires a verified original raw source archive"):
        remediation_publish.publish_plan(
            plan,
            "fixture",
            str(tmp_path / "evidence"),
            fixture_license(plan.source_id, plan.source_dataset_id, plan.source_revision),
            None,
        )


@pytest.fixture
def staged_publication(tmp_path, monkeypatch, frozen_population):
    original, specs, _ = frozen_population
    plan = original.model_copy(update={"source_dataset_id": "open-athena/repaired-fixture"})
    measurement = decontamination_evidence(plan.source_dataset_id, plan.source_revision, 3).model_copy(
        update={
            "receipt": None,
            "converter_revision": plan.converter_revision,
            "contaminated_tasks": 1,
            "contaminated_fraction": 1 / 3,
            "max_overlap": 1,
            "mean_max_overlap": 1 / 3,
            "p95_max_overlap": 1,
        }
    )
    receipt = LexicalMeasurementReceipt(
        source_id=plan.source_id,
        input=ActorCensusInput(
            dataset_id=plan.source_dataset_id,
            dataset_revision=plan.source_revision,
            converter_version=plan.converter_version,
            converter_revision=plan.converter_revision,
            measured_tasks=3,
            task_specs_sha256=plan.task_specs_sha256,
            evidence_url=plan.census_evidence_url,
        ),
        use=plan.lexical_use,
        measurement=measurement,
    )
    metadata = {
        "dataset_id": plan.source_dataset_id,
        "dataset_revision": plan.source_revision,
        "converter_revision": plan.converter_revision,
        "source_version": plan.converter_version,
        "task_specs_sha256": plan.task_specs_sha256,
        "task_count": plan.raw_count,
        "source_id": plan.source_id,
        "specs_output_dir": str(specs),
        "text_renderer_version": measurement.text_renderer_version,
    }
    (tmp_path / ".artifact.json").write_text(json.dumps({"result": metadata}))
    canonical = {plan.source_id: receipt}
    monkeypatch.setattr(remediation_publish, "lexical_measurements", lambda: canonical)
    hub = tmp_path / "hub"
    upstream = hub / plan.source_dataset_id
    upstream.mkdir(parents=True)
    license_bytes = b"SPDX-License-Identifier: MIT\n"
    (upstream / "LICENSE").write_bytes(license_bytes)

    class LocalHub:
        def create_repo(self, *, repo_id, repo_type, private, exist_ok):
            (hub / repo_id).mkdir(parents=True, exist_ok=exist_ok)

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=True, sha=kwargs.get("revision", "c" * 40))

        def list_repo_tree(self, repo_id, *, repo_type, revision, recursive):
            return [
                RepoFile(path=path.name, size=path.stat().st_size, oid="b" * 40) for path in (hub / repo_id).iterdir()
            ]

        def upload_folder(self, *, repo_id, repo_type, folder_path, commit_message, **kwargs):
            shutil.copytree(folder_path, hub / repo_id, dirs_exist_ok=True)
            return SimpleNamespace(oid="d" * 40)

    def download(repo_id, filename, *, repo_type, revision):
        return str(hub / repo_id / filename)

    for module in (remediation_publish, license_publication):
        monkeypatch.setattr(module, "HfApi", LocalHub)
        monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(
        license_publication,
        "resolve_license",
        lambda target: fixture_license(target.source_id, target.dataset_id, target.revision),
    )
    # The chokepoint reads the parent's receipt from the registry and re-pins it:
    # the fixture registers the pinned upstream grant and records the registry write.
    prefix = f"https://huggingface.co/datasets/{plan.source_dataset_id}/resolve/{plan.source_revision}/"
    parent_receipt = LicenseReceipt(
        resolver_version=LICENSE_RESOLVER_VERSION,
        source_id=plan.source_id,
        dataset_id=plan.source_dataset_id,
        revision=plan.source_revision,
        resolved_at=datetime(2026, 11, 15, tzinfo=UTC).isoformat(),
        evidence=[license_evidence(prefix + "LICENSE", license_bytes, "license_file")],
        bundled_third_party=False,
        model_generated_outputs=False,
    )
    monkeypatch.setattr(
        license_publication, "read_license_receipts", lambda index_path: {plan.source_id: parent_receipt}
    )
    return plan, receipt, canonical, hub


@pytest.mark.parametrize("entrypoint", ["canonical", "api", "cli"])
def test_repaired_population_publishes_from_exact_receipt_without_replacing_selected_evidence(
    tmp_path, monkeypatch, staged_publication, entrypoint
):
    plan, receipt, canonical, hub = staged_publication
    if entrypoint != "canonical":
        canonical[plan.source_id] = LexicalMeasurementReceipt.model_validate(
            {
                **receipt.model_dump(),
                "input": receipt.input.model_copy(update={"dataset_id": "open-athena/selected-fixture"}),
                "measurement": receipt.measurement.model_copy(
                    update={"input_dataset_id": "open-athena/selected-fixture"}
                ),
            }
        )
    selected_before = canonical[plan.source_id].model_dump_json()
    evidence = tmp_path / "evidence"
    license_receipt = fixture_license(plan.source_id, plan.source_dataset_id, plan.source_revision).model_copy(
        update={
            "evidence": [
                license_evidence(
                    f"https://huggingface.co/datasets/{plan.source_dataset_id}/resolve/{plan.source_revision}/LICENSE",
                    b"SPDX-License-Identifier: MIT\n",
                    "license_file",
                )
            ]
        }
    )
    if entrypoint == "cli":
        plans, licenses, measurements = tmp_path / "plans.json", tmp_path / "licenses", tmp_path / "measurements"
        licenses.mkdir()
        measurements.mkdir()
        plans.write_text(json.dumps([plan.model_dump(mode="json")]))
        (licenses / "part-000.json").write_text(json.dumps({plan.source_id: license_receipt.model_dump(mode="json")}))
        (measurements / "part-000.json").write_text(json.dumps({plan.source_id: receipt.model_dump(mode="json")}))
        monkeypatch.setattr(remediation_publish, "ProcessPoolExecutor", ThreadPoolExecutor)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "remediation_publish",
                "--plans",
                str(plans),
                "--license-receipts",
                str(licenses),
                "--lexical-receipts",
                str(measurements),
                "--evidence-root",
                str(evidence),
                "--version",
                "fixture",
                "--workers",
                "1",
            ],
        )
        remediation_publish.main()
        (result,) = json.loads((evidence / "releases.json").read_text())
    else:
        result = remediation_publish.publish_plan(
            plan,
            "fixture",
            str(evidence),
            license_receipt,
            None,
            measurement_receipt=receipt if entrypoint == "api" else None,
        )
    published = hub / result["repo"]
    tasks = pq.read_table(published / "tasks.parquet").to_pylist()
    assert [(row["id"], row["runtime"]) for row in tasks] == [("retained", '{ "cpus": 2 }')]
    assert json.loads((published / "manifest.json").read_text())["lexical"]["input_dataset_id"] == plan.source_dataset_id
    assert canonical[plan.source_id].model_dump_json() == selected_before
    # The publication carried the licence forward mechanically: the card declares the
    # ingest record's provenance and nothing is written to the receipts registry.
    card = DatasetCard((hub / result["repo"] / "README.md").read_text()).data.to_dict()
    assert card["license_source"]["dataset_id"] == plan.source_dataset_id


def test_staged_receipt_for_same_count_but_different_tasks_cannot_publish(tmp_path, staged_publication):
    plan, receipt, _unused, hub = staged_publication
    stale = receipt.model_copy(update={"input": receipt.input.model_copy(update={"task_specs_sha256": "f" * 64})})
    with pytest.raises(ValueError, match="Frozen lexical input differs"):
        remediation_publish.publish_plan(
            plan,
            "fixture",
            str(tmp_path / "evidence"),
            fixture_license(plan.source_id, plan.source_dataset_id, plan.source_revision),
            None,
            measurement_receipt=stale,
        )
    assert not list(hub.glob("**/tasks.parquet"))


@pytest.mark.parametrize("mutation", [None, "missing_file", "changed_bytes"])
def test_raw_source_archive_readback_rejects_missing_or_changed_files(
    tmp_path, monkeypatch, frozen_population, mutation
):
    plan, _, _ = frozen_population
    repository = tmp_path / "raw-repository"
    (repository / "raw").mkdir(parents=True)
    content = b"original source bytes"
    (repository / "raw/data.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "source_id": plan.source_id,
        "dataset_id": plan.source_dataset_id,
        "revision": plan.source_revision,
        "source_files": ["data.json"],
        "files": [{"path": "data.json", "sha256": digest}],
    }
    (repository / "raw-snapshot.json").write_text(json.dumps(manifest))
    hashes = {
        "raw/data.json": digest,
        "raw-snapshot.json": file_sha256(StoragePath(str(repository / "raw-snapshot.json"))),
    }
    if mutation == "missing_file":
        hashes.pop("raw/data.json")
    if mutation == "changed_bytes":
        (repository / "raw/data.json").write_bytes(b"changed source bytes")
    archive = RawSourceArchive(
        source_id=plan.source_id,
        dataset_id=plan.source_dataset_id,
        source_revision=plan.source_revision,
        repo="open-athena/raw-fixture",
        revision="f" * 40,
        url="https://huggingface.co/datasets/open-athena/raw-fixture/tree/" + "f" * 40,
        private=True,
        scope="exact_pinned_raw_source_archive",
        source_artifact=plan.input_artifact,
        verified_raw_files=1,
        file_hashes=hashes,
        license_check={"verdict": "PASS"},
    )
    monkeypatch.setattr(
        remediation_publish,
        "HfApi",
        lambda: SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(private=True)),
    )
    monkeypatch.setattr(
        remediation_publish, "hf_hub_download", lambda repo, filename, **kwargs: str(repository / filename)
    )
    monkeypatch.setattr(
        license_publication,
        "resolve_license",
        lambda target: fixture_license(target.source_id, target.dataset_id, target.revision),
    )
    if mutation is not None:
        with pytest.raises(ValueError, match=r"complete frozen source snapshot|readback differs"):
            remediation_publish.verify_raw_archive(plan, archive)
        return
    release = remediation_publish.verify_raw_archive(plan, archive)
    assert (release.repo, release.revision) == (archive.repo, archive.revision)


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
    source = with_consumer_receipt(
        with_license(bind_retained_release(original, receipt.model_copy(update={"publication": publication})))
    )
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
    source = with_consumer_receipt(with_license(bind_retained_release(original, receipt)))
    changes = {
        "oversight": {"oversight": None},
        "actor": {"daytona_snapshot_count": None},
        "private_receipt": {"lexical_contamination": None},
        "excluded": {"tags": (*source.info.tags, "excluded")},
    }[missing]
    source = replace(source, info=replace(source.info, **changes))
    with pytest.raises(ValueError, match=r"cannot admit|excluded"):
        source.pipeline(source, PipelineOptions(SourceProcessingMode.FULL, CampaignRuntime()))


@pytest.mark.parametrize("empty", [False, True])
def test_retained_admission_rejects_missing_grader_or_empty_population(tmp_path, retained_receipt, monkeypatch, empty):
    original, receipt = retained_receipt
    task = TaskSpec.model_validate(json.loads(prebuilt_task("retained")["lowered_task_spec"])["task"])
    task = task.model_copy(update={"grader": NoGrader(reason="Preference labels cannot grade new replies")})
    encoded = task.model_dump_json()
    local = tmp_path / "tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_spec": [] if empty else [encoded],
                "task_sha256": [] if empty else [hashlib.sha256(encoded.encode()).hexdigest()],
                "runtime": [] if empty else ["{}"],
            }
        ),
        local,
    )
    publication = receipt.publication.model_copy(update={"tasks_parquet_sha256": file_sha256(StoragePath(str(local)))})
    source = with_consumer_receipt(
        with_license(bind_retained_release(original, receipt.model_copy(update={"publication": publication})))
    )
    assert source.config is not None
    monkeypatch.setattr(retained_release, "hf_hub_download", lambda *args, **kwargs: str(local))
    invocation = RetainedImport(
        replace(source.config, expected_rows=0 if empty else 1),
        str(tmp_path / "admitted"),
        SourceProcessingMode.FULL,
        source.info.oversight,
        source.info.inherited_admission,
        source.info.lexical_contamination,
    )
    with pytest.raises(ValueError, match="nonempty retained population" if empty else "no runnable grader"):
        run_retained_import(invocation)
    assert not (tmp_path / "admitted" / "final").exists()
    if not empty:
        # QUICK can retain unsupported contracts for diagnosis without admitting them.
        quick = run_retained_import(
            replace(invocation, mode=SourceProcessingMode.QUICK, output_path=str(tmp_path / "quick"))
        )
        assert "final" not in quick.result.outputs


def zero_drop_receipt(receipt):
    """The fixture receipt re-published with every task retained and nothing dropped."""
    plan = receipt.publication.plan.model_copy(
        update={
            "unique_count": 3,
            "within_source_duplicates": 0,
            "raw_lexical_count": 0,
            "unique_lexical_count": 0,
            "combined_drop_count": 0,
            "retained_count": 3,
        }
    )
    lexical = receipt.publication.lexical.model_copy(update={"contaminated_tasks": 0, "measured_tasks": 3})
    publication = receipt.publication.model_copy(update={"plan": plan, "verified_retained_tasks": 3, "lexical": lexical})
    dedup = receipt.retained_dedup.model_copy(update={"raw_count": 3, "unique_count": 3})
    actor = receipt.actor.model_copy(
        update={"input": receipt.actor.input.model_copy(update={"measured_tasks": 3, "task_specs_sha256": "1" * 64})}
    )
    return receipt.model_copy(update={"publication": publication, "actor": actor, "retained_dedup": dedup})


def test_lineage_proof_requires_a_zero_drop_readback(retained_receipt):
    _, receipt = retained_receipt
    assert retained_lineage(receipt) is None
    lineage = retained_lineage(zero_drop_receipt(receipt))
    assert lineage is not None
    assert lineage.source_id == receipt.source_id
    assert lineage.original_dataset_id == receipt.publication.plan.source_dataset_id
    assert lineage.original_dataset_revision == receipt.publication.plan.source_revision
    assert lineage.original_input == receipt.original_info.actor_census_input
    assert lineage.original_unique_count == receipt.original_info.within_source_dedup.unique_count
    assert lineage.retained_content_sha256 == receipt.publication.retained_content_sha256


def test_archived_retention_keeps_frozen_pin_after_live_recipe_advances(retained_receipt):
    source, record = retained_receipt
    advanced = replace(source, config=replace(source.config, version="next-conversion"))
    with pytest.raises(ValueError, match="different conversion input"):
        bind_retained_release(advanced, record)
    binding = ArchivedRetainedBinding(
        retained=record, authority=ContentEvidenceReference(uri="memory://archive-ruling", sha256="a" * 64)
    )
    frozen = bind_archived_retained_release(ArchivedRetainedBinding.model_validate_json(binding.model_dump_json()))
    assert frozen.current_release == record.publication.release
    assert frozen.config.tasks_parquet_sha256 == record.publication.tasks_parquet_sha256
    assert frozen.config.expected_rows == record.publication.verified_retained_tasks
    assert frozen.info.actor_census_input == record.actor.input
    assert frozen.pipeline is retained_release.import_retained
    assert advanced.version == "next-conversion" and record.original_pipeline.version == source.version
