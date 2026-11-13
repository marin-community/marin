# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configured source identity controls whether a census can publish a verdict."""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    DaytonaMeasurement,
    DaytonaSafety,
    IntegrityContractSummary,
    SourceDeclaration,
)
from finestore.schema import arrow_schema
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.models import NoGrader, TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.census_binding import (
    ActorCensusReceipt,
    bind_actor_census,
    integrity_summary_for_actor,
)
from experiments.post_training.task_curation.grading_binding import bind_grading_census
from experiments.post_training.task_curation.grading_census import GradingCensusInput, census_source
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def census_receipt(source) -> ActorCensusReceipt:
    return ActorCensusReceipt(
        source_id=source.info.id,
        input=ActorCensusInput(
            dataset_id=source.dataset.name,
            dataset_revision=source.dataset.revision,
            converter_version=source.version,
            converter_revision="a" * 40,
            measured_tasks=1,
            task_specs_sha256="b" * 64,
            evidence_url="s3://fixture/census/.artifact.json",
        ),
        measurement=DaytonaMeasurement(
            dataset_id=source.dataset.name,
            dataset_revision=source.dataset.revision,
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Full source census",
            daytona_unsafe_reasons=[],
            daytona_unsafe_details={},
            daytona_unsafe_counts={},
        ),
        integrity_contracts=None,
    )


def test_bound_zero_actor_receipt_survives_catalog_arrow_roundtrip(tmp_path):
    source = number_source(tmp_path / "input")
    bound = bind_actor_census(source, census_receipt(source))
    declaration = bound.declaration
    table = pa.Table.from_pylist([declaration.model_dump(mode="json")], schema=arrow_schema(SourceDeclaration))
    restored = SourceDeclaration.model_validate(table.to_pylist()[0])
    assert restored.daytona_safety is DaytonaSafety.NOT_APPLICABLE
    assert restored.actor_census_input.converter_version == source.version
    assert restored.actor_census_input.measured_tasks == 1
    changed = replace(
        bound,
        config=replace(bound.config, version="2"),
        info=replace(
            bound.info, within_source_dedup=deduplication_evidence(bound.dataset.name, bound.dataset.revision, "2")
        ),
    )
    with pytest.raises(ValueError, match="Actor census input or converter"):
        _ = changed.declaration


@pytest.mark.parametrize("changed", ["dataset", "revision", "converter", "missing"])
def test_changed_input_or_missing_receipt_clears_stale_inline_census(tmp_path, changed):
    source = number_source(tmp_path / "input")
    receipt = census_receipt(source)
    if changed in {"dataset", "revision"}:
        dataset = replace(
            source.dataset, **({"name": "open-athena/replaced"} if changed == "dataset" else {"revision": "c" * 40})
        )
        source = replace(source, config=replace(source.config, dataset=dataset))
    elif changed == "converter":
        source = replace(source, config=replace(source.config, version="2"))
    else:
        receipt = None
    bound = bind_actor_census(source, receipt)
    assert bound.catalog_declaration is None
    assert not bound.has_actor_census
    assert bound.info.daytona_snapshot_count is None
    assert bound.info.actor_census_input is None


def test_refreshed_contract_families_and_full_example_survive_arrow():
    actor = {
        "source_id": "fixture",
        "input_artifact": "s3://fixture/specs",
        "converter_version": "1",
        "converter_revision": "a" * 40,
        "task_count": 2,
    }
    contract = {"id": "native-shared", "native": {"family": "pytest"}, "harness": "protected-control"}
    raw = {
        "schema_version": 2,
        "distinct_groups": 2,
        "absent_contract_tasks": 0,
        "required_shared_tasks": 2,
        "shared_missing_contract_tasks": 0,
        "evidence_url": "s3://fixture/contracts.json",
        "dataset_id": "private/fixture",
        "dataset_revision": "b" * 40,
        "task_specs_sha256": "c" * 64,
        "example": {"task_id": "task-1", "contract": contract},
        "groups": [
            {
                "id": "native-shared",
                "version": 1,
                "dependency_state": "agent_mutable",
                "native_family": family,
                "present": True,
                "count": 1,
                "harness_policy": {"protected_root": "/protected", "family": family},
            }
            for family in ("pytest", "shell")
        ],
    }
    integrity = {**actor, "summary": raw}
    summary = integrity_summary_for_actor(actor, integrity)
    table = pa.Table.from_pylist([summary.model_dump(mode="json")], schema=arrow_schema(IntegrityContractSummary))
    restored = IntegrityContractSummary.model_validate(table.to_pylist()[0])
    assert {group.native_family for group in restored.groups} == {"pytest", "shell"}
    assert len({group.harness_policy_sha256 for group in restored.groups}) == 2
    assert json.loads(restored.example.contract_json) == contract
    assert restored.required_shared_tasks == 2 and restored.shared_missing_contract_tasks == 0
    with pytest.raises(ValueError, match="input differs"):
        integrity_summary_for_actor(actor, {**integrity, "converter_revision": "changed-converter"})


def test_grading_census_retains_task_exceptions_and_clears_evidence_for_changed_population(tmp_path):
    source = number_source(tmp_path / "input")
    actor = census_receipt(source)
    separate = TaskSpec.model_validate(json.loads(prebuilt_task("separate")["lowered_task_spec"])["task"])
    shared = separate.model_copy(
        update={
            "id": "shared",
            "grader": NoGrader(reason="Native verifier or unavailable fixture"),
            "resources": separate.resources.model_copy(
                update={"verifier": (inline_resource("task.toml", b'[verifier]\nenvironment_mode = "shared"\n'),)}
            ),
        }
    )
    mismatch = TaskSpec.model_validate(
        json.loads(prebuilt_task("mismatch", capabilities=("git_repository",))["lowered_task_spec"])["task"]
    )
    unavailable = separate.model_copy(
        update={"id": "unavailable", "grader": NoGrader(reason="Native verifier or unavailable fixture")}
    )
    encoded = [task.model_dump_json() for task in (separate, shared, mismatch, unavailable)]
    encoded.append('{"id":"invalid"}')
    specs = tmp_path / "specs"
    specs.mkdir()
    pq.write_table(pa.table({"task_spec": encoded}), specs / "part-000000.parquet")
    digest = hashlib.sha256("".join(encoded).encode()).hexdigest()
    binding = actor.input.model_copy(update={"task_specs_sha256": digest, "measured_tasks": 5})
    actor = actor.model_copy(update={"input": binding})
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id=source.info.id,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
        source_version=binding.converter_version,
        converter_revision=binding.converter_revision,
        text_renderer_version="fixture",
        task_count=5,
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
    receipt = census_source(GradingCensusInput(population=population, input=binding), str(tmp_path / "census"))
    assert receipt.evidence.required_counts == {"separate": 2, "shared": 2}
    assert receipt.evidence.status_counts == {"matched": 2, "mismatch": 1, "unavailable": 1}
    assert sum(receipt.unassessable_reasons.values()) == 1
    records = {row["task_id"]: row for row in pq.read_table(receipt.task_evidence_uri).to_pylist()}
    assert records["shared"]["declared"] == "shared"
    assert records["mismatch"]["status"] == "mismatch"
    assert records["invalid"]["status"] == "unassessable"
    assert receipt.scanned_task_specs_sha256 == digest
    bound = bind_grading_census(bind_actor_census(source, actor), receipt)
    table = pa.Table.from_pylist([bound.declaration.model_dump(mode="json")], schema=arrow_schema(SourceDeclaration))
    restored = SourceDeclaration.model_validate(table.to_pylist()[0])
    assert restored.grading_environment == receipt.evidence
    changed = replace(
        bound, info=replace(bound.info, actor_census_input=binding.model_copy(update={"task_specs_sha256": "d" * 64}))
    )
    assert bind_grading_census(changed, receipt).info.grading_environment is None
    # A changed stored population cannot produce a complete-population receipt.
    pq.write_table(pa.table({"task_spec": encoded[:-1]}), specs / "part-000000.parquet")
    with pytest.raises(ValueError, match="input count differs"):
        census_source(GradingCensusInput(population=population, input=binding), str(tmp_path / "changed-census"))
