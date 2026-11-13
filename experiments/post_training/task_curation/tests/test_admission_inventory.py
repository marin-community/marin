# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from taskcompendium.models import NoGrader, TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.admission_inventory import inventory_source
from experiments.post_training.task_curation.census_binding import bind_actor_census, receipt_from_census
from experiments.post_training.task_curation.grading_census import GradingCensusInput, census_source
from experiments.post_training.task_curation.lexical_artifacts import ActorCensus, SourceTextPopulation
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def test_two_shard_inventory_checks_native_contracts_and_rejects_changed_resume_panel(tmp_path):
    source = number_source(tmp_path / "input")
    task = TaskSpec.model_validate(json.loads(prebuilt_task("typed")["lowered_task_spec"])["task"])
    native = task.model_copy(
        update={
            "id": "native",
            "grader": NoGrader(reason="Native verifier"),
            "tags": ("harbor",),
            "resources": task.resources.model_copy(
                update={
                    "verifier": (
                        inline_resource("task.toml", b'[environment]\ndocker_image = "ubuntu:24.04"\n'),
                        inline_resource("instruction.md", b"Do the task"),
                        inline_resource("tests/test.sh", b"#!/bin/sh\nexit 0\n"),
                    )
                }
            ),
        }
    )
    encoded = [task.model_dump_json(), native.model_dump_json()]
    specs = tmp_path / "specs"
    specs.mkdir()
    for index, document in enumerate(encoded):
        pq.write_table(pa.table({"task_spec": [document], "runtime": ["{}"]}), specs / f"part-{index:06}.parquet")
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id=source.info.id,
        dataset_id=source.dataset.name,
        dataset_revision=source.dataset.revision,
        source_version=source.version,
        converter_revision="fixture",
        text_renderer_version="fixture",
        task_count=2,
        task_specs_sha256=hashlib.sha256("".join(encoded).encode()).hexdigest(),
        text_sha256="a" * 64,
        main_output_dir=str(tmp_path / "text"),
        specs_output_dir=str(specs),
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=population.path,
            result_type=result_type_name(type(population)),
            result=population.result_payload(),
        )
    )
    actor = ActorCensus(
        path=str(tmp_path / "actor"),
        source_id=source.info.id,
        dataset_id=source.dataset.name,
        dataset_revision=source.dataset.revision,
        converter_version=source.version,
        converter_revision="fixture",
        input_artifact=population.path,
        task_specs_sha256=population.task_specs_sha256,
        task_count=2,
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Frozen local fixture population",
        daytona_unsafe_reasons=[],
        daytona_unsafe_details={},
        daytona_unsafe_counts={},
        daytona_roles={},
        actor_environment_counts={},
        empty_declared_environment_tasks={},
        snapshot_sizes={},
        execution_failures={},
        mapping_failures={},
        integrity_contracts={},
    )
    write_record(
        ArtifactRecord(output_path=actor.path, result_type=result_type_name(type(actor)), result=actor.result_payload())
    )
    source = bind_actor_census(
        replace(
            source,
            info=replace(
                source.info,
                within_source_dedup=deduplication_evidence(
                    source.dataset.name, source.dataset.revision, source.version, 2
                ),
            ),
        ),
        receipt_from_census(actor.model_dump(mode="json"), None),
    )
    declaration = source.declaration
    result = inventory_source(declaration, str(tmp_path / "first"), None)
    assert result["retained_tasks"] == 2 and result["sample_size"] == 2
    assert result["scanned_task_specs_sha256"] == population.task_specs_sha256
    assert {row["task_id"] for row in pq.read_table(result["panel_path"]).to_pylist()} == {"typed", "native"}
    assert inventory_source(declaration, str(tmp_path / "resumed"), str(tmp_path / "first")) == result
    complete = census_source(
        GradingCensusInput(population=population, input=source.info.actor_census_input),
        str(tmp_path / "complete-grading"),
    )
    assert complete.evidence.assessed_tasks == 2
    assert complete.evidence.status_counts == {"matched": 2}
    with open(result["panel_path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="sample panel changed"):
        inventory_source(declaration, str(tmp_path / "changed"), str(tmp_path / "first"))
    pq.write_table(pa.table({"task_spec": [encoded[0]], "runtime": ["{}"]}), specs / "part-000001.parquet")
    with pytest.raises(ValueError, match="Duplicate task"):
        inventory_source(declaration, str(tmp_path / "duplicate"), None)
    (specs / "part-000001.parquet").unlink()
    with pytest.raises(ValueError, match="task count differs"):
        inventory_source(declaration, str(tmp_path / "incomplete"), None)
    malformed = native.model_copy(
        update={
            "resources": native.resources.model_copy(
                update={
                    "verifier": (inline_resource("task.toml", b"[verifier]\n"),),
                }
            )
        }
    )
    pq.write_table(
        pa.table({"task_spec": [malformed.model_dump_json()], "runtime": ["{}"]}), specs / "part-000001.parquet"
    )
    # A task.toml marker alone also cannot create complete grading admission evidence.
    changed_digest = hashlib.sha256((encoded[0] + malformed.model_dump_json()).encode()).hexdigest()
    population = population.model_copy(update={"task_specs_sha256": changed_digest})
    write_record(
        ArtifactRecord(
            output_path=population.path,
            result_type=result_type_name(type(population)),
            result=population.result_payload(),
        )
    )
    binding = source.info.actor_census_input.model_copy(update={"task_specs_sha256": changed_digest})
    receipt = census_source(GradingCensusInput(population=population, input=binding), str(tmp_path / "grading"))
    assert receipt.evidence.assessed_tasks == 1
    assert sum(receipt.unassessable_reasons.values()) == 1
