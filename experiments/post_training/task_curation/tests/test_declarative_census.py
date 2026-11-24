# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declarative execution preserves identity checks while avoiding named-snapshot holds."""

import hashlib
import json

import pytest
from finestore.rl.curation import ContentEvidenceReference
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage

from experiments.post_training.task_curation.census_binding import receipt_from_census
from experiments.post_training.task_curation.declarative_census import (
    DeclarativeProvisioning,
    DeclarativeTaskExecution,
    bind_declarative_census,
)
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.local import artifact_prefix
from experiments.post_training.task_curation.oracle_filter import row_runtime_hash
from experiments.post_training.task_curation.published_census import CensusConfig, measure_population, population_rows
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


def reference(path, value):
    path.write_text(json.dumps(value))
    return ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def executed_population(tmp_path):
    documents = []
    for index in range(11):
        row = prebuilt_task(f"task-{index}", image=f"image-{index}@sha256:" + "a" * 64)
        task = TaskSpec.model_validate(json.loads(row["lowered_task_spec"])["task"])
        task = task.model_copy(
            update={"context": ConversationInput(events=(TextMessage(role="user", content=f"Edit {index}."),))}
        )
        documents.append((task.model_dump(mode="json"), {}))
    with artifact_prefix(str(tmp_path)):
        files = write_text_population(documents, str(tmp_path / "population"))
        population = SourceTextPopulation(
            path=str(tmp_path / "population"),
            source_id="fixture",
            dataset_id="open-athena/fixture",
            dataset_revision="a" * 40,
            source_version="1",
            converter_revision="b" * 40,
            text_renderer_version="renderer-v1",
            task_count=files.task_count,
            text_sha256=files.text_sha256,
            task_specs_sha256=files.task_specs_sha256,
            main_output_dir=files.main_output_dir,
            specs_output_dir=files.specs_output_dir,
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
        census = measure_population(CensusConfig(population.path, str(tmp_path / "snapshots")))
        write_record(
            ArtifactRecord(
                output_path=census.path, result_type=result_type_name(type(census)), result=census.result_payload()
            )
        )
        start = reference(
            tmp_path / "start.json",
            {
                "provisioning": "declarative",
                "source_revision": "b" * 40,
                "task_ids": [f"task-{index}" for index in range(11)],
                "task_count": 11,
                "native_budgets_unchanged": True,
            },
        )
        tasks = []
        for index, row in enumerate(population_rows(population)):
            task = TaskSpec.model_validate_json(row["task_spec"])
            contract = {
                "task_id": task.id,
                "task_sha256": task_spec_sha256(task),
                "runtime_sha256": row_runtime_hash(row),
            }
            receipt = reference(
                tmp_path / f"receipt-{index}.json",
                {
                    "source_id": "fixture",
                    "contract": contract,
                    "ownership_status": "all_owned_ids_absent",
                    "ownership": {
                        "evidence": {"sandbox_ids": [f"own-{index}"], "unbound_requests": [], "lifecycle_sha256": {}},
                        "sandboxes": [{"sandbox_id": f"own-{index}", "absent": True}],
                    },
                },
            )
            tasks.append(DeclarativeTaskExecution(**contract, receipt=receipt, start=start))
        descriptor = StoragePath(census.path) / ".artifact.json"
        qualification = DeclarativeProvisioning(
            source_id="fixture",
            input=receipt_from_census(census.model_dump(mode="json"), None).input,
            factory="shellbox.backends.daytona.machine.DaytonaDeclarativeMachineFactory",
            executor_revision="b" * 40,
            executor_script=reference(tmp_path / "executor.json", {"factory": "declarative"}),
            snapshot_census=ContentEvidenceReference(
                uri=str(descriptor), sha256=hashlib.sha256(descriptor.read_bytes()).hexdigest()
            ),
            tasks=tuple(tasks),
        )
        yield census, qualification


def test_declarative_execution_clears_only_named_snapshot_accounting(executed_population, tmp_path):
    census, qualification = executed_population
    assert census.task_count == census.daytona_snapshot_count == 11
    assert census.daytona_unsafe_reasons == ["too_many_snapshots"]
    proof = reference(tmp_path / "qualification.json", qualification.model_dump(mode="json"))
    bound = bind_declarative_census(census, proof, str(tmp_path / "declarative"))
    assert bound.daytona_snapshot_count == 0 and bound.daytona_safety == "safe"
    assert bound.daytona_roles == census.daytona_roles
    assert bound.daytona_machine_slots == census.daytona_machine_slots == 11
    assert bound.task_specs_sha256 == census.task_specs_sha256 and bound.input_artifact == census.input_artifact
    assert not bound.daytona_unsafe_reasons
    assert census.daytona_snapshot_count == len(census.snapshot_sizes) == 11


@pytest.mark.parametrize("fault", ["runtime", "foreign_pin", "ownership", "snapshot_route"])
def test_declarative_accounting_refuses_foreign_or_unresolved_execution(executed_population, tmp_path, fault):
    census, qualification = executed_population
    if fault == "foreign_pin":
        qualification = qualification.model_copy(
            update={"input": qualification.input.model_copy(update={"dataset_revision": "c" * 40})}
        )
    elif fault == "snapshot_route":
        first = qualification.tasks[0]
        payload = json.loads(StoragePath(first.start.uri).read_bytes())
        payload["provisioning"] = "snapshot"
        first = first.model_copy(update={"start": reference(tmp_path / "snapshot-start.json", payload)})
        qualification = qualification.model_copy(update={"tasks": (first, *qualification.tasks[1:])})
    else:
        first = qualification.tasks[0]
        path = tmp_path / "bad-receipt.json"
        payload = json.loads(StoragePath(first.receipt.uri).read_bytes())
        if fault == "runtime":
            payload["contract"]["runtime_sha256"] = "c" * 64
        else:
            payload["ownership"]["sandboxes"][0]["absent"] = False
        first = first.model_copy(update={"receipt": reference(path, payload)})
        qualification = qualification.model_copy(update={"tasks": (first, *qualification.tasks[1:])})
    proof = reference(tmp_path / "qualification.json", qualification.model_dump(mode="json"))
    with pytest.raises(ValueError, match="another census|incomplete or belongs to another task|not declarative"):
        bind_declarative_census(census, proof, str(tmp_path / "refused"))
