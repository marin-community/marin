# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full QC inputs bind proposed retention to exact task and deployment contracts."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.full_qc_inputs import InputConfig, RetentionPlan, prepare_inputs
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task


@pytest.fixture
def qc_inputs(tmp_path):
    documents = [
        json.loads(prebuilt_task(name)["lowered_task_spec"])["task"] for name in ("kept", "dropped", "unshaped")
    ]
    oracle = TaskSpec.model_validate(documents[0])
    documents[0] = oracle.model_copy(
        update={
            "resources": oracle.resources.model_copy(update={"oracle": (inline_resource("solution/solve.sh", b"true"),)})
        }
    ).model_dump(mode="json")
    documents[2]["execution_requirements"] = None
    files = write_text_population(((task, {"deployment": 1}) for task in documents), str(tmp_path / "population"))
    population = SourceTextPopulation(
        path=str(tmp_path / "population"),
        source_id="TaskTrove:fixture",
        dataset_id="open-athena/fixture",
        dataset_revision="pin",
        source_version="1",
        converter_revision="converter-pin",
        text_renderer_version="fixture",
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
    drop_path = tmp_path / "drops.parquet"
    pq.write_table(pa.table({"task_id": ["dropped"]}), drop_path)
    plan = RetentionPlan(
        source_id=population.source_id,
        source_dataset_id=population.dataset_id,
        source_revision=population.dataset_revision,
        converter_revision=population.converter_revision,
        converter_version=population.source_version,
        input_artifact=population.path,
        task_specs_sha256=population.task_specs_sha256,
        raw_count=3,
        combined_drop_count=1,
        retained_count=2,
        combined_drop_manifest_uri=str(drop_path),
        lexical_use="train",
    )
    return (
        InputConfig(population.path, plan, file_sha256(StoragePath(str(drop_path))), str(tmp_path / "prepared")),
        documents,
    )


def test_prepared_contracts_cover_only_retained_tasks_and_keep_qualification_diagnostics(qc_inputs):
    config, documents = qc_inputs
    result = prepare_inputs(config)
    rows = pq.read_table(result.index_path).to_pylist()
    assert {row["task_id"] for row in rows} == {"kept", "unshaped"}
    assert {row["task_id"]: row["task_sha256"] for row in rows} == {
        task["id"]: canonical_sha256(TaskSpec.model_validate(task).model_dump(mode="json"))
        for task in (documents[0], documents[2])
    }
    assert {row["runtime_sha256"] for row in rows} == {canonical_sha256({"deployment": 1})}
    assert result.retained_count == 2
    assert result.oracle_tasks == 1
    assert result.missing_execution_requirements == 1
    assert result.grading_environments == {"separate:matched": 2}
    assert result.index_sha256 == file_sha256(StoragePath(result.index_path))
    retry = prepare_inputs(config)
    assert retry.index_path != result.index_path
    assert pq.read_table(result.index_path).to_pylist() == pq.read_table(retry.index_path).to_pylist()


def test_retention_manifest_mutation_cannot_change_review_population(qc_inputs):
    config, _ = qc_inputs
    pq.write_table(pa.table({"task_id": ["kept"]}), config.retention.combined_drop_manifest_uri)
    with pytest.raises(ValueError, match="Drop manifest changed"):
        prepare_inputs(config)


def test_unknown_drop_id_cannot_certify_full_retention_coverage(qc_inputs):
    config, _ = qc_inputs
    path = config.retention.combined_drop_manifest_uri
    pq.write_table(pa.table({"task_id": ["unknown"]}), path)
    changed = InputConfig(config.population_path, config.retention, file_sha256(StoragePath(path)), config.output_path)
    with pytest.raises(ValueError, match="complete retention plan"):
        prepare_inputs(changed)
