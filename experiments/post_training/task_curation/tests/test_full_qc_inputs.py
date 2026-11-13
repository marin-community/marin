# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full QC inputs bind proposed retention to exact task and deployment contracts."""

import json
import subprocess
import sys
from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import (
    Confidence,
    Defect,
    Quality,
    ReferenceStatus,
    ReviewRecord,
    ReviewRubric,
    ReviewStatus,
    ReviewVerdict,
)
from taskcompendium.pipeline.review import ReviewBatchResult
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.full_qc import qualify_inputs
from experiments.post_training.task_curation.full_qc_artifacts import FullQCInputs, FullQCReview, RetentionPlan
from experiments.post_training.task_curation.full_qc_inputs import InputConfig, prepare_inputs
from experiments.post_training.task_curation.full_qc_review import review_inputs
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from lib.taskcompendium.tests.oracle_filter_fixtures import oracle_receipt


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


def test_cli_artifact_can_be_reloaded_by_another_process(qc_inputs, tmp_path):
    config, _ = qc_inputs
    plans = tmp_path / "plans.json"
    sources = tmp_path / "sources.json"
    output = tmp_path / "outputs.json"
    plans.write_text(json.dumps([config.retention.model_dump(mode="json")]))
    sources.write_text(json.dumps([config.retention.source_id]))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.post_training.task_curation.full_qc_inputs",
            "--retention-plans",
            str(plans),
            "--source-ids",
            str(sources),
            "--artifact-root",
            str(tmp_path / "graph"),
            "--artifact-version",
            "2026.11.12.1",
            "--output",
            str(output),
            "--run",
        ],
        check=True,
    )
    result = FullQCInputs.raw_load(json.loads(output.read_text())[0]["path"])
    assert {row["task_id"] for row in pq.read_table(result.index_path).to_pylist()} == {"kept", "unshaped"}


@dataclass(frozen=True)
class ReviewService:
    available: bool = True
    issue_quality: Quality = Quality.BAD

    @property
    def identity(self):
        return {"model": "fixture-reviewer", "revision": "fixed"}

    def review(self, tasks, rubric, *, originals=None, cached=None):
        if not self.available:
            raise ConnectionError("Review service unavailable")
        records = [
            ReviewRecord(
                task_id=task.id,
                status=ReviewStatus.REVIEWED,
                verdict=ReviewVerdict(
                    task_id=task.id,
                    quality=self.issue_quality if task.id == "unshaped" else Quality.GOOD,
                    confidence=Confidence.HIGH,
                    reference_status=ReferenceStatus.CONSISTENT,
                    defects=[Defect.RUBRIC_MISMATCH] if task.id == "unshaped" else [],
                    evidence="The required answer is unspecified." if task.id == "unshaped" else "A coherent task.",
                ),
                detail="",
            )
            for task in tasks
        ]
        return ReviewBatchResult(records, [])

    def read_cache(self, batches, rubric):
        return [None] * len(batches)


@pytest.mark.parametrize("issue_quality", [Quality.BAD, Quality.ISSUES, Quality.GOOD])
def test_full_review_keeps_issue_tasks_and_resumes_without_the_provider(qc_inputs, tmp_path, issue_quality):
    config, _ = qc_inputs
    inputs = prepare_inputs(config)
    reviewer = ReviewService(issue_quality=issue_quality)
    rubric = ReviewRubric("quality", "1", ("Check the complete task contract.",))
    output = str(tmp_path / "reviews")
    result = review_inputs(inputs, reviewer, rubric, output)
    assert result.coverage_complete
    assert result.attempted_count == result.result.reviewed == 2
    assert result.result.issue_count == 1
    assert result.result.verdict == "bad"
    assert result.result.issue_rate == 0.5
    assert {row["task_id"] for row in pq.read_table(result.records_path).to_pylist()} == {"kept", "unshaped"}
    reloaded = FullQCReview.raw_load(result.path)
    assert reloaded.records_sha256 == file_sha256(StoragePath(reloaded.records_path))
    resumed = review_inputs(inputs, replace(reviewer, available=False), rubric, output)
    assert resumed.result == result.result
    assert resumed.records_path != result.records_path
    assert pq.read_table(resumed.records_path).to_pylist() == pq.read_table(result.records_path).to_pylist()


def test_deployment_mutation_invalidates_completed_review_evidence(qc_inputs, tmp_path):
    config, _ = qc_inputs
    inputs = prepare_inputs(config)
    reviewer = ReviewService()
    rubric = ReviewRubric("quality", "1", ("Check the complete task contract.",))
    output = str(tmp_path / "reviews")
    review_inputs(inputs, reviewer, rubric, output)
    shard = tmp_path / "population/specs/part-000000.parquet"
    table = pq.read_table(shard)
    rows = [{**row, "runtime": json.dumps({"deployment": 2})} for row in table.to_pylist()]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), shard)
    with pytest.raises(ValueError, match="differs from prepared contract"):
        review_inputs(inputs, replace(reviewer, available=False), rubric, output)


def test_qualification_requires_matching_full_population_and_retained_execution(qc_inputs, tmp_path):
    config, documents = qc_inputs
    tasks = [TaskSpec.model_validate(document) for document in documents]
    tasks[2] = tasks[2].model_copy(
        update={
            "execution_requirements": tasks[0].execution_requirements,
            "context": (
                tasks[2].context.model_copy(
                    update={
                        "events": (tasks[2].context.events[0].model_copy(update={"content": "A distinct public task."}),)
                    }
                )
            ),
        }
    )
    files = write_text_population(
        ((task.model_dump(mode="json"), {"deployment": 1}) for task in tasks),
        str(tmp_path / "qualified-population"),
    )
    old = SourceTextPopulation.raw_load(config.population_path)
    population = old.model_copy(
        update={
            "path": str(tmp_path / "qualified-population"),
            "main_output_dir": files.main_output_dir,
            "specs_output_dir": files.specs_output_dir,
            "text_sha256": files.text_sha256,
            "task_specs_sha256": files.task_specs_sha256,
        }
    )
    write_record(
        ArtifactRecord(
            output_path=population.path,
            result_type=result_type_name(type(population)),
            result=population.result_payload(),
        )
    )
    inputs = prepare_inputs(
        replace(
            config,
            population_path=population.path,
            retention=config.retention.model_copy(
                update={
                    "input_artifact": population.path,
                    "task_specs_sha256": population.task_specs_sha256,
                }
            ),
        )
    )
    receipt = oracle_receipt(tasks, failed=frozenset({"dropped"}))
    runtime_digest = canonical_sha256({"deployment": 1})
    entries = {}
    for task_id, entry in receipt.tasks.items():
        liveness = entry.no_oracle_liveness
        entries[task_id] = entry.model_copy(
            update={
                "runtime_sha256": runtime_digest,
                "no_oracle_liveness": (
                    liveness.model_copy(update={"runtime_sha256": runtime_digest}) if liveness else None
                ),
            }
        )
    receipt = receipt.model_copy(update={"tasks": entries})
    path = tmp_path / "qualification.json"
    path.write_text(receipt.model_dump_json())
    assert qualify_inputs(inputs, str(path)) == file_sha256(StoragePath(str(path)))
    stale = dict(entries)
    stale["kept"] = stale["kept"].model_copy(update={"runtime_sha256": canonical_sha256({"deployment": 2})})
    path.write_text(receipt.model_copy(update={"tasks": stale}).model_dump_json())
    with pytest.raises(ValueError, match="another task or runtime"):
        qualify_inputs(inputs, str(path))
