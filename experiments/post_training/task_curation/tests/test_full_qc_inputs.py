# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full QC inputs bind proposed retention to exact task and deployment contracts."""

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

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

from experiments.post_training.task_curation.full_qc_artifacts import (
    FullQCInputs,
    FullQCReview,
    FullQCReviewShard,
    RetentionPlan,
)
from experiments.post_training.task_curation.full_qc_inputs import InputConfig, prepare_inputs
from experiments.post_training.task_curation.full_qc_review import assemble_shards, review_inputs, review_shard
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
    invalid_task_ids: frozenset[str] = frozenset()
    unavailable_task_ids: frozenset[str] = frozenset()

    @property
    def identity(self):
        return {"model": "fixture-reviewer", "revision": "fixed"}

    def review(self, tasks, rubric, *, originals=None, cached=None):
        if not self.available or any(task.id in self.unavailable_task_ids for task in tasks):
            raise ConnectionError("Review service unavailable")
        records = [
            (
                ReviewRecord(task_id=task.id, status=ReviewStatus.INVALID, verdict=None, detail="Truncated response")
                if task.id in self.invalid_task_ids
                else ReviewRecord(
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


@pytest.fixture
def multi_batch_inputs(qc_inputs, tmp_path):
    config, documents = qc_inputs
    for index in range(64):
        document = json.loads(prebuilt_task(f"extra-{index}")["lowered_task_spec"])["task"]
        document["context"]["events"][0]["content"] = f"Compute fixture answer {index}."
        documents.append(document)
    population_path = str(tmp_path / "multi-population")
    files = write_text_population(((task, {"deployment": 1}) for task in documents), population_path)
    population = SourceTextPopulation.raw_load(config.population_path).model_copy(
        update={
            "path": population_path,
            "task_count": files.task_count,
            "text_sha256": files.text_sha256,
            "task_specs_sha256": files.task_specs_sha256,
            "main_output_dir": files.main_output_dir,
            "specs_output_dir": files.specs_output_dir,
        }
    )
    write_record(
        ArtifactRecord(
            output_path=population.path,
            result_type=result_type_name(type(population)),
            result=population.result_payload(),
        )
    )
    plan = config.retention.model_copy(
        update={
            "input_artifact": population.path,
            "task_specs_sha256": files.task_specs_sha256,
            "raw_count": len(documents),
            "retained_count": len(documents) - 1,
        }
    )
    return prepare_inputs(
        InputConfig(population.path, plan, config.drop_manifest_sha256, str(tmp_path / "multi-prepared"))
    )


def source_shards(inputs, tmp_path, shard_count, reviewer):
    rubric = ReviewRubric("quality", "1", ("Check the complete task contract.",))
    output = str(tmp_path / "batches")
    shards = [
        review_shard(
            inputs,
            reviewer,
            rubric,
            output,
            str(tmp_path / "shards" / str(index)),
            shard_index=index,
            shard_count=shard_count,
            batch_offset=5,
            max_passes=2,
            worker_jobs=(),
        )
        for index in range(shard_count)
    ]
    return reviewer, rubric, output, [FullQCReviewShard.raw_load(shard.path) for shard in shards]


@pytest.mark.parametrize("shard_count", [3, 5])
def test_shards_assemble_every_retained_contract_and_reuse_existing_batch_cache(
    multi_batch_inputs, tmp_path, shard_count
):
    inputs = multi_batch_inputs
    reviewer, rubric, output, shards = source_shards(inputs, tmp_path, shard_count, ReviewService())
    result = assemble_shards(inputs, shards, reviewer.identity, rubric, str(tmp_path / "assembled"))
    expected = {row["task_id"] for row in pq.read_table(inputs.index_path).to_pylist()}
    records = pq.read_table(result.records_path).to_pylist()
    assert result.coverage_complete
    assert result.attempted_count == len(records) == len(expected) == 66
    assert {row["task_id"] for row in records} == expected
    assert "dropped" not in expected
    assert result.result.issue_count == 1
    replay = review_inputs(inputs, replace(reviewer, available=False), rubric, output)
    assert replay.result == result.result
    assert pq.read_table(replay.records_path).to_pylist() == records


@pytest.mark.parametrize("damage", ["missing", "overlap", "changed_receipt", "foreign_reviewer"])
def test_shard_assembly_rejects_incomplete_or_changed_evidence(multi_batch_inputs, tmp_path, damage):
    inputs = multi_batch_inputs
    reviewer, rubric, _, shards = source_shards(inputs, tmp_path, 3, ReviewService())
    if damage == "missing":
        shards[0] = shards[0].model_copy(update={"receipts": []})
    elif damage == "overlap":
        shards[0] = shards[0].model_copy(update={"receipts": [*shards[0].receipts, *shards[1].receipts]})
    elif damage == "changed_receipt":
        path = StoragePath(shards[0].receipts[0].path)
        path.write_text(path.read_text() + " ")
    else:
        shards = [shard.model_copy(update={"reviewer": {"model": "another-reviewer"}}) for shard in shards]
    with pytest.raises(ValueError, match=r"Missing retained|overlap|evidence changed|exact source and reviewer"):
        assemble_shards(inputs, shards, reviewer.identity, rubric, str(tmp_path / "assembled"))
    assert not list((tmp_path / "assembled").rglob(".artifact.json"))


def test_shard_assembly_keeps_invalid_observations_out_of_clean_coverage(multi_batch_inputs, tmp_path):
    reviewer, rubric, _, shards = source_shards(
        multi_batch_inputs, tmp_path, 3, ReviewService(invalid_task_ids=frozenset({"unshaped"}))
    )
    result = assemble_shards(multi_batch_inputs, shards, reviewer.identity, rubric, str(tmp_path / "assembled"))
    assert result.attempted_count == 66
    assert result.result.reviewed == 65
    assert result.result.unavailable_count == 1
    assert not result.coverage_complete
    assert sum(shard.first_pass_status_counts.get("invalid", 0) for shard in shards) == 1
    assert sum(shard.status_counts.get("invalid", 0) for shard in shards) == 1
    records = pq.read_table(result.records_path).to_pylist()
    invalid = [json.loads(row["review"]) for row in records if row["task_id"] == "unshaped"]
    assert invalid[0]["status"] == "invalid"


def test_partial_batch_retry_preserves_valid_observations_without_resubmitting_them(qc_inputs, tmp_path):
    config, _ = qc_inputs
    inputs = prepare_inputs(config)
    rubric = ReviewRubric("quality", "1", ("Check the complete task contract.",))
    output = str(tmp_path / "reviews")
    first = review_inputs(inputs, ReviewService(invalid_task_ids=frozenset({"unshaped"})), rubric, output)
    assert not first.coverage_complete
    second = review_inputs(inputs, ReviewService(unavailable_task_ids=frozenset({"kept"})), rubric, output)
    assert second.coverage_complete
    assert second.result.reviewed == 2
    original = {r["task_id"]: r["review"] for r in pq.read_table(first.records_path).to_pylist()}
    resumed = {r["task_id"]: r["review"] for r in pq.read_table(second.records_path).to_pylist()}
    assert original["kept"] == resumed["kept"]
    assert file_sha256(StoragePath(first.records_path)) == first.records_sha256


def test_completed_attempt_replays_after_interrupted_completion_marker_write(qc_inputs, tmp_path):
    config, _ = qc_inputs
    inputs = prepare_inputs(config)
    rubric = ReviewRubric("quality", "1", ("Check the complete task contract.",))
    output = str(tmp_path / "reviews")
    first = review_inputs(inputs, ReviewService(), rubric, output)
    (Path(first.batch_receipts[0]).parent.parent / "complete.json").unlink()
    resumed = review_inputs(inputs, ReviewService(available=False), rubric, output)
    assert resumed.coverage_complete
    assert resumed.result == first.result
