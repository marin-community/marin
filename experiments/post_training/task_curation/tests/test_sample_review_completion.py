# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Completion reviews freeze top-up tasks and record typed receipts."""

import json

import pyarrow.parquet as pq
import pytest
from marin.execution.artifact import read_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import ReviewRecord, ReviewStatus
from taskcompendium.pipeline.review import ReviewBatchResult

from experiments.post_training.task_curation.sample_review_completion import (
    CompletionReceipt,
    CompletionTarget,
    complete_target,
    freeze_topup_tasks,
)
from experiments.post_training.task_curation.tests.test_review_carry_forward import (
    CURRENT_PIN,
    CURRENT_REVISION,
    SOURCE_ID,
    good_review,
    write_population,
)

FIELD = "execution_requirements"


def current_tasks() -> dict[str, dict]:
    tasks = {}
    for index in range(6):
        task_id = f"task-{index:03d}"
        spec = TaskSpec(
            id=task_id,
            context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=NoGrader(reason="fixture"),
            source=Source(dataset="d", revision="r", row=task_id, importer_revision="1"),
        )
        tasks[task_id] = spec.model_dump(mode="json")
    return tasks


class StubReviewer:
    """A reviewer whose observations follow the supplied statuses."""

    def __init__(self, records: dict[str, str]) -> None:
        self.records = records

    @property
    def identity(self) -> dict:
        return {"mode": "chat", "model": "stub", "payload_revision": "6"}

    def review(self, tasks, rubric, *, originals=None, cached=None):
        reviews = []
        for task in tasks:
            if self.records.get(task.id, "reviewed") == "reviewed":
                reviews.append(good_review(task.id))
            else:
                reviews.append(
                    ReviewRecord(task_id=task.id, status=ReviewStatus.UNAVAILABLE, verdict=None, detail="judge down")
                )
        return ReviewBatchResult(reviews, [])


def test_freeze_topup_tasks_requires_every_task(tmp_path):
    population = write_population(
        tmp_path, "current", current_tasks(), dataset_id=CURRENT_PIN, dataset_revision=CURRENT_REVISION
    )
    tasks, runtimes = freeze_topup_tasks(population, ("task-001", "task-004"))
    assert [task.id for task in tasks] == ["task-001", "task-004"]
    assert runtimes == [{}, {}]
    with pytest.raises(ValueError, match="absent from the population"):
        freeze_topup_tasks(population, ("task-001", "task-999"))


def test_complete_target_records_typed_receipt(tmp_path):
    population = write_population(
        tmp_path, "current", current_tasks(), dataset_id=CURRENT_PIN, dataset_revision=CURRENT_REVISION
    )
    target = CompletionTarget(
        source_id=SOURCE_ID,
        population=population.path,
        task_ids=("task-001", "task-004"),
        topup_evidence=str(tmp_path / "recovery-record.json"),
    )
    receipt = complete_target(target, tmp_path.as_posix(), StubReviewer({}))
    assert receipt.source_id == SOURCE_ID
    assert receipt.task_ids == ("task-001", "task-004")
    assert receipt.qualification_status == "pending"
    assert receipt.result["reviewed"] == 2
    assert receipt.reviewer["model"] == "stub"
    record = read_record(receipt.path)
    assert record is not None and record.result["source_id"] == SOURCE_ID
    loaded = CompletionReceipt.raw_load(receipt.path)
    assert loaded.task_ids == ("task-001", "task-004")
    with StoragePath(loaded.records_path).open("rb") as stream:
        rows = pq.read_table(stream).to_pylist()
    assert {row["task_id"] for row in rows} == {"task-001", "task-004"}
    assert all(json.loads(row["review"])["status"] == "reviewed" for row in rows)


def test_complete_target_records_unavailable_observations(tmp_path):
    population = write_population(
        tmp_path, "current", current_tasks(), dataset_id=CURRENT_PIN, dataset_revision=CURRENT_REVISION
    )
    target = CompletionTarget(
        source_id=SOURCE_ID,
        population=population.path,
        task_ids=("task-002",),
        topup_evidence=str(tmp_path / "recovery-record.json"),
    )
    receipt = complete_target(target, tmp_path.as_posix(), StubReviewer({"task-002": "unavailable"}))
    assert receipt.result["reviewed"] == 0
    assert receipt.result["unavailable_count"] == 1
