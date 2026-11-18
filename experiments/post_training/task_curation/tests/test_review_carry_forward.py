# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Carry-forward plans key reviews on exact task-contract identity."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    Confidence,
    ContentEvidenceReference,
    OversightLevel,
    QualityResultFinalization,
    ReferenceStatus,
)
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256, task_identity
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
from taskcompendium.pipeline.models import Quality, ReviewRecord, ReviewStatus, ReviewVerdict

from experiments.post_training.task_curation.full_qc_review import RECORD_SCHEMA
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.review_carry_forward import (
    CarriedReview,
    contract_identity,
    earlier_panel_evidence,
    merged_review_result,
    plan_review_carry_forward,
    population_identities,
)
from experiments.post_training.task_curation.review_quality import quality_result
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import SampleQCReceipt

SOURCE_ID = "OpenAthena:carry-forward-source"
EARLIER_PIN = "open-athena/carry-forward-release"
EARLIER_REVISION = "a" * 40
CURRENT_PIN = "open-athena/carry-forward-release-repinned"
CURRENT_REVISION = "b" * 40
CONVERTER_VERSION = "2026.11.16.1"

REVIEWER = {"mode": "chat", "model": "glm-5.3", "payload_revision": "4", "temperature": 0.7}
RELABEL = {**REVIEWER, "payload_revision": "6"}

TASKS = {f"task-{index:03d}": {"id": f"task-{index:03d}", "prompt": "x" * index} for index in range(6)}


def write_population(
    tmp_path,
    name,
    tasks_by_id,
    *,
    dataset_id=EARLIER_PIN,
    dataset_revision=EARLIER_REVISION,
) -> SourceTextPopulation:
    """Write a specs-shard population artifact with the given tasks."""
    root = tmp_path / name
    specs = root / "specs"
    root.mkdir(parents=True, exist_ok=True)
    specs.mkdir(parents=True, exist_ok=True)
    items = list(tasks_by_id.items())
    task_specs_sha256 = hashlib.sha256()
    for shard in range((len(items) + 1) // 2):
        members = items[shard * 2 : (shard + 1) * 2]
        rows = [{"id": key, "task_spec": json.dumps(doc), "runtime": "{}"} for key, doc in members]
        with (specs / f"part-{shard:06d}.parquet").open("wb") as stream:
            pq.write_table(pa.Table.from_pylist(rows), stream)
        for _, doc in members:
            task_specs_sha256.update(json.dumps(doc).encode())
    population = SourceTextPopulation(
        path=str(root),
        source_id=SOURCE_ID,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        source_version=CONVERTER_VERSION,
        converter_revision="2f4b27ba2e",
        text_renderer_version="v1",
        task_count=len(tasks_by_id),
        text_sha256="0" * 64,
        task_specs_sha256=task_specs_sha256.hexdigest(),
        main_output_dir=str(root / "outputs"),
        specs_output_dir=str(specs),
        converter_rejections={},
        admission_rejections={},
    )
    write_record(
        ArtifactRecord(
            output_path=str(root),
            result_type=result_type_name(SourceTextPopulation),
            result=population.result_payload(),
        )
    )
    return population


def good_review(task_id: str) -> ReviewRecord:
    verdict = ReviewVerdict(
        task_id=task_id,
        quality=Quality.GOOD,
        confidence=Confidence.HIGH,
        reference_status=ReferenceStatus.CONSISTENT,
        defects=[],
        evidence="Clean public contract.",
    )
    return ReviewRecord(task_id=task_id, status=ReviewStatus.REVIEWED, verdict=verdict, detail="ok")


def write_receipt(
    tmp_path,
    population: SourceTextPopulation,
    tasks_by_id,
    reviews: dict[str, ReviewRecord],
    *,
    reviewer=REVIEWER,
) -> SampleQCReceipt:
    """Write a frozen SAMPLE panel receipt with one observation per panel member."""
    root = tmp_path / "receipt"
    root.mkdir(parents=True)
    panel_ids = sorted(tasks_by_id)
    panel_rows = [
        {"task_id": task_id, "task_spec": json.dumps(tasks_by_id[task_id]), "runtime": "{}"} for task_id in panel_ids
    ]
    panel_path = root / "panel.parquet"
    with panel_path.open("wb") as stream:
        pq.write_table(pa.Table.from_pylist(panel_rows), stream)
    record_rows = [
        {
            "task_id": task_id,
            "task_sha256": canonical_sha256(tasks_by_id[task_id]),
            "runtime_sha256": canonical_sha256({}),
            "review": reviews[task_id].model_dump_json(),
            "batch_receipt": "memory://batch",
        }
        for task_id in panel_ids
    ]
    records_path = root / "attempts" / "reviews.parquet"
    records_path.parent.mkdir(parents=True)
    with records_path.open("wb") as stream:
        pq.write_table(pa.Table.from_pylist(record_rows, schema=RECORD_SCHEMA), stream)
    review_records = list(reviews.values())
    panel = SamplePanel(
        source_id=population.source_id,
        dataset_id=population.dataset_id,
        dataset_revision=population.dataset_revision,
        converter_version=population.source_version,
        input_artifact=population.path,
        task_specs_sha256=population.task_specs_sha256,
        scanned_task_specs_sha256=population.task_specs_sha256,
        retained_tasks=population.task_count,
        sample_seed=0,
        sample_size=len(panel_ids),
        panel_path=str(panel_path),
        panel_sha256=file_sha256(StoragePath(str(panel_path))),
        evidence_url=str(root / "summary.json"),
    )
    review = SampleQCReview(
        path=str(root),
        panel=panel,
        reviewer=reviewer,
        rubric={},
        panel_coverage_complete=all(row.status is ReviewStatus.REVIEWED for row in review_records),
        attempted_count=len(panel_ids),
        result=quality_result(review_records),
        records_path=str(records_path),
        records_sha256=file_sha256(StoragePath(str(records_path))),
        batch_receipts=["memory://batch"],
    )
    return SampleQCReceipt(source_id=population.source_id, review=review, artifact_sha256="c" * 64)


def earlier_evidence(tmp_path, tasks_by_id, reviews, *, reviewer=REVIEWER):
    population = write_population(tmp_path, "earlier", tasks_by_id)
    receipt = write_receipt(tmp_path, population, tasks_by_id, reviews, reviewer=reviewer)
    reference = ContentEvidenceReference(uri=str(tmp_path / "receipt-document.json"), sha256="d" * 64)
    return population, receipt, earlier_panel_evidence(receipt, reference)


def build_plan(tmp_path, earlier_population, evidence, current_tasks, **kwargs):
    current_population = write_population(
        tmp_path, "current", current_tasks, dataset_id=CURRENT_PIN, dataset_revision=CURRENT_REVISION
    )
    return plan_review_carry_forward(
        SOURCE_ID,
        evidence,
        earlier_population,
        current_population,
        population_identities(earlier_population),
        population_identities(current_population),
        **kwargs,
    )


def test_contract_identity_matches_task_identity():
    spec = TaskSpec(
        id="task-001",
        tags=("fixture",),
        labels={"family": "fixture"},
        context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
        source=Source(dataset="d", revision="r", row="task-001", importer_revision="1"),
    )
    document = spec.model_dump(mode="json")
    assert contract_identity(json.loads(json.dumps(document))) == task_identity(spec)


def test_drop_only_repin_carries_everything(tmp_path):
    reviews = {task_id: good_review(task_id) for task_id in TASKS}
    earlier_population, _, evidence = earlier_evidence(tmp_path, TASKS, reviews)
    current_tasks = {task_id: TASKS[task_id] for task_id in ("task-000", "task-001", "task-002", "task-003")}
    plan = build_plan(tmp_path, earlier_population, evidence, current_tasks)
    assert plan.population.model_dump() == {"carried": 4, "changed": 0, "new": 0, "dropped": 2}
    assert len(plan.carried_reviews) == 4
    for carried in plan.carried_reviews:
        assert carried.current_task_id == carried.original_task_id
        assert carried.review == reviews[carried.current_task_id]
    assert plan.kept_members == tuple(sorted(current_tasks))
    assert plan.replaced_members == 0
    assert plan.needs_review == ()
    assert plan.reviewer == REVIEWER


def test_changed_task_needs_review(tmp_path):
    reviews = {task_id: good_review(task_id) for task_id in TASKS}
    earlier_population, _, evidence = earlier_evidence(tmp_path, TASKS, reviews)
    current_tasks = dict(TASKS)
    current_tasks["task-002"] = {"id": "task-002", "prompt": "changed"}
    plan = build_plan(tmp_path, earlier_population, evidence, current_tasks)
    assert plan.population.model_dump() == {"carried": 5, "changed": 1, "new": 0, "dropped": 0}
    assert len(plan.carried_reviews) == 5
    assert plan.kept_members == tuple(sorted(set(TASKS) - {"task-002"}))
    assert plan.replaced_members == 1
    assert plan.needs_review == ("task-002",)


def test_nonconclusive_member_keeps_slot_and_needs_review(tmp_path):
    reviews = {task_id: good_review(task_id) for task_id in TASKS}
    unavailable = ReviewRecord(task_id="task-005", status=ReviewStatus.UNAVAILABLE, verdict=None, detail="judge down")
    reviews["task-005"] = unavailable
    earlier_population, _, evidence = earlier_evidence(tmp_path, TASKS, reviews)
    plan = build_plan(tmp_path, earlier_population, evidence, dict(TASKS))
    assert plan.population.model_dump() == {"carried": 6, "changed": 0, "new": 0, "dropped": 0}
    assert len(plan.carried_reviews) == 5
    assert plan.kept_members == tuple(sorted(TASKS))
    assert plan.replaced_members == 0
    assert plan.needs_review == ("task-005",)


def test_relabeled_reviewer_is_refused(tmp_path):
    reviews = {task_id: good_review(task_id) for task_id in TASKS}
    earlier_population, _, evidence = earlier_evidence(tmp_path, TASKS, reviews)
    current_tasks = dict(TASKS)
    with pytest.raises(ValueError, match="refusing to relabel"):
        build_plan(tmp_path, earlier_population, evidence, current_tasks, reviewer=RELABEL)
    recorded = build_plan(tmp_path, earlier_population, evidence, current_tasks, reviewer=REVIEWER)
    default = build_plan(tmp_path, earlier_population, evidence, current_tasks)
    assert recorded == default
    assert recorded.reviewer == REVIEWER


def test_replacement_draw_is_seeded_and_reproducible(tmp_path):
    reviews = {task_id: good_review(task_id) for task_id in TASKS}
    earlier_population, _, evidence = earlier_evidence(tmp_path, TASKS, reviews)
    current_tasks = dict(TASKS)
    current_tasks["task-002"] = {"id": "task-002", "prompt": "changed"}
    first = build_plan(tmp_path, earlier_population, evidence, current_tasks)
    second = build_plan(tmp_path, earlier_population, evidence, current_tasks)
    assert first == second
    pool = sorted(set(current_tasks) - set(first.kept_members))
    replacements = set(first.needs_review) - {"task-002"}
    assert replacements <= set(pool)


def test_merged_review_result_rules():
    carried = tuple(
        CarriedReview(
            task_identity_sha256="a" * 64,
            original_task_id=f"task-{index:03d}",
            current_task_id=f"task-{index:03d}",
            review=good_review(f"task-{index:03d}"),
        )
        for index in range(100)
    )
    complete = merged_review_result(100, carried, ())
    assert complete.level is OversightLevel.FULL
    assert complete.finalization is QualityResultFinalization.COMPLETE
    assert complete.coverage_complete is True
    with pytest.raises(ValueError, match="cannot carry a waiver"):
        merged_review_result(100, carried, (), waiver=ContentEvidenceReference(uri="memory://w", sha256="e" * 64))
    at_bound = merged_review_result(
        100, carried[:99], (), waiver=ContentEvidenceReference(uri="memory://w", sha256="e" * 64)
    )
    assert at_bound.level is OversightLevel.FULL
    assert at_bound.finalization is QualityResultFinalization.ACCEPTED_INCOMPLETE
    without_waiver = merged_review_result(100, carried[:99], ())
    assert without_waiver.level is OversightLevel.SAMPLE
    assert without_waiver.finalization is None
    with pytest.raises(ValueError, match="unreviewed bound"):
        merged_review_result(100, carried[:90], (), waiver=ContentEvidenceReference(uri="memory://w", sha256="e" * 64))
    duplicate = carried[:1]
    with pytest.raises(ValueError, match="both a carried and a fresh review"):
        merged_review_result(6, carried[:5], (duplicate[0].review,))
