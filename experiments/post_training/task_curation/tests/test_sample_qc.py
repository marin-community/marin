# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ActorCensusInput, ContentEvidenceReference, SourceQualityResult
from marin.execution.artifact import read_record, write_record
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation.admission_inventory import PANEL_SCHEMA
from experiments.post_training.task_curation.difficulty_samples import relocated_panel, sample_oracle_state
from experiments.post_training.task_curation.full_qc import ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.published_census import population_documents
from experiments.post_training.task_curation.sample_qc import SamplePanel, panel_tasks, review_panel
from experiments.post_training.task_curation.sample_qc_artifacts import SampleInventory
from experiments.post_training.task_curation.sample_qc_binding import bind_sample_result
from experiments.post_training.task_curation.sample_qc_receipts import sample_qc_receipt
from experiments.post_training.task_curation.source import SourceOrigin, SourceReference
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


@pytest.fixture
def sample_panel(qc_inputs, tmp_path):
    config, _ = qc_inputs
    population = SourceTextPopulation.raw_load(config.population_path)
    path = tmp_path / "panel.parquet"
    rows = [
        {
            "task_id": task["id"],
            "task_spec": TaskSpec.model_validate(task).model_dump_json(),
            "runtime": json.dumps(runtime),
        }
        for task, runtime in population_documents(population)
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=PANEL_SCHEMA), path)
    summary = tmp_path / "summary.json"
    panel = SamplePanel(
        source_id=population.source_id,
        dataset_id=population.dataset_id,
        dataset_revision=population.dataset_revision,
        converter_version=population.source_version,
        input_artifact=population.path,
        task_specs_sha256=population.task_specs_sha256,
        scanned_task_specs_sha256=population.task_specs_sha256,
        retained_tasks=len(rows),
        sample_seed=0,
        sample_size=len(rows),
        panel_path=str(path),
        panel_sha256=file_sha256(StoragePath(str(path))),
        evidence_url=str(summary),
    )
    inventory = SampleInventory(
        **panel.model_dump(),
        oracle_tasks=1,
        execution_qualification="not_assessed",
        runnable_grader_tasks=3,
        no_grader_tasks=0,
        grader_kinds={"script": 3},
        missing_grader_reasons={},
    )
    summary.write_text(inventory.model_dump_json(exclude_none=True))
    return panel


def test_oracle_bearing_resample_leaves_pre_filter_queue_when_filter_evidence_is_bound(sample_panel):
    inventory = SampleInventory.model_validate_json(StoragePath(sample_panel.evidence_url).read_text())
    assert sample_oracle_state(inventory, None)["resample_after_oracle_filter"]
    receipt = ContentEvidenceReference(uri="s3://fixture/oracle-filter.json", sha256="f" * 64)
    state = sample_oracle_state(inventory, receipt)
    assert state == {
        "pre_oracle_filter": False,
        "resample_after_oracle_filter": False,
        "oracle_filter_receipt": receipt,
    }


def test_sample_review_keeps_full_population_identity_and_reuses_exact_panel_evidence(
    sample_panel, review_server, tmp_path
):
    reviewer = ChatReviewer(
        OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2),
        local_reviewer(ReviewerVersion.V3),
        query_cache_root=str(tmp_path / "cache"),
    )
    output = str(tmp_path / "reviews")
    first = review_panel(sample_panel, reviewer, output)
    requests = len(review_server.requests)
    repeated = review_panel(sample_panel, reviewer, output)
    assert first.panel.retained_tasks == 3 and first.attempted_count == 3
    assert first.result.reviewed == 3 and first.result.issue_count == 0
    assert first.panel_coverage_complete and first.qualification_status == "pending"
    records = pq.read_table(first.records_path).to_pylist()
    assert {row["task_id"] for row in records} == {"kept", "dropped", "unshaped"}
    hashes = {row["task_id"]: row["task_sha256"] for row in records}
    assert hashes["kept"] != hashes["unshaped"]
    assert repeated.path != first.path and repeated.batch_receipts == first.batch_receipts
    assert len(review_server.requests) == requests == 2
    receipt = sample_qc_receipt(first.path)
    source = number_source(tmp_path / "numbers")
    source = replace(
        source,
        config=replace(
            source.config, dataset=SourceReference(sample_panel.dataset_id, sample_panel.dataset_revision, "fixture")
        ),
        info=replace(
            source.info,
            id=sample_panel.source_id,
            origin=SourceOrigin.TASKTROVE,
            upstream_dataset="fixture",
            oversight=None,
            actor_census_input=ActorCensusInput(
                dataset_id=sample_panel.dataset_id,
                dataset_revision=sample_panel.dataset_revision,
                converter_version=sample_panel.converter_version,
                converter_revision="fixture",
                measured_tasks=sample_panel.retained_tasks,
                task_specs_sha256=sample_panel.task_specs_sha256,
                evidence_url=sample_panel.evidence_url,
            ),
        ),
    )
    bound = bind_sample_result(source, receipt)
    assert bound.info.quality_result == first.result
    assert bound.info.oversight is None
    assert bound.admission_blockers == source.admission_blockers
    stale = replace(source, config=replace(source.config, dataset=replace(source.dataset, revision="changed")))
    with pytest.raises(ValueError, match="another current population"):
        bind_sample_result(stale, receipt)


def test_changed_sample_bytes_are_refused_before_review(sample_panel):
    StoragePath(sample_panel.panel_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen sample panel changed"):
        panel_tasks(sample_panel)


def test_relocated_difficulty_panel_reuses_the_shared_verified_reader(sample_panel, tmp_path):
    original_tasks, original_contracts = panel_tasks(sample_panel)
    inventory = SampleInventory.model_validate_json(StoragePath(sample_panel.evidence_url).read_text())
    root = StoragePath(str(tmp_path / "difficulty"))
    root.mkdirs()
    staged_path = str(root / "panel.parquet")
    shutil.copyfile(sample_panel.panel_path, staged_path)
    relocated = relocated_panel(inventory, staged_path, root)
    tasks, contracts = panel_tasks(relocated)
    assert tasks == original_tasks and contracts == original_contracts
    assert SampleInventory.model_validate_json((root / "original-inventory.json").read_text()) == inventory


def test_sample_result_cannot_claim_issue_counts_absent_from_observations(sample_panel, review_server, tmp_path):
    reviewer = ChatReviewer(
        OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2),
        local_reviewer(ReviewerVersion.V3),
    )
    result = review_panel(sample_panel, reviewer, str(tmp_path / "reviews"))
    record = read_record(result.path)
    claimed = SourceQualityResult(reviewed=3, issue_count=1, issues=[], repair_recommendation="Inspect issues")
    write_record(record.model_copy(update={"result": {**record.result, "result": claimed.model_dump(mode="json")}}))
    with pytest.raises(ValueError, match="differs from its task-level observations"):
        sample_qc_receipt(result.path)
