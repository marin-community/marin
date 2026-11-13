# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.review import ChatReviewer

from experiments.post_training.task_curation.admission_inventory import PANEL_SCHEMA
from experiments.post_training.task_curation.full_qc import ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.published_census import population_documents
from experiments.post_training.task_curation.sample_qc import SamplePanel, panel_tasks, review_panel
from experiments.post_training.task_curation.tests.test_full_qc_inputs import qc_inputs as qc_inputs


@pytest.fixture
def sample_panel(request, tmp_path):
    config, _ = request.getfixturevalue("qc_inputs")
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
    summary.write_text(panel.model_dump_json())
    return panel


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


def test_changed_sample_bytes_are_refused_before_review(sample_panel):
    StoragePath(sample_panel.panel_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen sample panel changed"):
        panel_tasks(sample_panel)
