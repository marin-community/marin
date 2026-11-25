# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from dataclasses import asdict, replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    ContentEvidenceReference,
    SamplePanelCurrency,
    SourceQualityResult,
)
from marin.execution.artifact import ArtifactRecord, read_record, result_type_name, write_record
from marin.inference.openai_chat import OpenAIChatClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ConversationInput, TaskSpec, TextMessage
from taskcompendium.pipeline.models import ReviewRecord
from taskcompendium.pipeline.oracle_filter import OracleTaskReceipt
from taskcompendium.pipeline.review import ChatReviewer
from taskcompendium.pipeline.sampling import seeded_order, seeded_sample

from experiments.post_training.task_curation.admission_inventory import PANEL_SCHEMA
from experiments.post_training.task_curation.difficulty_samples import (
    OracleFilterBinding,
    prepare_sample,
    relocated_panel,
)
from experiments.post_training.task_curation.full_content_binding import RetainedLineageProof
from experiments.post_training.task_curation.full_qc import REVIEW_RUBRIC, ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.full_qc_review import ContractBatch, write_observations
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.lexical_contamination import write_text_population
from experiments.post_training.task_curation.published_census import population_documents, population_rows
from experiments.post_training.task_curation.review_quality import quality_result
from experiments.post_training.task_curation.sample_qc import SamplePanel, panel_tasks, review_panel
from experiments.post_training.task_curation.sample_qc_artifacts import SampleInventory, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import bind_sample_result
from experiments.post_training.task_curation.sample_qc_receipts import sample_qc_receipt
from experiments.post_training.task_curation.source import SourceOrigin, SourceReference
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence
from lib.taskcompendium.tests.oracle_filter_fixtures import oracle_receipt


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


def test_sample_filter_proof_must_cover_unselected_retained_tasks(sample_panel, tmp_path):
    original = next(task for task in panel_tasks(sample_panel)[0] if task.id == "kept")
    tasks = [
        original.model_copy(
            update={
                "id": f"task-{index:03d}",
                "context": ConversationInput(events=(TextMessage(role="user", content=f"Return {index}."),)),
            }
        )
        for index in range(101)
    ]
    files = write_text_population(((task.model_dump(mode="json"), {}) for task in tasks), str(tmp_path / "full"))
    source = number_source(tmp_path / "numbers")
    population = SourceTextPopulation.raw_load(sample_panel.input_artifact).model_copy(
        update={
            "path": str(tmp_path / "full"),
            "source_id": source.info.id,
            "dataset_id": source.dataset.name,
            "dataset_revision": source.dataset.revision,
            "task_count": 101,
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
    rows = [{"task_id": json.loads(row["task_spec"])["id"], **row} for row in population_rows(population)]
    _, selected = seeded_sample(rows, size=100, key=lambda row: seeded_order(row["task_id"], 0))
    panel_path = tmp_path / "panel-100.parquet"
    pq.write_table(pa.Table.from_pylist(selected, schema=PANEL_SCHEMA), panel_path)
    inventory = SampleInventory.model_validate_json(StoragePath(sample_panel.evidence_url).read_text()).model_copy(
        update={
            "source_id": source.info.id,
            "dataset_id": population.dataset_id,
            "dataset_revision": population.dataset_revision,
            "input_artifact": population.path,
            "task_specs_sha256": population.task_specs_sha256,
            "scanned_task_specs_sha256": hashlib.sha256("".join(row["task_spec"] for row in rows).encode()).hexdigest(),
            "retained_tasks": 101,
            "oracle_tasks": 101,
            "sample_size": 100,
            "panel_path": str(panel_path),
            "panel_sha256": file_sha256(StoragePath(str(panel_path))),
            "evidence_url": str(tmp_path / "inventory-101.json"),
        }
    )
    StoragePath(inventory.evidence_url).write_text(inventory.model_dump_json())
    dedup = deduplication_evidence(population.dataset_id, population.dataset_revision, "1", 101).model_copy(
        update={"drop_manifest_uri": str(tmp_path / "drops.parquet")}
    )
    source = replace(
        source,
        info=replace(
            source.info,
            within_source_dedup=dedup,
            actor_census_input=ActorCensusInput(
                dataset_id=population.dataset_id,
                dataset_revision=population.dataset_revision,
                converter_version="1",
                converter_revision=population.converter_revision,
                measured_tasks=101,
                task_specs_sha256=population.task_specs_sha256,
                evidence_url=inventory.evidence_url,
            ),
        ),
    )
    pq.write_table(pa.table({"task_id": pa.array([], type=pa.string())}), dedup.drop_manifest_uri)
    selected_ids = {row["task_id"] for row in selected}
    for name, cohort in (("partial", [task for task in tasks if task.id in selected_ids]), ("complete", tasks)):
        proof = oracle_receipt(cohort)
        entries = {}
        for key, entry in proof.tasks.items():
            assert isinstance(entry, OracleTaskReceipt) and entry.negative_control is not None
            entries[key] = entry.model_copy(
                update={
                    "runtime_sha256": canonical_sha256({}),
                    "negative_control": entry.negative_control.model_copy(
                        update={"runtime_sha256": canonical_sha256({})}
                    ),
                }
            )
        proof = proof.model_copy(update={"tasks": entries})
        assert all(proof.admits(task, {}) for task in cohort)
        path = StoragePath(str(tmp_path / f"{name}-filter.json"))
        path.write_text(proof.model_dump_json())
        reference = ContentEvidenceReference(uri=str(path), sha256=file_sha256(path))
        binding = {
            source.info.id: OracleFilterBinding(
                population_count=101, task_specs_sha256=population.task_specs_sha256, receipt=reference
            )
        }
        if name == "partial":
            with pytest.raises(ValueError, match="does not cover the retained population"):
                prepare_sample(source.declaration, inventory, str(tmp_path / name), None, binding)
        else:
            result = prepare_sample(source.declaration, inventory, str(tmp_path / name), None, binding)
            assert not result.sample.pre_oracle_filter and not result.sample.resample_after_oracle_filter
            assert result.sample.oracle_filter_receipt == reference


@pytest.mark.parametrize("payload_revision", ["4", "5", "6"])
def test_sample_review_keeps_full_population_identity_and_reuses_exact_panel_evidence(
    sample_panel, review_server, tmp_path, payload_revision
):
    reviewer = ChatReviewer(
        OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2),
        local_reviewer(ReviewerVersion.V3, payload_revision=payload_revision),
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
            title=None,
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
    # A population whose bytes actually changed refuses: the panel never reviewed them.
    changed = replace(
        source,
        config=replace(source.config, dataset=replace(source.dataset, revision="changed")),
        info=replace(
            source.info,
            actor_census_input=source.info.actor_census_input.model_copy(
                update={
                    "dataset_revision": "changed",
                    "task_specs_sha256": "changed-" + source.info.actor_census_input.task_specs_sha256,
                },
            ),
        ),
    )
    with pytest.raises(ValueError, match="another current population"):
        bind_sample_result(changed, receipt)


@pytest.mark.parametrize("payload_revision", ["4"])
def test_byte_identical_repin_panel_binds_current_pin_over_the_lineage_label(
    sample_panel, review_server, tmp_path, payload_revision
):
    """A panel whose digest equals the new pin's census binds CURRENT_PIN, not archived."""
    reviewer = ChatReviewer(
        OpenAIChatClient(f"http://127.0.0.1:{review_server.server_port}/v1", "unused", timeout=2),
        local_reviewer(ReviewerVersion.V3, payload_revision=payload_revision),
        query_cache_root=str(tmp_path / "cache"),
    )
    receipt = sample_qc_receipt(review_panel(sample_panel, reviewer, str(tmp_path / "reviews")).path)
    source = number_source(tmp_path / "numbers")
    source = replace(
        source,
        config=replace(
            source.config,
            dataset=SourceReference("fixture/repinned", "repinned-revision", "fixture"),
        ),
        info=replace(
            source.info,
            id=sample_panel.source_id,
            origin=SourceOrigin.TASKTROVE,
            upstream_dataset="fixture",
            oversight=None,
            title=None,
            actor_census_input=ActorCensusInput(
                dataset_id="fixture/repinned",
                dataset_revision="repinned-revision",
                converter_version=sample_panel.converter_version,
                converter_revision="fixture",
                measured_tasks=sample_panel.retained_tasks,
                task_specs_sha256=sample_panel.task_specs_sha256,
                evidence_url=sample_panel.evidence_url,
            ),
        ),
    )
    lineage = RetainedLineageProof(
        source_id=sample_panel.source_id,
        original_dataset_id=sample_panel.dataset_id,
        original_dataset_revision=sample_panel.dataset_revision,
        original_unique_count=sample_panel.retained_tasks,
        retained_content_sha256=sample_panel.task_specs_sha256,
        original_input=ActorCensusInput(
            dataset_id=sample_panel.dataset_id,
            dataset_revision=sample_panel.dataset_revision,
            converter_version=sample_panel.converter_version,
            converter_revision="fixture",
            measured_tasks=sample_panel.retained_tasks,
            task_specs_sha256=sample_panel.task_specs_sha256,
            evidence_url=sample_panel.evidence_url,
        ),
    )
    bound = bind_sample_result(source, receipt, retained_lineage=lineage)
    assert bound.info.independent_qc.panel_currency is SamplePanelCurrency.CURRENT_PIN
    assert str(sample_panel.task_specs_sha256[:12]) in bound.info.independent_qc.method
    assert "re-pin" in bound.info.independent_qc.method


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


@pytest.mark.parametrize("missing_count", [5, 6])
def test_sample_missing_tolerance_preserves_frozen_panel_and_missing_evidence(sample_panel, tmp_path, missing_count):
    template = panel_tasks(sample_panel)[0][0]
    tasks = [template.model_copy(update={"id": f"task-{index:03d}"}) for index in range(100)]
    files = write_text_population(((task.model_dump(mode="json"), {}) for task in tasks), str(tmp_path / "hundred"))
    population = SourceTextPopulation.raw_load(sample_panel.input_artifact).model_copy(
        update={
            "path": str(tmp_path / "hundred"),
            "task_count": 100,
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
    panel_path = tmp_path / "hundred-panel.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"task_id": task.id, "task_spec": task.model_dump_json(), "runtime": "{}"} for task in tasks],
            schema=PANEL_SCHEMA,
        ),
        panel_path,
    )
    inventory = SampleInventory.model_validate_json(StoragePath(sample_panel.evidence_url).read_text()).model_copy(
        update={
            "input_artifact": population.path,
            "retained_tasks": 100,
            "sample_size": 100,
            "task_specs_sha256": population.task_specs_sha256,
            "scanned_task_specs_sha256": population.task_specs_sha256,
            "panel_path": str(panel_path),
            "panel_sha256": file_sha256(StoragePath(str(panel_path))),
            "evidence_url": str(tmp_path / "hundred-inventory.json"),
        }
    )
    StoragePath(inventory.evidence_url).write_text(inventory.model_dump_json())
    panel = inventory.qc_panel()
    _, contracts = panel_tasks(panel)
    reviews = [
        ReviewRecord.model_validate_json(
            json.dumps(
                {
                    "task_id": task.id,
                    "status": "invalid" if index < missing_count else "reviewed",
                    "detail": "Truncated response after bounded attempts" if index < missing_count else "",
                    "verdict": (
                        None
                        if index < missing_count
                        else {
                            "task_id": task.id,
                            "quality": "good",
                            "confidence": "high",
                            "reference_status": "consistent",
                            "defects": [],
                            "evidence": "Reference agrees with the task.",
                        }
                    ),
                }
            )
        )
        for index, task in enumerate(tasks)
    ]
    batch = tmp_path / "batch.json"
    batch.write_text(json.dumps({"reviews": [row.model_dump(mode="json") for row in reviews], "attempts": []}))
    observations = write_observations(
        str(tmp_path / "observations"),
        [
            (ContractBatch(0, tasks, contracts), reviews, str(batch)),
        ],
    )
    reviewer = ChatReviewer(
        OpenAIChatClient("http://localhost:1/v1", "unused", timeout=2), local_reviewer(ReviewerVersion.V3)
    )
    result = SampleQCReview(
        path=str(observations.records_path.parent),
        panel=panel,
        reviewer=reviewer.identity,
        rubric=asdict(REVIEW_RUBRIC),
        panel_coverage_complete=False,
        attempted_count=100,
        result=quality_result(reviews),
        records_path=str(observations.records_path),
        records_sha256=file_sha256(observations.records_path),
        batch_receipts=observations.batch_receipts,
    )
    write_record(
        ArtifactRecord(
            output_path=result.path, result_type=result_type_name(type(result)), result=result.result_payload()
        )
    )
    if missing_count == 6:
        with pytest.raises(ValueError, match="at most 5% missing"):
            sample_qc_receipt(result.path)
        return
    receipt = sample_qc_receipt(result.path)
    assert receipt.review.panel == panel and not receipt.review.panel_coverage_complete
    assert receipt.review.result.reviewed == 95 and receipt.review.result.unavailable_count == 5
    assert receipt.review.result.issue_rate == 0
    assert {row.review.task_id for row in receipt.missing_observations} == {task.id for task in tasks[:5]}
    assert all(
        row.review.status.value == "invalid"
        and row.review.detail == "Truncated response after bounded attempts"
        and row.batch_receipt.uri == str(batch)
        and row.batch_receipt.sha256 == file_sha256(StoragePath(str(batch)))
        for row in receipt.missing_observations
    )
