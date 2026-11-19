# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-identity receipts bind byte-identical re-serializations exactly."""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import ActorCensusInput, SourceQualityResult
from marin.execution.artifact import ArtifactRecord, result_type_name, write_record
from pydantic import ValidationError
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.content_identity import (
    ContentIdentityReceipt,
    PopulationDigest,
    build_identity_receipt,
    read_content_identity_receipts,
)
from experiments.post_training.task_curation.full_content_binding import RetainedLineageProof
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.sample_qc_artifacts import SamplePanel, SampleQCReview
from experiments.post_training.task_curation.sample_qc_binding import (
    SampleQCReceipt,
    bind_sample_result,
    sample_matches_lineage,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

SOURCE_ID = "OpenAthena:identity-source"
PIN = "open-athena/identity-release-example"
REVISION = "a" * 40
CONVERTER_VERSION = "2026.11.13.1"

TASKS = {f"task-{index:03d}": json.dumps({"id": f"task-{index:03d}", "prompt": "x" * index}) for index in range(5)}


def write_population(tmp_path, name, tasks_by_id, *, rows_per_shard=2):
    """Write a specs-shard population artifact with the given per-shard row layout."""
    root = tmp_path / name
    specs = root / "specs"
    specs.mkdir(parents=True)
    items = list(tasks_by_id.items())
    task_specs_sha256 = hashlib.sha256()
    for shard in range((len(items) + rows_per_shard - 1) // rows_per_shard):
        rows = [
            {"id": key, "task_spec": tasks_by_id[key], "runtime": "{}"}
            for key, _ in items[shard * rows_per_shard : (shard + 1) * rows_per_shard]
        ]
        with (specs / f"part-{shard:06d}.parquet").open("wb") as stream:
            pq.write_table(pa.Table.from_pylist(rows), stream)
        for key, _ in items[shard * rows_per_shard : (shard + 1) * rows_per_shard]:
            task_specs_sha256.update(tasks_by_id[key].encode())
    population = SourceTextPopulation(
        path=str(root),
        source_id=SOURCE_ID,
        dataset_id=PIN,
        dataset_revision=REVISION,
        source_version=CONVERTER_VERSION,
        converter_revision="2f4b27ba2e",
        text_renderer_version="v1",
        task_count=len(TASKS),
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


@pytest.fixture
def reserialized_pair(tmp_path):
    """The same tasks written under two different shard layouts."""
    original = write_population(tmp_path, "original", TASKS, rows_per_shard=2)
    reordered = dict(reversed(list(TASKS.items())))
    reissued = write_population(tmp_path, "reissued", reordered, rows_per_shard=3)
    assert original.task_specs_sha256 != reissued.task_specs_sha256
    return original, reissued


def identity_receipt(original, reissued, tmp_path):
    return ContentIdentityReceipt(
        source_id=SOURCE_ID,
        dataset_id=PIN,
        dataset_revision=REVISION,
        original=PopulationDigest(
            artifact=original.path,
            task_specs_sha256=original.task_specs_sha256,
            task_count=original.task_count,
        ),
        reissued=PopulationDigest(
            artifact=reissued.path,
            task_specs_sha256=reissued.task_specs_sha256,
            task_count=reissued.task_count,
        ),
        identity_sha256="e" * 64,
        evidence_url=str(tmp_path / "digests.parquet"),
    )


def test_builder_accepts_byte_identical_re_serialization(reserialized_pair, tmp_path):
    original, reissued = reserialized_pair
    receipt = build_identity_receipt(
        SOURCE_ID,
        PIN,
        REVISION,
        original.path,
        reissued.path,
        str(tmp_path / "digests.parquet"),
    )
    assert receipt.original.task_specs_sha256 == original.task_specs_sha256
    assert receipt.reissued.task_specs_sha256 == reissued.task_specs_sha256
    assert receipt.original.task_count == receipt.reissued.task_count == len(TASKS)
    assert receipt.binds(SOURCE_ID, original.task_specs_sha256, reissued.task_specs_sha256)
    assert not receipt.binds(SOURCE_ID, original.task_specs_sha256, "f" * 64)
    assert not receipt.binds("Other:source", original.task_specs_sha256, reissued.task_specs_sha256)


def test_builder_refuses_one_changed_task(reserialized_pair, tmp_path):
    original, _ = reserialized_pair
    changed = dict(TASKS)
    changed["task-002"] = json.dumps({"id": "task-002", "prompt": "changed"})
    mutated = write_population(tmp_path, "mutated", changed, rows_per_shard=2)
    with pytest.raises(ValueError, match="1 tasks differ"):
        build_identity_receipt(
            SOURCE_ID,
            PIN,
            REVISION,
            original.path,
            mutated.path,
            str(tmp_path / "digests.parquet"),
        )


def test_builder_refuses_count_mismatch(reserialized_pair, tmp_path):
    original, _ = reserialized_pair
    subset = dict(list(TASKS.items())[:-1])
    shorter = write_population(tmp_path, "shorter", subset, rows_per_shard=2)
    with pytest.raises(ValueError, match="task digests"):
        build_identity_receipt(
            SOURCE_ID,
            PIN,
            REVISION,
            original.path,
            shorter.path,
            str(tmp_path / "digests.parquet"),
        )


def test_receipt_requires_equal_task_counts(tmp_path):
    with pytest.raises(ValidationError, match="equal task counts"):
        ContentIdentityReceipt(
            source_id=SOURCE_ID,
            dataset_id=PIN,
            dataset_revision=REVISION,
            original=PopulationDigest(artifact="a", task_specs_sha256="1" * 64, task_count=5),
            reissued=PopulationDigest(artifact="b", task_specs_sha256="2" * 64, task_count=4),
            identity_sha256="e" * 64,
            evidence_url="memory://evidence",
        )


def lineage_for(original, reissued):
    """A retained lineage whose original input digests like the reissued artifact."""
    actor = ActorCensusInput(
        dataset_id=PIN,
        dataset_revision=REVISION,
        converter_version=CONVERTER_VERSION,
        converter_revision="2f4b27ba2e",
        measured_tasks=len(TASKS),
        task_specs_sha256=reissued.task_specs_sha256,
        evidence_url="memory://census/artifact.json",
    )
    return RetainedLineageProof(
        source_id=SOURCE_ID,
        original_dataset_id=PIN,
        original_dataset_revision=REVISION,
        original_input=actor,
        original_unique_count=len(TASKS),
        retained_content_sha256="f" * 64,
    )


def panel_receipt(sha256_value):
    panel = SamplePanel(
        source_id=SOURCE_ID,
        dataset_id=PIN,
        dataset_revision=REVISION,
        converter_version=CONVERTER_VERSION,
        input_artifact="memory://sample/input",
        task_specs_sha256=sha256_value,
        scanned_task_specs_sha256=sha256_value,
        retained_tasks=len(TASKS),
        sample_seed=0,
        sample_size=3,
        panel_path="memory://sample/panel.parquet",
        panel_sha256="b" * 64,
        evidence_url="memory://sample/summary.json",
    )
    return SampleQCReceipt(
        source_id=SOURCE_ID,
        artifact_sha256="c" * 64,
        review=SampleQCReview(
            path="memory://sample/review",
            panel=panel,
            reviewer={},
            rubric={},
            panel_coverage_complete=True,
            attempted_count=3,
            result=SourceQualityResult(reviewed=3, issue_count=0, issues=[], repair_recommendation="None"),
            records_path="memory://sample/reviews.parquet",
            records_sha256="d" * 64,
            batch_receipts=[],
        ),
    )


def test_lineage_accepts_bound_digest_pair(reserialized_pair, tmp_path):
    original, reissued = reserialized_pair
    lineage = lineage_for(original, reissued)
    receipt = panel_receipt(original.task_specs_sha256)
    assert not sample_matches_lineage(receipt, lineage)
    assert sample_matches_lineage(receipt, lineage, identity_receipt(original, reissued, tmp_path))


def test_lineage_refuses_pair_outside_the_receipt(reserialized_pair, tmp_path):
    original, reissued = reserialized_pair
    lineage = lineage_for(original, reissued)
    receipt = panel_receipt("9" * 64)
    assert not sample_matches_lineage(receipt, lineage, identity_receipt(original, reissued, tmp_path))


def test_lineage_refuses_identity_for_another_pin(reserialized_pair, tmp_path):
    original, reissued = reserialized_pair
    lineage = lineage_for(original, reissued)
    receipt = panel_receipt(original.task_specs_sha256)
    moved = identity_receipt(original, reissued, tmp_path).model_copy(update={"dataset_revision": "b" * 40})
    assert not sample_matches_lineage(receipt, lineage, moved)


def declared_source(reissued, tmp_path):
    actor = ActorCensusInput(
        dataset_id=PIN,
        dataset_revision=REVISION,
        converter_version=CONVERTER_VERSION,
        converter_revision="2f4b27ba2e",
        measured_tasks=len(TASKS),
        task_specs_sha256=reissued.task_specs_sha256,
        evidence_url="memory://census/artifact.json",
    )
    info = SourceInfo(
        id=SOURCE_ID,
        origin=SourceOrigin.OPENATHENA,
        upstream_dataset="identity-source",
        upstream_name="identity-source",
        build_path="Test",
        actor_census_input=actor,
        within_source_dedup=deduplication_evidence(PIN, REVISION, CONVERTER_VERSION, len(TASKS)),
    )
    config = PrebuiltCohort(
        intended_use=IntendedUse.TRAIN,
        name="identity-source",
        version=CONVERTER_VERSION,
        repo=PIN,
        revision=REVISION,
        expected_rows=len(TASKS),
    )
    return RlDataSource(info=replace(info, dataset=config.dataset), config=config, pipeline=None)


def test_current_pin_panel_binds_through_identity(reserialized_pair, tmp_path):
    original, reissued = reserialized_pair
    source = declared_source(reissued, tmp_path)
    receipt = panel_receipt(original.task_specs_sha256)
    with pytest.raises(ValueError, match="another current population"):
        bind_sample_result(source, receipt)
    bound = bind_sample_result(
        source,
        receipt,
        content_identity=identity_receipt(original, reissued, tmp_path),
    )
    assert bound.info.quality_result == receipt.review.result
    assert "content-identity receipt" in bound.info.independent_qc.method
    assert bound.info.independent_qc.panel_currency.value == "current-pin"


def test_current_pin_panel_refuses_changed_task(reserialized_pair, tmp_path):
    original, reissued = reserialized_pair
    source = declared_source(reissued, tmp_path)
    receipt = panel_receipt(original.task_specs_sha256)
    forged = identity_receipt(original, reissued, tmp_path).model_copy(
        update={
            "reissued": PopulationDigest(
                artifact=reissued.path,
                task_specs_sha256="9" * 64,
                task_count=len(TASKS),
            ),
        },
    )
    with pytest.raises(ValueError, match="another current population"):
        bind_sample_result(source, receipt, content_identity=forged)


def test_pointer_loader_reads_the_pinned_document(tmp_path, reserialized_pair):
    original, reissued = reserialized_pair
    receipt = identity_receipt(original, reissued, tmp_path)
    document = tmp_path / "content_identity_receipts.json"
    document.write_text(json.dumps({SOURCE_ID: receipt.model_dump(mode="json")}) + "\n")
    loaded = read_content_identity_receipts(document, load_json=lambda path: json.loads(path.read_text()))
    assert loaded[SOURCE_ID] == receipt
