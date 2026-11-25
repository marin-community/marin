# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import pytest
from finestore.rl.curation import (
    ASSESSMENTS,
    ATTEMPTS,
    CURRENT,
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    INVENTORY,
    ITEMS,
    POPULATIONS,
    SNAPSHOTS,
    Assessment,
    AssessmentAttempt,
    AssessmentItem,
    AssessmentPopulation,
    AssessmentStatus,
    CurationCatalog,
    CurationSummary,
    DaytonaSafety,
    IndependentQC,
    OracleValidation,
    SnapshotSource,
    Verdict,
    register_catalog,
)
from finestore.rl.source_identity import canonical_source_id
from finestore.rl.status_taxonomy import DataRestriction
from finestore.store import DataStore
from rigging.filesystem.distributed_lock import LeaseLostError, LocalFileLease

from experiments.post_training.task_curation.catalog_snapshot import publish_catalog
from experiments.post_training.task_curation.curation_catalog import RUNS
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, SourceReference
from experiments.post_training.task_curation.tests.numbers_pipeline import NumbersConfig, number_source, numbers_pipeline


def record_attempt(
    root, source, run_id, started_at, key, status=AssessmentStatus.COMPLETE, *, source_count=1, source_id=None
):
    """Record a terminal campaign with an already archived model result.

    ``source_id`` stores the attempt under a historical published ID, as written
    before the identity rewrite.
    """
    published_id = source_id or source.info.id
    summary = CurationSummary(
        status="completed",
        mode="full",
        source_revision=source.dataset.revision,
        recipe_revision="1",
        raw_population_count=1,
        raw_sample_count=1,
        processed_rows=1,
        unprocessed_rows=0,
        admission_counts={"admitted": 1},
        verification=None,
    )
    with DataStore.open(root) as store:
        register_catalog(store)
        if store.read_view().point(RUNS, run_id=run_id) is None:
            store.table(RUNS, primary_key=("run_id",)).append(
                {"run_id": run_id, "status": "completed", "source_count": source_count, "started_at": started_at}
            )
        uri = store.write_object(f"curation/{key}.json", summary.model_dump_json().encode())
        archived = store.read_view().point(ASSESSMENTS, pipeline_key=key)
        if status == AssessmentStatus.COMPLETE and archived is None:
            packet = store.write_object(f"packets/{key}.json", json.dumps({"task": {"id": key}}).encode())
            store.flush()
            with store.transaction() as transaction:
                transaction.table(ASSESSMENTS).add(
                    Assessment(
                        pipeline_key=key,
                        source_id=published_id,
                        scope="full",
                        model={
                            "model": "fixture",
                            "revision": "1",
                            "max_tokens": 100,
                            "max_attempts": 1,
                            "retry_max_tokens": 100,
                        },
                        rubric="fixture",
                        curation=summary,
                        curation_uri=uri,
                        traces=[],
                    ).model_dump(mode="json")
                )
                transaction.table(POPULATIONS).extend(
                    AssessmentPopulation(
                        pipeline_key=key, cohort=cohort, population=count, sample_count=count
                    ).model_dump(mode="json")
                    for cohort, count in (("admitted", 1), ("excluded", 0))
                )
                transaction.table(ITEMS).add(
                    AssessmentItem(
                        pipeline_key=key,
                        task_id=key,
                        cohort="admitted",
                        status="reviewed",
                        verdict="good",
                        explanation="fixture",
                        evidence_ids=["task"],
                        packet_uri=packet,
                        has_task=True,
                        omissions=[],
                        exclusion_reason=None,
                    ).model_dump(mode="json")
                )
        store.table(ATTEMPTS, primary_key=("run_id", "name")).append(
            AssessmentAttempt(
                run_id=run_id,
                name=source.name,
                source_id=published_id,
                pipeline_key=key,
                status=status,
                reason="assessment failed" if status == AssessmentStatus.FAILED else None,
                paths=None,
                curation_uri=uri,
            ).model_dump(mode="json")
        )


def selected_sources(root):
    catalog = CurationCatalog(root)
    return {row.id: row for row in catalog.sources(catalog.current().snapshot_id)}


def store_historical_snapshot(root, run_id, created_at, rows):
    """Store a snapshot published before the identity rewrite, under historical IDs."""
    with DataStore.open(root) as store:
        register_catalog(store)
        store.table(SNAPSHOTS, primary_key=("snapshot_id",)).append(
            {"snapshot_id": run_id, "created_at": created_at, "source_count": len(rows)}
        )
        store.table(INVENTORY, primary_key=("snapshot_id", "id")).extend(row.model_dump(mode="json") for row in rows)
        store.table(CURRENT, primary_key=("name",)).append({"name": "latest", "snapshot_id": run_id})


HISTORICAL_VERIFIABLE_CODE = "MarinSkyRL:verifiable_code"


def renamed_runnable(path: Path) -> RlDataSource:
    """The Open-R1 code source whose published rows predate the identity rewrite."""
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="verifiable_code",
            upstream_name="verifiable_code",
            build_path="test",
            origin=SourceOrigin.OPEN_R1,
        ),
        config=NumbersConfig("verifiable_code", path),
        pipeline=numbers_pipeline,
    )


def historical_row(source: RlDataSource, run_id: str, historical_id: str, assessment_key: str | None) -> SnapshotSource:
    """One inventory row as stored before the identity rewrite."""
    if source.pipeline is None:
        status, reason = AssessmentStatus.SKIPPED, "No conversion pipeline"
    else:
        status, reason = AssessmentStatus.COMPLETE, "assessment published"
    return SnapshotSource(
        snapshot_id=run_id,
        id=historical_id,
        declaration=source.declaration.model_copy(update={"id": historical_id}),
        assessment_key=assessment_key,
        latest_attempt_key=assessment_key,
        latest_attempt_status=status,
        reason=reason,
    )


def test_disjoint_campaigns_accumulate_even_with_an_older_inventory(tmp_path):
    root = str(tmp_path / "production")
    first = number_source(tmp_path / "input", "first")
    second = number_source(tmp_path / "input", "second")
    record_attempt(root, second, "newer", "2026-10-16T12:00:00Z", "second-key")
    publish_catalog(root, "newer", [first, second])
    record_attempt(root, first, "older", "2026-10-16T11:00:00Z", "first-key")
    snapshot = publish_catalog(root, "older", [first])
    rows = selected_sources(root)
    assert {key: row.assessment_key for key, row in rows.items()} == {
        first.info.id: "first-key",
        second.info.id: "second-key",
    }
    assert CurationCatalog(root).packet("second-key", "second-key") == {"task": {"id": "second-key"}}
    assert publish_catalog(root, "older", [first]) == snapshot


def test_a_newer_failed_key_supersedes_an_older_success_and_exact_key_retry_reuses_evidence(tmp_path):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [source])
    record_attempt(root, source, "failed", "2026-10-16T12:00:00Z", "key-b", AssessmentStatus.FAILED)
    publish_catalog(root, "failed", [source])
    assert selected_sources(root)[source.info.id].assessment_key is None
    record_attempt(root, source, "superseded", "2026-10-16T11:00:00Z", "key-a")
    publish_catalog(root, "superseded", [source])
    row = selected_sources(root)[source.info.id]
    assert row.assessment_key is None
    assert row.latest_attempt_key == "key-b"
    assert "superseded" in row.reason
    record_attempt(root, source, "retry", "2026-10-16T13:00:00Z", "key-b")
    publish_catalog(root, "retry", [source])
    record_attempt(root, source, "same-key-failure", "2026-10-16T14:00:00Z", "key-b", AssessmentStatus.FAILED)
    publish_catalog(root, "same-key-failure", [source])
    assert selected_sources(root)[source.info.id].assessment_key == "key-b"
    assert CurationCatalog(root).packet("key-a", "key-a") == {"task": {"id": "key-a"}}


def test_selected_new_pin_invalidates_the_prior_grade(tmp_path):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [source])
    changed = replace(source, config=replace(source.config, dataset=replace(source.dataset, revision="new-pin")))
    record_attempt(root, changed, "failed", "2026-10-16T12:00:00Z", "key-b", AssessmentStatus.FAILED)
    publish_catalog(root, "failed", [changed])
    row = selected_sources(root)[source.info.id]
    assert row.declaration.dataset_revision == "new-pin"
    assert row.assessment_key is None


def test_a_superseded_source_does_not_block_the_other_sources_in_its_campaign(tmp_path):
    root = str(tmp_path / "production")
    first = number_source(tmp_path / "input", "first")
    second = number_source(tmp_path / "input", "second")
    record_attempt(root, first, "newer", "2026-10-16T12:00:00Z", "new-key", AssessmentStatus.FAILED)
    publish_catalog(root, "newer", [first])
    for source in (first, second):
        record_attempt(root, source, "older", "2026-10-16T11:00:00Z", source.name, source_count=2)
    snapshot = publish_catalog(root, "older", [first, second])
    rows = selected_sources(root)
    assert snapshot.source_count == 2
    assert rows[first.info.id].assessment_key is None
    assert rows[first.info.id].latest_attempt_key == "new-key"
    assert rows[second.info.id].assessment_key == "second"


def test_a_complete_attempt_publishes_its_measured_task_count(tmp_path):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [source])
    row = selected_sources(root)[source.info.id]
    assert row.declaration.task_count == 1


def test_an_unmeasured_or_moved_declaration_publishes_an_empty_count(tmp_path):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "failed", "2026-10-16T10:00:00Z", "key-a", AssessmentStatus.FAILED)
    publish_catalog(root, "failed", [source])
    row = selected_sources(root)[source.info.id]
    assert row.declaration.task_count is None
    record_attempt(root, source, "first", "2026-10-16T11:00:00Z", "key-b")
    publish_catalog(root, "first", [source])
    moved_pin = replace(source, config=replace(source.config, dataset=replace(source.dataset, revision="new-pin")))
    record_attempt(root, source, "second", "2026-10-16T12:00:00Z", "key-c")
    publish_catalog(root, "second", [moved_pin])
    row = selected_sources(root)[source.info.id]
    assert row.declaration.dataset_revision == "new-pin"
    assert row.assessment_key == "key-c"
    assert row.declaration.task_count is None
    bumped = replace(source, config=replace(source.config, version="2"))
    record_attempt(root, source, "third", "2026-10-16T13:00:00Z", "key-d")
    publish_catalog(root, "third", [bumped])
    row = selected_sources(root)[source.info.id]
    assert row.declaration.pipeline.version == "2"
    assert row.declaration.task_count is None


def test_a_failed_attempt_keeps_the_measured_count_at_its_pin(tmp_path):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [source])
    record_attempt(root, source, "failed", "2026-10-16T11:00:00Z", "key-b", AssessmentStatus.FAILED)
    publish_catalog(root, "failed", [source])
    row = selected_sources(root)[source.info.id]
    assert row.assessment_key is None
    assert row.declaration.task_count == 1


def test_republication_refreshes_an_inventory_only_declaration(tmp_path):
    root = str(tmp_path / "production")
    runnable = number_source(tmp_path / "input")
    pointer = RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="pointer",
            upstream_name="pointer",
            build_path="test",
            origin=SourceOrigin.MARIN,
        )
    )
    record_attempt(root, runnable, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [runnable, pointer])
    qc = IndependentQC(verdict=Verdict.GOOD, method="release oracle", reviewed=3, admitted=2, evidence_url="pinned")
    edited = RlDataSource(info=replace(pointer.info, independent_qc=qc))
    record_attempt(root, runnable, "second", "2026-10-16T11:00:00Z", "key-a")
    publish_catalog(root, "second", [runnable, edited])
    assert selected_sources(root)[pointer.info.id].declaration.independent_qc == qc


def test_publication_round_trips_daytona_safety_to_the_inventory(tmp_path):
    root = str(tmp_path / "production")
    runnable = number_source(tmp_path / "input")
    agent = RlDataSource(
        info=replace(
            runnable.info,
            upstream_dataset="agent",
            upstream_name="agent",
            tags=("agentic",),
            daytona_snapshot_count=25,
            daytona_count_basis="shellbox-identity over pinned payloads",
        )
    )
    record_attempt(root, runnable, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [runnable, agent])
    rows = selected_sources(root)
    assert rows[runnable.info.id].declaration.daytona_safety is DaytonaSafety.SAFE
    assert rows["Marin:agent"].declaration.daytona_safety is DaytonaSafety.UNSAFE
    assert rows["Marin:agent"].declaration.daytona_snapshot_threshold == DAYTONA_SAFE_SNAPSHOT_LIMIT


def test_an_unselected_runnable_source_picks_up_the_current_safety_verdict(tmp_path):
    root = str(tmp_path / "production")
    first = number_source(tmp_path / "first")
    pinned = SourceReference("agent-data", "pin-1", "https://example.org/agent")
    agent = RlDataSource(
        info=replace(
            first.info,
            upstream_dataset="agent",
            upstream_name="agent",
            tags=("agentic",),
            dataset=pinned,
        )
    )
    record_attempt(root, first, "first", "2026-10-16T10:00:00Z", "key-a", source_count=2)
    record_attempt(root, agent, "first", "2026-10-16T10:00:00Z", "key-b")
    publish_catalog(root, "first", [first, agent])
    reassessed = replace(first, config=replace(first.config, name="numbers-second"))
    record_attempt(root, reassessed, "second", "2026-10-16T11:00:00Z", "key-a")
    repaired = RlDataSource(
        info=replace(
            agent.info,
            daytona_snapshot_count=5,
            daytona_count_basis="shellbox-identity over pinned payloads",
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        )
    )
    publish_catalog(root, "second", [reassessed, repaired])
    rows = selected_sources(root)
    assert rows["Marin:agent"].declaration.daytona_safety is DaytonaSafety.SAFE
    assert rows["Marin:agent"].declaration.oracle_validation is OracleValidation.VERIFIER_EXECUTED
    assert rows["Marin:agent"].assessment_key == "key-b"
    assert rows[reassessed.info.id].assessment_key == "key-a"


def test_a_row_stored_under_a_historical_id_republishes_under_its_generated_identity(tmp_path):
    """A prior row matched through the alias table publishes under the generated ID."""
    root = str(tmp_path / "production")
    renamed = renamed_runnable(tmp_path / "renamed")
    record_attempt(root, renamed, "prior", "2026-10-16T10:00:00Z", "key-a", source_id=HISTORICAL_VERIFIABLE_CODE)
    store_historical_snapshot(
        root, "prior", "2026-10-16T10:00:00Z", [historical_row(renamed, "prior", HISTORICAL_VERIFIABLE_CODE, "key-a")]
    )
    first = number_source(tmp_path / "first", "first")
    second = number_source(tmp_path / "second", "second")
    record_attempt(root, first, "second", "2026-10-16T11:00:00Z", "key-f", source_count=2)
    record_attempt(root, second, "second", "2026-10-16T11:00:00Z", "key-s")
    publish_catalog(root, "second", [first, second, renamed])
    rows = selected_sources(root)
    assert HISTORICAL_VERIFIABLE_CODE not in rows
    row = rows[renamed.info.id]
    assert row.id == "Open-R1:verifiable_code"
    assert row.assessment_key == "key-a"


def test_an_inventory_row_stored_under_a_historical_id_reconciles_with_its_declaration(tmp_path):
    """A carried inventory row publishes its generated row ID and declaration ID together."""
    root = str(tmp_path / "production")
    runnable = number_source(tmp_path / "input")
    old_pointer = RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="skill2env",
            upstream_name="skill2env",
            build_path="test",
            origin=SourceOrigin.HARBOR_HUB,
        )
    )
    new_pointer = RlDataSource(info=replace(old_pointer.info, subset="skill2env"))
    store_historical_snapshot(
        root, "prior", "2026-10-16T10:00:00Z", [historical_row(old_pointer, "prior", old_pointer.info.id, None)]
    )
    record_attempt(root, runnable, "second", "2026-10-16T11:00:00Z", "key-r")
    publish_catalog(root, "second", [runnable, new_pointer])
    rows = selected_sources(root)
    assert old_pointer.info.id not in rows
    row = rows[new_pointer.info.id]
    assert row.id == row.declaration.id == "Harbor Hub:skill2env/skill2env"


def test_a_skipped_campaign_attempt_keeps_the_renamed_sources_prior_assessment(tmp_path):
    """A renamed source selected with a SKIPPED attempt keeps its prior assessment, and
    the stored assessment joins the generated row through the alias table."""
    root = str(tmp_path / "production")
    renamed = renamed_runnable(tmp_path / "renamed")
    record_attempt(root, renamed, "prior", "2026-10-16T10:00:00Z", "key-a", source_id=HISTORICAL_VERIFIABLE_CODE)
    store_historical_snapshot(
        root, "prior", "2026-10-16T10:00:00Z", [historical_row(renamed, "prior", HISTORICAL_VERIFIABLE_CODE, "key-a")]
    )
    record_attempt(root, renamed, "second", "2026-10-16T11:00:00Z", "key-b", AssessmentStatus.SKIPPED)
    publish_catalog(root, "second", [renamed])
    row = selected_sources(root)[renamed.info.id]
    assert row.id == "Open-R1:verifiable_code"
    assert row.assessment_key == "key-a"
    assert row.latest_attempt_status is AssessmentStatus.SKIPPED
    assessment = CurationCatalog(root).assessment("key-a")
    assert assessment.source_id == HISTORICAL_VERIFIABLE_CODE
    assert canonical_source_id(assessment.source_id) == canonical_source_id(row.id)


class ExpiredLease(LocalFileLease):
    def refresh(self):
        self.release()
        super().refresh()


def test_a_lost_publication_lease_cannot_replace_the_selected_snapshot(tmp_path, monkeypatch):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "key-a")
    first = publish_catalog(root, "first", [source])
    record_attempt(root, source, "next", "2026-10-16T11:00:00Z", "key-b")
    with monkeypatch.context() as context:
        context.setattr(
            "experiments.post_training.task_curation.catalog_snapshot.create_lock",
            lambda path, worker_id: ExpiredLease(path, worker_id),
        )
        with pytest.raises(LeaseLostError):
            publish_catalog(root, "next", [source])
    assert CurationCatalog(root).current() == first
    publish_catalog(root, "next", [source])
    assert selected_sources(root)[source.info.id].assessment_key == "key-b"
