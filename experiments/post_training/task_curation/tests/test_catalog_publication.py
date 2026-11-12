# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from finestore.rl.curation import (
    ASSESSMENTS,
    ATTEMPTS,
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    ITEMS,
    POPULATIONS,
    RUNS,
    Assessment,
    AssessmentAttempt,
    AssessmentItem,
    AssessmentPopulation,
    AssessmentStatus,
    CurationCatalog,
    CurationSummary,
    DaytonaSafety,
    OracleValidation,
    register_catalog,
)
from finestore.store import DataStore
from rigging.filesystem.distributed_lock import LeaseLostError, LocalFileLease

from experiments.post_training.task_curation.catalog_snapshot import (
    publish_catalog,
    refresh_catalog,
    retire_catalog_sources,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceReference
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


def record_attempt(root, source, run_id, started_at, key, status=AssessmentStatus.COMPLETE, *, source_count=1):
    """Record a terminal campaign with an already archived model result."""
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
                        source_id=source.info.id,
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
                source_id=source.info.id,
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


def test_publication_round_trips_daytona_safety_to_the_inventory(tmp_path):
    root = str(tmp_path / "production")
    runnable = number_source(tmp_path / "input")
    agent = replace(
        runnable,
        config=replace(runnable.config, name="agent"),
        info=replace(
            runnable.info,
            id="fixture:agent",
            tags=("agentic",),
            daytona_snapshot_count=25,
            daytona_count_basis="shellbox-identity over pinned payloads",
        ),
    )
    record_attempt(root, runnable, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [runnable, agent])
    rows = selected_sources(root)
    assert rows[runnable.info.id].declaration.daytona_safety is DaytonaSafety.SAFE
    assert rows["fixture:agent"].declaration.daytona_safety is DaytonaSafety.UNSAFE
    assert rows["fixture:agent"].declaration.daytona_snapshot_threshold == DAYTONA_SAFE_SNAPSHOT_LIMIT


def test_an_unselected_runnable_source_picks_up_the_current_safety_verdict(tmp_path):
    root = str(tmp_path / "production")
    first = number_source(tmp_path / "first")
    pinned = SourceReference("agent-data", "pin-1", "https://example.org/agent")
    agent = replace(
        first,
        config=replace(first.config, name="agent"),
        info=replace(first.info, id="fixture:agent", tags=("agentic",), dataset=pinned),
    )
    record_attempt(root, first, "first", "2026-10-16T10:00:00Z", "key-a", source_count=2)
    record_attempt(root, agent, "first", "2026-10-16T10:00:00Z", "key-b")
    publish_catalog(root, "first", [first, agent])
    reassessed = replace(first, config=replace(first.config, name="numbers-second"))
    record_attempt(root, reassessed, "second", "2026-10-16T11:00:00Z", "key-a")
    repaired = replace(
        agent,
        info=replace(
            agent.info,
            daytona_snapshot_count=5,
            daytona_count_basis="shellbox-identity over pinned payloads",
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        ),
    )
    publish_catalog(root, "second", [reassessed, repaired])
    rows = selected_sources(root)
    assert rows["fixture:agent"].declaration.daytona_safety is DaytonaSafety.SAFE
    assert rows["fixture:agent"].declaration.oracle_validation is OracleValidation.VERIFIER_EXECUTED
    assert rows["fixture:agent"].assessment_key == "key-b"
    assert rows[reassessed.info.id].assessment_key == "key-a"


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


def test_metadata_refresh_retires_sources_without_removing_archived_assessments(tmp_path):
    root = str(tmp_path / "production")
    kept = number_source(tmp_path / "input", "kept")
    retired = number_source(tmp_path / "input", "retired")
    record_attempt(root, kept, "first", "2026-10-16T10:00:00Z", "kept-key", source_count=2)
    record_attempt(root, retired, "first", "2026-10-16T10:00:00Z", "retired-key")
    original = publish_catalog(root, "first", [kept, retired])
    edited = replace(kept, info=replace(kept.info, notes="Updated measured evidence"))
    retired_snapshot = retire_catalog_sources(root, "retired-snapshot", frozenset({retired.info.id}))
    assert retired_snapshot.source_count == 1
    refreshed = refresh_catalog(root, "metadata-refresh", [edited], retired_source_ids=frozenset())
    current = selected_sources(root)
    assert set(current) == {kept.info.id}
    assert current[kept.info.id].assessment_key == "kept-key"
    assert current[kept.info.id].declaration.notes == "Updated measured evidence"
    assert refreshed.source_count == 1
    catalog = CurationCatalog(root)
    assert {row.id for row in catalog.sources(original.snapshot_id)} == {kept.info.id, retired.info.id}
    assert catalog.packet("retired-key", "retired-key") == {"task": {"id": "retired-key"}}


def test_metadata_refresh_cannot_reuse_a_grade_for_a_changed_pipeline(tmp_path):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "key")
    original = publish_catalog(root, "first", [source])
    changed = replace(source, config=replace(source.config, name="changed-pipeline"))
    with pytest.raises(ValueError, match="Source pin changed"):
        refresh_catalog(root, "invalid-refresh", [changed], retired_source_ids=frozenset())
    assert CurationCatalog(root).current() == original


def test_publication_rejects_a_source_without_a_registered_converter(tmp_path):
    root = str(tmp_path / "production")
    pointer = RlDataSource(info=SourceInfo(id="new:pointer", title="Pointer", origin="fixture"))
    with pytest.raises(ValueError, match="no registered conversion pipeline"):
        publish_catalog(root, "invalid-source", [pointer])
    assert not (tmp_path / "production").exists()
