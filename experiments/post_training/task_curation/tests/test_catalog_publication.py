# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

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
    ConvertedExamples,
    CurationCatalog,
    CurationSummary,
    DaytonaMeasurement,
    DaytonaSafety,
    DaytonaUnsafeReason,
    OracleValidation,
    SnapshotSource,
    register_catalog,
)
from finestore.rl.source_identity import canonical_source_id
from finestore.store import DataStore
from rigging.filesystem.distributed_lock import LeaseLostError, LocalFileLease

from experiments.post_training.task_curation.catalog_snapshot import (
    publish_catalog,
    refresh_catalog,
    refresh_daytona_measurements,
    retire_catalog_sources,
)
from experiments.post_training.task_curation.curation_catalog import RUNS
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, SourceReference
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
    changed = replace(
        source,
        config=replace(
            source.config,
            dataset=replace(source.dataset, revision="b" * 40, url=source.dataset.url.replace("a" * 40, "b" * 40)),
        ),
    )
    record_attempt(root, changed, "failed", "2026-10-16T12:00:00Z", "key-b", AssessmentStatus.FAILED)
    publish_catalog(root, "failed", [changed])
    row = selected_sources(root)[source.info.id]
    assert row.declaration.dataset_revision == "b" * 40
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
            id="Marin:agent",
            upstream_dataset="agent",
            upstream_name="agent",
            title=None,
            tags=("agentic",),
            daytona_snapshot_count=25,
            daytona_machine_slots=25,
            daytona_unsafe_reasons=(DaytonaUnsafeReason.TOO_MANY_SNAPSHOTS,),
            daytona_unsafe_details={"too_many_snapshots": "25 measured identities"},
            daytona_unsafe_counts={"too_many_snapshots": 25},
            daytona_execution_failures=0,
            daytona_count_basis="shellbox-identity over pinned payloads",
        ),
    )
    record_attempt(root, runnable, "first", "2026-10-16T10:00:00Z", "key-a")
    publish_catalog(root, "first", [runnable, agent])
    rows = selected_sources(root)
    assert rows[runnable.info.id].declaration.daytona_safety is DaytonaSafety.NOT_APPLICABLE
    assert rows["Marin:agent"].declaration.daytona_safety is DaytonaSafety.UNSAFE
    assert rows["Marin:agent"].declaration.daytona_snapshot_threshold == DAYTONA_SAFE_SNAPSHOT_LIMIT


def test_an_unselected_runnable_source_picks_up_the_current_safety_verdict(tmp_path):
    root = str(tmp_path / "production")
    first = number_source(tmp_path / "first")
    pinned = SourceReference("agent-data", "pin-1", "https://example.org/agent")
    agent = replace(
        first,
        config=replace(first.config, name="agent"),
        info=replace(
            first.info,
            id="Marin:agent",
            upstream_dataset="agent",
            upstream_name="agent",
            title=None,
            tags=("agentic",),
            dataset=pinned,
            daytona_snapshot_count=11,
            daytona_machine_slots=11,
            daytona_execution_failures=0,
            daytona_count_basis="complete actor census",
            daytona_unsafe_reasons=(DaytonaUnsafeReason.TOO_MANY_SNAPSHOTS,),
            daytona_unsafe_details={"too_many_snapshots": "11 identities"},
            daytona_unsafe_counts={"too_many_snapshots": 11},
        ),
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
            daytona_unsafe_reasons=(),
            daytona_unsafe_details={},
            daytona_unsafe_counts={},
            daytona_snapshot_count=5,
            daytona_machine_slots=5,
            daytona_execution_failures=0,
            daytona_count_basis="shellbox-identity over pinned payloads",
            oracle_validation=OracleValidation.VERIFIER_EXECUTED,
        ),
    )
    publish_catalog(root, "second", [reassessed, repaired])
    rows = selected_sources(root)
    assert rows["Marin:agent"].declaration.daytona_safety is DaytonaSafety.SAFE
    assert rows["Marin:agent"].declaration.oracle_validation is OracleValidation.VERIFIER_EXECUTED
    assert rows["Marin:agent"].assessment_key == "key-b"
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
    assert current[kept.info.id].declaration.quality_result.reviewed == 1
    assert current[kept.info.id].declaration.quality_result.verdict == "good"
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
    pointer = RlDataSource(
        info=SourceInfo(
            id="Marin:pointer",
            upstream_dataset=("Marin:pointer").partition(":")[2],
            upstream_name=("Marin:pointer").partition(":")[2],
            build_path="test",
            origin=SourceOrigin.MARIN,
        )
    )
    with pytest.raises(ValueError, match="no registered conversion pipeline"):
        publish_catalog(root, "invalid-source", [pointer])
    assert not (tmp_path / "production").exists()


def test_retirement_materializes_legacy_missing_defaults_without_changing_archived_rows(tmp_path):
    root = str(tmp_path / "production")
    kept = number_source(tmp_path / "input", "kept")
    retired = number_source(tmp_path / "input", "retired")
    record_attempt(root, kept, "first", "2026-10-16T10:00:00Z", "kept-key", source_count=2)
    record_attempt(root, retired, "first", "2026-10-16T10:00:00Z", "retired-key")
    publish_catalog(root, "first", [kept, retired])
    with DataStore.open(root) as store:
        register_catalog(store)
        records = [
            {
                key: value
                for key, value in store.read_view().point(INVENTORY, snapshot_id="first", id=source.info.id).items()
                if key in SnapshotSource.model_fields
            }
            for source in (kept, retired)
        ]

        records[0]["declaration"]["daytona_safety"] = None
        records[0]["declaration"]["oversight"] = None
        with store.transaction() as transaction:
            transaction.table(INVENTORY).extend({**row, "snapshot_id": "legacy"} for row in records)
            transaction.table(SNAPSHOTS).add(
                {"snapshot_id": "legacy", "created_at": "2026-10-16T10:30:00Z", "source_count": 2}
            )
            transaction.table(CURRENT).add({"name": "latest", "snapshot_id": "legacy"})
    retire_catalog_sources(root, "retired", frozenset({retired.info.id}))
    (current,) = CurationCatalog(root).sources("retired")
    assert current.declaration.daytona_safety is None
    assert current.declaration.oversight.value == "none"
    assert current.declaration.daytona_snapshot_count == 0
    original = CurationCatalog(root).view.point(INVENTORY, snapshot_id="legacy", id=kept.info.id)
    assert original["declaration"]["daytona_safety"] is None
    assert original["declaration"]["oversight"] is None


def test_measured_refresh_preserves_assessments_and_roundtrips_arrow_maps(tmp_path):
    root = str(tmp_path / "production")
    source = number_source(tmp_path / "input")
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "reviewed")
    publish_catalog(root, "first", [source])
    with pytest.raises(ValueError, match="Actor measurement is missing"):
        refresh_daytona_measurements(root, "unmeasured", {}, retired_source_ids=frozenset())
    assert CurationCatalog(root).current().snapshot_id == "first"
    measurement = DaytonaMeasurement(
        dataset_id=source.dataset.name,
        dataset_revision=source.dataset.revision,
        daytona_snapshot_count=14,
        daytona_machine_slots=20,
        daytona_execution_failures=3,
        daytona_count_basis="Complete production identity census",
        daytona_unsafe_reasons=["too_many_snapshots", "unbuilt_image"],
        daytona_unsafe_counts={"too_many_snapshots": 20, "unbuilt_image": 3},
        daytona_unsafe_details={"too_many_snapshots": "14 distinct identities", "unbuilt_image": "3 unbuilt actors"},
        daytona_roles={"agent": {"daytona": {"machine_slots": 20, "failures": {"unbuilt_image": 3}}}},
    )
    counts = ConvertedExamples(
        count=18,
        input_rows=20,
        rejections={"missing_environment": 2},
        admission_rejections={},
        source_dataset_id=source.dataset.name,
        source_revision=source.dataset.revision,
        converter_revision="converter-pin",
        converter_version=source.version,
        evidence_url="s3://test/converted.json",
    )
    refresh_daytona_measurements(
        root,
        "measured",
        {source.info.id: measurement},
        retired_source_ids=frozenset(),
        converted_examples={source.info.id: counts},
    )
    (row,) = CurationCatalog(root).sources("measured")
    assert row.assessment_key == "reviewed"
    assert row.declaration.pipeline == source.declaration.pipeline
    assert row.declaration.daytona_safety is DaytonaSafety.UNSAFE
    assert row.declaration.daytona_roles["agent"]["daytona"].failures == {"unbuilt_image": 3}
    assert row.declaration.daytona_unsafe_details == measurement.daytona_unsafe_details
    assert row.declaration.converted_examples == counts
    with pytest.raises(ValueError, match="different dataset pin"):
        refresh_daytona_measurements(
            root,
            "wrong-pin",
            {source.info.id: measurement.model_copy(update={"dataset_revision": "wrong"})},
            retired_source_ids=frozenset(),
        )
    assert CurationCatalog(root).current().snapshot_id == "measured"
    with pytest.raises(ValueError, match="different converter version"):
        refresh_daytona_measurements(
            root,
            "wrong-converter",
            {source.info.id: measurement},
            retired_source_ids=frozenset(),
            converted_examples={source.info.id: counts.model_copy(update={"converter_version": "stale"})},
        )
    assert CurationCatalog(root).current().snapshot_id == "measured"


def test_alias_metadata_refresh_keeps_archived_rows_and_packet_bytes(tmp_path):
    old = "MarinSkyRL:aime24"
    current = canonical_source_id(old)
    root = str(tmp_path / "production")
    fixture = number_source(tmp_path / "input")
    source = replace(
        fixture, info=replace(fixture.info, id=current, origin=SourceOrigin.HUGGINGFACEH4, upstream_dataset="aime24")
    )
    record_attempt(root, source, "first", "2026-10-16T10:00:00Z", "opaque-assessment-key")
    publish_catalog(root, "first", [source])
    with DataStore.open(root) as store:
        register_catalog(store)
        record = {
            key: value
            for key, value in store.read_view().point(INVENTORY, snapshot_id="first", id=current).items()
            if key in SnapshotSource.model_fields
        }
        record.update(snapshot_id="legacy", id=old, declaration={**record["declaration"], "id": old})
        assessment = {
            key: value
            for key, value in store.read_view().point(ASSESSMENTS, pipeline_key="opaque-assessment-key").items()
            if key in Assessment.model_fields
        }
        assessment["source_id"] = old
        with store.transaction() as transaction:
            transaction.table(INVENTORY).add(record)
            transaction.table(ASSESSMENTS).add(assessment)
            transaction.table(SNAPSHOTS).add(
                {"snapshot_id": "legacy", "created_at": "2026-10-16T10:30:00Z", "source_count": 1}
            )
            transaction.table(CURRENT).add({"name": "latest", "snapshot_id": "legacy"})
    catalog = CurationCatalog(root)
    before = {table: list(catalog.view.iter_rows(table)) for table in (ASSESSMENTS, POPULATIONS, ITEMS)}
    packet = catalog.packet("opaque-assessment-key", "opaque-assessment-key")
    refreshed = refresh_catalog(root, "canonical", [source], retired_source_ids=frozenset())
    assert refreshed.source_count == 1
    catalog = CurationCatalog(root)
    (selected,) = catalog.sources("canonical")
    assert selected.id == current and selected.assessment_key == "opaque-assessment-key"
    assert catalog.assessment("opaque-assessment-key").source_id == current
    assert catalog.view.point(INVENTORY, snapshot_id="legacy", id=old)["declaration"]["id"] == old
    assert {table: list(catalog.view.iter_rows(table)) for table in before} == before
    assert catalog.packet("opaque-assessment-key", "opaque-assessment-key") == packet
