# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of EvalDash's durable serving catalog and scheduled reconciliation."""

import asyncio
import math
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy
from evaldash import app as evaldash_app
from evaldash import fixtures, ingest, results_db
from evaldash.db_migrations import apply_migrations
from evaldash.metrics import record_headline
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import EvalRunRecord, TaskCoverage, list_records, write_record
from marin.execution.step_status import STATUS_RUNNING, STATUS_SUCCESS, StatusFile
from sqlalchemy.pool import StaticPool


def _engine() -> sqlalchemy.Engine:
    return sqlalchemy.create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def migrate_schema(engine: sqlalchemy.Engine) -> None:
    with engine.begin() as connection:
        apply_migrations(connection)
    results_db.verify_schema(engine)


def _record(tmp_path) -> EvalRunRecord:
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    return list_records(str(source))[0]


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 16, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _store(engine: sqlalchemy.Engine) -> evaldash_app.PgRecordStore:
    return evaldash_app.PgRecordStore(engine)


def _stored(store: evaldash_app.PgRecordStore, run_id: str) -> dict:
    store.reload_if_changed()
    record = store.get_record(run_id)
    assert record is not None, f"{run_id} is not in the catalog"
    return record


def test_migrations_adopt_existing_serving_rows_and_boot_without_object_storage(tmp_path):
    engine = _engine()
    for table in (results_db.eval_runs, results_db.eval_metrics, results_db.model_state):
        table.create(engine)
    record = _record(tmp_path).model_copy(update={"description": "seeded before migration"})
    results_db.upsert_legacy_record(engine, record)

    migrate_schema(engine)
    migrate_schema(engine)
    store = _store(engine)

    assert _stored(store, record.run_id)["run_id"] == record.run_id
    assert store.store_info().record_count == 1
    results_db.upsert_legacy_record(engine, record.model_copy(update={"description": "old revision write"}))
    assert _stored(_store(engine), record.run_id)["description"] == "seeded before migration"
    with engine.begin() as conn:
        migrations = set(conn.execute(sqlalchemy.text("SELECT name FROM schema_migrations")).scalars())
    assert migrations == {"0001_initial", "0002_record_sources", "0003_repeat_aggregates"}


def test_older_binary_rejects_a_database_with_unknown_migrations():
    engine = _engine()
    migrate_schema(engine)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("INSERT INTO schema_migrations (name) VALUES ('9999_future')"))

    with pytest.raises(RuntimeError, match="9999_future"):
        migrate_schema(engine)


@pytest.mark.parametrize("source_count,source_correct", [(30, 21), (300, 221)])
def test_catalog_projection_reports_repeat_items_and_trials(tmp_path, source_count, source_correct):
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    record = next(record for record in list_records(str(source)) if record.evaluation.name == "aime24")
    record = record.model_copy(
        update={
            "coverage": {
                "aime24": TaskCoverage(
                    n_benchmark=source_count,
                    n_attempted=source_count,
                    n_scored=source_count,
                    n_correct=source_correct,
                )
            }
        }
    )

    projected = EvalRunRecord.model_validate(results_db.run_row(record)["record"])
    repeat = projected.repeat_aggregates["aime24"]
    measurement = record_headline(projected)
    assert measurement is not None

    assert projected.coverage["aime24"].n_scored == 30
    assert projected.coverage["aime24"].n_correct is None
    assert (repeat.n_items, repeat.n_repeats, repeat.n_trials) == (30, 10, 300)
    assert repeat.source_coverage.n_scored == source_count
    assert measurement["metric_kind"] == "continuous"
    assert measurement["repeat_aggregation"]["n_trials"] == 300
    assert repeat.stderr == pytest.approx(math.sqrt(0.42 * 0.58 / 29))
    assert measurement["low"] == pytest.approx(0.42 - 1.96 * repeat.stderr, abs=1e-5)


def test_capped_repeat_uses_full_benchmark_in_item_units(tmp_path):
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    record = next(record for record in list_records(str(source)) if record.evaluation.name == "aime24")
    metrics = {**record.metrics["aime24"], "num_total": 29.0, "solved_avg": 12.0, "accuracy_avg": 12.0 / 29}
    record = record.model_copy(
        update={
            "metrics": {"aime24": metrics},
            "coverage": {"aime24": TaskCoverage(n_benchmark=300, n_attempted=290, n_scored=290)},
        }
    )

    projected = EvalRunRecord.model_validate(results_db.run_row(record)["record"])
    headline = record_headline(projected)
    assert headline is not None
    assert projected.coverage["aime24"].n_benchmark == 30
    assert (headline["n_attempted"], headline["n_scored"]) == (29, 29)
    assert headline["benchmark_rate"] == pytest.approx(29 / 30)


@pytest.mark.parametrize("source_count", [30, 300, None])
def test_catalog_projection_recovers_historical_ungraded_or_absent_coverage(tmp_path, source_count):
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    record = next(record for record in list_records(str(source)) if record.evaluation.name == "aime24")
    old_coverage = (
        {"aime24": TaskCoverage(n_attempted=source_count, n_scored=0, errors={"ungraded": source_count})}
        if source_count is not None
        else {}
    )
    record = record.model_copy(update={"coverage": old_coverage})

    projected = EvalRunRecord.model_validate(results_db.run_row(record)["record"])
    headline = record_headline(projected)
    assert headline is not None

    assert projected.coverage["aime24"].n_scored == 30
    assert projected.coverage["aime24"].n_attempted == (30 if source_count is not None else None)
    assert headline["value"] == pytest.approx(0.42)
    assert headline["n_scored"] == 30
    assert headline["errors"] == {}


def test_partial_trial_coverage_does_not_abort_catalog_projection(tmp_path):
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    record = next(record for record in list_records(str(source)) if record.evaluation.name == "aime24")

    nearly_complete = record.model_copy(
        update={
            "coverage": {
                "aime24": TaskCoverage(n_benchmark=300, n_attempted=300, n_scored=299, errors={"request_failed": 1})
            }
        }
    )
    projected = EvalRunRecord.model_validate(results_db.run_row(nearly_complete)["record"])
    assert projected.coverage["aime24"].n_scored == 30
    assert projected.repeat_aggregates["aime24"].n_scored_trials == 299
    assert projected.repeat_aggregates["aime24"].source_coverage.errors == {"request_failed": 1}

    ambiguous = record.model_copy(update={"coverage": {"aime24": TaskCoverage(n_attempted=300, n_scored=200)}})
    projected_partial = EvalRunRecord.model_validate(results_db.run_row(ambiguous)["record"])
    partial_headline = record_headline(projected_partial)
    complete_headline = record_headline(projected)
    assert partial_headline is not None and complete_headline is not None
    assert projected_partial.repeat_aggregates["aime24"].coverage_basis == "item_lower_bound"
    assert projected_partial.coverage["aime24"].n_scored == 20
    assert partial_headline["high"] - partial_headline["low"] >= complete_headline["high"] - complete_headline["low"]


@pytest.mark.parametrize("source_count", [30, 300])
def test_all_unanswered_repeated_trials_remain_flagged(tmp_path, source_count):
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    record = next(record for record in list_records(str(source)) if record.evaluation.name == "aime24")
    metrics = {**record.metrics["aime24"], "accuracy_avg": 0.0, "solved_avg": 0.0}
    record = record.model_copy(
        update={
            "metrics": {"aime24": metrics},
            "coverage": {
                "aime24": TaskCoverage(
                    n_benchmark=source_count,
                    n_attempted=source_count,
                    n_scored=source_count,
                    n_unanswered=source_count,
                )
            },
        }
    )

    projected = EvalRunRecord.model_validate(results_db.run_row(record)["record"])
    headline = record_headline(projected)
    assert headline is not None
    assert projected.coverage["aime24"].n_unanswered == 30
    assert "no_answers" in headline["flags"]


def test_migration_rematerializes_existing_repeat_records(tmp_path):
    engine = _engine()
    results_db.metadata.create_all(engine)
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    record = next(record for record in list_records(str(source)) if record.evaluation.name == "aime24")
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMP)"))
        conn.execute(
            sqlalchemy.text("INSERT INTO schema_migrations (name) VALUES ('0001_initial'), ('0002_record_sources')")
        )
        conn.execute(
            results_db.catalog_state.insert().values(singleton=True, generation=1, updated_at=datetime.now(UTC))
        )
        conn.execute(
            results_db.catalog_runs.insert(),
            {**results_db.run_row(record), "record": record.model_dump(mode="json", by_alias=True)},
        )

    migrate_schema(engine)
    projected = results_db.fetch_snapshot(engine).records[0]

    assert projected.repeat_aggregates["aime24"].n_trials == 300
    assert projected.coverage["aime24"].n_scored == 30
    assert results_db.fetch_snapshot(engine).generation == 2


def test_snapshot_reads_record_written_before_tokenizer_revision(tmp_path):
    engine = _engine()
    migrate_schema(engine)
    record = _record(tmp_path)
    stored = record.model_dump(mode="json")
    config = asdict(ModelConfig(name=record.model.name, location=record.model.location))
    config.pop("tokenizer_revision")
    stored["model"]["config"] = config
    with engine.begin() as conn:
        conn.execute(results_db.catalog_runs.insert(), {**results_db.run_row(record), "record": stored})

    loaded = results_db.fetch_snapshot(engine).records[0]

    assert loaded.run_id == record.run_id
    assert loaded.model.config is not None
    assert loaded.model.config.tokenizer_revision is None
    assert loaded.model.config.location == record.model.location


def test_reconciler_discovers_new_paths_and_only_rereads_known_paths_when_due(tmp_path):
    engine = _engine()
    migrate_schema(engine)
    clock = Clock()
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    write_record(record, str(prefix))
    store = _store(engine)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(prefix),), 600, 86400, now=clock)

    asyncio.run(ingestor.run_once())
    initial = store.get_record(record.run_id)
    assert initial is not None
    generation = store.store_info().catalog_generation
    state = next(iter(results_db.source_states(engine, str(prefix)).values()))

    clock.now += timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert store.store_info().catalog_generation == generation

    changed = record.model_copy(update={"description": "rewritten"})
    write_record(changed, str(prefix))
    added = record.model_copy(update={"run_id": f"{record.run_id}-new", "description": "new"})
    write_record(added, str(prefix))
    clock.now = state.next_verify_at.replace(tzinfo=UTC) - timedelta(seconds=1)
    asyncio.run(ingestor.run_once())

    assert _stored(store, record.run_id)["description"] == initial["description"]
    assert _stored(store, added.run_id)["description"] == "new"

    clock.now += timedelta(seconds=2)
    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["description"] == "rewritten"


def test_step_completion_updates_a_cataloged_run_without_rewriting_its_record(tmp_path):
    engine = _engine()
    migrate_schema(engine)
    clock = Clock()
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    write_record(record, str(prefix))
    status_file = StatusFile(str(prefix / record.run_id), worker_id="test")
    status_file.write_status(STATUS_RUNNING)
    store = _store(engine)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(prefix),), 600, 86400, now=clock)

    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["step_status"] == STATUS_RUNNING
    assert store.fetch_runs()[0]["status"] == "running"

    status_file.write_status(STATUS_SUCCESS)
    state = next(iter(results_db.source_states(engine, str(prefix)).values()))
    clock.now = state.next_verify_at.replace(tzinfo=UTC) + timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["step_status"] == STATUS_SUCCESS
    assert store.fetch_runs()[0]["status"] == record.status.value


def test_reconciler_promotes_duplicate_only_after_two_successful_absences(tmp_path):
    engine = _engine()
    migrate_schema(engine)
    clock = Clock()
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    record = _record(tmp_path)
    write_record(record.model_copy(update={"description": "canonical"}), str(canonical))
    write_record(record.model_copy(update={"description": "legacy"}), str(legacy))
    store = _store(engine)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(canonical), str(legacy)), 600, 86400, now=clock)

    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["description"] == "canonical"

    canonical_state = next(iter(results_db.source_states(engine, str(canonical)).values()))
    (canonical / record.run_id / "record.json").unlink()
    clock.now = canonical_state.next_verify_at.replace(tzinfo=UTC) + timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["description"] == "canonical"

    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["description"] == "canonical"

    clock.now += timedelta(seconds=600)
    asyncio.run(ingestor.run_once())
    assert not results_db.source_states(engine, str(canonical))
    assert results_db.source_states(engine, str(legacy))
    assert _stored(store, record.run_id)["description"] == "legacy"


def test_reconciler_reappearance_resets_missing_confirmation(tmp_path):
    engine = _engine()
    migrate_schema(engine)
    clock = Clock()
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    write_record(record, str(prefix))
    store = _store(engine)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(prefix),), 600, 86400, now=clock)
    asyncio.run(ingestor.run_once())

    state = next(iter(results_db.source_states(engine, str(prefix)).values()))
    path = prefix / record.run_id / "record.json"
    path.unlink()
    clock.now = state.next_verify_at.replace(tzinfo=UTC) + timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert next(iter(results_db.source_states(engine, str(prefix)).values())).missing_since is not None

    write_record(record, str(prefix))
    clock.now += timedelta(seconds=600)
    asyncio.run(ingestor.run_once())
    state = next(iter(results_db.source_states(engine, str(prefix)).values()))
    assert state.missing_since is None

    path.unlink()
    clock.now = state.next_verify_at.replace(tzinfo=UTC) + timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert store.get_record(record.run_id) is not None


def test_first_inventory_failure_does_not_downgrade_migrated_row(tmp_path, monkeypatch):
    engine = _engine()
    for table in (results_db.eval_runs, results_db.eval_metrics, results_db.model_state):
        table.create(engine)
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    record = _record(tmp_path)
    results_db.upsert_legacy_record(engine, record.model_copy(update={"description": "migrated canonical"}))
    migrate_schema(engine)
    write_record(record.model_copy(update={"description": "legacy"}), str(legacy))
    store = _store(engine)
    real_list_record_paths = evaldash_app.list_record_paths

    def list_unless_canonical(prefix: str) -> list[str]:
        if prefix == str(canonical):
            raise OSError("canonical unavailable")
        return real_list_record_paths(prefix)

    monkeypatch.setattr(evaldash_app, "list_record_paths", list_unless_canonical)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(canonical), str(legacy)), 600, 86400, now=Clock())
    assert asyncio.run(ingestor.run_once()) == (str(canonical),)

    assert _stored(store, record.run_id)["description"] == "migrated canonical"


def test_stale_failure_cannot_overwrite_a_later_success():
    engine = _engine()
    migrate_schema(engine)
    prefix = "file:///records"
    results_db.configure_prefixes(engine, (prefix,))
    probe_at = datetime(2026, 8, 16, tzinfo=UTC)
    results_db.reconcile_prefix(engine, prefix, [], [], probe_at + timedelta(seconds=1), 600)

    results_db.mark_prefix_failed(engine, prefix, probe_at, "stale failure")

    status = results_db.prefix_statuses(engine)[0]
    assert status.last_probe_at.replace(tzinfo=UTC) == probe_at + timedelta(seconds=1)
    assert status.error is None


def test_reconciler_keeps_last_valid_row_when_a_prefix_or_rewritten_record_fails(tmp_path, monkeypatch):
    engine = _engine()
    migrate_schema(engine)
    clock = Clock()
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    write_record(record, str(prefix))
    store = _store(engine)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(prefix),), 600, 86400, now=clock)
    asyncio.run(ingestor.run_once())

    state = next(iter(results_db.source_states(engine, str(prefix)).values()))
    (prefix / record.run_id / "record.json").write_text("not json", encoding="utf-8")
    clock.now = state.next_verify_at.replace(tzinfo=UTC) + timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["run_id"] == record.run_id
    assert ingestor.status()["prefixes"][0]["parse_failures"]

    def fail_listing(_prefix: str) -> list[str]:
        raise OSError("object store unavailable")

    monkeypatch.setattr(evaldash_app, "list_record_paths", fail_listing)
    clock.now += timedelta(seconds=600)
    assert asyncio.run(ingestor.run_once()) == (str(prefix),)

    assert _stored(store, record.run_id)["run_id"] == record.run_id
    failed_probe = ingestor.status()["prefixes"][0]
    assert failed_probe["last_probe_time"] != failed_probe["last_success_time"]
    assert failed_probe["error"]


def test_reconciler_preserves_seeded_row_when_first_object_read_is_invalid(tmp_path):
    engine = _engine()
    for table in (results_db.eval_runs, results_db.eval_metrics, results_db.model_state):
        table.create(engine)
    record = _record(tmp_path).model_copy(update={"description": "last valid"})
    results_db.upsert_legacy_record(engine, record)
    migrate_schema(engine)
    prefix = tmp_path / "records"
    path = prefix / record.run_id / "record.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    store = _store(engine)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(prefix),), 600, 86400, now=Clock())

    asyncio.run(ingestor.run_once())

    assert _stored(store, record.run_id)["description"] == "last valid"
    assert ingestor.status()["prefixes"][0]["parse_failures"]


def test_configured_prefix_order_and_membership_select_retained_sources(tmp_path):
    engine = _engine()
    migrate_schema(engine)
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    record = _record(tmp_path)
    write_record(record.model_copy(update={"description": "canonical"}), str(canonical))
    write_record(record.model_copy(update={"description": "legacy"}), str(legacy))
    store = _store(engine)
    ingestor = evaldash_app.PostgresIngestor(engine, (str(canonical), str(legacy)), 600, 86400, now=Clock())
    asyncio.run(ingestor.run_once())
    assert _stored(store, record.run_id)["description"] == "canonical"

    reordered = evaldash_app.PostgresIngestor(engine, (str(legacy), str(canonical)), 600, 86400, now=Clock())
    reordered.status()
    assert _stored(store, record.run_id)["description"] == "canonical"
    asyncio.run(reordered.run_once())
    assert _stored(store, record.run_id)["description"] == "legacy"

    asyncio.run(evaldash_app.PostgresIngestor(engine, (str(canonical),), 600, 86400, now=Clock()).run_once())
    assert _stored(store, record.run_id)["description"] == "canonical"


def test_scheduled_ingest_publishes_multiple_batches_without_loading_a_serving_snapshot(tmp_path, monkeypatch):
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    migrate_schema(engine)
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    expected = {f"run-{index}" for index in range(257)}
    for run_id in expected:
        write_record(record.model_copy(update={"run_id": run_id}), str(prefix))
    monkeypatch.setenv("RECORDS_PREFIXES", str(prefix))
    monkeypatch.setenv("MARINA_DATABASE_URL", str(engine.url))
    monkeypatch.setenv("EVALDASH_STORE", "postgres")
    monkeypatch.setattr(ingest, "job_engine", lambda: engine)

    materialized_sizes = []

    def inspect_catalog_queries(_conn, _cursor, _statement, parameters, context, _many):
        if context.compiled is None:
            return  # Schema reflection uses raw driver SQL, not catalog projections.
        query = context.compiled.statement
        if not isinstance(query, sqlalchemy.sql.Select):
            return
        if query.selected_columns.contains_column(results_db.catalog_runs.c.record) and query.whereclause is None:
            raise AssertionError("ingestion must not load the serving catalog")
        if query.selected_columns.contains_column(results_db.record_sources.c.record):
            materialized_sizes.append(len(parameters))

    sqlalchemy.event.listen(engine, "before_cursor_execute", inspect_catalog_queries)
    assert ingest.main() == 0
    with engine.begin() as conn:
        stored_ids = set(conn.execute(sqlalchemy.select(results_db.catalog_runs.c.run_id)).scalars())
    assert stored_ids == expected
    assert sum(materialized_sizes) == len(expected)
    assert max(materialized_sizes) < len(expected)
    assert results_db.prefix_statuses(engine)[0].error is None
