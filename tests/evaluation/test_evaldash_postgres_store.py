# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of EvalDash's durable serving catalog and background reconciliation."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
import sqlalchemy
from marin.evaluation.records import EvalRunRecord, list_records, write_record
from sqlalchemy.pool import StaticPool

from infra.evaldash.src import fixtures, results_db, server


def _engine() -> sqlalchemy.Engine:
    return sqlalchemy.create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def _record(tmp_path) -> EvalRunRecord:
    source = tmp_path / "fixtures"
    fixtures.build_fixtures(str(source))
    return list_records(str(source))[0]


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 16, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _store(engine: sqlalchemy.Engine) -> server.PgRecordStore:
    return server.PgRecordStore(engine, instance="test", database="evals")


def test_migrations_adopt_existing_serving_rows_and_boot_without_object_storage(tmp_path):
    engine = _engine()
    for table in (results_db.eval_runs, results_db.eval_metrics, results_db.model_state):
        table.create(engine)
    record = _record(tmp_path).model_copy(update={"description": "seeded before migration"})
    results_db.upsert_legacy_record(engine, record)

    results_db.ensure_schema(engine)
    results_db.ensure_schema(engine)
    store = _store(engine)

    assert store.get_record(record.run_id)["run_id"] == record.run_id
    assert store.store_info().record_count == 1
    results_db.upsert_legacy_record(engine, record.model_copy(update={"description": "old revision write"}))
    assert _store(engine).get_record(record.run_id)["description"] == "seeded before migration"
    with engine.begin() as conn:
        migrations = set(conn.execute(sqlalchemy.text("SELECT name FROM schema_migrations")).scalars())
    assert migrations == {"0001_initial", "0002_record_sources"}


def test_older_binary_rejects_a_database_with_unknown_migrations():
    engine = _engine()
    results_db.ensure_schema(engine)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("INSERT INTO schema_migrations (name) VALUES ('9999_future')"))

    with pytest.raises(RuntimeError, match="9999_future"):
        results_db.ensure_schema(engine)


def test_reconciler_discovers_new_paths_and_only_rereads_known_paths_when_due(tmp_path):
    engine = _engine()
    results_db.ensure_schema(engine)
    clock = Clock()
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    write_record(record, str(prefix))
    store = _store(engine)
    ingestor = server.PostgresIngestor(store, (str(prefix),), 600, 86400, now=clock)

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

    assert store.get_record(record.run_id)["description"] == initial["description"]
    assert store.get_record(added.run_id)["description"] == "new"

    clock.now += timedelta(seconds=2)
    asyncio.run(ingestor.run_once())
    assert store.get_record(record.run_id)["description"] == "rewritten"


def test_reconciler_promotes_duplicate_only_after_two_successful_absences(tmp_path):
    engine = _engine()
    results_db.ensure_schema(engine)
    clock = Clock()
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    record = _record(tmp_path)
    write_record(record.model_copy(update={"description": "canonical"}), str(canonical))
    write_record(record.model_copy(update={"description": "legacy"}), str(legacy))
    store = _store(engine)
    ingestor = server.PostgresIngestor(store, (str(canonical), str(legacy)), 600, 86400, now=clock)

    asyncio.run(ingestor.run_once())
    assert store.get_record(record.run_id)["description"] == "canonical"

    canonical_state = next(iter(results_db.source_states(engine, str(canonical)).values()))
    (canonical / record.run_id / "record.json").unlink()
    clock.now = canonical_state.next_verify_at.replace(tzinfo=UTC) + timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert store.get_record(record.run_id)["description"] == "canonical"

    asyncio.run(ingestor.run_once())
    assert store.get_record(record.run_id)["description"] == "canonical"

    clock.now += timedelta(seconds=600)
    asyncio.run(ingestor.run_once())
    assert not results_db.source_states(engine, str(canonical))
    assert results_db.source_states(engine, str(legacy))
    assert store.get_record(record.run_id)["description"] == "legacy"


def test_reconciler_reappearance_resets_missing_confirmation(tmp_path):
    engine = _engine()
    results_db.ensure_schema(engine)
    clock = Clock()
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    write_record(record, str(prefix))
    store = _store(engine)
    ingestor = server.PostgresIngestor(store, (str(prefix),), 600, 86400, now=clock)
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
    results_db.ensure_schema(engine)
    write_record(record.model_copy(update={"description": "legacy"}), str(legacy))
    store = _store(engine)
    real_list_record_paths = server.list_record_paths

    def list_unless_canonical(prefix: str) -> list[str]:
        if prefix == str(canonical):
            raise OSError("canonical unavailable")
        return real_list_record_paths(prefix)

    monkeypatch.setattr(server, "list_record_paths", list_unless_canonical)
    ingestor = server.PostgresIngestor(store, (str(canonical), str(legacy)), 600, 86400, now=Clock())
    asyncio.run(ingestor.run_once())

    assert store.get_record(record.run_id)["description"] == "migrated canonical"


def test_stale_failure_cannot_overwrite_a_later_success():
    engine = _engine()
    results_db.ensure_schema(engine)
    prefix = "file:///records"
    results_db.configure_prefixes(engine, (prefix,))
    probe_at = datetime(2026, 8, 16, tzinfo=UTC)
    results_db.reconcile_prefix(engine, prefix, [], [], probe_at + timedelta(seconds=1), 600)

    results_db.mark_prefix_failed(engine, prefix, probe_at, "stale failure")

    status = results_db.prefix_statuses(engine)[0]
    assert status["last_probe_at"].replace(tzinfo=UTC) == probe_at + timedelta(seconds=1)
    assert status["error"] is None


def test_reconciler_keeps_last_valid_row_when_a_prefix_or_rewritten_record_fails(tmp_path, monkeypatch):
    engine = _engine()
    results_db.ensure_schema(engine)
    clock = Clock()
    prefix = tmp_path / "records"
    record = _record(tmp_path)
    write_record(record, str(prefix))
    store = _store(engine)
    ingestor = server.PostgresIngestor(store, (str(prefix),), 600, 86400, now=clock)
    asyncio.run(ingestor.run_once())

    state = next(iter(results_db.source_states(engine, str(prefix)).values()))
    (prefix / record.run_id / "record.json").write_text("not json", encoding="utf-8")
    clock.now = state.next_verify_at.replace(tzinfo=UTC) + timedelta(seconds=1)
    asyncio.run(ingestor.run_once())
    assert store.get_record(record.run_id)["run_id"] == record.run_id
    assert ingestor.status()["prefixes"][0]["parse_failures"]

    def fail_listing(_prefix: str) -> list[str]:
        raise OSError("object store unavailable")

    monkeypatch.setattr(server, "list_record_paths", fail_listing)
    clock.now += timedelta(seconds=600)
    asyncio.run(ingestor.run_once())

    assert store.get_record(record.run_id)["run_id"] == record.run_id
    assert "object store unavailable" in ingestor.status()["prefixes"][0]["error"]


def test_reconciler_preserves_seeded_row_when_first_object_read_is_invalid(tmp_path):
    engine = _engine()
    for table in (results_db.eval_runs, results_db.eval_metrics, results_db.model_state):
        table.create(engine)
    record = _record(tmp_path).model_copy(update={"description": "last valid"})
    results_db.upsert_legacy_record(engine, record)
    results_db.ensure_schema(engine)
    prefix = tmp_path / "records"
    path = prefix / record.run_id / "record.json"
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    store = _store(engine)
    ingestor = server.PostgresIngestor(store, (str(prefix),), 600, 86400, now=Clock())

    asyncio.run(ingestor.run_once())

    assert store.get_record(record.run_id)["description"] == "last valid"
    assert ingestor.status()["prefixes"][0]["parse_failures"]


def test_configured_prefix_order_and_membership_select_retained_sources(tmp_path):
    engine = _engine()
    results_db.ensure_schema(engine)
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    record = _record(tmp_path)
    write_record(record.model_copy(update={"description": "canonical"}), str(canonical))
    write_record(record.model_copy(update={"description": "legacy"}), str(legacy))
    store = _store(engine)
    ingestor = server.PostgresIngestor(store, (str(canonical), str(legacy)), 600, 86400, now=Clock())
    asyncio.run(ingestor.run_once())
    assert store.get_record(record.run_id)["description"] == "canonical"

    server.PostgresIngestor(store, (str(legacy), str(canonical)), 600, 86400, now=Clock())
    assert store.get_record(record.run_id)["description"] == "legacy"

    server.PostgresIngestor(store, (str(canonical),), 600, 86400, now=Clock())
    assert store.get_record(record.run_id)["description"] == "canonical"
