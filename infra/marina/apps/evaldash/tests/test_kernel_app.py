# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EvalDash over the kernel's contract: the engine it is handed, on real Postgres.

The rest of the suite exercises the stores over SQLite. This walks the path the deploy takes --
``marina migrate`` applies the schema, the kernel builds the app on an engine scoped to it, and the
mounted API answers from the committed catalog -- so the Postgres dialect and the app's own schema
are covered rather than assumed.
"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Event

import httpx
import pytest
import sqlalchemy
from evaldash import app as evaldash_app
from evaldash import fixtures, results_db
from marin.evaluation.records import list_records, write_record
from marina.db import UrlDatabase, engine_for
from marina.testing import SourceServices
from starlette.testclient import TestClient

APP = "evaldash"


@pytest.fixture
def engine(database_url):
    engine = engine_for(UrlDatabase(url=database_url), APP)
    with engine.begin() as conn:
        for table in reversed(results_db.metadata.sorted_tables):
            conn.exec_driver_sql(f'DROP TABLE IF EXISTS "{table.name}" CASCADE')
        conn.exec_driver_sql("DROP TABLE IF EXISTS schema_migrations")
    yield engine
    engine.dispose()


@pytest.fixture
def records(tmp_path) -> str:
    prefix = tmp_path / "records"
    fixtures.build_fixtures(str(prefix))
    return str(prefix)


@pytest.fixture
def catalog(engine, records):
    with engine.begin() as conn:
        evaldash_app.migrate(conn)
    config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
    asyncio.run(
        evaldash_app.PostgresIngestor(
            engine, config.prefixes, config.ingest_interval, config.revalidate_after
        ).run_once()
    )
    return results_db.fetch_snapshot(engine)


def test_snapshot_reads_committed_generation_while_writer_is_uncommitted(engine, catalog):
    changed = catalog.records[0].model_copy(update={"description": "uncommitted rewrite"})

    def bound_lock_wait(conn):
        conn.exec_driver_sql("SET LOCAL lock_timeout = '1s'")

    with engine.begin() as writer:
        writer.execute(results_db.catalog_state.update().values(generation=catalog.generation + 1))
        writer.execute(
            results_db.catalog_runs.update()
            .where(results_db.catalog_runs.c.run_id == changed.run_id)
            .values(record=changed.model_dump(mode="json", by_alias=True))
        )
        sqlalchemy.event.listen(engine, "begin", bound_lock_wait)
        try:
            snapshot = results_db.fetch_snapshot(engine)
        finally:
            sqlalchemy.event.remove(engine, "begin", bound_lock_wait)
        assert snapshot == catalog

    committed = results_db.fetch_snapshot(engine)
    assert committed.generation == catalog.generation + 1
    assert next(record for record in committed.records if record.run_id == changed.run_id) == changed


def test_snapshot_keeps_generation_and_records_consistent_when_writer_commits_between_reads(engine, catalog):
    changed = catalog.records[0].model_copy(update={"description": "committed between snapshot reads"})

    def commit_after_generation_read(_conn, _cursor, _statement, _parameters, context, _many):
        if context.compiled is None:
            return
        query = context.compiled.statement
        if not isinstance(query, sqlalchemy.sql.Select):
            return
        if not query.selected_columns.contains_column(results_db.catalog_state.c.updated_at):
            return
        # The real writer must commit while the reader still holds its transaction open.
        with engine.begin() as writer:
            writer.exec_driver_sql("SET LOCAL lock_timeout = '1s'")
            writer.execute(
                results_db.catalog_runs.update()
                .where(results_db.catalog_runs.c.run_id == changed.run_id)
                .values(record=changed.model_dump(mode="json", by_alias=True))
            )
            writer.execute(results_db.catalog_state.update().values(generation=catalog.generation + 1))

    sqlalchemy.event.listen(engine, "after_cursor_execute", commit_after_generation_read)
    try:
        snapshot = results_db.fetch_snapshot(engine)
    finally:
        sqlalchemy.event.remove(engine, "after_cursor_execute", commit_after_generation_read)

    assert snapshot == catalog
    committed = results_db.fetch_snapshot(engine)
    assert committed.generation == catalog.generation + 1
    assert next(record for record in committed.records if record.run_id == changed.run_id) == changed


def test_metadata_database_wait_does_not_block_other_requests(engine, catalog):
    store = evaldash_app.PgRecordStore(engine)
    config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": " "})
    api = evaldash_app.build_api(store, evaldash_app.NullClusterGateway(), config)
    metadata_query = Event()

    def bound_metadata_wait(_conn, cursor, _statement, _parameters, context, _many):
        if context.compiled is None:
            return
        query = context.compiled.statement
        if isinstance(query, sqlalchemy.sql.Select) and query.selected_columns.contains_column(
            results_db.model_state.c.model_name
        ):
            cursor.execute("SET LOCAL statement_timeout = '3s'")
            metadata_query.set()

    async def request_while_metadata_waits():
        transport = httpx.ASGITransport(app=api.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            with engine.begin() as writer:
                writer.exec_driver_sql("LOCK TABLE model_state IN ACCESS EXCLUSIVE MODE")
                metadata = asyncio.create_task(client.get("/meta"))
                try:
                    assert await asyncio.to_thread(metadata_query.wait, 5)
                    runs = await client.get("/runs?limit=1")
                    assert runs.status_code == 200
                    assert runs.json()[0]["run_id"] in {record.run_id for record in catalog.records}
                    assert not metadata.done()
                finally:
                    writer.rollback()
                    response = await metadata
                assert response.status_code == 200
                assert "run_facets" in response.json()

    sqlalchemy.event.listen(engine, "before_cursor_execute", bound_metadata_wait)
    try:
        asyncio.run(request_while_metadata_waits())
    finally:
        sqlalchemy.event.remove(engine, "before_cursor_execute", bound_metadata_wait)


def test_metadata_refreshes_with_catalog_and_archive_updates_without_a_new_generation(engine, catalog, records):
    store = evaldash_app.PgRecordStore(engine)
    assert "added-model" not in store.meta()["models"]
    added = catalog.records[0].model_copy(
        update={
            "run_id": "metadata-added-run",
            "model": catalog.records[0].model.model_copy(update={"name": "added-model"}),
            "user": "added-user",
            "version": "added-cohort",
        }
    )
    write_record(added, records)
    config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
    asyncio.run(
        evaldash_app.PostgresIngestor(
            engine, config.prefixes, config.ingest_interval, config.revalidate_after
        ).run_once()
    )
    store.reload_if_changed()
    updated = store.meta()
    assert "added-model" in updated["models"]
    assert "added-user" in updated["facets"]["user"]
    assert "added-cohort" in updated["run_facets"]["version"]
    generation = store.store_info().catalog_generation
    assert generation > catalog.generation

    store.set_model_archived("added-model", True, "operator")
    assert store.meta()["archived_models"] == ["added-model"]
    assert store.store_info().catalog_generation == generation
    store.set_model_archived("added-model", False, "operator")
    assert store.meta()["archived_models"] == []
    assert store.store_info().catalog_generation == generation


def test_reconciliation_batches_verification_writes_and_preserves_last_valid_records(engine, records):
    with engine.begin() as conn:
        evaldash_app.migrate(conn)
    record = list_records(records)[0]
    prefix = "file:///batch-records"
    results_db.configure_prefixes(engine, (prefix,))
    now = datetime(2026, 10, 20, tzinfo=UTC)
    initial = [
        results_db.RecordObservation(
            path=f"{prefix}/{index}/record.json",
            object_version=f"version-{index}",
            verified_at=now,
            next_verify_at=now + timedelta(days=1),
            kind=results_db.ObservationKind.CHANGED,
            record=record.model_copy(update={"run_id": f"batch-{index}"}),
        )
        for index in range(259)
    ]
    paths = [observation.path for observation in initial]
    results_db.reconcile_prefix(engine, prefix, paths, initial, now, 600)
    generation = results_db.catalog_generation(engine)
    probe_at = now + timedelta(days=1)
    observations = [
        replace(
            observation,
            kind=results_db.ObservationKind.UNCHANGED,
            record=None,
            verified_at=probe_at,
            next_verify_at=probe_at + timedelta(days=1),
        )
        for observation in initial
    ]
    observations[0] = replace(observations[0], kind=results_db.ObservationKind.MISSING, error="not found")
    observations[1] = replace(
        observations[1], kind=results_db.ObservationKind.CHANGED, object_version="invalid-version", error="invalid JSON"
    )
    invalid_path = f"{prefix}/new-invalid/record.json"
    observations.append(
        replace(observations[1], path=invalid_path, run_id="invalid-new", object_version=None, error="new invalid JSON")
    )
    updates = []

    def count_source_updates(_conn, _cursor, _statement, parameters, context, many):
        if context.compiled is None:
            return
        query = context.compiled.statement
        if isinstance(query, sqlalchemy.sql.dml.Update) and query.table.name == results_db.record_sources.name:
            updates.append(len(parameters) if many else 1)

    sqlalchemy.event.listen(engine, "before_cursor_execute", count_source_updates)
    try:
        results_db.reconcile_prefix(engine, prefix, [*paths, invalid_path], observations, probe_at, 600)
    finally:
        sqlalchemy.event.remove(engine, "before_cursor_execute", count_source_updates)

    states = results_db.source_states(engine, prefix)
    for observation in observations:
        state = states[observation.path]
        assert state.last_verified_at == probe_at
        assert state.next_verify_at == probe_at + timedelta(days=1)
        assert state.object_version == observation.object_version
        assert state.error == observation.error
    assert states[initial[0].path].missing_since == probe_at
    assert states[initial[1].path].missing_since is None
    assert states[initial[1].path].run_id == "batch-1"
    snapshot = results_db.fetch_snapshot(engine)
    assert snapshot.generation == generation
    assert {row.run_id for row in snapshot.records} == {f"batch-{index}" for index in range(259)}
    assert next(row for row in snapshot.records if row.run_id == "batch-1") == initial[1].record
    # Count statements at the database boundary: per-row executemany would exceed 250 writes.
    assert sum(updates) < 10


def test_an_unmigrated_schema_is_named_rather_than_failing_on_the_first_query(engine):
    with pytest.raises(RuntimeError, match="marina migrate"):
        results_db.verify_schema(engine)


def test_the_mounted_api_serves_what_the_reconciler_committed(engine, records, database_url, monkeypatch):
    with engine.begin() as migration_connection:
        evaldash_app.migrate(migration_connection)
    monkeypatch.setenv("RECORDS_PREFIXES", records)
    monkeypatch.delenv("EVALDASH_STORE", raising=False)
    monkeypatch.setenv("EVALDASH_INGEST_JOB", "projects/hai-gcp-models/locations/us-central1/jobs/marina-evaldash")
    triggered = []
    monkeypatch.setattr(
        evaldash_app,
        "trigger_cloud_run_job",
        lambda job: triggered.append(job) or "operations/evaldash-ingest",
    )
    services = SourceServices(name=APP, data_url="memory://evaldash", database=UrlDatabase(url=database_url))

    api = evaldash_app.create_api(services)
    with TestClient(api.app) as client:
        assert client.get("/runs").json() == []
        config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
        ingestor = evaldash_app.PostgresIngestor(
            engine, config.prefixes, config.ingest_interval, config.revalidate_after
        )
        assert asyncio.run(ingestor.run_once()) == ()
        response = client.post("/refresh")
        assert response.status_code == 202
        assert response.json()["store"]["backend"] == "postgres"
        assert triggered == ["projects/hai-gcp-models/locations/us-central1/jobs/marina-evaldash"]

        runs = client.get("/runs?limit=100").json()
        assert len(runs) == 21
        assert client.get("/runs/snowball-2026.07.20-mmlu").json()["status"] == "succeeded"
        assert client.get("/status").json()["store"]["catalog_generation"] > 0

        snowball = next(
            row for row in client.get("/panel", params={"cohort": "all"}).json()["rows"] if row["model"] == "snowball"
        )
        assert snowball["last_updated"] == max(cell["created_at"] for cell in snowball["cells"].values())


def test_a_second_instance_serves_the_generation_the_first_committed(engine, records):
    with engine.begin() as migration_connection:
        evaldash_app.migrate(migration_connection)
    config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
    writer = evaldash_app.PgRecordStore(engine)
    now = [0.0]
    reader = evaldash_app.PgRecordStore(engine, now=lambda: now[0])
    ingestor = evaldash_app.PostgresIngestor(engine, config.prefixes, config.ingest_interval, config.revalidate_after)
    assert reader.store_info().record_count == 0

    asyncio.run(ingestor.run_once())

    # The reader keeps its snapshot until the short generation-check interval expires.
    assert reader.store_info().record_count == 0
    now[0] = evaldash_app.CATALOG_CHECK_INTERVAL
    assert reader.get_record("snowball-2026.07.20-mmlu") is not None
    assert reader.store_info().record_count == writer.store_info().record_count

    added = writer.get_record("snowball-2026.07.20-mmlu")
    write_record(
        results_db.EvalRunRecord.model_validate(added).model_copy(update={"run_id": "added-run"}),
        records,
    )
    asyncio.run(ingestor.run_once())
    now[0] += evaldash_app.CATALOG_CHECK_INTERVAL
    assert reader.get_record("added-run") is not None


def test_catalog_check_failure_serves_cached_snapshot_until_retry(engine, records, monkeypatch):
    with engine.begin() as migration_connection:
        evaldash_app.migrate(migration_connection)
    config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
    asyncio.run(
        evaldash_app.PostgresIngestor(
            engine, config.prefixes, config.ingest_interval, config.revalidate_after
        ).run_once()
    )
    now = [0.0]
    reader = evaldash_app.PgRecordStore(engine, now=lambda: now[0])
    cached = reader.get_record("snowball-2026.07.20-mmlu")
    real_catalog_generation = evaldash_app.catalog_generation
    monkeypatch.setattr(
        evaldash_app,
        "catalog_generation",
        lambda _engine: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    now[0] += evaldash_app.CATALOG_CHECK_INTERVAL
    assert reader.get_record("snowball-2026.07.20-mmlu") == cached
    assert reader.store_info().catalog_error is not None

    monkeypatch.setattr(evaldash_app, "catalog_generation", real_catalog_generation)
    now[0] += evaldash_app.CATALOG_CHECK_INTERVAL
    assert reader.get_record("snowball-2026.07.20-mmlu") == cached
    assert reader.store_info().catalog_error is None
