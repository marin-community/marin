# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EvalDash over the kernel's contract: the engine it is handed, on real Postgres.

The rest of the suite exercises the stores over SQLite. This walks the path the deploy takes --
``marina migrate`` applies the schema, the kernel builds the app on an engine scoped to it, and the
mounted API answers from the committed catalog -- so the Postgres dialect and the app's own schema
are covered rather than assumed.
"""

import asyncio

import pytest
from evaldash import app as evaldash_app
from evaldash import fixtures, results_db
from marin.evaluation.records import write_record
from marina.apps import Services
from marina.db import UrlDatabase, engine_for
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


def test_an_unmigrated_schema_is_named_rather_than_failing_on_the_first_query(engine):
    with pytest.raises(RuntimeError, match="marina migrate"):
        results_db.verify_schema(engine)


def test_the_mounted_api_serves_what_the_reconciler_committed(engine, records, database_url, monkeypatch):
    evaldash_app.migrate(engine)
    monkeypatch.setenv("RECORDS_PREFIXES", records)
    monkeypatch.delenv("EVALDASH_STORE", raising=False)
    monkeypatch.setenv("EVALDASH_INGEST_JOB", "projects/hai-gcp-models/locations/us-central1/jobs/marina-evaldash")
    triggered = []
    monkeypatch.setattr(
        evaldash_app,
        "trigger_cloud_run_job",
        lambda job: triggered.append(job) or "operations/evaldash-ingest",
    )
    services = Services(name=APP, data_url="memory://evaldash", database=UrlDatabase(url=database_url))

    api = evaldash_app.create_api(services)
    with TestClient(api) as client:
        assert client.get("/runs").json() == []
        config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
        writer = evaldash_app.PgRecordStore(engine)
        ingestor = evaldash_app.PostgresIngestor(
            writer, config.prefixes, config.ingest_interval, config.revalidate_after
        )
        assert asyncio.run(ingestor.run_once()) == ()
        response = client.post("/refresh")
        assert response.status_code == 202
        assert response.json()["store"]["backend"] == "postgres"
        assert triggered == ["projects/hai-gcp-models/locations/us-central1/jobs/marina-evaldash"]

        runs = client.get("/runs?limit=100").json()
        assert len(runs) == 15
        assert client.get("/runs/snowball-2026.07.20-mmlu").json()["status"] == "succeeded"
        assert client.get("/status").json()["store"]["catalog_generation"] > 0


def test_a_second_instance_serves_the_generation_the_first_committed(engine, records):
    evaldash_app.migrate(engine)
    config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
    writer = evaldash_app.PgRecordStore(engine)
    now = [0.0]
    reader = evaldash_app.PgRecordStore(engine, now=lambda: now[0])
    ingestor = evaldash_app.PostgresIngestor(writer, config.prefixes, config.ingest_interval, config.revalidate_after)
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
    evaldash_app.migrate(engine)
    config = evaldash_app.EvaldashConfig.from_env({"RECORDS_PREFIXES": records})
    writer = evaldash_app.PgRecordStore(engine)
    asyncio.run(
        evaldash_app.PostgresIngestor(
            writer, config.prefixes, config.ingest_interval, config.revalidate_after
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
