# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Playground reports: ingest from the serve dashboard's objects, listing, and triage."""

import json
from pathlib import Path

import sqlalchemy
from evaldash import app as evaldash_app
from evaldash import results_db
from evaldash.db_migrations import apply_migrations
from evaldash.playground_reports import ingest_reports
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient


def _engine() -> sqlalchemy.Engine:
    engine = sqlalchemy.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as connection:
        apply_migrations(connection)
    results_db.verify_schema(engine)
    return engine


def _write_report(prefix: Path, day: str, report_id: str, created_at: str, **report) -> None:
    record = {
        "version": 1,
        "id": report_id,
        "created_at": created_at,
        "serving": {"model": "marin/demo", "backend": "vllm", "tensor_parallel_size": 4},
        "report": {"model": "marin/demo", "reporter": "", "conversation": None, **report},
    }
    (prefix / day).mkdir(parents=True, exist_ok=True)
    (prefix / day / f"{report_id}.json").write_text(json.dumps(record))


def _client(engine: sqlalchemy.Engine, prefix: Path) -> TestClient:
    config = evaldash_app.EvaldashConfig.from_env({"PLAYGROUND_REPORT_PREFIXES": str(prefix)})
    config = evaldash_app.EvaldashConfig(**{**config.__dict__, "prefixes": ()})
    store = evaldash_app.PgRecordStore(engine)
    return TestClient(evaldash_app.build_api(store, evaldash_app.NullClusterGateway(), config).app)


def test_ingest_copies_new_reports_once_and_skips_malformed_objects(tmp_path):
    engine = _engine()
    prefix = tmp_path / "model-feedback"
    conversation = {"title": "Capital", "messages": [{"role": "user", "content": "Capital of Australia?"}]}
    _write_report(
        prefix,
        "2026-11-17",
        "205725-aaaa",
        "2026-11-17T20:57:25+00:00",
        description="Said Sydney.",
        reporter="Ada",
        conversation=conversation,
    )
    _write_report(prefix, "2026-11-18", "090000-bbbb", "2026-11-18T09:00:00+00:00", description="Too verbose.")
    (prefix / "2026-11-18" / "broken.json").write_text("{not json")

    assert ingest_reports(engine, (str(prefix),)) == ()
    assert ingest_reports(engine, (str(prefix),)) == ()

    with _client(engine, prefix) as client:
        summaries = client.get("/playground-reports").json()
        detail = client.get("/playground-reports/2026-11-17-205725-aaaa").json()

    assert [(row["report_id"], row["status"]) for row in summaries] == [
        ("2026-11-18-090000-bbbb", "new"),
        ("2026-11-17-205725-aaaa", "new"),
    ]
    assert summaries[1]["reporter"] == "Ada"
    assert summaries[1]["has_conversation"] is True
    assert "report" not in summaries[1]
    assert detail["report"]["conversation"] == conversation
    assert detail["serving"]["tensor_parallel_size"] == 4
    assert detail["source_url"].endswith("2026-11-17/205725-aaaa.json")


def test_triage_survives_reingest_and_rejects_unknown_reports_and_statuses(tmp_path):
    engine = _engine()
    prefix = tmp_path / "model-feedback"
    _write_report(prefix, "2026-11-18", "090000-bbbb", "2026-11-18T09:00:00+00:00", description="Too verbose.")
    ingest_reports(engine, (str(prefix),))

    with _client(engine, prefix) as client:
        updated = client.post(
            "/playground-reports/2026-11-18-090000-bbbb/triage",
            json={"status": "filed", "notes": "Same as the verbosity regression.", "github_issue": " #12345 "},
        )
        ingest_reports(engine, (str(prefix),))
        after_reingest = client.get("/playground-reports").json()[0]
        unknown = client.post("/playground-reports/missing/triage", json={"status": "triaged"})
        bad_status = client.post("/playground-reports/2026-11-18-090000-bbbb/triage", json={"status": "done"})

    assert updated.status_code == 200
    assert updated.json()["notes"] == "Same as the verbosity regression."
    assert (after_reingest["status"], after_reingest["github_issue"]) == ("filed", "#12345")
    assert unknown.status_code == 404
    assert bad_status.status_code == 422


def test_local_store_reports_that_playground_reports_need_postgres():
    config = evaldash_app.EvaldashConfig.from_env({"EVALDASH_STORE": "local", "RECORDS_PREFIXES": ""})
    config = evaldash_app.EvaldashConfig(**{**config.__dict__, "prefixes": ()})
    api = evaldash_app.build_api(evaldash_app.MemoryRecordStore(), evaldash_app.NullClusterGateway(), config)

    with TestClient(api.app) as client:
        assert client.get("/playground-reports").status_code == 503
