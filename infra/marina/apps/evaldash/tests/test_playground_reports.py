# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Playground reports: ingest from the serve dashboard's objects, listing, and detail."""

import json
from pathlib import Path

import sqlalchemy
from evaldash import app as evaldash_app
from evaldash import results_db
from evaldash.db_migrations import apply_migrations
from evaldash.playground_reports import DEFAULT_REPORT_PREFIXES, fetch_report_summaries, ingest_reports
from marin.inference.model_feedback import COREWEAVE_FEEDBACK_PREFIX, GCS_FEEDBACK_PREFIX, save_feedback_report
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
    (prefix / "2026-11-18" / "binary.json").write_bytes(b"\xff\xfe")

    assert ingest_reports(engine, (str(prefix),)) == ()
    assert ingest_reports(engine, (str(prefix),)) == ()

    with _client(engine, prefix) as client:
        listing = client.get("/playground-reports").json()
        detail = client.get("/playground-reports/2026-11-17-205725-aaaa").json()

    summaries = listing["reports"]
    assert listing["truncated"] is False
    assert [row["report_id"] for row in summaries] == ["2026-11-18-090000-bbbb", "2026-11-17-205725-aaaa"]
    assert summaries[1]["reporter"] == "Ada"
    assert summaries[1]["has_conversation"] is True
    assert "report" not in summaries[1]
    assert detail["report"]["conversation"] == conversation
    assert detail["serving"]["tensor_parallel_size"] == 4
    assert detail["source_url"].endswith("2026-11-17/205725-aaaa.json")


def test_local_store_reports_that_playground_reports_need_postgres():
    config = evaldash_app.EvaldashConfig.from_env({"EVALDASH_STORE": "local", "RECORDS_PREFIXES": ""})
    config = evaldash_app.EvaldashConfig(**{**config.__dict__, "prefixes": ()})
    api = evaldash_app.build_api(evaldash_app.MemoryRecordStore(), evaldash_app.NullClusterGateway(), config)

    with TestClient(api.app) as client:
        assert client.get("/playground-reports").status_code == 503


def test_unreadable_report_fails_its_prefix_without_blocking_other_reports(tmp_path):
    engine = _engine()
    first, second = tmp_path / "first", tmp_path / "second"
    _write_report(first, "2026-11-18", "090000-aaaa", "2026-11-18T09:00:00+00:00", description="Readable.")
    _write_report(first, "2026-11-18", "091500-bbbb", "2026-11-18T09:15:00+00:00", description="Unreadable.")
    _write_report(second, "2026-11-18", "100000-cccc", "2026-11-18T10:00:00+00:00", description="Other bucket.")
    unreadable = first / "2026-11-18" / "091500-bbbb.json"
    unreadable.chmod(0)
    try:
        failed = ingest_reports(engine, (str(first), str(second)))
    finally:
        unreadable.chmod(0o644)

    assert failed == (str(first),)
    with _client(engine, first) as client:
        ids = [row["report_id"] for row in client.get("/playground-reports").json()["reports"]]
    assert ids == ["2026-11-18-100000-cccc", "2026-11-18-090000-aaaa"]

    assert ingest_reports(engine, (str(first), str(second))) == ()
    with _client(engine, first) as client:
        assert len(client.get("/playground-reports").json()["reports"]) == 3


def test_summaries_say_when_older_reports_were_left_out(tmp_path):
    engine = _engine()
    prefix = tmp_path / "model-feedback"
    for minute in range(3):
        _write_report(
            prefix, "2026-11-18", f"09{minute:02d}00-aaaa", f"2026-11-18T09:{minute:02d}:00+00:00", description="x"
        )
    ingest_reports(engine, (str(prefix),))

    assert fetch_report_summaries(engine, limit=3)["truncated"] is False
    capped = fetch_report_summaries(engine, limit=2)
    assert capped["truncated"] is True
    assert [row["report_id"] for row in capped["reports"]] == ["2026-11-18-090200-aaaa", "2026-11-18-090100-aaaa"]


def test_reports_saved_by_the_serve_dashboard_appear_in_evaldash(tmp_path):
    """EvalDash keeps its own copy of the record format; this fails if the writer's format drifts."""
    assert set(DEFAULT_REPORT_PREFIXES) == {GCS_FEEDBACK_PREFIX, COREWEAVE_FEEDBACK_PREFIX}
    engine = _engine()
    prefix = tmp_path / "model-feedback"
    conversation = {"title": "Capital", "messages": [{"role": "user", "content": "Capital of Australia?"}]}
    report = {"description": "Said Sydney.", "reporter": "Ada", "model": "marin/demo", "conversation": conversation}
    serving = {"model": "marin/demo", "backend": "vllm", "tensor_parallel_size": 4}
    location = save_feedback_report(str(prefix), report, serving)

    assert ingest_reports(engine, (str(prefix),)) == ()

    with _client(engine, prefix) as client:
        [summary] = client.get("/playground-reports").json()["reports"]
        detail = client.get(f"/playground-reports/{summary['report_id']}").json()
    assert (summary["model"], summary["reporter"], summary["description"]) == ("marin/demo", "Ada", "Said Sydney.")
    assert summary["has_conversation"] is True
    assert detail["source_url"].endswith(Path(location).name)
    assert detail["report"]["conversation"] == conversation
    assert detail["serving"] == serving
