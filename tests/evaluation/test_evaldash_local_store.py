# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaldash local (Postgres-free) store and its API, exercised over generated fixtures.

The server runs with ``infra/evaldash/src`` on ``sys.path`` (the image layout), so its modules use
bare sibling imports; the tests put that directory on the path the same way to import ``server`` and
``fixtures``. These cover the ``MemoryRecordStore`` reads, the null-metric sample-reader fix, and the
HTTP surface a local dashboard serves with no database or cluster.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_EVALDASH_SRC = Path(__file__).resolve().parents[2] / "infra" / "evaldash" / "src"
if str(_EVALDASH_SRC) not in sys.path:
    sys.path.insert(0, str(_EVALDASH_SRC))

import fixtures  # noqa: E402
import samples  # noqa: E402
import server  # noqa: E402
from marin.evaluation.records import list_records  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


@pytest.fixture
def store(tmp_path) -> server.MemoryRecordStore:
    fixtures.build_fixtures(str(tmp_path))
    store = server.MemoryRecordStore()
    store.refresh(list_records(str(tmp_path)))
    return store


@pytest.fixture
def client(store) -> TestClient:
    # prefixes=() disables the background ingestor: the store is already populated from fixtures, and
    # this keeps the test hermetic (it never scans the remote gs://+s3:// defaults).
    app = server.create_app(store, server._dashboard_dist(), server.NullClusterGateway(), prefixes=())
    with TestClient(app) as client:
        yield client


def test_memory_store_matrix_uses_latest_version_cohort(store):
    matrix = store.matrix()
    qwen = next(row for row in matrix["rows"] if row["model"] == "qwen3-8b")
    # qwen3-8b has a 2026.07.19 and a 2026.07.21 launch; the matrix shows only the newer cohort.
    assert qwen["version"] == "2026.07.21"
    assert qwen["cells"]["mmlu"]["value"] == pytest.approx(0.719)


def test_leaderboard_reports_coverage_not_just_score(store):
    board = {entry["model"]: entry for entry in store.matrix()["leaderboard"]}
    # snowball ran all four headline suites; llama3-8b only mmlu. Coverage makes that visible even
    # though a narrow model can post a higher mean.
    assert board["snowball"]["covered"] == 4
    assert board["llama3-8b"]["covered"] == 1
    assert board["snowball"]["total"] == board["llama3-8b"]["total"]


def test_groups_roll_up_mixed_launch_status(store):
    groups = {group["group_id"]: group for group in store.groups()}
    # tootsie-8b's launch has a success, an eval failure, and an infra failure -> mixed.
    assert groups["tootsie-8b-2026.07.20"]["status"] == "mixed"
    assert groups["snowball-2026.07.20"]["status"] == "succeeded"
    assert groups["snowball-2026.07.20"]["n_succeeded"] == 4


def test_fetch_runs_filters_and_get_record_round_trip(store):
    tootsie = store.fetch_runs(model="tootsie-8b")
    assert {row["status"] for row in tootsie} == {"succeeded", "failed", "infra_failed"}
    record = store.get_record("snowball-2026.07.20-mmlu")
    assert record is not None
    # The record serializes the evaluation field under its ``eval`` wire alias.
    assert record["eval"]["name"] == "mmlu"


def test_ungraded_sample_does_not_count_as_incorrect_answer(store):
    # The snowball mmlu fixture includes one ungraded row (correct=None, no primary metric). It must
    # read back without a validation error and keep an empty metrics map rather than a null value.
    results_path = store.get_record("snowball-2026.07.20-mmlu")["results_path"]
    page = samples.fetch_samples(results_path, "mmlu", offset=0, limit=50, correct="all")
    ungraded = [row for row in page.rows if row.correct is None]
    assert len(ungraded) == 1
    assert ungraded[0].metrics == {}

    # The ungraded row is its own bucket: it is excluded from both correct and incorrect, so the
    # three counts partition the graded rows and `ungraded` accounts for the rest.
    assert page.counts.model_dump() == {"all": 5, "correct": 3, "incorrect": 1, "ungraded": 1}

    only_ungraded = samples.fetch_samples(results_path, "mmlu", offset=0, limit=50, correct="ungraded")
    assert [row.doc_id for row in only_ungraded.rows] == [ungraded[0].doc_id]

    incorrect = samples.fetch_samples(results_path, "mmlu", offset=0, limit=50, correct="incorrect")
    assert all(row.correct is False for row in incorrect.rows)


def test_api_surface_over_fixtures(client):
    assert client.get("/healthz").json()["store"] == "memory"

    meta = client.get("/api/meta").json()
    assert meta["store"] == "memory"
    assert "snowball" in meta["models"]

    matrix = client.get("/api/matrix").json()
    assert set(matrix["tasks"]) >= {"mmlu", "arc-challenge", "gsm8k-0shot"}

    runs = client.get("/api/runs?limit=100").json()
    assert len(runs) == 13

    detail = client.get("/api/runs/snowball-2026.07.20-mmlu").json()
    assert detail["status"] == "succeeded"
    # The detail endpoint attaches the rolled-up headline grade and the captured timing window.
    assert detail["headline"]["metric"] == "acc,none"
    assert detail["headline"]["value"] == pytest.approx(0.741)
    assert detail["timing"]["started_at"] and detail["timing"]["finished_at"]

    tasks = client.get("/api/runs/snowball-2026.07.20-mmlu/samples/tasks").json()
    assert tasks["available"] is True

    samples_page = client.get("/api/runs/snowball-2026.07.20-mmlu/samples", params={"task": "mmlu"}).json()
    assert samples_page["total"] == 5
    assert samples_page["primary_metric"] == "acc,none"


def test_run_detail_headline_is_null_for_a_failed_run(client):
    # The math500 run failed before producing metrics, so there is no grade to roll up; timing is
    # still present because the run had a wall-clock window before it failed.
    detail = client.get("/api/runs/tootsie-8b-2026.07.20-math500").json()
    assert detail["status"] == "failed"
    assert detail["headline"] is None
    assert detail["timing"]["finished_at"]


def test_api_agentic_artifact_is_run_local(client):
    rid = "snowball-2026.07.20-aime"
    page = client.get(f"/api/runs/{rid}/samples", params={"task": "aime"}).json()
    trajectory_uri = page["rows"][0]["trajectory_uri"]
    assert trajectory_uri
    artifact = client.get(f"/api/runs/{rid}/samples/artifact", params={"uri": trajectory_uri}).json()
    assert artifact["available"] is True
    assert artifact["media_type"] == "application/json"


def test_ingestor_surfaces_parse_failures(tmp_path):
    # The fixtures include one malformed record.json; an ingest pass keeps the good records and reports
    # the bad one on its prefix probe rather than dropping it silently -- what the Debug view shows.
    fixtures.build_fixtures(str(tmp_path))
    store = server.MemoryRecordStore()
    ingestor = server.Ingestor(store, (str(tmp_path),), interval=999)

    asyncio.run(ingestor.run_once())

    probe = ingestor.status()["prefixes"][0]
    assert probe["record_count"] == 13
    assert probe["error"] is None
    assert len(probe["parse_failures"]) == 1
    assert probe["parse_failures"][0]["path"].endswith("20260722-000000-legacy-mmlu-broken/record.json")
    assert "eval_runtime" in probe["parse_failures"][0]["error"]


def test_api_jobs_degrade_without_a_cluster(client):
    # The local NullClusterGateway reports every role unreachable rather than reaching Iris.
    jobs = client.get("/api/runs/snowball-2026.07.20-mmlu/jobs").json()
    assert jobs["roles"]
    assert all(role["reachable"] is False for role in jobs["roles"])
