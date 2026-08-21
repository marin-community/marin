# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaldash local (Postgres-free) store and its API, exercised over generated fixtures.

These cover the ``MemoryRecordStore`` reads, the null-metric sample-reader fix, and the HTTP surface
a local dashboard serves with no database or cluster.
"""

import asyncio

import pytest
from marin.evaluation.records import list_records, write_record
from starlette.testclient import TestClient

from infra.evaldash.src import fixtures, metrics, samples, server


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


def _panel(store, **kwargs):
    return store.panel(metrics.panel_request(**kwargs), None, False)


def test_memory_store_panel_takes_each_benchmark_from_its_newest_run(store):
    panel = _panel(store)
    qwen = next(row for row in panel["rows"] if row["model"] == "qwen3-8b")
    # qwen3-8b has a 2026.07.19 and a 2026.07.21 launch; mmlu comes from the newer one.
    assert qwen["cells"]["mmlu"]["version"] == "2026.07.21"
    assert qwen["cells"]["mmlu"]["value"] == pytest.approx(0.719)


def test_panel_reports_coverage_of_the_selected_benchmarks(store):
    rows = {row["model"]: row for row in _panel(store)["rows"]}
    # snowball ran every headline suite; llama3-8b only mmlu. Coverage makes that visible, and no
    # cross-benchmark mean is offered to paper over the difference.
    assert rows["snowball"]["covered"] == 5
    assert rows["llama3-8b"]["covered"] == 1
    assert rows["snowball"]["aggregate"] is None


def test_panel_keeps_only_models_with_the_full_selection_when_asked(store):
    complete = _panel(store, benchmarks=("mmlu", "gsm8k-0shot"), completeness=metrics.Completeness.COMPLETE_PANEL)

    models = {row["model"] for row in complete["rows"]}
    assert "snowball" in models
    assert "llama3-8b" not in models


def test_agentic_cell_carries_its_coverage_and_a_wider_interval(store):
    """The aime fixture lost one of ten trials to a timeout: its interval covers the ungraded trial
    rather than assuming it would have gone either way."""
    row = next(row for row in _panel(store)["rows"] if row["model"] == "snowball")

    aime = row["cells"]["aime"]
    assert aime["interval_kind"] == "identified"
    assert aime["coverage"] == pytest.approx(0.9)
    assert aime["errors"] == {"AgentTimeoutError": 1}
    assert aime["high"] - aime["low"] >= 0.1


def test_groups_roll_up_mixed_launch_status(store):
    groups = {group["group_id"]: group for group in store.groups()}
    # tootsie-8b's launch has a success, an eval failure, and an infra failure -> mixed.
    assert groups["tootsie-8b-2026.07.20"]["status"] == "mixed"
    assert groups["snowball-2026.07.20"]["status"] == "succeeded"
    assert groups["snowball-2026.07.20"]["n_succeeded"] == 5


def test_status_rollup_does_not_invent_evaluator_failure():
    assert server._status_rollup({"artifact_failed", "infra_failed"}) == "mixed"
    assert server._status_rollup({"failed", "artifact_failed", "infra_failed"}) == "failed"


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

    panel = client.get("/api/panel").json()
    assert set(panel["benchmarks"]) >= {"mmlu", "arc-challenge", "gsm8k-0shot"}
    assert panel["request"]["min_coverage"] == pytest.approx(0.9)

    runs = client.get("/api/runs?limit=100").json()
    assert len(runs) == 15
    # Rows carry version (from the record jsonb) so the client can facet on it.
    assert any(row["version"] == "2026.07.20" for row in runs)
    assert {row["version"] for row in runs} >= {"2026.07.19", "2026.07.20", "2026.07.21"}

    detail = client.get("/api/runs/snowball-2026.07.20-mmlu").json()
    assert detail["status"] == "succeeded"
    # The detail endpoint attaches the rolled-up headline grade and the captured timing window.
    assert detail["headline"]["metric"] == "acc,none"
    assert detail["headline"]["value"] == pytest.approx(0.741)
    assert detail["timing"]["started_at"] and detail["timing"]["finished_at"]
    # Serving params round-trip through the record and reach the detail response.
    assert detail["serving"]["tensor_parallel_size"] == 8
    assert detail["serving"]["extra"]["temperature"] == "0.0"

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


def test_api_compare_reports_shared_benchmarks_and_their_difference_intervals(client):
    comparison = client.get("/api/compare", params={"models": "snowball,qwen3-8b"}).json()

    assert set(comparison["shared"]) >= {"mmlu", "arc-challenge"}
    mmlu = next(row for row in comparison["rows"] if row["benchmark"] == "mmlu")
    assert mmlu["leader"] in ("snowball", "qwen3-8b")
    # The leader is not compared with itself; every other model gets an interval for its gap.
    assert set(mmlu["differences"]) == set(mmlu["cells"]) - {mmlu["leader"]}
    (gap,) = mmlu["differences"].values()
    assert gap["low"] <= gap["high"]


def test_api_compare_applies_the_selection_it_is_given(client):
    comparison = client.get("/api/compare", params={"models": "snowball,qwen3-8b", "benchmarks": "mmlu"}).json()

    assert comparison["benchmarks"] == ["mmlu"]
    assert comparison["shared"] == ["mmlu"]


def test_api_compare_rejects_a_request_it_cannot_answer(client):
    assert client.get("/api/compare", params={"models": "snowball"}).status_code == 400
    assert client.get("/api/compare", params={"models": "a,b,c,d,e"}).status_code == 400


def test_api_panel_rejects_an_unusable_query_rather_than_answering_a_different_one(client):
    """A typo'd aggregate policy and "no aggregate" are different questions, so the server does not
    silently substitute one for the other."""
    bad_policy = client.get("/api/panel", params={"aggregate": "mena"})
    assert bad_policy.status_code == 400
    assert "unknown aggregate policy" in bad_policy.json()["error"]

    assert client.get("/api/panel", params={"min_coverage": "ninety"}).status_code == 400
    assert client.get("/api/panel", params={"min_coverage": "90"}).status_code == 400


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
    assert probe["record_count"] == 15
    assert probe["error"] is None
    assert len(probe["parse_failures"]) == 1
    assert probe["parse_failures"][0]["path"].endswith("20260722-000000-legacy-mmlu-broken/record.json")
    assert "launch_host" in probe["parse_failures"][0]["error"]


def test_memory_store_deduplicates_migrated_runs_with_canonical_precedence(tmp_path):
    source = tmp_path / "source"
    fixtures.build_fixtures(str(source))
    record = list_records(str(source))[0]
    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    write_record(record.model_copy(update={"description": "canonical"}), str(canonical))
    write_record(record.model_copy(update={"description": "legacy"}), str(legacy))
    store = server.MemoryRecordStore()
    store.refresh(list_records(str(canonical)) + list_records(str(legacy)))

    assert len(store.fetch_runs()) == 1
    stored = store.get_record(record.run_id)
    assert stored is not None
    assert stored["description"] == "canonical"


def test_api_jobs_degrade_without_a_cluster(client):
    # The local NullClusterGateway reports every role unreachable rather than reaching Iris.
    jobs = client.get("/api/runs/snowball-2026.07.20-mmlu/jobs").json()
    assert jobs["roles"]
    assert all(role["reachable"] is False for role in jobs["roles"])
