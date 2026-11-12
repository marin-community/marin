# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from starlette.testclient import TestClient

from infra.marina.apps.knotmeter.ingest.adapter import prepare_bundle
from infra.marina.apps.knotmeter.ingest.calibration import prepare_calibrations
from infra.marina.apps.knotmeter.ingest.load import load_corpus, load_run
from infra.marina.apps.knotmeter.ingest.replay import prepare_requests
from infra.marina.apps.knotmeter.server.app import create_api


@pytest.mark.parametrize("hosts,expected_gpus", [(1, 8), (4, 16)])
def test_bundle_preserves_processor_metrics_and_exact_timestamps(bundle, hosts, expected_gpus):
    path, actual = bundle(hosts)
    (prepared,) = prepare_bundle(path, actual)
    assert prepared.run["physical_gpus"] == expected_gpus
    assert prepared.run["metrics"]["throughput"]["per_gpu"]["output_tput_tps"] == 40.5
    assert prepared.run["metrics"]["latency"]["ttft"]["p90"] == 1.25
    assert prepared.run["source"]["run_attempt"] == 2
    assert prepared.run["date"] == "2026-10-28"
    assert prepared.run["origin_ns"] == "1793157479513724276"
    assert [r["start_ns"] for r in prepared.requests] == [0, 51, 12800]
    assert [r["end_ns"] for r in prepared.requests] == [123456789, 123456840, 123469589]
    assert prepared.requests[0]["phase"] == "warmup"
    assert prepared.requests[1]["corpus_id"] == "conversation"
    assert prepared.requests[1]["source_outer_idx"] == 7


def test_calibration_repeats_survive_native_import_and_keep_baselines_separate(calibration, bundle, knotmeter_database):
    services, query = knotmeter_database
    path, actual = bundle()
    (baseline,) = prepare_bundle(path, actual)
    load_run(baseline, query)
    points = list(prepare_calibrations(calibration))
    assert len(points) == 2  # The invalid screen has no artifacts and never enters this catalog.
    assert points[0].key != points[1].key
    assert points[0].run["calibration"]["curve_fingerprint"] == points[1].run["calibration"]["curve_fingerprint"]
    for point in points:
        assert load_run(point, query)["changed"]
    with TestClient(create_api(services).app) as client:
        catalog = client.get("/runs").json()
        assert len(catalog) == 3
        for index, point in enumerate(points):
            saved = client.get(f"/runs/{point.key}/data").json()
            run = saved["run"]
            assert run["source"]["kind"] == "private-calibration"
            assert run["source"]["job_id"] == f"/test/calibration-repeat-{index}"
            assert "run_id" not in run["source"]
            assert run["date"] == f"2026-11-{10 + index}"
            assert run["measurement_seconds"] == 900
            assert run["topology"] == {"tp": 6, "dp": 2, "ep": 12, "pp": 1}
            assert run["aggregate"]["tp"] == run["physical_gpus"] == 12
            assert run["metrics"]["throughput"]["per_gpu"]["output_tput_tps"] == 40.5
            assert run["calibration"]["profiling_cancelled"] == 1
            assert saved["requests"] == point.requests
            assert not load_run(point, query)["changed"]
        assert client.get(f"/runs/{baseline.key}").json()["measurement_seconds"] == 3600
    # A serving-control change must split the comparison curve even at the same load.
    receipt = calibration / "point-1/restored-point/attempt.json"
    changed = json.loads(receipt.read_text())
    changed["server_command"][4] = "256"
    receipt.write_text(json.dumps(changed))
    first, second = prepare_calibrations(calibration)
    assert first.run["calibration"]["curve_fingerprint"] != second.run["calibration"]["curve_fingerprint"]
    # The separate GitHub path keeps its full-hour coverage requirement.
    source_path = path / "agentx-source.json"
    submission = json.loads(source_path.read_text())
    submission["points"][0]["receipt"]["validation"]["coverage"]["expected_duration_seconds"] = 900
    source_path.write_text(json.dumps(submission))
    with pytest.raises(ValueError, match="full-hour"):
        prepare_bundle(path, actual)


def test_multihost_import_rejects_per_host_gpu_normalization(bundle):
    path, actual = bundle(4)
    aggregate_path = path / "bmk_point/agg.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["num_gpus"] = 4
    aggregate_path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="physical GPU"):
        prepare_bundle(path, actual)


def test_import_replay_and_concurrent_selection_preserve_complete_serving_run(bundle, knotmeter_database):
    path, actual = bundle()
    (original,) = prepare_bundle(path, actual)
    original = replace(original, series={"kvCacheUsage": [{"t": i, "value": 0.25} for i in range(10000)]})
    services, query = knotmeter_database
    assert load_run(original, query)["changed"]
    with TestClient(create_api(services).app) as client:
        before = client.get(f"/runs/{original.key}/requests").json()
        assert client.get(f"/runs/{original.key}/series").json() == original.series
        assert [r["start_ns"] for r in before["requests"]] == [0, 51, 12800]
        assert not load_run(original, query)["changed"]
        assert client.get(f"/runs/{original.key}/requests").json() == before
        assert client.get(f"/runs/{original.key}/series").json() == original.series
        assert query("SELECT count(*) AS count FROM knotmeter_requests", {})["rows"][0]["count"] == 3
        winner = replace(original, digest="winner", run=original.run | {"runtime": "winning-runtime"})
        loser = replace(original, digest="loser", run=original.run | {"runtime": "stale-runtime"})

        def racing_query(sql, parameters):
            if sql.startswith("INSERT INTO knotmeter_current"):
                load_run(winner, query)
            return query(sql, parameters)

        with pytest.raises(RuntimeError, match="another loader"):
            load_run(loser, racing_query)
        assert client.get(f"/runs/{original.key}").json()["runtime"] == "winning-runtime"
        assert client.get(f"/runs/{original.key}/requests").json() == before
        snapshot = client.get(f"/runs/{original.key}/data").json()
        assert snapshot["run"]["runtime"] == "winning-runtime"
        assert snapshot["requests"] == before["requests"]
        assert snapshot["series"] == original.series
        assert not load_run(winner, query)["changed"]


def test_replay_ancestry_distinguishes_repeated_sources_and_preserves_queue_time(bundle):
    path, _ = bundle()
    records = [
        json.loads(line) for line in (path / "point/aiperf_artifacts/profile_export.jsonl").read_text().splitlines()
    ]
    # Parent links must join a child to its replay even when the child appears first.
    records[0]["metadata"].update(x_correlation_id="child", parent_correlation_id="parent", source_inner_idx=3)
    records[1]["metadata"].update(x_correlation_id="parent", root_correlation_id="root-a")
    records[2]["metadata"].update(x_correlation_id="repeated-source", root_correlation_id="root-b", worker_id="worker-2")
    records[1]["metadata"]["credit_issued_ns"] = records[1]["metadata"]["request_start_ns"] - 19
    records[0]["metadata"]["request_ack_ns"] = records[0]["metadata"]["request_start_ns"] + 7
    records[0]["metadata"]["was_cancelled"] = True
    _, requests = prepare_requests(lambda: records)
    parent, repeated, child = requests
    assert parent["replay_index"] == child["replay_index"]
    assert repeated["replay_index"] != parent["replay_index"]
    assert parent["credit_ns"] == 0
    assert parent["start_ns"] == 19
    assert child["ack_ns"] is not None
    assert child["ack_ns"] - child["start_ns"] == 7
    assert child["source_inner_idx"] == 3
    assert child["cancelled"]


def test_corpus_summary_counts_nested_requests_and_keeps_revision_immutable(tmp_path, knotmeter_database):
    turn = {"kind": "turn", "turnIndex": 0, "in": 100, "out": 10, "cached": 50, "uncached": 50, "model": "a"}
    child = turn | {"in": 40, "out": 4, "cached": 20, "uncached": 20, "model": "b"}
    structure = {
        "nodes": [turn, {"kind": "subagent", "children": [child, child]}],
        "totals": {"in": 180, "out": 18, "cached": 90, "uncached": 90, "numTurns": 1, "numSubagentGroups": 1},
    }
    path = tmp_path / "corpus.jsonl"
    rows = [
        {"dataset": "example/corpus", "revision": "r1", "conversation_id": cid, "structure": structure}
        for cid in ["one", "two"]
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    services, query = knotmeter_database
    load_corpus(path, query)
    with TestClient(create_api(services).app) as client:
        before = client.get("/datasets").json()
        summary = before[0]["summary"]
        assert summary["requests"] == 6
        assert summary["models"] == {"a": 2, "b": 4}
        assert summary["in"] == 360
        inputs = before[0]["distributions"]["Input tokens per request"]
        assert inputs["stats"]["mean"] == 60
        assert inputs["stats"]["median"] == 40
        assert sum(bucket["count"] for bucket in inputs["bins"]) == 6
        load_corpus(path, query)
        assert client.get("/datasets").json() == before
        assert len(client.get("/corpus?dataset=example/corpus&revision=r1").json()) == 2
        rows[0]["structure"] = structure | {"totals": structure["totals"] | {"in": 999}}
        path.write_text("\n".join(json.dumps(row) for row in rows))
        with pytest.raises(ValueError, match="immutable"):
            load_corpus(path, query)
        assert client.get("/datasets").json() == before
