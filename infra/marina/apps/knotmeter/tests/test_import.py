# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from marina.db import UrlDatabase
from marina.testing import SourceServices
from marina.testing import test_database as disposable_database
from sqlalchemy import text
from starlette.testclient import TestClient

from infra.marina.apps.knotmeter.ingest.adapter import prepare_bundle
from infra.marina.apps.knotmeter.ingest.load import load_run
from infra.marina.apps.knotmeter.server.app import create_api, migrate


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


def test_multihost_import_rejects_per_host_gpu_normalization(bundle):
    path, actual = bundle(4)
    aggregate_path = path / "bmk_point/agg.json"
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["num_gpus"] = 4
    aggregate_path.write_text(json.dumps(aggregate))
    with pytest.raises(ValueError, match="physical GPU"):
        prepare_bundle(path, actual)


def test_import_replay_and_concurrent_selection_preserve_complete_serving_run(bundle):
    path, actual = bundle()
    (original,) = prepare_bundle(path, actual)
    with disposable_database() as url:
        services = SourceServices(name="knotmeter", data_url="", database=UrlDatabase(url))
        engine = services.engine()
        with engine.begin() as connection:
            migrate(connection)

        def query(sql, parameters):
            with engine.begin() as connection:
                result = connection.execute(text(sql), parameters)
                return {
                    "rows": [dict(r) for r in result.mappings()] if result.returns_rows else [],
                    "row_count": result.rowcount,
                }

        assert load_run(original, query)["changed"]
        with TestClient(create_api(services).app) as client:
            before = client.get(f"/runs/{original.key}/requests").json()
            assert [r["start_ns"] for r in before["requests"]] == [0, 51, 12800]
            assert not load_run(original, query)["changed"]
            assert client.get(f"/runs/{original.key}/requests").json() == before
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
            assert not load_run(winner, query)["changed"]
        engine.dispose()
