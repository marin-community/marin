# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import timedelta

import pytest
from rl_producers import RL_PRODUCER_NAMESPACES
from rl_run_fixtures import (
    _COLUMNS,
    _NOW_MS,
    _WINDOW_START_MS,
    CLUSTER,
    NODES,
    RUN_ID,
    WINDOW_START,
    _census,
    _fresh_store,
    _millis,
    _overview_rows,
    _panel_sql,
    _request_rows,
    _row,
    _seed_rows,
)


def test_the_engine_panels_select_by_metric_name_alone(rl_run_store) -> None:
    # The engine path is the metric_source: the fixture's embedded-engine rows publish
    # under service='marinskyrl' with metric_source='vllm', and other producers' rows
    # are filtered out before the window rather than after bucketing.
    throughput = rl_run_store.execute(_panel_sql("vLLM token throughput")).fetchall()
    # A rate over a CUMULATIVE counter: the panel takes the delta between consecutive samples,
    # so a fixture growing by 1024 per bucket per engine yields a constant rate -- and the first
    # bucket has no predecessor to difference against, so it drops out.
    rates = [round(row[1], 4) for row in throughput]
    assert rates == [round(2 * 1024.0 / 300.0, 4)] * len(rates)
    assert len(rates) == 5

    queue = rl_run_store.execute(_panel_sql("vLLM queue and KV-cache usage")).fetchall()
    assert [(row[1], row[2]) for row in queue] == [(48.0, 12.0)] * 6

    latency = rl_run_store.execute(_panel_sql("vLLM request latency (mean by stage)")).fetchall()
    # Prometheus histograms arrive as `_sum` and `_count`, so the panel reports a mean per stage.
    # A p90 would need bucket interpolation.
    assert {row[1] for row in latency} == {
        "request_queue_time_seconds",
        "request_decode_time_seconds",
    }


def test_the_producer_census_separates_the_engine_series_from_the_trainer(rl_run_store) -> None:
    # Each engine actor publishes its own registry under `service='vllm'`, so the census spans
    # both streams.
    rows = _census(rl_run_store, frozenset(RL_PRODUCER_NAMESPACES))

    assert {(row["producer"], row["role"]) for row in rows} >= {("marinskyrl", "trainer"), ("vllm", "inference")}


def test_engine_rates_keep_two_actors_that_both_report_engine_zero_apart(rl_run_store) -> None:
    # `engine` indexes engines within the registry that published them, so its scope depends on how
    # they were registered and two rows can carry the same index from different processes. The
    # window keys on the resource; GREATEST(delta, 0) would hide half of a cross-resource
    # subtraction as zero.
    for bucket in range(6):
        rl_run_store.execute(
            f'INSERT INTO "telemetry_v1.vllm" VALUES ({", ".join("?" for _ in _COLUMNS)})',
            list(
                _row(
                    service="vllm",
                    name="generation_tokens_total",
                    value=50_000.0 + 1024.0 * (bucket + 1),
                    moment=WINDOW_START + timedelta(minutes=5 * bucket),
                    seq=bucket,
                    run_id=RUN_ID,
                    node_name=NODES[0],
                    role="inference",
                    attributes={"engine": "0"},
                )
            ),
        )

    throughput = rl_run_store.execute(_panel_sql("vLLM token throughput")).fetchall()
    rates = [round(row[1], 4) for row in throughput]

    assert rates == [round(3 * 1024.0 / 300.0, 4)] * len(rates)
    assert len(rates) == 5


@pytest.mark.parametrize(
    ("values", "growth"),
    [
        ((1000.0, 1010.0, 0.0, 10.0), 20),
        ((0.0, 100.0, 0.0, 200.0), 300),
        ((100.0, 200.0, 0.0, 300.0), 400),
        ((10.0, 100.0, 20.0, 30.0), 100),
        ((0.0, 100.0, 0.0, 50.0, 0.0, 200.0), 350),
    ],
)
def test_engine_growth_counts_resets_inside_its_bucket(values, growth) -> None:
    database = _fresh_store()
    _seed_rows(
        database,
        [
            _row(
                service="vllm",
                name="generation_tokens_total",
                value=value,
                moment=WINDOW_START + timedelta(seconds=10 * index),
                seq=index,
                run_id=RUN_ID,
                node_name=NODES[0],
                role="inference",
                attributes={"metric_source": "vllm", "engine": "0"},
            )
            for index, value in enumerate(values)
        ],
    )

    (row,) = _overview_rows(database, "engine_tokens")

    assert row["t"] == _WINDOW_START_MS
    assert row["generation"] == pytest.approx(growth / 300)


def test_engine_rates_keep_identical_series_in_different_clusters_apart() -> None:
    database = _fresh_store()
    rows = []
    for cluster in (CLUSTER, "other-cluster"):
        for index, value in enumerate((0.0, 100.0)):
            row = _row(
                service="vllm",
                name="generation_tokens_total",
                value=value,
                moment=WINDOW_START + timedelta(seconds=10 * index),
                seq=index,
                run_id=RUN_ID,
                node_name=NODES[0],
                role="inference",
                attributes={"engine": "0"},
            )
            rows.append((cluster, *row[1:]))
    _seed_rows(database, rows)
    (row,) = _request_rows(
        database,
        "/v1/rl/overview",
        {
            "clusters": f"{CLUSTER},other-cluster",
            "run": RUN_ID,
            "from": _WINDOW_START_MS,
            "to": _NOW_MS,
            "bucket_ms": 300000,
            "view": "engine_tokens",
        },
    )
    assert row["generation"] == pytest.approx(200 / 300)


def test_engine_rates_keep_two_actors_that_differ_only_by_resource() -> None:
    database = _fresh_store()
    _seed_rows(
        database,
        [
            (
                CLUSTER,
                "vllm",
                RUN_ID,
                None,
                "iris:two-actors",
                NODES[0],
                None,
                "generation_tokens_total",
                base + 1024.0 * (bucket + 1),
                _millis(WINDOW_START + timedelta(minutes=5 * bucket)),
                bucket,
                json.dumps({"role": "inference", "training_type": "sync", "actor_uid": actor}),
                json.dumps({"metric_source": "vllm", "engine": "0"}),
            )
            for actor, base in (("a", 0.0), ("b", 50_000.0))
            for bucket in range(6)
        ],
    )

    rates = [round(row["generation"], 4) for row in _overview_rows(database, "engine_tokens")]

    assert rates == [round(2 * 1024.0 / 300.0, 4)] * len(rates)
    assert len(rates) == 5


def test_engine_panels_read_the_embedded_stream_as_well_as_the_standalone_one(rl_run_store) -> None:
    # An engine embedded in the trainer publishes through the trainer's exporter, so its rows land
    # under service='marinskyrl' with metric_source='vllm'. Reading only telemetry_v1.vllm would
    # leave every engine panel blank on such a run.
    for bucket in range(6):
        rl_run_store.execute(
            f'INSERT INTO "telemetry_v1.marinskyrl" VALUES ({", ".join("?" for _ in _COLUMNS)})',
            list(
                _row(
                    service="marinskyrl",
                    name="generation_tokens_total",
                    value=1024.0 * (bucket + 1),
                    moment=WINDOW_START + timedelta(minutes=5 * bucket),
                    seq=bucket,
                    run_id=RUN_ID,
                    node_name=NODES[0],
                    role="trainer",
                    attributes={"engine": "GPU-abc", "metric_source": "vllm"},
                )
            ),
        )

    rates = [round(row[1], 4) for row in rl_run_store.execute(_panel_sql("vLLM token throughput")).fetchall()]

    assert rates == [round(3 * 1024.0 / 300.0, 4)] * len(rates)
    assert len(rates) == 5
