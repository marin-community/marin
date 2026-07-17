# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Telltale registry -> finelog forwarding: row construction and a round-trip."""

import itertools

import pytest
from finelog.client import FlushResult, schema_from_dataclass
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from iris.runtime.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV
from iris.runtime.telltale_forward import (
    TELLTALE_NAMESPACE,
    TelltaleMetric,
    _identity_labels,
    _source_for,
    scrape_rows,
)
from rigging import telltale
from rigging.timing import Timestamp

_names = itertools.count()
_TS = Timestamp.now().as_naive_utc()


@pytest.fixture
def name() -> str:
    """A metric name unique per test (the registry is process-global)."""
    return f"telltale_fwd_test_{next(_names)}"


@pytest.fixture
def clean_global_labels():
    saved = telltale.get_global_labels()
    telltale._global_labels.clear()
    yield
    telltale._global_labels.clear()
    telltale._global_labels.update(saved)


def _row(name: str, rows: list[TelltaleMetric]) -> TelltaleMetric:
    matching = [r for r in rows if r.name == name]
    assert len(matching) == 1, f"expected one {name!r} row, got {len(matching)}"
    return matching[0]


# --- source resolution -----------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "labels", "expected"),
    [
        ("levanter_train_loss", {}, "levanter"),
        ("zephyr_item_count", {}, "zephyr"),
        ("iris_task_reconciles", {}, "iris"),
        ("process_cpu_seconds_total", {}, "process"),
        ("python_gc_objects_collected", {}, "process"),
        ("levanter_train_loss", {"source": "custom"}, "custom"),
    ],
)
def test_source_prefers_explicit_label_then_name_prefix(metric_name, labels, expected):
    assert _source_for(metric_name, labels) == expected


# --- identity --------------------------------------------------------------


def test_identity_labels_carry_job_and_process_index(monkeypatch):
    monkeypatch.setenv(IRIS_MULTIGPU_PROCESS_INDEX_ENV, "2")
    info = JobInfo(task_id=JobName.from_wire("/alice/train/worker/3"), worker_id="w-7", attempt_id=1)

    identity = _identity_labels(info)

    assert identity["task_id"] == "/alice/train/worker/3"
    assert identity["job_id"] == "/alice/train"
    assert identity["worker"] == "w-7"
    assert identity["attempt"] == "1"
    assert identity["process_index"] == "2"


# --- scrape_rows -----------------------------------------------------------


def test_scrape_stamps_identity_and_lifts_source_and_run(name, clean_global_labels):
    telltale.gauge(name, "d").set(2.0)
    telltale.set_global_labels(run="r1", source="levanter")

    row = _row(name, scrape_rows({"task_id": "t", "job_id": "j"}, telltale.get_global_labels(), _TS))

    assert row.value == 2.0
    assert row.kind == "gauge"
    assert row.source == "levanter"  # from the global label
    assert row.run == "r1"
    # run/source are promoted to columns, not duplicated in the map.
    assert "run" not in row.labels and "source" not in row.labels
    # identity is stamped into the map.
    assert row.labels["task_id"] == "t" and row.labels["job_id"] == "j"


def test_identity_overrides_a_colliding_metric_label(name):
    # A metric that carries its own `task_id` label must not spoof job identity.
    telltale.counter(name, "d", ["task_id"]).labels("evil").inc()

    row = _row(f"{name}_total", scrape_rows({"task_id": "real"}, {}, _TS))

    assert row.labels["task_id"] == "real"


def test_scrape_drops_created_series(name):
    telltale.counter(name, "d").inc()

    names = {r.name for r in scrape_rows({}, {}, _TS)}

    assert f"{name}_total" in names
    assert f"{name}_created" not in names


def test_scrape_keeps_histogram_le_label(name):
    telltale.histogram(name, "d").observe(0.5)

    buckets = [r for r in scrape_rows({}, {}, _TS) if r.name == f"{name}_bucket"]

    assert buckets and all(r.kind == "histogram" for r in buckets)
    assert "+Inf" in {r.labels["le"] for r in buckets}


# --- schema ----------------------------------------------------------------


def test_metric_schema_keys_on_name_with_a_native_map():
    schema = schema_from_dataclass(TelltaleMetric)

    assert schema.key_column == "name"
    by_type = {c.name: c.type for c in schema.columns}
    assert by_type["name"] == stats_pb2.COLUMN_TYPE_STRING
    assert by_type["value"] == stats_pb2.COLUMN_TYPE_FLOAT64
    assert by_type["labels"] == stats_pb2.COLUMN_TYPE_MAP
    assert by_type["run"] == stats_pb2.COLUMN_TYPE_STRING
    assert by_type["ts"] == stats_pb2.COLUMN_TYPE_TIMESTAMP_MS


# --- round-trip through a real finelog server ------------------------------


def _require_native_map(client) -> None:
    """Skip when the embedded finelog server predates the native Map column type.

    The map type + json_get were added recently; an older pinned ``finelog_server``
    wheel drops a map-carrying write with "unknown column type". The deployed
    server (built from main) supports it, and the Rust suite covers map behavior
    directly — so we skip here rather than fail on a stale local extension.
    """
    probe = "telltale_map_probe"
    table = client.get_table(probe, TelltaleMetric)
    table.write(
        [
            TelltaleMetric(
                name="p",
                value=0.0,
                labels={"k": "v"},
                kind="gauge",
                source="iris",
                run=None,
                ts=Timestamp.now().as_naive_utc(),
            )
        ]
    )
    table.flush(timeout=10.0)
    try:
        landed = client.query(f"SELECT count(*) AS n FROM {probe}").to_pylist()
    except Exception:
        landed = []
    if not landed or landed[0]["n"] < 1:
        pytest.skip("embedded finelog server predates the native Map column type")


def test_forwarded_rows_round_trip_and_json_get_reads_the_map(log_client, name, clean_global_labels):
    """Write scraped rows to an embedded finelog and read them back via json_get."""
    _require_native_map(log_client)
    telltale.gauge(name, "d").set(3.5)
    telltale.set_global_labels(run="run-xyz", source="levanter")

    table = log_client.get_table(TELLTALE_NAMESPACE, TelltaleMetric)
    rows = [
        r
        for r in scrape_rows({"task_id": "t1"}, telltale.get_global_labels(), Timestamp.now().as_naive_utc())
        if r.name == name
    ]
    table.write(rows)
    assert table.flush(timeout=10.0) == FlushResult.SUCCEEDED

    result = log_client.query(
        f"SELECT value, source, run, json_get(labels, 'task_id') AS task_id "
        f"FROM {TELLTALE_NAMESPACE} WHERE name = '{name}'"
    ).to_pylist()

    assert result == [{"value": 3.5, "source": "levanter", "run": "run-xyz", "task_id": "t1"}]
