# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from finelog.benchmarks.layout_candidates import (
    PartitionedSegment,
    SegmentManifestEntry,
    _aggregate_rollup_batch,
    _merge_rollup_states,
    _rollup_difference_count,
    select_manifest_entries,
    select_partitioned_entries,
    table_digest,
)
from finelog.benchmarks.query_measurement import parse_explain_metrics
from finelog.benchmarks.telemetry_workload_corpus import (
    RUN_METRIC_NAMES,
    SERIES_COUNT,
    DatasetSpec,
    SignalGroup,
    Workload,
    WorkloadName,
    build_workloads,
    dataset_facts,
    generate_batch,
    generate_batches,
)


def test_generated_rows_cover_signal_groups_and_time_range() -> None:
    spec = DatasetSpec(rows=SERIES_COUNT * 2, duration_days=2, batch_rows=377)
    table = pa.Table.from_batches(list(generate_batches(spec)))

    assert table.num_rows == spec.rows
    assert set(table.column("source").to_pylist()) == {group.value for group in SignalGroup}
    timestamps = table.column("ts").to_pylist()
    assert timestamps[0].timestamp() * 1000 == spec.start_ms
    assert timestamps[-1].timestamp() * 1000 == spec.end_ms - 1
    assert "run-000007" in set(table.column("run").to_pylist())


def test_workload_corpus_covers_decision_queries_and_bounds_every_scan() -> None:
    spec = DatasetSpec(rows=1_000, duration_days=14, batch_rows=100)
    workloads = build_workloads(spec)

    assert {workload.name for workload in workloads} == set(WorkloadName) - {WorkloadName.RUN_PREFIX_SEARCH}
    for workload in workloads:
        assert "ts >=" in workload.sql
        assert "ts <" in workload.sql
    run_workloads = [workload for workload in workloads if workload.name.value.startswith("run_drilldown")]
    assert len(run_workloads) == 3
    assert len(RUN_METRIC_NAMES) == 10
    assert {spec.end_ms - workload.start_ms for workload in run_workloads} == {
        15 * 60 * 1_000,
        24 * 60 * 60 * 1_000,
        7 * 24 * 60 * 60 * 1_000,
    }
    errors = next(workload for workload in workloads if workload.name == WorkloadName.ERRORS)
    assert "LIKE '%OOM%'" in errors.sql


def test_node_history_selects_worker_bucket_seven() -> None:
    spec = DatasetSpec(rows=500_000, duration_days=14, batch_rows=25_000)
    workload = next(workload for workload in build_workloads(spec) if workload.name == WorkloadName.NODE_DEVICE_HISTORY)
    entries = tuple(
        PartitionedSegment(
            entry=SegmentManifestEntry(
                path=f"/tmp/segment-{bucket}.parquet",
                min_ts_ms=workload.start_ms,
                max_ts_ms=workload.end_ms - 1,
                min_name="a",
                max_name="z",
                rows=10,
                row_groups=1,
                bytes=100,
            ),
            day=13,
            bucket=bucket,
        )
        for bucket in range(64)
    )

    selected = select_partitioned_entries(entries, workload)

    assert workload.hash_bucket == 7
    assert [entry.path for entry in selected] == ["/tmp/segment-7.parquet"]


@pytest.mark.parametrize(
    ("rows", "scrapes", "interval_minutes", "runs"),
    [
        (500_000, 230, 88.03493449781659, 11_027),
        (1_000_000, 460, 43.92156862745098, 22_053),
    ],
)
def test_dataset_facts_report_achieved_corpus_shape(
    rows: int,
    scrapes: int,
    interval_minutes: float,
    runs: int,
) -> None:
    facts = dataset_facts(DatasetSpec(rows=rows, duration_days=14, batch_rows=25_000))

    assert facts["signal_count"] == 34
    assert facts["achieved_scrapes"] == scrapes
    assert facts["scrape_interval_minutes"] == pytest.approx(interval_minutes)
    assert facts["achieved_distinct_run_ids"] == runs


def test_parse_explain_metrics_sums_data_sources() -> None:
    # DataFusion prints every one of these as a counter: decimal scaling with a
    # bare ` K`/` M`/` B` suffix, `bytes_scanned` included.
    plan = """
DataSourceExec: metrics=[files_ranges_pruned_statistics=12 total → 2 matched,
row_groups_pruned_statistics=64 total → 3 matched, bytes_scanned=42.5 K,
pushdown_rows_matched=4.10 K, pushdown_rows_pruned=127.0 K,
row_pushdown_eval_time=1.79ms,
metadata_load_time=10.2ms, time_elapsed_opening=23.1ms]
DataSourceExec: metrics=[files_ranges_pruned_statistics=8 total → 1 matched,
row_groups_pruned_statistics=1.16 K total → 2 matched, bytes_scanned=1.5 M,
pushdown_rows_matched=900, pushdown_rows_pruned=1.16 B,
row_pushdown_eval_time=3.21s,
metadata_load_time=400µs, time_elapsed_opening=1.1ms]
"""
    metrics = parse_explain_metrics(plan)

    assert metrics.files_total == 20
    assert metrics.files_matched == 3
    assert metrics.row_groups_total == 64 + 1_160
    assert metrics.row_groups_matched == 5
    assert metrics.bytes_scanned == 42_500 + 1_500_000
    assert metrics.metadata_load_ms == 10.6
    assert metrics.file_open_ms == pytest.approx(24.2)
    assert metrics.pushdown_rows_matched == 4_100 + 900
    assert metrics.pushdown_rows_pruned == 127_000 + 1_160_000_000
    assert metrics.pushdown_eval_ms == pytest.approx(3_211.79)


def test_manifest_selection_keeps_only_timestamp_overlap() -> None:
    entries = tuple(
        SegmentManifestEntry(
            path=f"/tmp/segment-{index}.parquet",
            min_ts_ms=index * 100,
            max_ts_ms=index * 100 + 99,
            min_name="a",
            max_name="z",
            rows=10,
            row_groups=1,
            bytes=100,
        )
        for index in range(4)
    )
    workload = Workload(
        name=WorkloadName.RUN_DRILLDOWN_15M,
        sql="SELECT 1",
        start_ms=150,
        end_ms=300,
    )

    selected = select_manifest_entries(entries, workload)

    assert [entry.path for entry in selected] == [
        "/tmp/segment-1.parquet",
        "/tmp/segment-2.parquet",
    ]


def test_partition_selection_maps_principal_to_configured_bucket_count() -> None:
    entries = tuple(
        PartitionedSegment(
            entry=SegmentManifestEntry(
                path=f"/tmp/segment-{bucket}.parquet",
                min_ts_ms=0,
                max_ts_ms=999,
                min_name="a",
                max_name="z",
                rows=10,
                row_groups=1,
                bytes=100,
            ),
            day=0,
            bucket=bucket,
        )
        for bucket in range(2)
    )
    workload = Workload(
        name=WorkloadName.RUN_DRILLDOWN_15M,
        sql="SELECT 1",
        start_ms=0,
        end_ms=1_000,
        hash_bucket=3,
    )

    selected = select_partitioned_entries(entries, workload)

    assert [entry.path for entry in selected] == ["/tmp/segment-1.parquet"]


def test_segment_rollup_states_merge_to_full_batch_result() -> None:
    spec = DatasetSpec(rows=4_096, duration_days=1, batch_rows=4_096)
    whole = _aggregate_rollup_batch(generate_batch(spec, 0, spec.rows), "hour")
    partials = [
        _aggregate_rollup_batch(generate_batch(spec, 0, 2_048), "hour"),
        _aggregate_rollup_batch(generate_batch(spec, 2_048, spec.rows), "hour"),
    ]

    merged = _merge_rollup_states(partials, "hour")

    assert _rollup_difference_count(whole, merged) == 0


def test_result_digest_ignores_insignificant_float_aggregation_order() -> None:
    left = pa.table({"key": ["a"], "value": [1.00000000001]})
    right = pa.table({"key": ["a"], "value": [1.00000000002]})

    assert table_digest(left) == table_digest(right)
