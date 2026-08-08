# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The generated `log` corpus must have the properties its workloads assume.

A corpus that drifts from its workloads measures the wrong thing silently: a
target job confined to one segment turns a whole-namespace needle into a
single-file scan, a task key no row carries makes the tail workload fast because
it matches nothing, and a key column with too few distinct values per segment
makes substring pruning look better than it is on real data.
"""

from collections import Counter

import pyarrow as pa
from finelog.benchmarks.log_workload_corpus import (
    KEYS_PER_SEGMENT,
    LEVEL_ERROR,
    RECENT_WINDOW_MS,
    START_MS,
    TARGET_JOB,
    TARGET_JOB_ROW_STRIDE,
    TASKS_PER_JOB,
    LogDatasetSpec,
    WorkloadName,
    build_workloads,
    generate_batches,
    target_key,
)


def _corpus(rows: int = 80_000, segments: int = 4) -> tuple[LogDatasetSpec, pa.Table]:
    spec = LogDatasetSpec(rows=rows, segments=segments, batch_rows=3_000)
    return spec, pa.Table.from_batches(list(generate_batches(spec)))


def test_target_job_is_a_needle_spread_over_every_segment() -> None:
    spec, table = _corpus()
    keys = table.column("key").to_pylist()

    assert table.num_rows == spec.rows
    matching = [index for index, key in enumerate(keys) if TARGET_JOB in key]
    # A job-scoped query only measures pruning when the job is a small share of
    # every segment rather than a contiguous block of one.
    assert 0 < len(matching) < spec.rows // 100
    assert {index // spec.rows_per_segment for index in matching} == set(range(spec.segments))


def test_target_job_appears_only_through_the_stride_injection() -> None:
    spec, table = _corpus()
    keys = table.column("key").to_pylist()

    # One segment's ordinary key band covers the target job's slots. If the
    # generator walked them, that segment would hold thousands of extra target
    # rows and the job would stop being an evenly spread needle.
    per_segment = Counter(index // spec.rows_per_segment for index, key in enumerate(keys) if TARGET_JOB in key)
    expected = spec.rows_per_segment // TARGET_JOB_ROW_STRIDE + 1
    assert set(per_segment.values()) == {expected}


def test_each_segment_carries_its_own_band_of_thousands_of_keys() -> None:
    spec, table = _corpus()
    keys = table.column("key").to_pylist()

    bands = []
    for segment in range(spec.segments):
        start = segment * spec.rows_per_segment
        bands.append(set(keys[start : start + spec.rows_per_segment]))
        # The band covering the target job's slots gives them up, and a corpus
        # this small injects only a few of that job's tasks back.
        assert len(bands[-1]) >= KEYS_PER_SEGMENT - TASKS_PER_JOB

    # Segments share only the target job, so each one's key band is distinct —
    # the layout `(key, seq)` compaction produces on the deployed namespace.
    shared = set.intersection(*bands)
    assert {key for key in shared if TARGET_JOB in key} == shared


def test_task_scoped_workload_reads_a_key_the_corpus_emits() -> None:
    _, table = _corpus()
    workloads = {workload.name: workload for workload in build_workloads(START_MS)}

    assert target_key() in set(table.column("key").to_pylist())
    assert f"key = '{target_key()}'" in workloads[WorkloadName.TASK_TAIL].sql


def test_recent_window_filters_a_real_window_of_the_corpus() -> None:
    spec, table = _corpus()
    latest = max(table.column("epoch_ms").to_pylist())
    workloads = {workload.name: workload for workload in build_workloads(latest)}

    # The cutoff anchors on the newest row, not on a corpus dimension, so
    # `measure` over production segments filters the same relative window.
    cutoff = latest - RECENT_WINDOW_MS
    assert f"epoch_ms >= {cutoff}" in workloads[WorkloadName.JOB_RECENT_WINDOW].sql
    assert sum(1 for ms in table.column("epoch_ms").to_pylist() if ms >= cutoff) == min(spec.rows, RECENT_WINDOW_MS)


def test_error_lines_exist_for_the_first_error_and_body_search_workloads() -> None:
    _, table = _corpus()
    bodies = table.column("data").to_pylist()

    assert any("CUDA_ERROR" in body for body in bodies)
    assert any("Traceback" in body for body in bodies)


def test_the_target_job_carries_both_error_shapes_it_is_searched_for() -> None:
    _, table = _corpus()
    rows = zip(
        table.column("key").to_pylist(),
        table.column("level").to_pylist(),
        table.column("data").to_pylist(),
        strict=True,
    )
    target = [(level, body) for key, level, body in rows if TARGET_JOB in key]

    # JOB_FIRST_ERROR filters on level and JOB_TEXT_SEARCH on the body, so a
    # corpus where the job's rows are never errors measures two empty results.
    assert any(level >= LEVEL_ERROR for level, _ in target)
    assert any("CUDA_ERROR" in body for _, body in target)


def test_every_workload_name_is_measured_once() -> None:
    assert [workload.name for workload in build_workloads(START_MS)] == list(WorkloadName)
