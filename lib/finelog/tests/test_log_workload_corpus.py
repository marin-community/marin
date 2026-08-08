# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The generated `log` corpus must have the properties its workloads assume.

A corpus that drifts from its workloads measures the wrong thing silently: a
target job confined to one segment turns a whole-namespace needle into a
single-file scan, a task key no row carries makes the tail workload fast because
it matches nothing, and a key column with too few distinct values per segment
makes substring pruning look better than it is on real data.
"""

import pyarrow as pa
from finelog.benchmarks.log_workload_corpus import (
    ERROR_ROW_STRIDE,
    KEYS_PER_SEGMENT,
    LEVEL_ERROR,
    TARGET_JOB,
    LogDatasetSpec,
    WorkloadName,
    build_workloads,
    generate_batches,
    target_key,
)


def _corpus(rows: int = 40_000, segments: int = 4) -> tuple[LogDatasetSpec, pa.Table]:
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


def test_each_segment_carries_its_own_band_of_thousands_of_keys() -> None:
    spec, table = _corpus()
    keys = table.column("key").to_pylist()

    bands = []
    for segment in range(spec.segments):
        start = segment * spec.rows_per_segment
        bands.append(set(keys[start : start + spec.rows_per_segment]))
        assert len(bands[-1]) >= KEYS_PER_SEGMENT

    # Segments share only the target job, so each one's key band is distinct —
    # the layout `(key, seq)` compaction produces on the deployed namespace.
    shared = set.intersection(*bands)
    assert {key for key in shared if TARGET_JOB in key} == shared


def test_task_scoped_workload_reads_a_key_the_corpus_emits() -> None:
    spec, table = _corpus()
    workloads = {workload.name: workload for workload in build_workloads(spec)}

    assert target_key() in set(table.column("key").to_pylist())
    assert f"key = '{target_key()}'" in workloads[WorkloadName.TASK_TAIL].sql


def test_error_lines_exist_for_the_first_error_and_body_search_workloads() -> None:
    spec, table = _corpus()
    levels = table.column("level").to_pylist()
    bodies = table.column("data").to_pylist()

    errors = sum(1 for level in levels if level >= LEVEL_ERROR)
    assert 0 < errors <= spec.rows // ERROR_ROW_STRIDE + 1
    assert any("CUDA_ERROR" in body for body in bodies)
    assert any("Traceback" in body for body in bodies)


def test_every_workload_name_is_measured_once() -> None:
    spec, _ = _corpus(rows=1_000, segments=2)

    assert [workload.name for workload in build_workloads(spec)] == list(WorkloadName)
