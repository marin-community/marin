# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Request budgets and concurrency behaviour on the simulated S3 backend."""

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from finestore.compaction import CompactionResult, compact_table
from finestore.reader import ReadView
from finestore.store import DataStore, OnConflict
from finestore.testing.simulated_s3 import Request, SimulatedConditionalObject, simulated_s3
from rigging.filesystem.conditional_object import ConditionalWriteError
from rigging.filesystem.storage_path import StoragePath

ROOT = "s3://finestore-tests/run"


def test_overlapping_conditional_writes_let_exactly_one_writer_win():
    with simulated_s3(latency=0.01) as fs:
        head = SimulatedConditionalObject(f"{ROOT}/HEAD", fs)
        version = head.write(b"v0", expected_version=None)
        barrier = threading.Barrier(2)
        outcomes = []

        def attempt(payload: bytes) -> None:
            barrier.wait(timeout=5)
            try:
                outcomes.append(head.write(payload, expected_version=version))
            except ConditionalWriteError as error:
                outcomes.append(error)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in [pool.submit(attempt, b"a"), pool.submit(attempt, b"b")]:
                future.result(timeout=5)

    assert sum(isinstance(outcome, str) for outcome in outcomes) == 1
    assert fs.log.cas_conflicts == 1
    with pytest.raises(ConditionalWriteError):
        head.write(b"c", expected_version=version)


def test_single_row_flush_fits_the_request_budget(simulated_bucket):
    fs = simulated_bucket
    with DataStore.open(ROOT, writer_id="w1", flush_interval=3600) as store:
        table = store.table("samples", primary_key=("id",))
        table.append({"id": 1})
        before = fs.log.snapshot()
        store.flush()
        delta = fs.log.delta(before)
    # The shard PUT, the manifest PUT, and the compare-and-swap.
    assert delta.requests[Request.CAS_PUT] == 1
    assert delta.total() <= 3, dict(delta.requests)


def test_read_view_costs_two_requests(simulated_bucket):
    fs = simulated_bucket
    with DataStore.open(ROOT, writer_id="w1", flush_interval=3600) as store:
        store.table("samples", primary_key=("id",)).append({"id": 1})
        store.flush()
    before = fs.log.snapshot()
    ReadView(ROOT)
    # HEAD and the manifest it names.
    delta = fs.log.delta(before)
    assert delta.total() <= 2, dict(delta.requests)


@pytest.mark.parametrize("payload_bytes", [0, 16 * 1024])
def test_small_compaction_fetches_shards_concurrently_once_and_preserves_latest_rows(
    simulated_bucket, monkeypatch, payload_bytes
):
    fs = simulated_bucket
    store = DataStore.open(ROOT, writer_id="w1", flush_interval=3600, compaction_levels=(1000,))
    table = store.table("samples", primary_key=("id",), on_conflict=OnConflict.SUPERSEDE)
    for revision in range(17):
        rows = [
            {"id": index, "revision": revision, "payload": hashlib.shake_256(str(index).encode()).digest(payload_bytes)}
            for index in range(8)
        ]
        table.extend(rows)
        store.flush()
    store.request_stop()
    inputs = ReadView(ROOT).list_shards("samples")
    assert len(inputs) == 17

    concurrent_paths = {shard.path for shard in inputs[:2]}
    barrier = threading.Barrier(2)
    cat_file = fs.cat_file

    def read(path, **kwargs):
        if path in concurrent_paths:
            barrier.wait(timeout=5)
        return cat_file(path, **kwargs)

    monkeypatch.setattr(fs, "cat_file", read)
    before = fs.log.snapshot()
    result = compact_table(ROOT, "samples")
    delta = fs.log.delta(before)

    assert result == CompactionResult(written=8, superseded=128)
    # Two snapshot GETs, table metadata, one GET per input, and three publication writes.
    assert delta.total() <= len(inputs) + 6, dict(delta.requests)
    assert delta.requests[Request.HEAD] == 0
    assert list(ReadView(ROOT).iter_rows("samples", columns=["id", "revision", "payload"])) == rows


def test_listing_a_simulated_bucket_returns_reopenable_s3_paths(simulated_bucket):
    with DataStore.open(ROOT, writer_id="w1", flush_interval=3600) as store:
        store.table("samples", primary_key=("id",)).append({"id": 1})
        store.flush()
    entries = StoragePath(ROOT).ls()
    assert entries and all(str(entry).startswith(f"{ROOT}/") for entry in entries)
    assert (
        StoragePath(f"{ROOT}/HEAD").read_bytes() == next(entry for entry in entries if entry.name == "HEAD").read_bytes()
    )


def test_point_lookup_reads_only_shards_whose_bounds_match(simulated_bucket):
    fs = simulated_bucket
    store = DataStore.open(ROOT, writer_id="w1", flush_interval=3600)
    table = store.table("samples", primary_key=("id",))
    for first in (0, 100, 200, 300):
        table.extend({"id": first + offset} for offset in range(10))
        store.flush()
    store.request_stop()
    view = ReadView(ROOT)
    assert len(view.list_shards("samples")) == 4
    before = fs.log.snapshot()
    found = view.point("samples", id=205)
    assert found is not None and found["id"] == 205
    delta = fs.log.delta(before)
    # The table metadata plus the footer and rows of the one shard that can hold the key;
    # manifest shards are never stat'ed before they are opened.
    assert delta.requests[Request.HEAD] == 0
    assert delta.total() <= 3, dict(delta.requests)
    assert view.point("samples", id=999) is None


def test_concurrent_writers_publish_every_row_through_conflicts():
    writers = 6
    commits = 3
    with simulated_s3(latency=0.002, time_scale=0.001) as fs:

        def run(index: int) -> None:
            with DataStore.open(ROOT, writer_id=f"w{index}", flush_interval=3600) as store:
                table = store.table("samples", primary_key=("writer", "row"))
                for row in range(commits):
                    table.append({"writer": index, "row": row})
                    store.flush()

        with ThreadPoolExecutor(max_workers=writers) as pool:
            for future in [pool.submit(run, index) for index in range(writers)]:
                future.result(timeout=60)

        rows = ReadView(ROOT).scan("samples", columns=["writer", "row"]).to_pylist()
    assert sorted((row["writer"], row["row"]) for row in rows) == [
        (index, row) for index in range(writers) for row in range(commits)
    ]
    assert fs.log.cas_successes >= writers * commits
