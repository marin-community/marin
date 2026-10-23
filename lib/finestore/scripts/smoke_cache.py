# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise cache compaction and expiry recovery using disposable local archives.

Run from the repository root with:
    uv run --no-sync python lib/finestore/scripts/smoke_cache.py

Archives live temporarily under .cache and are removed even when a check fails.
"""

import json
import logging
import os
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
from finestore.cache import PersistentKvCache
from finestore.commit import read_snapshot
from finestore.compaction import compact_table
from finestore.layout import CACHE_HISTORY_FLOOR_FEATURE, BlobTables, FineStoreLayout, SystemColumns
from finestore.reader import ReadView
from finestore.store import INLINE_BLOB_BYTES, OBJECT_PART_BYTES, DataStore

SEED = 71203
COMMITS = 450


def check_cache(base: Path, expected: dict[str, bytes]) -> None:
    with closing(PersistentKvCache.at(str(base))) as cache:
        names = [*expected, "absent", *expected]
        assert cache.load_many(names) == expected, f"persisted values differ at {base}"
        assert cache.load_many(names) == expected, f"memory values differ at {base}"


def stress_compaction(base: Path) -> dict:
    rng = random.Random(SEED)
    with closing(PersistentKvCache.at(str(base))) as cache:
        root = cache.location()
    assert root is not None
    expected = {}
    sizes = (0, 19, 4096, INLINE_BLOB_BYTES, INLINE_BLOB_BYTES + 1)
    with DataStore.open(root, flush_interval=3600, max_buffer_bytes=32 * 1024 * 1024) as store:
        multipart = rng.randbytes(OBJECT_PART_BYTES + 113)
        store.write_object("multipart", multipart)
        store.flush()
        expected["multipart"] = multipart
        pinned = ReadView(root)
        for index in range(COMMITS):
            key = f"kernel-{index % 41:03d}"
            value = rng.randbytes(sizes[index % len(sizes)])
            store.write_object(key, value)
            store.flush()
            store.maintain()
            expected[key] = value
            if index % 85 == 0:
                check_cache(base, expected)
        assert pinned.read_blob("multipart") == expected["multipart"]

    check_cache(base, expected)
    view = ReadView(root)
    generations = sorted({shard.generation for shard in view.list_shards(BlobTables.DESCRIPTORS)})
    assert max(generations) >= 3, f"did not reach generation three: {generations}"
    for table in view.table_names():
        for shard in view.list_shards(table):
            commits = pq.ParquetFile(shard.path).read(columns=[SystemColumns.COMMIT]).column(0).to_pylist()
            assert shard.min_row_commit == min(commits)
            assert shard.max_row_commit == max(commits)
            assert max(commits) <= shard.commit_sequence
        compact_table(root, table)
    check_cache(base, expected)
    assert len(ReadView(root).keys(BlobTables.DESCRIPTORS)) == len(expected)
    return {
        "commits": COMMITS + 1,
        "keys": len(expected),
        "generations": generations,
        "multipart_bytes": OBJECT_PART_BYTES + 113,
    }


def recover_expiry(base: Path, expired: str) -> dict:
    with closing(PersistentKvCache.at(str(base))) as cache:
        root = cache.location()
    assert root is not None
    old = b"a" * (INLINE_BLOB_BYTES + 1)
    new = b"b" * len(old)
    with DataStore.open(root, flush_interval=3600, compaction_levels=(1000,)) as store:
        for value in (old, new):
            store.write_object("overwritten", value)
            store.flush()
            store.compact(BlobTables.DESCRIPTORS, generation=0)
            store.compact(BlobTables.PARTS, generation=0)
        before_healthy = ReadView(root)
        store.write_object("healthy", b"survives")
        store.flush()
    if expired == "schema":
        missing = before_healthy.table_metadata_path(BlobTables.DESCRIPTORS)
    else:
        table = BlobTables.DESCRIPTORS if expired == "descriptor" else BlobTables.PARTS
        missing = max(before_healthy.list_shards(table), key=lambda shard: shard.commit_sequence).path
    assert missing is not None
    Path(missing).unlink()
    expected = {"healthy": b"survives"}
    if expired == "schema":
        expected["overwritten"] = new
    token = ReadView(root).token
    with closing(PersistentKvCache.at(str(base))) as cache:
        assert cache.load_many(["overwritten", "healthy"]) == expected
        assert ReadView(root).token == token, "read unexpectedly repaired storage"
        cache.store("overwritten", b"recompiled")
        cache.store("after", b"fresh")
    expected.update(overwritten=b"recompiled", after=b"fresh")
    check_cache(base, expected)
    snapshot = read_snapshot(FineStoreLayout(root))
    if expired != "schema":
        assert snapshot.manifest.cache_history_floor > 0
        assert CACHE_HISTORY_FLOOR_FEATURE in snapshot.manifest.required_features
    for table in ReadView(root).table_names():
        compact_table(root, table)
    check_cache(base, expected)
    return {"expired": expired, "history_floor": snapshot.manifest.cache_history_floor, "keys": len(expected)}


def concurrent_writers(base: Path) -> dict:
    def write_batch(writer: int) -> dict[str, bytes]:
        values = {f"writer-{writer}/kernel-{index}": f"code-{writer}-{index}".encode() for index in range(24)}
        with closing(PersistentKvCache.at(str(base))) as cache:
            for key, value in values.items():
                cache.store(key, value)
        return values

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(write_batch, range(4)))
    expected = {key: value for batch in batches for key, value in batch.items()}
    check_cache(base, expected)
    return {"writers": 4, "keys": len(expected)}


def refresh_and_isolate(base: Path) -> dict:
    with DataStore.open(str(base)) as legacy:
        legacy.write_object("kernel", b"legacy")
    legacy_token = ReadView(str(base)).token
    with closing(PersistentKvCache.at(str(base))) as cache:
        assert cache.load("kernel") is None
        cache.store("kernel", b"v2")
        root = cache.location()
    assert root == str(base / "v2")
    view = ReadView(root)
    paths = [Path(shard.path) for shard in view.list_shards(BlobTables.DESCRIPTORS)]
    schema = view.table_metadata_path(BlobTables.DESCRIPTORS)
    assert schema is not None
    paths.append(Path(schema))
    aged = (datetime.now(UTC) - timedelta(days=20)).timestamp()
    for path in paths:
        os.utime(path, (aged, aged))
    timestamps = {path: path.stat().st_mtime_ns for path in paths}
    check_cache(base, {"kernel": b"v2"})
    assert {path: path.stat().st_mtime_ns for path in paths} == timestamps, "read unexpectedly refreshed storage"
    with closing(PersistentKvCache.at(str(base))) as cache:
        cache.store("new", b"fresh")
    cutoff = (datetime.now(UTC) - timedelta(days=15)).timestamp()
    assert all(path.stat().st_mtime >= cutoff for path in paths)
    assert view.read_blob("kernel") == b"v2"
    assert ReadView(str(base)).token == legacy_token
    assert ReadView(str(base)).read_blob("kernel") == b"legacy"
    check_cache(base, {"kernel": b"v2", "new": b"fresh"})
    return {"refreshed_objects": len(paths), "legacy_unchanged": True}


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    scratch = Path(__file__).resolve().parents[3] / ".cache"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="finestore-smoke-", dir=scratch) as directory:
        root = Path(directory)
        results = {}
        scenarios = {
            "compaction": lambda: stress_compaction(root / "compaction"),
            **{
                f"expiry-{kind}": lambda kind=kind: recover_expiry(root / kind, kind)
                for kind in ("descriptor", "parts", "schema")
            },
            "concurrent-writers": lambda: concurrent_writers(root / "concurrent"),
            "retention-and-version": lambda: refresh_and_isolate(root / "ttl=30d" / "retention"),
        }
        for name, run in scenarios.items():
            results[name] = run()
            print(json.dumps({"scenario": name, "result": results[name]}), flush=True)
    print(json.dumps({"status": "passed", "seed": SEED, "scenarios": len(results)}))


if __name__ == "__main__":
    main()
