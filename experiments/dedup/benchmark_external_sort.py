# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark three external sort strategies on synthetic LSH-like data.

Approaches
----------
old        list(heapq.merge(*batch)) in pass-1 — materialises everything, OOMs.
current    Streaming pickle, one item per pickle.dump/load — no OOM but slow.
improved   Batched pickle + zstd compression — no OOM, less storage, faster I/O.

Usage (must be inside a memory-limited Docker container):
    # Build the image once:
    docker build -t ext-sort-bench - <<'EOF'
    FROM python:3.11-slim
    RUN pip install zstandard psutil
    EOF

    # Run all three (or one at a time):
    docker run --rm --memory=10g --memory-swap=10g \\
        -v /home/rav/marin/experiments/dedup:/work -w /work \\
        ext-sort-bench python benchmark_external_sort.py --approach all

    # Trigger OOM with old approach (expect exit code 137):
    docker run --rm --memory=10g --memory-swap=10g \\
        -v /home/rav/marin/experiments/dedup:/work -w /work \\
        ext-sort-bench python benchmark_external_sort.py --approach old
"""

import argparse
import hashlib
import heapq
import logging
import os
import pickle
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from itertools import islice

import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── synthetic data parameters ──────────────────────────────────────────────

# Use enough data to trigger OOM in 10 GB with the old approach.
# OLD approach materialises N_ITERS x ITEMS_PER_ITER Python dicts in a list.
# Each Python dict is ~600 bytes in memory, so:
#   500 iters x 40 000 items x 600 B ~ 12 GB  ->  OOM in 10 GB container.
# NEW / IMPROVED approaches stream items and stay well under 1 GB.
N_ITERS = 500  # simulates EXTERNAL_SORT_FAN_IN number of scatter chunks
ITEMS_PER_ITER = 40_000  # items per scatter chunk
N_BUCKETS = 5_000  # unique LSH bucket values (many items share a bucket → realistic)

# ── external sort constants ────────────────────────────────────────────────
EXTERNAL_SORT_FAN_IN = 500  # max concurrent iterators per pass-1 batch
_CURRENT_BATCH_SIZE = 1  # items per pickle.dump in "current" approach
_IMPROVED_BATCH_SIZE = 10_000  # items per pickle.dump in "improved" approach


# ── helpers ────────────────────────────────────────────────────────────────


def _memory_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def _cgroup_limit_gb() -> float:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                val = f.read().strip()
            if val != "max":
                return int(val) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return psutil.virtual_memory().total / 1e9


class MemorySampler:
    """Background thread sampling RSS every 0.5 s; hard-aborts on limit."""

    def __init__(self, limit_gb: float | None = None):
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._limit_gb = limit_gb
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def peak_gb(self) -> float:
        return max(self._samples) if self._samples else 0.0

    def _run(self):
        while not self._stop.wait(0.5):
            gb = _memory_gb()
            self._samples.append(gb)
            if self._limit_gb and gb > self._limit_gb * 0.95:
                logger.error("RSS %.1f GB ≥ 95%% of limit %.1f GB — aborting", gb, self._limit_gb)
                os._exit(137)


# ── data generation ────────────────────────────────────────────────────────


def _make_bucket(i: int) -> str:
    """Deterministic 32-char bucket string (simulates MinHash LSH output)."""
    return hashlib.md5(f"b{i}".encode()).hexdigest()


# Pre-generate sorted bucket list once so each iterator is in bucket order.
_SORTED_BUCKETS: list[str] = sorted(_make_bucket(i) for i in range(N_BUCKETS))


def make_sorted_iterator(iter_idx: int, items_per_iter: int) -> Iterator[dict]:
    """Yield items in sorted bucket order without materialising the full list."""
    items_per_bucket, remainder = divmod(items_per_iter, N_BUCKETS)
    item_idx = 0
    for b_idx, bucket in enumerate(_SORTED_BUCKETS):
        count = items_per_bucket + (1 if b_idx < remainder else 0)
        for _j in range(count):
            yield {"bucket": bucket, "id": f"i{iter_idx}j{item_idx}", "file_idx": iter_idx}
            item_idx += 1


# ── old approach (list materialisation) ───────────────────────────────────


def external_sort_old(iters_gen: Iterator[Iterator], tmp_dir: str) -> Iterator[dict]:
    """OLD pass-1: list(heapq.merge(*batch)) — OOMs on large data."""
    run_paths: list[str] = []
    batch_idx = 0
    while True:
        batch = list(islice(iters_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = os.path.join(tmp_dir, f"run-{batch_idx:04d}.pkl")
        logger.info("[old] pass-1 batch %d: materialising …", batch_idx)
        merged = list(heapq.merge(*batch, key=lambda x: x["bucket"]))  # ← OOM here
        with open(run_path, "wb") as f:
            pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        del merged
        run_paths.append(run_path)
        batch_idx += 1

    def _read(path: str) -> Iterator[dict]:
        with open(path, "rb") as f:
            yield from pickle.load(f)

    yield from heapq.merge(*[_read(p) for p in run_paths], key=lambda x: x["bucket"])


# ── current approach (streaming pickle, 1 item/call) ─────────────────────


def external_sort_current(iters_gen: Iterator[Iterator], tmp_dir: str) -> Iterator[dict]:
    """CURRENT pass-1: one pickle.dump per item — no OOM, baseline speed."""
    run_paths: list[str] = []
    batch_idx = 0
    while True:
        batch = list(islice(iters_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = os.path.join(tmp_dir, f"run-{batch_idx:04d}.pkl")
        n = 0
        with open(run_path, "wb") as f:
            for item in heapq.merge(*batch, key=lambda x: x["bucket"]):
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
                n += 1
        size_mb = os.path.getsize(run_path) / 1e6
        logger.info("[current] pass-1 batch %d: %d items, %.1f MB", batch_idx, n, size_mb)
        run_paths.append(run_path)
        batch_idx += 1

    def _read(path: str) -> Iterator[dict]:
        with open(path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    yield from heapq.merge(*[_read(p) for p in run_paths], key=lambda x: x["bucket"])


# ── improved approach (batched pickle + zstd) ─────────────────────────────


def external_sort_improved(
    iters_gen: Iterator[Iterator],
    tmp_dir: str,
    batch_size: int = _IMPROVED_BATCH_SIZE,
) -> Iterator[dict]:
    """IMPROVED pass-1: batched pickle + zstd — no OOM, less storage, faster I/O."""
    import zstandard as zstd

    run_paths: list[str] = []
    batch_idx = 0
    cctx = zstd.ZstdCompressor(level=3)

    while True:
        batch = list(islice(iters_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = os.path.join(tmp_dir, f"run-{batch_idx:04d}.pkl.zst")
        n = 0
        pending: list[dict] = []
        with open(run_path, "wb") as raw_f:
            with cctx.stream_writer(raw_f, closefd=False) as f:
                for item in heapq.merge(*batch, key=lambda x: x["bucket"]):
                    pending.append(item)
                    if len(pending) >= batch_size:
                        pickle.dump(pending, f, protocol=pickle.HIGHEST_PROTOCOL)
                        n += len(pending)
                        pending = []
                if pending:
                    pickle.dump(pending, f, protocol=pickle.HIGHEST_PROTOCOL)
                    n += len(pending)
        size_mb = os.path.getsize(run_path) / 1e6
        logger.info("[improved] pass-1 batch %d: %d items, %.1f MB", batch_idx, n, size_mb)
        run_paths.append(run_path)
        batch_idx += 1

    def _read(path: str) -> Iterator[dict]:
        import zstandard as _zstd

        with open(path, "rb") as raw_f:
            with _zstd.ZstdDecompressor().stream_reader(raw_f) as f:
                while True:
                    try:
                        items: list[dict] = pickle.load(f)
                        yield from items
                    except EOFError:
                        break

    yield from heapq.merge(*[_read(p) for p in run_paths], key=lambda x: x["bucket"])


# ── zstd streaming, no batching ──────────────────────────────────────────


def external_sort_zstd_stream(
    iters_gen: Iterator[Iterator],
    tmp_dir: str,
) -> Iterator[dict]:
    """ZSTD-STREAM: one pickle.dump per item inside a zstd stream — no batch in memory."""
    import zstandard as zstd

    run_paths: list[str] = []
    batch_idx = 0
    cctx = zstd.ZstdCompressor(level=3)
    while True:
        batch = list(islice(iters_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = os.path.join(tmp_dir, f"run-{batch_idx:04d}.pkl.zst")
        n = 0
        with open(run_path, "wb") as raw_f:
            with cctx.stream_writer(raw_f, closefd=False) as f:
                for item in heapq.merge(*batch, key=lambda x: x["bucket"]):
                    pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
                    n += 1
        size_mb = os.path.getsize(run_path) / 1e6
        logger.info("[zstd-stream] pass-1 batch %d: %d items, %.1f MB", batch_idx, n, size_mb)
        run_paths.append(run_path)
        batch_idx += 1

    def _read(path: str) -> Iterator[dict]:
        import zstandard as _zstd

        with open(path, "rb") as raw_f:
            with _zstd.ZstdDecompressor().stream_reader(raw_f) as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        break

    yield from heapq.merge(*[_read(p) for p in run_paths], key=lambda x: x["bucket"])


# ── adaptive batched zstd ────────────────────────────────────────────────


_READ_MEMORY_FRACTION = 0.25


def _safe_read_batch_size(n_runs: int, sample_run_path: str, dctx) -> int:
    """Compute a safe pass-2 read batch size from cgroup memory and item size."""
    with open(sample_run_path, "rb") as raw_f:
        with dctx.stream_reader(raw_f) as f:
            try:
                batch = pickle.load(f)
            except EOFError:
                return 100
    sample = batch[:100]
    item_bytes = max(64, len(pickle.dumps(sample)) // len(sample) * 3)

    # cgroup-aware memory limit
    available = 0
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                val = f.read().strip()
            if val != "max":
                v = int(val)
                if v < 2**62:
                    available = v
                    break
        except (FileNotFoundError, ValueError):
            pass
    if not available:
        import psutil

        available = psutil.virtual_memory().total

    budget = int(available * _READ_MEMORY_FRACTION)
    size = budget // max(1, n_runs * item_bytes)
    result = max(100, min(size, _IMPROVED_BATCH_SIZE))
    logger.info(
        "Adaptive batch_size: %d runs x ~%d bytes/item, budget=%.1f GB -> batch_size=%d",
        n_runs,
        item_bytes,
        budget / 1e9,
        result,
    )
    return result


def external_sort_adaptive(
    iters_gen: Iterator[Iterator],
    tmp_dir: str,
) -> Iterator[dict]:
    """ADAPTIVE: fixed write batch, memory-safe read batch computed from cgroup limit."""
    import zstandard as zstd

    run_paths: list[str] = []
    batch_idx = 0
    item_counts: list[int] = []
    cctx = zstd.ZstdCompressor(level=3)
    dctx = zstd.ZstdDecompressor()

    # Pass-1: fixed write batch (same as improved)
    while True:
        batch = list(islice(iters_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = os.path.join(tmp_dir, f"run-{batch_idx:04d}.pkl.zst")
        n = 0
        pending: list[dict] = []
        with open(run_path, "wb") as raw_f:
            with cctx.stream_writer(raw_f, closefd=False) as f:
                for item in heapq.merge(*batch, key=lambda x: x["bucket"]):
                    pending.append(item)
                    if len(pending) >= _IMPROVED_BATCH_SIZE:
                        pickle.dump(pending, f, protocol=pickle.HIGHEST_PROTOCOL)
                        n += len(pending)
                        pending = []
                if pending:
                    pickle.dump(pending, f, protocol=pickle.HIGHEST_PROTOCOL)
                    n += len(pending)
        size_mb = os.path.getsize(run_path) / 1e6
        logger.info("[adaptive] pass-1 batch %d: %d items, %.1f MB", batch_idx, n, size_mb)
        run_paths.append(run_path)
        item_counts.append(n)
        batch_idx += 1

    # Compute safe read batch size from actual memory and run count
    read_batch_size = _safe_read_batch_size(len(run_paths), run_paths[0], dctx)

    def _read(path: str) -> Iterator[dict]:
        import zstandard as _zstd

        _dctx = _zstd.ZstdDecompressor()
        with open(path, "rb") as raw_f:
            with _dctx.stream_reader(raw_f) as f:
                while True:
                    try:
                        items: list[dict] = pickle.load(f)
                        # Yield in read_batch_size chunks; del from front so
                        # consumed items are freed even while generator is suspended.
                        while items:
                            chunk = items[:read_batch_size]
                            del items[:read_batch_size]
                            yield from chunk
                    except EOFError:
                        break

    yield from heapq.merge(*[_read(p) for p in run_paths], key=lambda x: x["bucket"])


# ── parquet approach (bucket column + pickled item column) ────────────────


def external_sort_parquet(
    iters_gen: Iterator[Iterator],
    tmp_dir: str,
    batch_size: int = _IMPROVED_BATCH_SIZE,
) -> Iterator[dict]:
    """PARQUET pass-1: bucket (string, dict-encoded) + item (binary, pickled).

    Bucket column gets parquet dictionary encoding since there are few unique
    values.  Row groups are batch_size rows, compressed with zstd.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    _schema = pa.schema([("bucket", pa.string()), ("item", pa.large_binary())])
    run_paths: list[str] = []
    batch_idx = 0

    while True:
        batch = list(islice(iters_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = os.path.join(tmp_dir, f"run-{batch_idx:04d}.parquet")
        writer = pq.ParquetWriter(run_path, _schema, compression="zstd")
        n = 0
        buckets: list[str] = []
        items_bin: list[bytes] = []
        for item in heapq.merge(*batch, key=lambda x: x["bucket"]):
            buckets.append(item["bucket"])
            items_bin.append(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))
            if len(buckets) >= batch_size:
                writer.write_batch(
                    pa.record_batch(
                        [pa.array(buckets, type=pa.string()), pa.array(items_bin, type=pa.large_binary())],
                        schema=_schema,
                    )
                )
                n += len(buckets)
                buckets = []
                items_bin = []
        if buckets:
            writer.write_batch(
                pa.record_batch(
                    [pa.array(buckets, type=pa.string()), pa.array(items_bin, type=pa.large_binary())],
                    schema=_schema,
                )
            )
            n += len(buckets)
        writer.close()
        size_mb = os.path.getsize(run_path) / 1e6
        logger.info("[parquet] pass-1 batch %d: %d items, %.1f MB", batch_idx, n, size_mb)
        run_paths.append(run_path)
        batch_idx += 1

    def _read(path: str) -> Iterator[dict]:
        import pyarrow.parquet as _pq

        pf = _pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch_size, columns=["item"]):
            for item_bytes in rb.column("item").to_pylist():
                yield pickle.loads(item_bytes)

    yield from heapq.merge(*[_read(p) for p in run_paths], key=lambda x: x["bucket"])


# ── vortex approach ───────────────────────────────────────────────────────


def external_sort_vortex(
    iters_gen: Iterator[Iterator],
    tmp_dir: str,
    batch_size: int = _IMPROVED_BATCH_SIZE,
) -> Iterator[dict]:
    """VORTEX pass-1: bucket (string) + item (binary, pickled), written via vortex.

    Uses vortex's native compression pipeline on the bucket column.
    """
    import pyarrow as pa
    import vortex

    run_paths: list[str] = []
    batch_idx = 0

    _vschema = pa.schema([("bucket", pa.string()), ("item", pa.large_binary())])

    while True:
        batch = list(islice(iters_gen, EXTERNAL_SORT_FAN_IN))
        if not batch:
            break
        run_path = os.path.join(tmp_dir, f"run-{batch_idx:04d}.vortex")
        n = 0

        # Stream via RecordBatchReader so vortex writes without materialising.
        def _record_batch_gen(fan_in_batch=batch):
            nonlocal n
            buckets: list[str] = []
            items_bin: list[bytes] = []
            for item in heapq.merge(*fan_in_batch, key=lambda x: x["bucket"]):
                buckets.append(item["bucket"])
                items_bin.append(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL))
                if len(buckets) >= batch_size:
                    yield pa.record_batch(
                        [pa.array(buckets, type=pa.string()), pa.array(items_bin, type=pa.large_binary())],
                        schema=_vschema,
                    )
                    n += len(buckets)
                    buckets = []
                    items_bin = []
            if buckets:
                yield pa.record_batch(
                    [pa.array(buckets, type=pa.string()), pa.array(items_bin, type=pa.large_binary())],
                    schema=_vschema,
                )
                n += len(buckets)

        reader = pa.RecordBatchReader.from_batches(_vschema, _record_batch_gen())
        vortex.io.write(reader, run_path)
        size_mb = os.path.getsize(run_path) / 1e6
        logger.info("[vortex] pass-1 batch %d: %d items, %.1f MB", batch_idx, n, size_mb)
        run_paths.append(run_path)
        batch_idx += 1

    def _read(path: str) -> Iterator[dict]:
        arr = vortex.io.read_url(f"file://{path}", projection=["item"])
        for rb in arr.to_arrow_table().to_batches(max_chunksize=batch_size):
            for item_bytes in rb.column("item").to_pylist():
                yield pickle.loads(item_bytes)

    yield from heapq.merge(*[_read(p) for p in run_paths], key=lambda x: x["bucket"])


# ── runner ─────────────────────────────────────────────────────────────────


def run_benchmark(
    approach: str,
    mem_limit_gb: float | None,
    n_iters: int = N_ITERS,
    items_per_iter: int = ITEMS_PER_ITER,
) -> dict:
    total_items = n_iters * items_per_iter
    logger.info(
        "=== %s | %d iters x %d items = %d total items ===",
        approach.upper(),
        n_iters,
        items_per_iter,
        total_items,
    )

    tmp_dir = tempfile.mkdtemp(prefix=f"ext_sort_{approach}_")
    sampler = MemorySampler(limit_gb=mem_limit_gb)
    sampler.start()
    t0 = time.perf_counter()

    try:
        iters_gen = (make_sorted_iterator(i, items_per_iter) for i in range(n_iters))

        if approach == "old":
            sort_iter = external_sort_old(iters_gen, tmp_dir)
        elif approach == "current":
            sort_iter = external_sort_current(iters_gen, tmp_dir)
        elif approach == "improved":
            sort_iter = external_sort_improved(iters_gen, tmp_dir)
        elif approach == "zstd-stream":
            sort_iter = external_sort_zstd_stream(iters_gen, tmp_dir)
        elif approach == "adaptive":
            sort_iter = external_sort_adaptive(iters_gen, tmp_dir)
        elif approach == "parquet":
            sort_iter = external_sort_parquet(iters_gen, tmp_dir)
        else:
            sort_iter = external_sort_vortex(iters_gen, tmp_dir)

        count = sum(1 for _ in sort_iter)
    finally:
        sampler.stop()
        run_files = [f for f in os.listdir(tmp_dir) if f.startswith("run-")]
        storage_mb = sum(os.path.getsize(os.path.join(tmp_dir, f)) for f in run_files) / 1e6
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    peak_gb = sampler.peak_gb()

    result = {
        "approach": approach,
        "items": count,
        "elapsed_s": round(elapsed, 1),
        "peak_mem_gb": round(peak_gb, 2),
        "storage_mb": round(storage_mb, 1),
        "throughput_mps": round(count / elapsed / 1e6, 2),
    }
    logger.info(
        "RESULT %s: items=%d elapsed=%.1fs peak_mem=%.2fGB storage=%.1fMB throughput=%.2f M/s",
        approach,
        count,
        elapsed,
        peak_gb,
        storage_mb,
        result["throughput_mps"],
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--approach",
        choices=["old", "current", "improved", "zstd-stream", "adaptive", "parquet", "vortex", "all"],
        default="all",
        help="Which approach to benchmark (default: all — skips 'old' OOM test)",
    )
    ap.add_argument("--include-old", action="store_true", help="Include old OOM test when --approach all")
    ap.add_argument(
        "--n-iters",
        type=int,
        default=N_ITERS,
        help=f"Number of iterators (default: {N_ITERS})",
    )
    ap.add_argument(
        "--items-per-iter",
        type=int,
        default=ITEMS_PER_ITER,
        help=f"Items per iterator (default: {ITEMS_PER_ITER})",
    )
    args = ap.parse_args()

    # Allow overriding data sizes
    n_iters = args.n_iters
    items_per_iter = args.items_per_iter

    mem_limit = _cgroup_limit_gb()
    logger.info("Memory limit: %.1f GB", mem_limit)
    logger.info(
        "Data: %d iters x %d items = %dM items, ~%.1f GB raw",
        n_iters,
        items_per_iter,
        n_iters * items_per_iter // 1_000_000,
        n_iters * items_per_iter * 150 / 1e9,
    )

    if args.approach == "all":
        approaches = ["current", "improved", "zstd-stream", "adaptive", "parquet", "vortex"]
        if args.include_old:
            approaches = ["old", *approaches]
    else:
        approaches = [args.approach]

    results = []
    for ap_name in approaches:
        try:
            results.append(
                run_benchmark(ap_name, mem_limit_gb=mem_limit, n_iters=n_iters, items_per_iter=items_per_iter)
            )
        except SystemExit as e:
            if e.code == 137:
                logger.error("%s OOMed as expected (exit 137)", ap_name)
                results.append({"approach": ap_name, "items": 0, "error": "OOM"})
            else:
                raise

    print("\n── SUMMARY ──────────────────────────────────────────────────")
    print(f"{'approach':<10} {'items':>10} {'elapsed':>10} {'peak_mem':>10} {'storage':>10} {'M/s':>6}")
    print("-" * 62)
    for r in results:
        if "error" in r:
            print(f"{r['approach']:<10} {'OOM':>10}")
        else:
            print(
                f"{r['approach']:<10} {r['items']:>10,} "
                f"{r['elapsed_s']:>9.1f}s {r['peak_mem_gb']:>9.2f}G "
                f"{r['storage_mb']:>9.1f}M {r['throughput_mps']:>5.2f}M/s"
            )


if __name__ == "__main__":
    main()
