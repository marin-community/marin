#!/usr/bin/env python3
"""Benchmark: how fast is glob(detail=True) + parallel parquet footer reads?

Key insight: fsspec glob(detail=True) returns file sizes from the same
list-objects API call — no individual stats needed. The only extra cost
for parquet splitting is reading footers (last ~few KB of each file).
"""

import concurrent.futures
import time
from dataclasses import dataclass

import fsspec
import pyarrow.parquet as pq
from braceexpand import braceexpand
from rigging.filesystem import url_to_fs


NEMOTRON_JSONL = "gs://marin-eu-west4/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl/quality=high/kind=actual/**/*.jsonl.*"
FINEWEB_PARQUET = "gs://marin-us-central2/raw/fineweb-edu-87f0914/data/**/*.parquet"

# Only read footers for this many files (keeps benchmark fast)
FOOTER_LIMIT = 50


@dataclass
class FileInfo:
    path: str
    size: int


@dataclass
class ParquetFileInfo:
    path: str
    size: int
    num_row_groups: int
    num_rows: int


def glob_with_sizes(pattern: str) -> list[FileInfo]:
    """Glob returning (path, size) using detail=True — single list-objects call."""
    fs, _ = url_to_fs(pattern)
    protocol = fsspec.core.split_protocol(pattern)[0]

    results = []
    for expanded in braceexpand(pattern):
        detail = fs.glob(expanded, detail=True)
        for path, info in detail.items():
            full = f"{protocol}://{path}" if protocol else path
            results.append(FileInfo(path=full, size=info.get("size", 0)))
    results.sort(key=lambda f: f.path)
    return results


def _read_parquet_footer(path: str) -> ParquetFileInfo:
    """Read only the parquet footer (metadata) — no data."""
    meta = pq.read_metadata(path)
    return ParquetFileInfo(
        path=path,
        size=sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)),
        num_row_groups=meta.num_row_groups,
        num_rows=meta.num_rows,
    )


def bench_parallel_footers(files: list[str], max_workers: int) -> list[ParquetFileInfo]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_read_parquet_footer, files))


def timed(label: str):
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self.start = time.monotonic()
            return self
        def __exit__(self, *_):
            self.elapsed = time.monotonic() - self.start
            print(f"  {label}: {self.elapsed:.2f}s")
    return Timer()


def print_size_dist(files: list[FileInfo]):
    sizes = sorted(f.size for f in files)
    total_gb = sum(sizes) / 1e9
    print(f"  min: {sizes[0] / 1e6:.1f} MB, median: {sizes[len(sizes)//2] / 1e6:.1f} MB, "
          f"max: {sizes[-1] / 1e6:.1f} MB, total: {total_gb:.1f} GB")


def main():
    print("=" * 60)
    print("Zephyr filescan benchmark")
    print("=" * 60)

    # --- JSONL glob ---
    print(f"\n--- JSONL glob: nemotron hq_actual ---")
    with timed("glob+sizes") as t_jsonl:
        jsonl_files = glob_with_sizes(NEMOTRON_JSONL)
    print(f"  {len(jsonl_files)} files")
    print_size_dist(jsonl_files)

    # --- Parquet glob ---
    print(f"\n--- Parquet glob: fineweb-edu full ---")
    with timed("glob+sizes") as t_pq:
        pq_files = glob_with_sizes(FINEWEB_PARQUET)
    print(f"  {len(pq_files)} files")
    if pq_files:
        print_size_dist(pq_files)

    # --- Parquet footer reads (the only thing that needs a threadpool) ---
    if pq_files:
        subset = [f.path for f in pq_files[:FOOTER_LIMIT]]
        print(f"\n--- Parquet footer reads ({len(subset)} files) ---")
        for workers in [1, 16, 64]:
            with timed(f"footers @ {workers:>2} workers") as t:
                results = bench_parallel_footers(subset, workers)
            per_file_ms = t.elapsed / len(subset) * 1000
            print(f"    per-file: {per_file_ms:.0f}ms, throughput: {len(subset) / t.elapsed:.0f} files/s")

        # Extrapolate
        rate_64 = len(subset) / t.elapsed
        est_all = len(pq_files) / rate_64
        print(f"\n--- Extrapolated: footers for all {len(pq_files)} parquet files @ 64 workers ---")
        print(f"  ~{est_all:.1f}s")

    # --- Total ---
    print(f"\n--- Total plan-time cost with proposed approach ---")
    print(f"  JSONL (glob only):    ~{t_jsonl.elapsed:.1f}s for {len(jsonl_files)} files")
    if pq_files:
        print(f"  Parquet (glob+footer): ~{t_pq.elapsed + est_all:.1f}s for {len(pq_files)} files")
    print(f"  + one more glob for skip_existing output check")


if __name__ == "__main__":
    main()
