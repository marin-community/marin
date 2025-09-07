import argparse
import json
import math
import time
import uuid
import ray

import fsspec
import hashlib

from marin.processing.classification.inference import (
    AsyncJSONLWriter,
    write_dataset_streaming,
)


def parse_size_to_bytes(size_str: str) -> int:
    s = size_str.strip().upper()
    multiplier = 1024**2
    if s.endswith("KB"):
        multiplier = 1024
        s = s[:-2]
    elif s.endswith("MB"):
        multiplier = 1024**2
        s = s[:-2]
    elif s.endswith("GB"):
        multiplier = 1024**3
        s = s[:-2]
    elif s.endswith("B"):
        multiplier = 1
        s = s[:-1]
    return int(float(s)) * multiplier


def estimate_jsonl_row_bytes(row: dict) -> int:
    return len(json.dumps(row)) + 1


def build_batches(
    total_target_bytes: int,
    row_bytes: int,
    batch_rows: int,
) -> tuple[int, int, list[int]]:
    sample_row = {"text": "x" * row_bytes, "id": 0}
    approx_row_len = estimate_jsonl_row_bytes(sample_row)
    total_rows = math.ceil(total_target_bytes / max(1, approx_row_len))
    num_batches = math.ceil(total_rows / batch_rows)
    batch_sizes: list[int] = []
    remaining = total_rows
    for _ in range(num_batches):
        take = min(batch_rows, remaining)
        batch_sizes.append(take)
        remaining -= take
    return total_rows, num_batches, batch_sizes


def make_rows(start_id: int, count: int, row_bytes: int) -> list[dict]:
    payload = "x" * row_bytes
    return [{"id": start_id + i, "text": payload} for i in range(count)]


def run_sync(output_url: str, batch_sizes: list[int], row_bytes: int) -> float:
    start = time.perf_counter()
    offset = 0
    for idx, size in enumerate(batch_sizes):
        rows = make_rows(offset, size, row_bytes)
        write_dataset_streaming(iter(rows), output_url, append=idx > 0)
        offset += size
    end = time.perf_counter()
    return end - start


def run_async(output_url: str, batch_sizes: list[int], row_bytes: int) -> float:
    start = time.perf_counter()
    writer = AsyncJSONLWriter(output_url, append=False)
    offset = 0
    for size in batch_sizes:
        rows = make_rows(offset, size, row_bytes)
        writer.submit_rows(rows)
        offset += size
    writer.close()
    end = time.perf_counter()
    return end - start


def get_remote_size(url: str) -> int:
    fs, _ = fsspec.core.url_to_fs(url)
    if fs.exists(url):
        return int(fs.size(url))
    return 0


def mbps(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (num_bytes / (1024 * 1024)) / seconds


def benchmark(
    base_path: str,
    size_bytes: int,
    row_bytes: int,
    batch_rows: int,
    repeats: int,
    ext: str,
    cleanup: bool,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    uid = uuid.uuid4().hex[:8]

    total_rows, num_batches, batch_sizes = build_batches(size_bytes, row_bytes, batch_rows)

    def run_variant(variant: str) -> tuple[float, float, int]:
        times: list[float] = []
        last_size = 0
        for r in range(repeats):
            filename = f"bench_async_{variant}_{uid}_{r}.{ext}"
            url = base_path.rstrip("/") + "/" + filename
            if variant == "sync":
                elapsed = run_sync(url, batch_sizes, row_bytes)
            else:
                elapsed = run_async(url, batch_sizes, row_bytes)
            times.append(elapsed)
            last_size = get_remote_size(url)
            if cleanup:
                try:
                    fs, _ = fsspec.core.url_to_fs(url)
                    if fs.exists(url):
                        fs.rm(url)
                except Exception:
                    pass
        avg_time = sum(times) / len(times)
        return avg_time, mbps(last_size, avg_time), last_size

    sync_time, sync_mbps, bytes_written = run_variant("sync")
    async_time, async_mbps, _ = run_variant("async")

    results["sync"] = {
        "seconds": float(sync_time),
        "MBps": float(sync_mbps),
        "bytes_written": float(bytes_written),
        "batches": float(num_batches),
        "rows": float(total_rows),
    }
    results["async"] = {
        "seconds": float(async_time),
        "MBps": float(async_mbps),
        "speedup_x": float(sync_time / async_time if async_time > 0 else float("inf")),
    }

    # Correctness check: write one file with sync and one with async, compare bytes and sha256
    sync_url = base_path.rstrip("/") + "/" + f"bench_async_correct_sync_{uid}.{ext}"
    async_url = base_path.rstrip("/") + "/" + f"bench_async_correct_async_{uid}.{ext}"

    run_sync(sync_url, batch_sizes, row_bytes)
    run_async(async_url, batch_sizes, row_bytes)

    def sha256sum(url: str, decompressed: bool = True) -> tuple[str, int]:
        h = hashlib.sha256()
        total = 0
        # When decompressed=True, read logical content using compression inference
        # so gzip multi-member vs single-stream produce identical bytes.
        open_kwargs = {"compression": "infer"} if decompressed else {}
        with fsspec.open(url, "rb", **open_kwargs) as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                total += len(chunk)
        return h.hexdigest(), total

    # Compute hashes of decompressed logical content for correctness
    sync_hash, sync_bytes = sha256sum(sync_url, decompressed=True)
    async_hash, async_bytes = sha256sum(async_url, decompressed=True)
    equal = (sync_bytes == async_bytes) and (sync_hash == async_hash)

    results["correctness"] = {
        "equal_bytes_and_hash": bool(equal),
        "sync_bytes": float(sync_bytes),  # decompressed logical bytes
        "async_bytes": float(async_bytes),  # decompressed logical bytes
        "sync_sha256": sync_hash,
        "async_sha256": async_hash,
    }

    # IDs-based correctness: ensure both files contain the same set of ids
    def collect_ids(url: str) -> list[int]:
        ids: list[int] = []
        # Read as text with compression inference to safely iterate lines
        with fsspec.open(url, "rt", compression="infer", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                obj = json.loads(line)
                if "id" in obj:
                    ids.append(obj["id"])
        return ids

    sync_ids = set(collect_ids(sync_url))
    async_ids = set(collect_ids(async_url))
    ids_equal = sync_ids == async_ids

    missing_in_async = list(sync_ids - async_ids)
    missing_in_sync = list(async_ids - sync_ids)

    results["correctness_ids"] = {
        "ids_equal": bool(ids_equal),
        "sync_unique_ids": float(len(sync_ids)),
        "async_unique_ids": float(len(async_ids)),
        "missing_in_async_count": float(len(missing_in_async)),
        "missing_in_sync_count": float(len(missing_in_sync)),
        "missing_in_async_examples": missing_in_async[:10],
        "missing_in_sync_examples": missing_in_sync[:10],
    }

    if cleanup:
        for url in (sync_url, async_url):
            try:
                fs, _ = fsspec.core.url_to_fs(url)
                if fs.exists(url):
                    fs.rm(url)
            except Exception:
                pass
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark async JSONL writer vs sync append uploads.")
    parser.add_argument("--base-path", required=True, help="Destination base path (local dir or remote fs URI).")
    parser.add_argument("--size", default="256MB", help="Approximate target output size (e.g., 256MB, 1GB).")
    parser.add_argument("--row-bytes", type=int, default=1024, help="Payload bytes per row (default: 1024).")
    parser.add_argument("--batch-rows", type=int, default=10000, help="Rows per batch append (default: 10000).")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per variant to average.")
    parser.add_argument("--ext", default="jsonl", choices=["jsonl", "jsonl.gz", "jsonl.zst"], help="File ext.")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep output files for inspection.")
    parser.add_argument("--ray", action="store_true", help="Run benchmark in Ray.")
    args = parser.parse_args()

    size_bytes = parse_size_to_bytes(args.size)

    if args.ray:
        benchmark_remote = ray.remote(benchmark)
        results = ray.get(
            benchmark_remote.options(
                num_cpus=1,
                memory=16 * 1024 * 1024 * 1024,
            ).remote(
                base_path=args.base_path,
                size_bytes=size_bytes,
                row_bytes=args.row_bytes,
                batch_rows=args.batch_rows,
                repeats=args.repeats,
                ext=args.ext,
                cleanup=not args.no_cleanup,
            )
        )
    else:
        results = benchmark(
            base_path=args.base_path,
            size_bytes=size_bytes,
            row_bytes=args.row_bytes,
            batch_rows=args.batch_rows,
            repeats=args.repeats,
            ext=args.ext,
            cleanup=not args.no_cleanup,
        )

    print(
        json.dumps(
            {
                "base_path": args.base_path,
                "size_target_bytes": size_bytes,
                "row_bytes": args.row_bytes,
                "batch_rows": args.batch_rows,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
