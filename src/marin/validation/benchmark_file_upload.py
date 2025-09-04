import argparse
import json
import os
import time
import uuid
from collections.abc import Iterable

import fsspec


def parse_size_to_bytes(size_str: str) -> int:
    """Parse a human-friendly size string into bytes.

    Supports suffixes: B, KB, MB, GB (case-insensitive). If no suffix, assume MB.
    Examples: "1MB", "100MB", "1GB", "256", "1048576B".
    """
    s = size_str.strip().upper()
    multiplier = 1024**2  # default MB
    if s.endswith("B"):
        if s.endswith("KB"):
            multiplier = 1024
            s = s[:-2]
        elif s.endswith("MB"):
            multiplier = 1024**2
            s = s[:-2]
        elif s.endswith("GB"):
            multiplier = 1024**3
            s = s[:-2]
        elif s.endswith("B") and not s.endswith(("KB", "MB", "GB")):
            multiplier = 1
            s = s[:-1]
    else:
        # No suffix -> assume MB
        multiplier = 1024**2
    return int(float(s)) * multiplier


def iter_random_bytes(total_size_bytes: int, chunk_size: int = 8 * 1024 * 1024) -> Iterable[bytes]:
    """Yield random bytes until total_size_bytes is produced.

    Uses os.urandom in chunked fashion to avoid holding data in memory.
    """
    bytes_remaining = total_size_bytes
    while bytes_remaining > 0:
        to_produce = min(chunk_size, bytes_remaining)
        yield os.urandom(to_produce)
        bytes_remaining -= to_produce


def timed_upload(url: str, total_size_bytes: int, chunk_size: int) -> float:
    """Upload random data of size total_size_bytes to url, returning seconds elapsed."""
    start = time.perf_counter()
    with fsspec.open(url, "wb") as f:
        for chunk in iter_random_bytes(total_size_bytes, chunk_size):
            f.write(chunk)
        f.flush()
    end = time.perf_counter()
    return end - start


def timed_download(url: str, chunk_size: int) -> tuple[float, int]:
    """Download data from url, discarding it, returning (seconds elapsed, total bytes read)."""
    total = 0
    start = time.perf_counter()
    with fsspec.open(url, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            total += len(buf)
    end = time.perf_counter()
    return end - start, total


def mb_per_sec(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return (num_bytes / (1024 * 1024)) / seconds


def benchmark_transfer(
    base_path: str,
    sizes_bytes: list[int],
    repeats: int = 1,
    chunk_size: int = 8 * 1024 * 1024,
    cleanup: bool = True,
) -> dict[str, dict[str, float]]:
    """Run upload and download benchmarks for each requested size.

    Returns a dict mapping human size labels to metrics.
    """
    results: dict[str, dict[str, float]] = {}
    unique_run_id = uuid.uuid4().hex[:8]

    for size in sizes_bytes:
        size_label = f"{round(size / (1024 * 1024))}MB" if size < 1024**3 else f"{round(size / (1024 ** 3), 2)}GB"
        filename = f"bench_{unique_run_id}_{size}.bin"
        url = base_path.rstrip("/") + "/" + filename

        upload_times: list[float] = []
        download_times: list[float] = []

        # Repeat measurements
        for _ in range(repeats):
            upload_sec = timed_upload(url, size, chunk_size)
            upload_times.append(upload_sec)

            download_sec, downloaded_bytes = timed_download(url, chunk_size)
            # Guard: remote may change content length representation; trust read bytes
            download_times.append(download_sec)

        # Aggregate
        avg_upload = sum(upload_times) / len(upload_times)
        avg_download = sum(download_times) / len(download_times)
        results[size_label] = {
            "bytes": float(size),
            "upload_sec_avg": float(avg_upload),
            "download_sec_avg": float(avg_download),
            "upload_MBps_avg": float(mb_per_sec(size, avg_upload)),
            "download_MBps_avg": float(mb_per_sec(size, avg_download)),
            "repeats": float(repeats),
        }

        # Cleanup artifact
        if cleanup:
            try:
                fs, path_in_fs = fsspec.core.url_to_fs(url)
                if fs.exists(url):
                    fs.rm(url)
            except Exception:
                pass

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark upload/download throughput using fsspec.")
    parser.add_argument(
        "--base-path",
        required=True,
        help=("Destination base path (e.g., gs://bucket/prefix, s3://bucket/prefix, or local dir like /tmp/bench)."),
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        default=["1MB", "100MB", "256MB", "512MB", "1GB"],
        help="List of sizes to test. Accepts B, KB, MB, GB suffixes. Default: 1MB 100MB 256MB 512MB 1GB",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per size (averaged).",
    )
    parser.add_argument(
        "--chunk-size",
        type=str,
        default="8MB",
        help="Chunk size for streaming reads/writes (default: 8MB).",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not delete benchmark files after completion.",
    )
    args = parser.parse_args()

    sizes_bytes = [parse_size_to_bytes(s) for s in args.sizes]
    chunk_size = parse_size_to_bytes(args.chunk_size)
    cleanup = not args.no_cleanup

    # Ensure local directories exist if using a local path
    try:
        fs, _ = fsspec.core.url_to_fs(args.base_path)
        if args.base_path.startswith("/"):
            # Local filesystem target directory creation
            os.makedirs(args.base_path, exist_ok=True)
    except Exception:
        pass

    results = benchmark_transfer(
        base_path=args.base_path,
        sizes_bytes=sizes_bytes,
        repeats=args.repeats,
        chunk_size=chunk_size,
        cleanup=cleanup,
    )

    print(
        json.dumps(
            {
                "base_path": args.base_path,
                "chunk_size_bytes": chunk_size,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
