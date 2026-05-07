# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a ~50 GB synthetic skewed-distribution dataset for the tier2 ferry.

Output schema mirrors starcoder2_extras: parquet with ``id`` (string) and
``content`` (string) columns. Source text is drawn from the staged FineWeb-Edu
``sample/10BT`` shards; per-doc length is sampled from a mixture:

- 75% log-normal centered at ``NORMAL_MEAN_BYTES`` (~5 KB / ~1000 GPT-2 tokens)
- 25% truncated Pareto with ``MAX_DOC_BYTES`` cap (heavy tail up to 256 MB)

Run on iris in us-central1 (where the source lives) and write to a TTL-managed
temp prefix. Output path is provided by the caller; suggested layout::

    gs://marin-us-central1/tmp/ttl=3d/datakit-tier2-skew-v1/data/part-NNNNN.parquet
"""

from __future__ import annotations

import argparse
import logging
import secrets
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO

import gcsfs
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import marin_region
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_GLOB = "marin-us-central1/raw/fineweb-edu-87f0914/sample/10BT/*.parquet"

DEFAULT_TARGET_BYTES = 100 * 1024**3  # 100 GB
DEFAULT_SHARD_BYTES = 1 * 1024**3  # 1 GB target per parquet shard
DEFAULT_HEAVY_FRAC = 0.30
DEFAULT_NORMAL_MEAN_BYTES = 5_000  # ~1000 GPT-2 tokens of English
DEFAULT_NORMAL_LOG_SIGMA = 0.5  # sigma in log-space; p10/p90 ~ 0.5x/2x the mean
DEFAULT_PARETO_ALPHA = 1.1  # heavy tail; mean is finite for alpha > 1
DEFAULT_PARETO_SCALE = 2_000  # 2 KB minimum heavy-doc size
DEFAULT_MAX_DOC_BYTES = 256 * 1024 * 1024  # 256 MB hard cap on Pareto draws
DEFAULT_MIN_DOC_BYTES = 100  # don't emit tiny degenerate docs

# Deterministic "mega" doc injection: a fixed count of docs sampled uniformly
# from a target byte range, sprinkled at random positions throughout the run.
# Guarantees the heaviest tail bucket is populated regardless of stochastic
# variance in the Pareto draws.
DEFAULT_MEGA_COUNT = 100
DEFAULT_MEGA_MIN_BYTES = 128 * 1024 * 1024  # 128 MB
DEFAULT_MEGA_MAX_BYTES = 256 * 1024 * 1024  # 256 MB

PROGRESS_LOG_EVERY_N_DOCS = 50_000

OUTPUT_SCHEMA = pa.schema([("id", pa.string()), ("content", pa.string())])


@dataclass
class GenStats:
    """Mutable counters for the generation loop.

    Fields prefixed ``shard_`` reset on each shard rollover; the rest are
    cumulative across the whole run. Bundled into one struct so the inner loop
    doesn't carry ten parallel locals.
    """

    shard_idx: int = 0
    shard_bytes: int = 0
    shard_doc_count: int = 0
    total_docs: int = 0
    total_bytes: int = 0
    heavy_count: int = 0
    mega_idx: int = 0
    mega_emitted: int = 0
    largest_doc: int = 0

    def reset_shard(self) -> None:
        self.shard_idx += 1
        self.shard_bytes = 0
        self.shard_doc_count = 0


def stream_source_text(fs: gcsfs.GCSFileSystem, paths: list[str]) -> Iterator[bytes]:
    """Yield UTF-8 source text bytes from a list of parquet files, cycling indefinitely.

    Reads one row group at a time so peak memory stays bounded regardless of
    source size. Empty/null text values are skipped.
    """
    while True:
        for p in paths:
            with fs.open(p, "rb") as f:
                pf = pq.ParquetFile(f)
                for rg_idx in range(pf.num_row_groups):
                    tbl = pf.read_row_group(rg_idx, columns=["text"])
                    for v in tbl.column("text").to_pylist():
                        if not v:
                            continue
                        yield v.encode("utf-8") if isinstance(v, str) else v


def materialize_bytes(target_bytes: int, source: Iterator[bytes]) -> bytes:
    """Build a single doc of approximately *target_bytes* by concatenating source docs."""
    parts: list[bytes] = []
    total = 0
    while total < target_bytes:
        b = next(source)
        parts.append(b)
        total += len(b)
    blob = b"".join(parts)
    if len(blob) > target_bytes:
        blob = blob[:target_bytes]
    return blob


def sample_doc_size(
    rng: np.random.Generator,
    *,
    heavy_frac: float,
    normal_mean: float,
    normal_log_sigma: float,
    pareto_alpha: float,
    pareto_scale: float,
    max_bytes: int,
    min_bytes: int,
) -> int:
    """Sample one doc length (in bytes) from the 75/25 mixture."""
    if rng.random() < heavy_frac:
        # Pareto(alpha, scale): X = scale * U^(-1/alpha) for U ~ Uniform(0,1).
        u = rng.random()
        size = pareto_scale * (u ** (-1.0 / pareto_alpha))
    else:
        size = rng.lognormal(np.log(normal_mean), normal_log_sigma)
    return max(min_bytes, min(int(size), max_bytes))


def open_shard_writer(fs: gcsfs.GCSFileSystem, uri: str) -> tuple[IO[bytes], pq.ParquetWriter]:
    """Open a parquet writer streaming to a GCS object."""
    fh = fs.open(uri.removeprefix("gs://"), "wb")
    writer = pq.ParquetWriter(fh, OUTPUT_SCHEMA, compression="zstd")
    return fh, writer


def flush_batch(writer: pq.ParquetWriter, ids: list[str], contents: list[bytes]) -> None:
    """Write a batch of (id, content) rows. Decodes bytes -> str at write time."""
    if not ids:
        return
    tbl = pa.table(
        {
            "id": ids,
            "content": [c.decode("utf-8", errors="replace") for c in contents],
        },
        schema=OUTPUT_SCHEMA,
    )
    writer.write_table(tbl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        required=True,
        help="GCS prefix for shard output (e.g., gs://.../tier2-skew-v1/data); no trailing slash.",
    )
    parser.add_argument("--source-glob", default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--target-bytes", type=int, default=DEFAULT_TARGET_BYTES)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--heavy-frac", type=float, default=DEFAULT_HEAVY_FRAC)
    parser.add_argument("--normal-mean", type=int, default=DEFAULT_NORMAL_MEAN_BYTES)
    parser.add_argument("--normal-log-sigma", type=float, default=DEFAULT_NORMAL_LOG_SIGMA)
    parser.add_argument("--pareto-alpha", type=float, default=DEFAULT_PARETO_ALPHA)
    parser.add_argument("--pareto-scale", type=int, default=DEFAULT_PARETO_SCALE)
    parser.add_argument("--max-doc-bytes", type=int, default=DEFAULT_MAX_DOC_BYTES)
    parser.add_argument("--min-doc-bytes", type=int, default=DEFAULT_MIN_DOC_BYTES)
    parser.add_argument(
        "--mega-count",
        type=int,
        default=DEFAULT_MEGA_COUNT,
        help="Number of guaranteed huge docs to inject uniformly across the run.",
    )
    parser.add_argument(
        "--mega-min-bytes",
        type=int,
        default=DEFAULT_MEGA_MIN_BYTES,
        help="Lower bound (inclusive) for mega-doc size in bytes.",
    )
    parser.add_argument(
        "--mega-max-bytes",
        type=int,
        default=DEFAULT_MEGA_MAX_BYTES,
        help="Upper bound (inclusive) for mega-doc size in bytes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-batch-rows", type=int, default=100)
    args = parser.parse_args()

    configure_logging()
    rng = np.random.default_rng(args.seed)
    fs = gcsfs.GCSFileSystem()

    # Resolve any "{region}" placeholders in --source-glob / --output-path
    # against the iris VM's region. Lets a single submit command target
    # multiple iris regions (e.g., --region=us-central1 --region=us-central2)
    # without pre-pinning the source/output bucket at submit time.
    if "{region}" in args.source_glob or "{region}" in args.output_path:
        region = marin_region()
        if region is None:
            raise RuntimeError("Cannot resolve {region} placeholder; metadata server unreachable.")
        logger.info("Resolved {region} -> %s", region)
        args.source_glob = args.source_glob.replace("{region}", region)
        args.output_path = args.output_path.replace("{region}", region)

    src_paths_raw = sorted(fs.glob(args.source_glob))
    if not src_paths_raw:
        raise FileNotFoundError(f"No source files matched {args.source_glob}")
    src_paths = [f"gs://{p}" if not p.startswith("gs://") else p for p in src_paths_raw]
    logger.info("Source: %d parquet files (e.g. %s)", len(src_paths), src_paths[0])
    src_iter = stream_source_text(fs, src_paths)

    out_prefix = args.output_path.rstrip("/")
    if args.mega_count > 0:
        if args.mega_min_bytes > args.mega_max_bytes:
            raise ValueError("--mega-min-bytes must be <= --mega-max-bytes")
        mega_trigger_bytes = sorted(rng.uniform(0, args.target_bytes, args.mega_count).tolist())
    else:
        mega_trigger_bytes = []
    logger.info(
        "Target %.2f GB -> %s (shard ~%.2f GB, heavy_frac=%.2f, pareto_alpha=%.2f, max_doc=%.0f MB)",
        args.target_bytes / 1e9,
        out_prefix,
        args.shard_bytes / 1e9,
        args.heavy_frac,
        args.pareto_alpha,
        args.max_doc_bytes / 1e6,
    )
    logger.info(
        "Mega injection: %d docs uniformly in [%.0f, %.0f] MB",
        args.mega_count,
        args.mega_min_bytes / 1e6,
        args.mega_max_bytes / 1e6,
    )

    stats = GenStats()
    out_uri = f"{out_prefix}/part-{stats.shard_idx:05d}.parquet"
    fh, writer = open_shard_writer(fs, out_uri)
    batch_ids: list[str] = []
    batch_content: list[bytes] = []
    start = time.time()

    try:
        while stats.total_bytes < args.target_bytes:
            # Inject a mega doc when we've crossed the next pre-sampled byte
            # trigger; otherwise sample from the normal/heavy mixture.
            if stats.mega_idx < len(mega_trigger_bytes) and stats.total_bytes >= mega_trigger_bytes[stats.mega_idx]:
                target_size = int(rng.uniform(args.mega_min_bytes, args.mega_max_bytes + 1))
                stats.mega_idx += 1
                stats.mega_emitted += 1
                is_mega = True
            else:
                target_size = sample_doc_size(
                    rng,
                    heavy_frac=args.heavy_frac,
                    normal_mean=args.normal_mean,
                    normal_log_sigma=args.normal_log_sigma,
                    pareto_alpha=args.pareto_alpha,
                    pareto_scale=args.pareto_scale,
                    max_bytes=args.max_doc_bytes,
                    min_bytes=args.min_doc_bytes,
                )
                is_mega = False
            content = materialize_bytes(target_size, src_iter)
            doc_id = secrets.token_hex(8)

            batch_ids.append(doc_id)
            batch_content.append(content)
            actual = len(content)
            stats.shard_bytes += actual
            stats.shard_doc_count += 1
            stats.total_bytes += actual
            stats.total_docs += 1
            if actual > args.normal_mean * 10:
                stats.heavy_count += 1
            if actual > stats.largest_doc:
                stats.largest_doc = actual
            if is_mega:
                logger.info(
                    "mega doc emitted: %.1f MB (mega %d/%d)",
                    actual / 1e6,
                    stats.mega_emitted,
                    args.mega_count,
                )

            if (
                len(batch_ids) >= args.write_batch_rows
                or stats.shard_bytes >= args.shard_bytes
                or actual >= 16 * 1024 * 1024  # flush large docs immediately
            ):
                flush_batch(writer, batch_ids, batch_content)
                batch_ids.clear()
                batch_content.clear()

            if stats.total_docs % PROGRESS_LOG_EVERY_N_DOCS == 0:
                elapsed = time.time() - start
                rate = stats.total_bytes / 1e9 / elapsed * 60  # GB/min
                logger.info(
                    "progress: %d docs, %.2f / %.2f GB (%.1f%%), heavy>=10x=%d, megas=%d/%d, "
                    "largest=%.1f MB, %.2f GB/min",
                    stats.total_docs,
                    stats.total_bytes / 1e9,
                    args.target_bytes / 1e9,
                    100 * stats.total_bytes / args.target_bytes,
                    stats.heavy_count,
                    stats.mega_emitted,
                    args.mega_count,
                    stats.largest_doc / 1e6,
                    rate,
                )

            if stats.shard_bytes >= args.shard_bytes:
                flush_batch(writer, batch_ids, batch_content)
                batch_ids.clear()
                batch_content.clear()
                writer.close()
                fh.close()
                logger.info(
                    "shard %d closed: %s | %d docs, %.2f GB",
                    stats.shard_idx,
                    out_uri,
                    stats.shard_doc_count,
                    stats.shard_bytes / 1e9,
                )
                stats.reset_shard()
                out_uri = f"{out_prefix}/part-{stats.shard_idx:05d}.parquet"
                fh, writer = open_shard_writer(fs, out_uri)

        # Final flush
        flush_batch(writer, batch_ids, batch_content)
    finally:
        writer.close()
        fh.close()

    elapsed = time.time() - start
    logger.info(
        "DONE: %d shards, %d docs, %.2f GB total, megas=%d/%d, largest=%.1f MB, %.1f min",
        stats.shard_idx + 1,
        stats.total_docs,
        stats.total_bytes / 1e9,
        stats.mega_emitted,
        args.mega_count,
        stats.largest_doc / 1e6,
        elapsed / 60,
    )


if __name__ == "__main__":
    main()
