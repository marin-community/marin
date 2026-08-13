# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off: join GLM-5.2 quality labels onto the harrier-oss 50M sample.

Stages a slim (text-dropped, ``glm52_``-renamed) copy of
``glm52_labels_88k.parquet``, then fans one Zephyr task per dataset shard.
Each task probes the shard's ``id`` column against the labels for the shard's
source directory (labels are stratified per source dir, usually one shard per
source), full-reads only shards with hits, inner-joins with polars on ``id``,
and writes matched rows co-partitioned under ``OUT/outputs/<source>/<basename>``.

Per-task I/O byte/time counters are emitted as zephyr counters and returned as
per-shard stats; the driver aggregates them into ``OUT/_join_stats.json``.
"""

import json
import logging
import os
import socket
import time
from io import BytesIO

import fsspec
import polars as pl
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

LABELS_URL = "s3://marin-us-east-02a/marin/user/rav/quality_v2/glm52_labels_88k.parquet"
BASE_URL = "s3://marin-us-east-02a/marin/datakit/samples/harrier-oss-v1-0.6b-50m-text-v1"
OUT_URL = "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_88k-x-harrier-oss-v1-0.6b-50m-text-v1"
SLIM_LABELS_URL = f"{OUT_URL}/_labels_slim.parquet"
STATS_URL = f"{OUT_URL}/_join_stats.json"

# Label columns renamed so they cannot collide with the dataset's own
# quality/source columns. ``text`` is dropped (the dataset carries the text).
LABEL_RENAME = {
    c: f"glm52_{c}"
    for c in ["source", "quality", "score_normalized", "content_type", "valid", "v0_score", "label_batch"]
}

MAX_WORKERS = 32
WORKER_RESOURCES = ResourceConfig(cpu=4, ram="16g")


class CountingFile:
    """File wrapper counting bytes returned by read() and time spent in it."""

    def __init__(self, inner):
        self._inner = inner
        self.bytes_read = 0
        self.read_time = 0.0

    def read(self, *args):
        t0 = time.monotonic()
        data = self._inner.read(*args)
        self.read_time += time.monotonic() - t0
        self.bytes_read += len(data)
        return data

    def __getattr__(self, name):
        return getattr(self._inner, name)


# Per-process lazy cache for the slim label table. Plain global (not
# functools.cache) so cloudpickle serializes ``process_shard`` by value:
# lru_cache wrappers pickle by reference to ``__main__`` and break on workers.
_SLIM_LABELS: pl.DataFrame | None = None


def _slim_labels() -> pl.DataFrame:
    global _SLIM_LABELS
    if _SLIM_LABELS is None:
        with fsspec.filesystem("s3").open(SLIM_LABELS_URL, "rb") as f:
            _SLIM_LABELS = pl.read_parquet(f)
    return _SLIM_LABELS


def process_shard(relpath: str) -> dict:
    """Probe one dataset shard for label ids; join and write matches if any."""
    fs = fsspec.filesystem("s3")
    labels = _slim_labels()
    source_dir = os.path.dirname(relpath)
    source_labels = labels.filter(pl.col("glm52_source") == source_dir)

    raw = fs.open(f"{BASE_URL}/{relpath}", "rb", cache_type="none")
    cf = CountingFile(raw)
    pf = pq.ParquetFile(cf)
    rows_in = pf.metadata.num_rows

    hit_ids: list[str] = []
    if source_labels.height:
        shard_ids = pl.from_arrow(pf.read(columns=["id"]))
        hit_ids = (
            shard_ids.filter(pl.col("id").is_in(source_labels.get_column("id").to_list()))
            .get_column("id")
            .unique()
            .to_list()
        )
    probe_bytes, probe_time = cf.bytes_read, cf.read_time

    matched = None
    if hit_ids:
        parts = []
        for rg in range(pf.metadata.num_row_groups):
            df = pl.from_arrow(pf.read_row_group(rg))
            sub = df.filter(pl.col("id").is_in(hit_ids))
            if sub.height:
                parts.append(sub)
        matched = pl.concat(parts).join(source_labels, on="id", how="inner")
        matched = matched.with_columns(pl.lit(relpath).alias("shard"))
    read_bytes, read_time = cf.bytes_read, cf.read_time
    cf.close()

    write_bytes = 0
    write_time = 0.0
    if matched is not None:
        buf = BytesIO()
        matched.write_parquet(buf)
        data = buf.getvalue()
        write_bytes = len(data)
        t0 = time.monotonic()
        fs.pipe_file(f"{OUT_URL}/outputs/{relpath}", data)
        write_time = time.monotonic() - t0

    stage = counters.pipeline
    stage.update_counter("join/rows_in", rows_in)
    stage.update_counter("join/rows_matched", matched.height if matched is not None else 0)
    stage.update_counter("join/bytes_read", read_bytes)
    stage.update_counter("join/read_time_ms", int(read_time * 1000))
    stage.update_counter("join/bytes_written", write_bytes)
    stage.update_counter("join/write_time_ms", int(write_time * 1000))
    stage.update_counter("join/shards_scanned", 1)
    stage.update_counter("join/shards_joined", 1 if matched is not None else 0)

    return {
        "shard": relpath,
        "source": source_dir,
        "rows_in": rows_in,
        "labels_for_source": source_labels.height,
        "rows_matched": matched.height if matched is not None else 0,
        "probe_bytes": probe_bytes,
        "probe_time": probe_time,
        "read_bytes": read_bytes,
        "read_time": read_time,
        "write_bytes": write_bytes,
        "write_time": write_time,
        "worker": f"{socket.gethostname()}:{os.getpid()}",
    }


def stage_slim_labels() -> tuple[int, int]:
    """Write the text-free, renamed label table next to the output. Returns (rows, bytes)."""
    fs = fsspec.filesystem("s3")
    with fs.open(LABELS_URL, "rb") as f:
        cols = [c for c in pq.ParquetFile(f).schema_arrow.names if c != "text"]
        f.seek(0)
        labels = pl.from_arrow(pq.read_table(f, columns=cols))
    slim = labels.rename(LABEL_RENAME)
    buf = BytesIO()
    slim.write_parquet(buf)
    fs.pipe_file(SLIM_LABELS_URL, buf.getvalue())
    return slim.height, len(buf.getvalue())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    fs = fsspec.filesystem("s3")
    base_key = BASE_URL.removeprefix("s3://")
    shard_paths = sorted(fs.glob(f"{base_key}/**/*.parquet"))
    relpaths = [p[len(base_key) + 1 :] for p in shard_paths]
    logger.info("Found %d dataset shards", len(relpaths))

    n_labels, slim_bytes = stage_slim_labels()
    logger.info("Staged %d slim labels (%d bytes) to %s", n_labels, slim_bytes, SLIM_LABELS_URL)

    ds = Dataset.from_list(relpaths).map(process_shard)
    ctx = ZephyrContext(
        resources=WORKER_RESOURCES,
        max_workers=MAX_WORKERS,
        name="join-glm52-harrier50m",
    )
    t0 = time.monotonic()
    outcome = ctx.execute(ds, verbose=True)
    wall = time.monotonic() - t0

    stats = sorted(outcome.results, key=lambda s: s["shard"])
    total_matched = sum(s["rows_matched"] for s in stats)
    total_read = sum(s["read_bytes"] for s in stats)
    total_read_time = sum(s["read_time"] for s in stats)
    total_written = sum(s["write_bytes"] for s in stats)
    total_write_time = sum(s["write_time"] for s in stats)
    summary = {
        "labels_total": n_labels,
        "shards_scanned": len(stats),
        "shards_with_matches": sum(1 for s in stats if s["rows_matched"]),
        "rows_in_total": sum(s["rows_in"] for s in stats),
        "rows_matched_total": total_matched,
        "bytes_read_total": total_read,
        "read_time_total": total_read_time,
        "read_mb_per_s_task_serial": (total_read / 1e6 / total_read_time) if total_read_time else 0.0,
        "bytes_written_total": total_written,
        "write_time_total": total_write_time,
        "write_mb_per_s_task_serial": (total_written / 1e6 / total_write_time) if total_write_time else 0.0,
        "pipeline_wall_time": wall,
        "aggregate_read_mb_per_s": total_read / 1e6 / wall,
        "max_workers": MAX_WORKERS,
        "worker_resources": {"cpu": WORKER_RESOURCES.cpu, "ram": WORKER_RESOURCES.ram},
        "counters": {k: v for k, v in sorted(outcome.counters.items())},
    }
    fs.pipe_file(STATS_URL, json.dumps({"summary": summary, "shards": stats}, indent=2).encode())
    logger.info("Join complete: %s", json.dumps(summary, indent=2))
    logger.info("Wrote stats to %s", STATS_URL)


if __name__ == "__main__":
    main()
