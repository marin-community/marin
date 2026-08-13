# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach harrier document embeddings to the 80k domain evaluation set.

The fusion quality scorer reads a 1024-d harrier document embedding alongside
the text, so it cannot score ``quality_v2/domain_eval/docs_v2`` — the population
every earlier quality comparison used — until that column is recovered. The
evaluation set was drawn from the same 50M sample harrier embedded, so the
embedding exists per document; it is simply not carried on the eval shards.

Fans one Zephyr task per harrier shard in a source directory the evaluation set
draws from, probes the shard's ``id`` column against the eval ids for that
source, and writes ``(id, embedding)`` for the matches under
``OUT/outputs/<source>/<basename>``. Both trees name sources by the same
directory path, so the probe never has to read a shard that cannot match.
"""

import argparse
import functools
import json
import logging
import os
from io import BytesIO

import fsspec
import polars as pl
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

EVAL_DOCS_URL = "s3://marin-us-east-02a/marin/user/rav/quality_v2/domain_eval/docs_v2"
HARRIER_URL = "s3://marin-us-east-02a/marin/datakit/samples/harrier-oss-v1-0.6b-50m-text-v1"
DEFAULT_OUT = "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/domain_eval_docs_v2-x-harrier-embeddings"

MAX_WORKERS = 32
WORKER_RESOURCES = ResourceConfig(cpu=4, ram="16g")

# Per-process cache of the staged id table. A plain global (not functools.cache)
# so cloudpickle serializes the task function by value.
_EVAL_IDS: pl.DataFrame | None = None


def _eval_ids(slim_url: str) -> pl.DataFrame:
    global _EVAL_IDS
    if _EVAL_IDS is None:
        with fsspec.filesystem("s3").open(slim_url, "rb") as fh:
            _EVAL_IDS = pl.read_parquet(fh)
    return _EVAL_IDS


def process_shard(relpath: str, *, harrier_url: str, out_url: str, slim_url: str) -> dict:
    """Probe one harrier shard for eval ids; write ``(id, embedding)`` for the hits."""
    fs = fsspec.filesystem("s3")
    source_dir = os.path.dirname(relpath)
    wanted = _eval_ids(slim_url).filter(pl.col("source") == source_dir).get_column("id")

    matched_rows = 0
    if wanted.len():
        with fs.open(f"{harrier_url}/{relpath}", "rb", cache_type="none") as raw:
            pf = pq.ParquetFile(raw)
            parts = []
            for rg in range(pf.metadata.num_row_groups):
                df = pl.from_arrow(pf.read_row_group(rg, columns=["id", "embedding"]))
                sub = df.filter(pl.col("id").is_in(wanted))
                if sub.height:
                    parts.append(sub)
        if parts:
            matched = pl.concat(parts).unique(subset=["id"])
            matched_rows = matched.height
            buf = BytesIO()
            matched.write_parquet(buf)
            fs.pipe_file(f"{out_url}/outputs/{relpath}", buf.getvalue())

    stage = counters.pipeline
    stage.update_counter("embed_join/shards_scanned", 1)
    stage.update_counter("embed_join/rows_matched", matched_rows)
    return {"shard": relpath, "source": source_dir, "eval_ids_for_source": wanted.len(), "rows_matched": matched_rows}


def stage_eval_ids(eval_docs_url: str, slim_url: str) -> tuple[int, dict[str, int]]:
    """Write the ``(id, source)`` table the workers probe with. Returns (rows, per-source counts)."""
    fs = fsspec.filesystem("s3")
    shards = sorted(fs.glob(f"{eval_docs_url.removeprefix('s3://')}/*.parquet"))
    frames = []
    for shard in shards:
        with fs.open(shard, "rb") as fh:
            frames.append(pl.from_arrow(pq.ParquetFile(fh).read(columns=["id", "source"])))
    table = pl.concat(frames).unique(subset=["id"])
    buf = BytesIO()
    table.write_parquet(buf)
    fs.pipe_file(slim_url, buf.getvalue())
    counts = dict(table.group_by("source").len().iter_rows())
    return table.height, counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-docs", default=EVAL_DOCS_URL)
    p.add_argument("--harrier", default=HARRIER_URL)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    fs = fsspec.filesystem("s3")
    out_url = args.out.rstrip("/").removeprefix("s3://")
    harrier_url = args.harrier.rstrip("/").removeprefix("s3://")
    slim_url = f"{out_url}/_eval_ids.parquet"

    rows, per_source = stage_eval_ids(args.eval_docs.rstrip("/"), slim_url)
    logger.info("staged %d eval ids across %d sources -> %s", rows, len(per_source), slim_url)

    shard_paths = sorted(fs.glob(f"{harrier_url}/**/*.parquet"))
    relpaths = [p[len(harrier_url) + 1 :] for p in shard_paths]
    covered = [r for r in relpaths if os.path.dirname(r) in per_source]
    logger.info("harrier shards: %d total, %d in evaluated sources", len(relpaths), len(covered))

    ds = Dataset.from_list(covered).map(
        functools.partial(process_shard, harrier_url=harrier_url, out_url=out_url, slim_url=slim_url)
    )
    ctx = ZephyrContext(resources=WORKER_RESOURCES, max_workers=MAX_WORKERS, name="join-evalset-harrier-embeddings")
    outcome = ctx.execute(ds, verbose=True)

    stats = sorted(outcome.results, key=lambda s: s["shard"])
    matched = sum(s["rows_matched"] for s in stats)
    by_source: dict[str, int] = {}
    for s in stats:
        by_source[s["source"]] = by_source.get(s["source"], 0) + s["rows_matched"]
    missing = {src: n - by_source.get(src, 0) for src, n in per_source.items() if n - by_source.get(src, 0) > 0}
    summary = {
        "eval_rows": rows,
        "eval_sources": len(per_source),
        "shards_scanned": len(stats),
        "rows_matched_total": matched,
        "coverage": matched / rows if rows else 0.0,
        "sources_short": missing,
        "counters": {k: v for k, v in sorted(outcome.counters.items())},
    }
    fs.pipe_file(f"{out_url}/_join_stats.json", json.dumps({"summary": summary, "shards": stats}, indent=2).encode())
    logger.info("embedding join complete: %s", json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
