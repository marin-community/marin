# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join the scale-up window labels onto the mined docs, in the 88k-join layout.

The mining extract stage already wrote every selected document's full corpus
row — embedding included — co-partitioned by shard under ``<round>/docs/``, so
this join never re-reads the corpus: each task reads one docs shard, attaches
the document's begin-window verdict from the scale-up labels (the same
convention as the 88k set, whose labels are all begin-window grades), and
writes the joined rows under ``<out>/outputs/<relpath>`` mirroring the source
layout. Columns follow the ``glm52_labels_88k-x-…`` join exactly: the source's
own columns, the ``glm52_*`` label columns, and ``shard`` —
``glm52_v0_score`` is null here because the scale-up's rubric never produced a
v0 score. Documents whose begin window was dropped by the labeler are skipped
and counted.

Only newly mined documents are joined. Topped-up documents from the 88k set
already have their join rows; their new middle/end windows are window-level
training data, not document labels.
"""

import argparse
import logging
from io import BytesIO

import fsspec
import polars as pl
from fray.types import ResourceConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

MAX_WORKERS = 32
WORKER_RESOURCES = ResourceConfig(cpu=4, ram="16g")
LABEL_COLUMNS = ["id", "source", "content_type", "valid", "quality", "score_normalized", "label_batch"]

# Per-process cache of the begin-window label table, keyed by path.
_LABEL_CACHE: dict[str, pl.DataFrame] = {}


def _begin_labels(path: str) -> pl.DataFrame:
    if path not in _LABEL_CACHE:
        with fsspec.filesystem("s3").open(path, "rb") as fh:
            labels = pl.read_parquet(fh, columns=[*LABEL_COLUMNS, "window"])
        _LABEL_CACHE[path] = (
            labels.filter(pl.col("window") == "begin")
            .drop("window")
            .rename({c: f"glm52_{c}" for c in LABEL_COLUMNS if c != "id"})
        )
    return _LABEL_CACHE[path]


def join_shard(task: dict) -> dict:
    """Join one docs shard against the begin-window labels."""
    relpath = task["relpath"]
    fs = fsspec.filesystem("s3")
    out_path = f"{task['out']}/outputs/{relpath}"
    if fs.exists(out_path):
        return {"shard": relpath, "skipped": True}
    with fs.open(f"{task['docs']}/{relpath}", "rb") as fh:
        docs = pl.read_parquet(fh).drop("pred_type", "pred_conf")
    labels = _begin_labels(task["labels"])
    joined = docs.join(labels, on="id", how="inner")
    joined = joined.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("glm52_v0_score"),
        pl.lit(relpath).alias("shard"),
    )
    buf = BytesIO()
    joined.write_parquet(buf)
    fs.pipe_file(out_path, buf.getvalue())
    return {"shard": relpath, "docs": docs.height, "joined": joined.height}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", required=True, nargs="+", help="mined docs prefixes (e.g. .../mine/docs)")
    parser.add_argument("--labels", required=True, help="the scale-up window labels parquet")
    parser.add_argument("--out", required=True, help="join output prefix (rows land under <out>/outputs/)")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    fs = fsspec.filesystem("s3")
    tasks = []
    for docs in args.docs:
        docs = docs.rstrip("/")
        base_key = docs.removeprefix("s3://")
        for path in sorted(fs.glob(f"{base_key}/**/*.parquet")):
            tasks.append({"relpath": path[len(base_key) + 1 :], "docs": docs, "labels": args.labels, "out": args.out})
    logger.info("join: %d docs shards across %d rounds", len(tasks), len(args.docs))

    outcome = ZephyrContext(
        name="join-scaleup-labels",
        resources=WORKER_RESOURCES,
        max_workers=MAX_WORKERS,
    ).execute(Dataset.from_list(tasks).map(join_shard), verbose=True)
    docs = sum(r.get("docs", 0) for r in outcome.results)
    joined = sum(r.get("joined", 0) for r in outcome.results)
    logger.info("join: %d of %d docs joined (%d lacked a begin verdict) -> %s", joined, docs, docs - joined, args.out)


if __name__ == "__main__":
    main()
