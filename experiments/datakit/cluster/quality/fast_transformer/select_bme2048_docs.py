# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the bme2048 regrade pool and cut its 2048-token grading windows.

The first scale-up graded 512-token windows and cut every begin window at the
window boundary with no marker, so the grader read the cut as damage: 36.5% of
new-document begin windows came back invalid, with the rationales blaming the
cut. This campaign regrades a fixed per-type sample of the already-labeled
corpus under a wider window and the rubric's excerpt-marker convention
(:func:`label_windows_openrouter.window_user_content` appends it to a begin
window whose document continues past ``token_end``).

The document pool is the ~212k documents that already carry a GLM-5.2 label and
a stored harrier embedding — the 88k join plus the scale-up mining joins — so
nothing is mined and no join is rebuilt. Selection draws
``--docs-per-type`` documents per content type, hash-ordered under a fixed seed
so the draw is reproducible and independent of shard order, after excluding the
seed-0 holdout ids: those documents are the evaluation set and their labels must
never become training data.

Three stages, each skipped when its output exists:

``census``   read ``(id, content_type, source)`` from every join shard; the
             scale-up pool's type is the oracle's own, taken from its begin
             window's grade (majority across the document's windows when no
             begin grade exists).
``select``   draw the per-type sample and write the selection manifest.
``cut``      one Zephyr task per shard holding selected documents: re-read their
             text, cut the :data:`~bme_windows.GEOMETRY_2048` windows with the
             parity-gated gemma tokenizer, and write the windows the labeling
             driver consumes.
"""

import argparse
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO

import fsspec
import polars as pl
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    GEOMETRY_2048,
    check_gigatoken_parity,
    doc_windows,
    encode_documents,
)
from experiments.datakit.cluster.quality.fast_transformer.embed_exp import holdout_id_set
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import walk_parquet
from experiments.datakit.cluster.quality.fast_transformer.rubric import CONTENT_TYPES

logger = logging.getLogger(__name__)

QUALITY_V2 = "s3://marin-us-east-02a/marin/user/muchanem/quality_v2"
LEGACY_JOINED = f"{QUALITY_V2}/glm52_labels_88k-x-harrier-oss-v1-0.6b-50m-text-v1"
SCALEUP_JOINED = f"{QUALITY_V2}/glm52_labels_scaleup-x-harrier-oss-v1-0.6b-50m-text-v1"
SCALEUP_WINDOW_LABELS = f"{QUALITY_V2}/glm52_labels_scaleup/labels/windows.parquet"
# The label parquet whose seed-0 permutation defines the evaluation holdout.
HOLDOUT_LABELS = "s3://marin-us-east-02a/marin/user/rav/quality_v2/glm52_labels_88k.parquet"

DOCS_PER_TYPE = 20_000
SELECT_SEED = 0
CENSUS_COLUMNS = ["id", "glm52_content_type", "glm52_source"]
CUT_COLUMNS = ["id", "text", "glm52_source"]
# Census reads three small columns per shard; the cost is object-store round
# trips, so read shards concurrently rather than one at a time.
CENSUS_THREADS = 16
MAX_WORKERS = 48
CUT_RESOURCES = ResourceConfig(cpu=4, ram="16g")
PARITY_DOCS = 256

LEGACY_POOL = "legacy88k"
SCALEUP_POOL = "scaleup"
BEGIN = "begin"


@dataclass(frozen=True)
class Shortfall:
    """A content type that could not fill its quota, and what it had."""

    content_type: str
    wanted: int
    available: int


def _write_parquet(fs: fsspec.AbstractFileSystem, frame: pl.DataFrame, path: str) -> None:
    buf = BytesIO()
    frame.write_parquet(buf)
    fs.pipe_file(path, buf.getvalue())


def census(joined_dir: str, pool: str) -> pl.DataFrame:
    """Every joined document of one pool: id, its shard, its label type and source."""
    root = joined_dir.rstrip("/")
    shards = walk_parquet(f"{root}/outputs")
    if not shards:
        raise ValueError(f"no parquet shards under {root}/outputs/")

    def read(shard: str) -> pl.DataFrame:
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=CENSUS_COLUMNS)
        return pl.from_arrow(table).with_columns(pl.lit(shard).alias("shard"), pl.lit(pool).alias("pool"))

    with ThreadPoolExecutor(max_workers=CENSUS_THREADS) as threads:
        frames = list(threads.map(read, shards))
    rows = pl.concat(frames).rename({"glm52_content_type": "content_type", "glm52_source": "source"})
    logger.info("census: %s has %d rows over %d shards", pool, rows.height, len(shards))
    return rows


def scaleup_types(window_labels: str) -> dict[str, str]:
    """Each scale-up document's oracle content type, from its window grades.

    The begin window's verdict is the document's type — the convention the
    scale-up join itself used. A document graded only on its middle/end windows
    falls back to the majority type across the windows it does have, ties broken
    by rubric order so the result does not depend on row order.
    """
    with StoragePath(window_labels).open("rb") as fh:
        table = pq.ParquetFile(fh).read(columns=["id", "window", "content_type"])
    begin: dict[str, str] = {}
    votes: dict[str, Counter] = {}
    for doc_id, window, content_type in zip(
        table.column("id").to_pylist(),
        table.column("window").to_pylist(),
        table.column("content_type").to_pylist(),
        strict=True,
    ):
        if window == BEGIN:
            begin.setdefault(doc_id, content_type)
        votes.setdefault(doc_id, Counter())[content_type] += 1
    resolved = {
        doc_id: begin.get(doc_id) or max(counter.items(), key=lambda kv: (kv[1], -CONTENT_TYPES.index(kv[0])))[0]
        for doc_id, counter in votes.items()
    }
    logger.info(
        "scale-up types: %d documents (%d from a begin grade, %d from a window majority)",
        len(resolved),
        len(begin),
        len(resolved) - len(begin),
    )
    return resolved


def order_key(doc_id: str, seed: int) -> bytes:
    """Stable per-document sort key: the draw is a hash order, not a shuffle."""
    return hashlib.blake2b(f"{seed}:{doc_id}".encode(), digest_size=8).digest()


def select_docs(
    pools: pl.DataFrame, holdout: set[str], docs_per_type: int, seed: int
) -> tuple[pl.DataFrame, list[Shortfall]]:
    """Draw ``docs_per_type`` documents of each content type, hash-ordered.

    A document present in both pools is kept once, from the pool that listed it
    first (the legacy join), and holdout ids are dropped before the draw so a
    short type reports a shortfall rather than silently pulling evaluation
    documents in.
    """
    pool = (
        pools.filter(~pl.col("id").is_in(list(holdout)))
        .unique(subset="id", keep="first", maintain_order=True)
        .with_columns(pl.col("id").map_elements(lambda i: order_key(i, seed), return_dtype=pl.Binary).alias("_order"))
    )
    logger.info("select: %d unique non-holdout documents in the pool", pool.height)

    picks: list[pl.DataFrame] = []
    shortfalls: list[Shortfall] = []
    for content_type in CONTENT_TYPES:
        available = pool.filter(pl.col("content_type") == content_type).sort("_order")
        take = min(docs_per_type, available.height)
        if take < docs_per_type:
            shortfalls.append(Shortfall(content_type, docs_per_type, available.height))
        picks.append(available.head(take))
        logger.info("select: %s available=%d taken=%d", content_type, available.height, take)
    return pl.concat(picks).drop("_order"), shortfalls


def cut_shard(task: dict) -> dict:
    """Cut the 2048-token windows of one shard's selected documents."""
    fs = fsspec.filesystem("s3")
    out_path = task["out_path"]
    if fs.exists(out_path):
        return {"shard": task["shard"], "skipped": True}
    with fs.open(f"{task['out']}/selected_docs.parquet", "rb") as fh:
        selected = pl.read_parquet(fh).filter(pl.col("shard") == task["shard"])
    wanted = set(selected.get_column("id").to_list())

    with fs.open(task["shard"], "rb", cache_type="none") as fh:
        pf = pq.ParquetFile(fh)
        parts = []
        for group in range(pf.metadata.num_row_groups):
            frame = pl.from_arrow(pf.read_row_group(group, columns=CUT_COLUMNS))
            hits = frame.filter(pl.col("id").is_in(wanted))
            if hits.height:
                parts.append(hits)
    if not parts:
        return {"shard": task["shard"], "docs": 0, "windows": 0}
    # A document id can repeat inside a shard; the selection counts it once.
    rows = pl.concat(parts).unique(subset="id", keep="first", maintain_order=True)
    types = dict(zip(selected.get_column("id").to_list(), selected.get_column("content_type").to_list(), strict=True))

    token_ids = encode_documents(rows.get_column("text").to_list())
    windows = []
    for doc_id, source, ids in zip(
        rows.get_column("id").to_list(), rows.get_column("glm52_source").to_list(), token_ids, strict=True
    ):
        for window in doc_windows(ids, GEOMETRY_2048):
            windows.append(
                {
                    "id": doc_id,
                    "source": source,
                    "window": window.position,
                    "token_start": window.token_start,
                    "token_end": window.token_end,
                    "text": window.text,
                    "doc_tokens": len(ids),
                    "content_type": types[doc_id],
                    "pool": task["pool"],
                }
            )
    _write_parquet(fs, pl.DataFrame(windows), out_path)
    return {"shard": task["shard"], "docs": rows.height, "windows": len(windows)}


def parity_gate(shard: str) -> None:
    """Gate the run on gigatoken reproducing HF gemma ids on this corpus's own text."""
    with StoragePath(shard).open("rb") as fh:
        texts = pq.ParquetFile(fh).read_row_group(0, columns=["text"]).column("text").to_pylist()
    check_gigatoken_parity(texts[:PARITY_DOCS], seed=SELECT_SEED)


def build_selection(fs: fsspec.AbstractFileSystem, out: str, docs_per_type: int, holdout_labels: str) -> None:
    """Census both pools, resolve types, draw the sample, write the manifest."""
    selected_path = f"{out}/selected_docs.parquet"
    if fs.exists(selected_path):
        logger.info("select: reusing %s", selected_path)
        return
    legacy = census(LEGACY_JOINED, LEGACY_POOL)
    scaleup = census(SCALEUP_JOINED, SCALEUP_POOL)
    oracle_types = scaleup_types(SCALEUP_WINDOW_LABELS)
    # The scale-up join carries the begin grade already; re-resolving from the
    # window labels also types documents the join could only give a null type.
    scaleup = scaleup.with_columns(
        pl.col("id").replace_strict(oracle_types, default=None).fill_null(pl.col("content_type")).alias("content_type")
    )
    pools = pl.concat([legacy, scaleup])

    holdout = holdout_id_set(holdout_labels)
    logger.info("select: %d holdout ids excluded from the pool", len(holdout))
    selected, shortfalls = select_docs(pools, holdout, docs_per_type, SELECT_SEED)
    _write_parquet(fs, selected, selected_path)

    manifest = {
        "docs_per_type_target": docs_per_type,
        "seed": SELECT_SEED,
        "window_tokens": GEOMETRY_2048.window_tokens,
        "long_doc_tokens": GEOMETRY_2048.long_doc_tokens,
        "holdout_labels": holdout_labels,
        "holdout_ids": len(holdout),
        "pools": {LEGACY_POOL: LEGACY_JOINED, SCALEUP_POOL: SCALEUP_JOINED},
        "pool_docs": {LEGACY_POOL: legacy.height, SCALEUP_POOL: scaleup.height},
        "selected_docs": selected.height,
        "selected_by_type": dict(sorted(Counter(selected.get_column("content_type").to_list()).items())),
        "selected_by_pool": dict(sorted(Counter(selected.get_column("pool").to_list()).items())),
        "shortfalls": [
            {"content_type": s.content_type, "wanted": s.wanted, "available": s.available} for s in shortfalls
        ],
    }
    fs.pipe_file(f"{out}/selection.json", json.dumps(manifest, indent=2).encode())
    logger.info("select: %s", json.dumps(manifest, indent=2))


def cut_windows(fs: fsspec.AbstractFileSystem, out: str) -> None:
    """Fan the window cut over the shards holding selected documents."""
    with fs.open(f"{out}/selected_docs.parquet", "rb") as fh:
        selected = pl.read_parquet(fh, columns=["shard", "pool"])
    shards = sorted(set(zip(selected.get_column("shard").to_list(), selected.get_column("pool").to_list(), strict=True)))
    parity_gate(shards[0][0])
    tasks = [
        {
            "shard": shard,
            "pool": pool,
            "out": out,
            # Flat output names, so the labeling driver's single-level glob finds
            # every part however deep the joins nest their own shards.
            "out_path": f"{out}/windows/{pool}-{index:05d}.parquet",
        }
        for index, (shard, pool) in enumerate(shards)
    ]
    logger.info("cut: %d shards hold selected documents", len(tasks))
    outcome = ZephyrContext(
        name="bme2048-cut-windows",
        resources=CUT_RESOURCES,
        max_workers=MAX_WORKERS,
    ).execute(Dataset.from_list(tasks).map(cut_shard), verbose=True)
    docs = sum(r.get("docs", 0) for r in outcome.results)
    windows = sum(r.get("windows", 0) for r in outcome.results)
    logger.info("cut: %d docs, %d windows over %d shards -> %s/windows/", docs, windows, len(tasks), out)


def summarize(out: str) -> dict:
    """Per-type and per-position window counts, read back from the cut output."""
    frames = []
    for path in sorted(str(p) for p in StoragePath(f"{out}/windows/*.parquet").glob()):
        with StoragePath(path).open("rb") as fh:
            frames.append(pl.read_parquet(fh, columns=["id", "window", "content_type", "doc_tokens"]))
    windows = pl.concat(frames)
    long_docs = windows.filter(pl.col("window") == BEGIN).filter(pl.col("doc_tokens") >= GEOMETRY_2048.long_doc_tokens)
    return {
        "windows": windows.height,
        "docs": windows.get_column("id").n_unique(),
        "windows_by_position": dict(sorted(Counter(windows.get_column("window").to_list()).items())),
        "windows_by_type": dict(sorted(Counter(windows.get_column("content_type").to_list()).items())),
        "long_docs": long_docs.height,
        "cut_begin_windows": (
            windows.filter((pl.col("window") == BEGIN) & (pl.col("doc_tokens") > GEOMETRY_2048.window_tokens)).height
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output prefix for the selection manifest and windows")
    parser.add_argument("--docs-per-type", type=int, default=DOCS_PER_TYPE)
    parser.add_argument("--holdout-labels", default=HOLDOUT_LABELS, help="label parquet defining the holdout id set")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    out = args.out.rstrip("/")
    fs = fsspec.filesystem("s3")
    build_selection(fs, out, args.docs_per_type, args.holdout_labels)
    cut_windows(fs, out)
    summary = summarize(out)
    fs.pipe_file(f"{out}/windows_summary.json", json.dumps(summary, indent=2).encode())
    logger.info("bme2048 windows: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
