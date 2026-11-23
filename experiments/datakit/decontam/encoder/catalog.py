# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The whole-catalog run: stage 1 over every shard, stage 2 scores, and the removal list.

``locate``: one row per keyed item stream of ``keys.parquet``: its key-1 hash, the key window and its
offset, and the item's question stream, so a scan can cut the excerpt around a hit.
``scan``: a Zephyr job over every shard of the sources (worker code in ``candidates.py``).
Scores come from ``catalog_score.py`` (GPU), which writes ``scores/<shard>.parquet``.
``collect``: a record is removed when one of its pairs has a mean score at or above ``--cutoff``. Writes
the removal list and counts per eval and per source next to the scores folder.
"""

import argparse
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon_keys import LOCATOR_COLUMNS, WINDOW, KeyStream, key_streams, text_stream
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.datakit.decontam.encoder.candidates import LOCATORS, Records, make_scan
from experiments.datakit.decontam.encoder.tables import (
    item_texts,
    parquet_files,
    read_frame,
    read_table,
    write_frame,
    write_table,
)

logger = logging.getLogger(__name__)


def locate(stage1: str, items_root: str, output: str, width: int = WINDOW) -> None:
    keys = read_table(prefix_join(stage1, "keys.parquet"), ["id", "kind", "key1_hash", "key1_offset"]).to_pandas()
    keys = keys[keys["key1_hash"] != 0]
    texts = item_texts(items_root)
    evals = {}
    for path in parquet_files(items_root):
        table = read_frame(path, ["id", "eval"])
        evals.update(zip(table["id"], table["eval"], strict=True))
    columns: dict[str, list] = {name: [] for name in LOCATOR_COLUMNS}
    for item, kind, key1_hash, offset in zip(
        keys["id"], keys["kind"], keys["key1_hash"], keys["key1_offset"], strict=True
    ):
        text = texts[item]
        stream = key_streams(text.question, text.choices, text.passage, width)[KeyStream(kind)]
        values = (
            item,
            kind,
            evals[item],
            int(key1_hash),
            stream[offset : offset + width],
            int(offset),
            text_stream(text.question),
            len(text.question),
        )
        for name, value in zip(LOCATOR_COLUMNS, values, strict=True):
            columns[name].append(value)
    table = pa.table(
        {
            name: pa.array(values, pa.uint64()) if name == "key1_hash" else pa.array(values)
            for name, values in columns.items()
        }
    )
    write_table(table, prefix_join(output, LOCATORS))
    logger.info("locate: %d keyed streams of %d items", table.num_rows, len(set(columns["item"])))


def catalog_shards(sources_path: str, input_root: str) -> list[str]:
    """Every shard of every source in the JSON list at *sources_path*."""
    with StoragePath(sources_path).open("r") as source:
        names = json.load(source)
    with ThreadPoolExecutor(16) as pool:
        listed = pool.map(lambda name: parquet_files(prefix_join(input_root, f"{name}/outputs/main")), names)
    return [shard for shards in listed for shard in shards]


def _removed_in(path: str, cutoff: float) -> tuple[int, pd.DataFrame]:
    """The number of scored pairs in one scores file, and its records whose best pair reaches *cutoff*."""
    frame = read_frame(path)
    best = frame.sort_values("score", ascending=False).drop_duplicates("record")
    return len(frame), best[best["score"] >= cutoff]


def collect(output: str, scores_dir: str, cutoff: float) -> None:
    """The records whose best pair in *scores_dir* reaches *cutoff*, written to ``<scores_dir>-removed.parquet``, and
    counts per eval and source in ``<scores_dir>-summary.json``."""
    with ThreadPoolExecutor(64) as pool:
        stats = list(
            pool.map(lambda path: json.loads(path.read_text()), StoragePath(prefix_join(output, "stats/*.json")).glob())
        )
        scored = list(pool.map(lambda path: _removed_in(path, cutoff), parquet_files(scores_dir)))
    shards = json.loads(StoragePath(prefix_join(output, "shards.json")).read_text())
    removed = pd.concat([part for _, part in scored], ignore_index=True)
    removed["source"] = removed["path"].str.split("/normalized/").str[1].str.split("/").str[0]
    result = scores_dir.rstrip("/")
    write_frame(removed, f"{result}-removed.parquet")
    totals: Counter = Counter()
    by_source: dict[str, Counter] = {}
    for row in stats:
        source = row["path"].split("/normalized/")[1].split("/")[0]
        counts = by_source.setdefault(source, Counter())
        for key in ("records", "chars", "candidate_records", "pairs"):
            counts[key] += row[key]
            totals[key] += row[key]
        counts["oversized"] += len(row["oversized"])
        totals["oversized"] += len(row["oversized"])
    for source, part in removed.groupby("source"):
        by_source[source]["removed_records"] = len(part)
        by_source[source]["removed_chars"] = int(part["record_chars"].sum())
    summary = {
        "cutoff": cutoff,
        "shards": len(shards),
        "shards_done": len(stats),
        "scored_pairs": sum(count for count, _ in scored),
        "totals": dict(totals),
        "removed_records": len(removed),
        "removed_chars": int(removed["record_chars"].sum()),
        "removed_by_eval": removed["eval"].value_counts().to_dict(),
        "by_source": {source: dict(counts) for source, counts in sorted(by_source.items())},
    }
    StoragePath(f"{result}-summary.json").write_text(json.dumps(summary, indent=1))
    logger.info(
        "collect: %d of %d shards; %d records removed of %d (%.4f%% of chars)",
        len(stats),
        len(shards),
        len(removed),
        totals["records"],
        100 * summary["removed_chars"] / max(totals["chars"], 1),
    )


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["locate", "scan", "collect"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage1", help="locate: stage-1 output with keys.parquet")
    parser.add_argument("--items", help="locate: the item table directory (items/*.parquet)")
    parser.add_argument("--sources", help="scan: JSON list of normalized source names, such as sources.json here")
    parser.add_argument("--input-root", help="scan: prefix that holds <source>/outputs/main")
    parser.add_argument("--files", help="scan: JSON list of shard paths, in place of every shard of --sources")
    parser.add_argument("--max-workers", type=int, default=1000, help="scan: Zephyr workers")
    parser.add_argument("--ram", default="16g", help="scan: memory per worker")
    parser.add_argument(
        "--records", type=Records, help="scan: all, short (over MAX_DOC_CHARS listed, not read), or oversized only"
    )
    parser.add_argument("--cutoff", type=float, help="collect: a record is removed at or above this mean score")
    parser.add_argument(
        "--scores",
        help="collect: scores folder (default <output>/scores); writes <scores>-removed.parquet and -summary.json",
    )
    parser.add_argument("--width", type=int, default=WINDOW)
    args = parser.parse_args()
    if args.step == "locate":
        locate(args.stage1, args.items, args.output, args.width)
    elif args.step == "collect":
        collect(args.output, args.scores or prefix_join(args.output, "scores"), args.cutoff)
    else:
        if args.records is None:
            parser.error("scan needs --records")
        if args.files:
            # A partial scan (a pilot, a rescan of a few shards) adds to the output; shards.json stays the catalog's.
            files = json.loads(StoragePath(args.files).read_text())
        else:
            files = catalog_shards(args.sources, args.input_root)
            StoragePath(prefix_join(args.output, "shards.json")).write_text(json.dumps(files))
        logger.info("scan: %d shards", len(files))
        resources = ResourceConfig(cpu=1, ram=args.ram)
        context = ZephyrContext(name="decon-encoder-catalog", resources=resources, max_workers=args.max_workers)
        outcome = context.execute(
            Dataset.from_list(files).map_shard(make_scan(args.output, args.width, args.records)),
            map_task_resources=resources,
        )
        logger.info(
            "scan done: %d shards, %d pairs", len(outcome.results), sum(r.get("pairs", 0) for r in outcome.results)
        )


if __name__ == "__main__":
    main()
