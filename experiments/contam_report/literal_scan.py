# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search hero training documents for the literal patterns of `literal_patterns.py` (gate 22).

Each batch of documents is first filtered with one RE2 alternation of every
pattern (`pyarrow.compute.match_substring_regex`, case-insensitive), which is
fast in C++; only matching documents are searched pattern by pattern in Python.
Writes, per input file, ``<OUT_PREFIX>/matches/<sha>.parquet`` (``path``, ``id``,
``pattern_id``, ``count``) and ``<OUT_PREFIX>/stats/<sha>.json`` (documents and
bytes read); a file with stats is skipped on resume.

``--every N`` scans every Nth file of each source, for a pilot.

Usage: literal_scan.py PLAN_JSON PATTERNS_JSON OUT_PREFIX [--every N] [--max-workers N] [--worker-ram R]
"""

import argparse
import hashlib
import json
import re

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.writers import write_parquet_file

BATCH_ROWS = 1024
SCHEMA = pa.schema([("path", pa.string()), ("id", pa.string()), ("pattern_id", pa.int64()), ("count", pa.int64())])


def file_key(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:16]


def document_matches(text: str, patterns: list[tuple[int, re.Pattern]]) -> list[tuple[int, int]]:
    out = []
    for pattern_id, pattern in patterns:
        count = len(pattern.findall(text))
        if count:
            out.append((pattern_id, count))
    return out


def make_searcher(patterns_path: str, out_prefix: str):
    def run(items, shard):
        with StoragePath(patterns_path).open("r") as f:
            specs = json.load(f)
        combined = "|".join(f"(?:{s['regex']})" for s in specs)
        patterns = [(s["pattern_id"], re.compile(s["regex"], re.IGNORECASE)) for s in specs]
        for item in items:
            path = item["path"]
            key = file_key(path)
            stats_path = f"{out_prefix}/stats/{key}.json"
            if StoragePath(stats_path).exists():
                continue
            stats = {"path": path, "docs": 0, "bytes": 0, "match_docs": 0}

            def rows(path: str, stats: dict):
                fs, local = filesystem_for(path)
                with fs.open(local, "rb") as fh:
                    for batch in pq.ParquetFile(fh).iter_batches(batch_size=BATCH_ROWS, columns=["id", "text"]):
                        texts = batch.column("text")
                        stats["docs"] += len(batch)
                        stats["bytes"] += pc.sum(pc.binary_length(texts)).as_py() or 0
                        mask = pc.fill_null(pc.match_substring_regex(texts, combined, ignore_case=True), False)
                        if not pc.any(mask).as_py():
                            continue
                        hit = batch.filter(mask)
                        for doc_id, text in zip(
                            hit.column("id").to_pylist(), hit.column("text").to_pylist(), strict=True
                        ):
                            found = document_matches(text, patterns)
                            stats["match_docs"] += bool(found)
                            for pattern_id, count in found:
                                yield {"path": path, "id": str(doc_id), "pattern_id": pattern_id, "count": count}

            yield write_parquet_file(rows(path, stats), output_path=f"{out_prefix}/matches/{key}.parquet", schema=SCHEMA)
            with StoragePath(stats_path).open("w") as f:
                json.dump(stats, f)

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("patterns")
    parser.add_argument("out_prefix")
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=500)
    parser.add_argument("--worker-ram", default="8g")
    args = parser.parse_args()
    with StoragePath(args.plan).open("r") as f:
        plan = json.load(f)
    items = [{"source": s, "path": p} for s, paths in sorted(plan.items()) for p in paths[:: args.every]]
    ctx = ZephyrContext(
        name="contam-literal-scan",
        resources=ResourceConfig(cpu=1, ram=args.worker_ram),
        max_workers=args.max_workers,
    )
    ctx.execute(Dataset.from_list(items).map_shard(make_searcher(args.patterns, args.out_prefix)))


if __name__ == "__main__":
    main()
