"""Record every target-eval Bloom hit in hero training documents (Gates 3 and 4).

For each input Parquet file this writes, under ``<out>/<source>/``:

* ``hits/<file>``: one row per document with at least one hit: ``id``, the
  distinct hit hashes, the document's feature count, its highest paragraph
  overlap, and ``marked`` (the hero rule: a paragraph with overlap >= 0.5 and
  >= 2 distinct hits).
* ``stats/<file>.json``: documents, features and hit documents in the file.

Features and hashes come from `marin.datakit.decon`, so they match the Bloom
that `build_bloom.py` built. Run as an Iris job; see `launch_scan.sh`.

Usage: scan.py PLAN_JSON OUT_PREFIX BLOOM_DIR [--max-workers N]
"""

import argparse
import json
import logging
import os
from typing import Any

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.decon import (
    _bloom_hash,
    _extract_ngrams,
    _iter_paragraphs,
    _load_bloom,
    _short_exact_feature_from_tokens,
    bloom_paths,
)
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

NGRAM_LENGTH = 13
OVERLAP_THRESHOLD = 0.5
MIN_DISTINCT_HITS = 2
PARAGRAPH_DELIMITER = "\n\n"

HITS_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("hits", pa.list_(pa.uint64())),
        ("n_features", pa.int64()),
        ("max_overlap", pa.float32()),
        ("marked", pa.bool_()),
    ]
)


def paragraph_features(paragraph: str) -> list[str]:
    tokens = paragraph.split()
    short = _short_exact_feature_from_tokens(tokens, NGRAM_LENGTH)
    if short is not None:
        return [short]
    return list(_extract_ngrams(paragraph, NGRAM_LENGTH, 0))


def document_hits(text: str, bloom) -> tuple[set[int], int, float, bool]:
    hits: set[int] = set()
    n_features = 0
    max_overlap = 0.0
    marked = False
    for paragraph in _iter_paragraphs(text, PARAGRAPH_DELIMITER):
        features = paragraph_features(paragraph)
        if not features:
            continue
        n_features += len(features)
        para_hits = [h for h in map(_bloom_hash, features) if h in bloom]
        if not para_hits:
            continue
        overlap = len(para_hits) / len(features)
        max_overlap = max(max_overlap, overlap)
        distinct = set(para_hits)
        hits |= distinct
        if overlap >= OVERLAP_THRESHOLD and (len(distinct) >= MIN_DISTINCT_HITS or len(features) == 1):
            marked = True
    return hits, n_features, max_overlap, marked


def make_scanner(bloom_path: str, out_prefix: str):
    def scan(items, shard) -> Any:
        bloom = _load_bloom(bloom_path)
        for item in items:
            source, path = item["source"], item["path"]
            name = os.path.basename(path)
            stats_path = f"{out_prefix}/{source}/stats/{name}.json"
            if StoragePath(stats_path).exists():
                yield {"source": source, "path": path, "skipped": True}
                continue
            stats = {"source": source, "path": path, "docs": 0, "features": 0, "hit_docs": 0, "marked_docs": 0}

            def rows():
                for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                    hits, n_features, max_overlap, marked = document_hits(str(record.get("text") or ""), bloom)
                    stats["docs"] += 1
                    stats["features"] += n_features
                    counters.pipeline.update_counter("scan/docs", 1)
                    if not hits:
                        continue
                    stats["hit_docs"] += 1
                    stats["marked_docs"] += marked
                    counters.pipeline.update_counter("scan/hit_docs", 1)
                    yield {
                        "id": str(record["id"]),
                        "hits": sorted(hits),
                        "n_features": n_features,
                        "max_overlap": max_overlap,
                        "marked": marked,
                    }

            write_parquet_file(rows(), output_path=f"{out_prefix}/{source}/hits/{name}", schema=HITS_SCHEMA)
            # The stats file is written last: its presence marks the file done, so a rerun skips it.
            with StoragePath(stats_path).open("w") as f:
                json.dump(stats, f)
            yield stats

    return scan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("out_prefix")
    parser.add_argument("bloom_dir")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--worker-ram", default="8g")
    args = parser.parse_args()

    with StoragePath(args.plan).open("r") as f:
        plan = json.load(f)
    items = [{"source": s, "path": p} for s, paths in plan.items() for p in paths]
    logger.info("scan: %d files from %d sources", len(items), len(plan))
    bloom_path, _ = bloom_paths(args.bloom_dir)
    resources = ResourceConfig(cpu=1, ram=args.worker_ram)
    ctx = ZephyrContext(name="contam-scan", resources=resources, max_workers=args.max_workers)
    outcome = ctx.execute(Dataset.from_list(items).map_shard(make_scanner(bloom_path, args.out_prefix)))
    logger.info("scan done: %s", dict(outcome.counters))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
