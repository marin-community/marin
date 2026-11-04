# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search hero training documents for the literal patterns of `literal_patterns.py` (gate 22).

Two cheap filters keep the per-document work small:

* canaries and task names (``LITERAL_KINDS``): one RE2 alternation over the raw
  text (`pyarrow.compute.match_substring_regex`), then a substring test per
  pattern on the matching documents;
* phrases: each phrase's two longest inner words (``ANCHOR_MIN_CHARS`` or more)
  are its anchors. A verbatim copy keeps its inner words intact, so a document
  can hold the phrase only if its lowercased word set contains both anchors.
  Only then is the phrase tested on the document's lowercased text with
  whitespace (and JSON ``\\n``, ``\\t``) collapsed. Phrases without two such
  anchors are not searched.

Writes, per input file, ``<OUT_PREFIX>/matches/<sha>.parquet`` (``path``, ``id``,
``pattern_id``, ``count``) and ``<OUT_PREFIX>/stats/<sha>.json``; a file with
stats is skipped on resume. ``--every N`` scans every Nth file of each source,
for a pilot.

Usage: literal_scan.py PLAN_JSON PATTERNS_JSON OUT_PREFIX [--every N] [--max-workers N] [--worker-ram R]
"""

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass

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
ANCHOR_MIN_CHARS = 7
LITERAL_KINDS = ("canary", "task_name")
SCHEMA = pa.schema([("path", pa.string()), ("id", pa.string()), ("pattern_id", pa.int64()), ("count", pa.int64())])


@dataclass(frozen=True)
class Matcher:
    literal_regex: str
    literals: list[tuple[int, str]]
    phrases: dict[int, str]
    anchors: dict[int, tuple[str, str]]
    by_anchor: dict[str, list[int]]
    anchor_words: frozenset[str]


def phrase_anchors(phrase: str) -> tuple[str, str] | None:
    words = phrase.split()
    inner = sorted({w for w in words[1:-1] if len(w) >= ANCHOR_MIN_CHARS}, key=len, reverse=True)
    return (inner[0], inner[1]) if len(inner) >= 2 else None


def build_matcher(specs: list[dict]) -> Matcher:
    literals = [(s["pattern_id"], s["text"]) for s in specs if s["kind"] in LITERAL_KINDS]
    phrases, anchors, by_anchor = {}, {}, defaultdict(list)
    for s in specs:
        if s["kind"] in LITERAL_KINDS:
            continue
        phrase = " ".join(s["text"].lower().split())
        pair = phrase_anchors(phrase)
        if pair is None:
            continue
        phrases[s["pattern_id"]], anchors[s["pattern_id"]] = phrase, pair
        for word in pair:
            by_anchor[word].append(s["pattern_id"])
    literal_regex = "|".join(re.escape(text) for _, text in literals)
    return Matcher(literal_regex, literals, phrases, anchors, dict(by_anchor), frozenset(by_anchor))


def lowered(texts: pa.ChunkedArray | pa.Array) -> list[str | None]:
    low = pc.utf8_lower(texts)
    return pc.replace_substring(pc.replace_substring(low, "\\n", " "), "\\t", " ").to_pylist()


def phrase_counts(low: str, matcher: Matcher) -> list[tuple[int, int]]:
    present = matcher.anchor_words.intersection(low.split())
    if len(present) < 2:
        return []
    candidates = {p for word in present for p in matcher.by_anchor[word]}
    candidates = [p for p in candidates if matcher.anchors[p][0] in present and matcher.anchors[p][1] in present]
    if not candidates:
        return []
    flat = " ".join(low.split())
    return [(p, flat.count(matcher.phrases[p])) for p in candidates if matcher.phrases[p] in flat]


def batch_matches(ids: list[str], texts: pa.Array, matcher: Matcher) -> list[tuple[str, int, int]]:
    out = []
    literal_mask = pc.fill_null(pc.match_substring_regex(texts, matcher.literal_regex), False).to_pylist()
    raw = texts.to_pylist()
    for doc_id, text, low, has_literal in zip(ids, raw, lowered(texts), literal_mask, strict=True):
        if not text:
            continue
        if has_literal:
            out += [(doc_id, p, text.count(lit)) for p, lit in matcher.literals if lit in text]
        out += [(doc_id, p, n) for p, n in phrase_counts(low, matcher)]
    return out


def file_key(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()[:16]


def make_searcher(patterns_path: str, out_prefix: str):
    def run(items, shard):
        with StoragePath(patterns_path).open("r") as f:
            matcher = build_matcher(json.load(f))
        for item in items:
            path = item["path"]
            key = file_key(path)
            stats_path = f"{out_prefix}/stats/{key}.json"
            if StoragePath(stats_path).exists():
                continue
            stats = {"path": path, "docs": 0, "bytes": 0, "seconds": 0.0}
            start = time.time()

            def rows(path: str, stats: dict):
                fs, local = filesystem_for(path)
                with fs.open(local, "rb") as fh:
                    for batch in pq.ParquetFile(fh).iter_batches(batch_size=BATCH_ROWS, columns=["id", "text"]):
                        texts = batch.column("text")
                        stats["docs"] += len(batch)
                        stats["bytes"] += pc.sum(pc.binary_length(texts)).as_py() or 0
                        ids = [str(i) for i in batch.column("id").to_pylist()]
                        for doc_id, pattern_id, count in batch_matches(ids, texts, matcher):
                            yield {"path": path, "id": doc_id, "pattern_id": pattern_id, "count": count}

            yield write_parquet_file(rows(path, stats), output_path=f"{out_prefix}/matches/{key}.parquet", schema=SCHEMA)
            stats["seconds"] = time.time() - start
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
