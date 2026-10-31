# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch the part of a training document that matches an eval item, for reading by hand.

Input: a JSON list of ``{"path", "decon_path", "id", "eval_id", "item"}``, where
``path`` is the normalized Parquet file that holds document ``id``, ``decon_path``
the hero decontamination attributes for that file, and ``item`` the eval item's
text. Writes one Parquet file per input file under OUT_PREFIX with, per request:
the document length, the character offset of the first item 13-gram found in
the document (-1 if none), a window of text around it, and ``hero_contaminated``,
the hero mark that made the store drop the document. Run as an Iris job, next to
the data.

Usage: excerpts.py REQUESTS_JSON OUT_PREFIX [--max-workers N] [--worker-ram R]
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

NGRAM_LENGTH = 13
WINDOW_BEFORE = 1_000
WINDOW_AFTER = 4_000
TOKEN = re.compile(r"\S+")
SCHEMA = pa.schema(
    [
        ("eval_id", pa.string()),
        ("path", pa.string()),
        ("id", pa.string()),
        ("doc_len", pa.int64()),
        ("match_offset", pa.int64()),
        ("window", pa.string()),
        ("hero_contaminated", pa.bool_()),
    ]
)


def first_match_offset(doc: str, item: str) -> int:
    """Return the character offset in ``doc`` of the first item n-gram it contains, or -1."""
    item_tokens = item.split()
    n = min(NGRAM_LENGTH, len(item_tokens))
    item_ngrams = {tuple(item_tokens[i : i + n]) for i in range(len(item_tokens) - n + 1)}
    matches = list(TOKEN.finditer(doc))
    tokens = [m.group() for m in matches]
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i : i + n]) in item_ngrams:
            return matches[i].start()
    return -1


def make_fetcher(out_prefix: str):
    def fetch(groups, shard):
        for group in groups:
            path, wanted = group["path"], group["wanted"]
            out_path = f"{out_prefix}/{hashlib.sha1(path.encode()).hexdigest()[:16]}.parquet"
            if StoragePath(out_path).exists():
                continue
            hero = {
                str(r["id"]): bool(r["contaminated"])
                for r in load_file(InputFileSpec(path=group["decon_path"], columns=["id", "contaminated"]))
                if str(r["id"]) in wanted
            }

            def rows():
                for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                    doc_id = str(record["id"])
                    if doc_id not in wanted:
                        continue
                    doc = str(record["text"])
                    for eval_id, item in wanted[doc_id]:
                        offset = first_match_offset(doc, item)
                        start = max(0, offset - WINDOW_BEFORE) if offset >= 0 else 0
                        yield {
                            "eval_id": eval_id,
                            "path": path,
                            "id": doc_id,
                            "doc_len": len(doc),
                            "match_offset": offset,
                            "window": doc[start : max(offset, 0) + WINDOW_AFTER],
                            "hero_contaminated": hero.get(doc_id),
                        }

            yield write_parquet_file(rows(), output_path=out_path, schema=SCHEMA)

    return fetch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests")
    parser.add_argument("out_prefix")
    parser.add_argument("--max-workers", type=int, default=50)
    parser.add_argument("--worker-ram", default="16g")
    args = parser.parse_args()
    with StoragePath(args.requests).open("r") as f:
        requests = json.load(f)
    by_path: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    decon = {}
    for r in requests:
        by_path[r["path"]][r["id"]].append((r["eval_id"], r["item"]))
        decon[r["path"]] = r["decon_path"]
    groups = [{"path": p, "decon_path": decon[p], "wanted": dict(w)} for p, w in sorted(by_path.items())]
    resources = ResourceConfig(cpu=1, ram=args.worker_ram)
    ctx = ZephyrContext(name="contam-excerpts", resources=resources, max_workers=args.max_workers)
    ctx.execute(Dataset.from_list(groups).map_shard(make_fetcher(args.out_prefix)))


if __name__ == "__main__":
    main()
