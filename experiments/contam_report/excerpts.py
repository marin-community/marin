# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch the text of chosen training documents, for reading matches by hand.

Input: a JSON list of ``{"path", "decon_path", "id", "eval_id"}``, where
``path`` is the normalized Parquet file that holds document ``id`` and
``decon_path`` the hero decontamination attributes for that file. Writes one
Parquet file per input file under OUT_PREFIX, with ``eval_id``, ``path``, ``id``,
``text`` (cut to ``MAX_CHARS``) and ``hero_contaminated``, the hero mark that made
the store drop the document. Run as an Iris job, next to the data.

Usage: excerpts.py REQUESTS_JSON OUT_PREFIX [--max-workers N]
"""

import argparse
import hashlib
import json
from collections import defaultdict

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

MAX_CHARS = 50_000
SCHEMA = pa.schema(
    [
        ("eval_id", pa.string()),
        ("path", pa.string()),
        ("id", pa.string()),
        ("text", pa.string()),
        ("hero_contaminated", pa.bool_()),
    ]
)


def make_fetcher(out_prefix: str):
    def fetch(groups, shard):
        for group in groups:
            path, wanted = group["path"], group["wanted"]
            hero = {
                str(r["id"]): bool(r["contaminated"])
                for r in load_file(InputFileSpec(path=group["decon_path"], columns=["id", "contaminated"]))
                if str(r["id"]) in wanted
            }

            def rows():
                for record in load_file(InputFileSpec(path=path, columns=["id", "text"])):
                    doc_id = str(record["id"])
                    for eval_id in wanted.get(doc_id, ()):
                        yield {
                            "eval_id": eval_id,
                            "path": path,
                            "id": doc_id,
                            "text": str(record["text"])[:MAX_CHARS],
                            "hero_contaminated": hero.get(doc_id),
                        }

            name = hashlib.sha1(path.encode()).hexdigest()[:16]
            yield write_parquet_file(rows(), output_path=f"{out_prefix}/{name}.parquet", schema=SCHEMA)

    return fetch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("requests")
    parser.add_argument("out_prefix")
    parser.add_argument("--max-workers", type=int, default=50)
    args = parser.parse_args()
    with StoragePath(args.requests).open("r") as f:
        requests = json.load(f)
    by_path: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    decon = {}
    for r in requests:
        by_path[r["path"]][r["id"]].append(r["eval_id"])
        decon[r["path"]] = r["decon_path"]
    groups = [{"path": p, "decon_path": decon[p], "wanted": dict(w)} for p, w in sorted(by_path.items())]
    ctx = ZephyrContext(name="contam-excerpts", resources=ResourceConfig(cpu=1, ram="4g"), max_workers=args.max_workers)
    ctx.execute(Dataset.from_list(groups).map_shard(make_fetcher(args.out_prefix)))


if __name__ == "__main__":
    main()
