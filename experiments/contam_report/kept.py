# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find which candidate documents the hero store kept (Gate 5).

The store ``store_4d2e363d`` drops a normalized document when its hero
decontamination mark has ``contaminated`` set, when the verified fuzzy-dedup
attributes list it (except for fuzzy-exempt sources), or when the exact-dedup
attributes list it. Each attribute file has the basename of its normalized file.

Input: FILTERS_JSON, ``{source: {"dir", "decon", "exact", "fuzzy"}}`` with the
normalized directory and each attribute directory (``fuzzy`` is null for exempt
sources), and REQUESTS_JSON, a list of ``{"source", "path", "id"}``. Writes one
Parquet file per normalized file under OUT_PREFIX with ``path``, ``id``,
``contaminated``, ``exact_dup``, ``fuzzy_dup`` and ``kept``.

Usage: kept.py FILTERS_JSON REQUESTS_JSON OUT_PREFIX [--max-workers N] [--worker-ram R]
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_file
from zephyr.writers import write_parquet_file

SCHEMA = pa.schema(
    [
        ("path", pa.string()),
        ("id", pa.string()),
        ("contaminated", pa.bool_()),
        ("exact_dup", pa.bool_()),
        ("fuzzy_dup", pa.bool_()),
        ("kept", pa.bool_()),
    ]
)


def listed_ids(path: str | None, wanted: set[str]) -> set[str]:
    """Return the wanted IDs a sparse duplicate file lists; a missing file lists none."""
    if path is None or not StoragePath(path).exists():
        return set()
    return {str(r["id"]) for r in load_file(InputFileSpec(path=path, columns=["id"])) if str(r["id"]) in wanted}


def make_checker(out_prefix: str):
    def check(groups, shard):
        for group in groups:
            path, wanted = group["path"], set(group["ids"])
            out_path = f"{out_prefix}/{hashlib.sha1(path.encode()).hexdigest()[:16]}.parquet"
            if StoragePath(out_path).exists():
                continue
            contaminated = {
                str(r["id"]): bool(r["contaminated"])
                for r in load_file(InputFileSpec(path=group["decon"], columns=["id", "contaminated"]))
                if str(r["id"]) in wanted
            }
            if missing := wanted - contaminated.keys():
                raise ValueError(f"{path}: {len(missing)} ids missing from {group['decon']}")
            exact = listed_ids(group["exact"], wanted)
            fuzzy = listed_ids(group["fuzzy"], wanted)
            rows = [
                {
                    "path": path,
                    "id": doc_id,
                    "contaminated": contaminated[doc_id],
                    "exact_dup": doc_id in exact,
                    "fuzzy_dup": doc_id in fuzzy,
                    "kept": not (contaminated[doc_id] or doc_id in exact or doc_id in fuzzy),
                }
                for doc_id in sorted(wanted)
            ]
            yield write_parquet_file(rows, output_path=out_path, schema=SCHEMA)

    return check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filters")
    parser.add_argument("requests")
    parser.add_argument("out_prefix")
    parser.add_argument("--max-workers", type=int, default=100)
    parser.add_argument("--worker-ram", default="4g")
    args = parser.parse_args()
    with StoragePath(args.filters).open("r") as f:
        filters = json.load(f)
    with StoragePath(args.requests).open("r") as f:
        requests = json.load(f)
    by_path: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in requests:
        by_path[(r["source"], r["path"])].append(r["id"])
    groups = []
    for (source, path), ids in sorted(by_path.items()):
        attrs = filters[source]
        base = os.path.basename(path)
        if path != f"{attrs['dir']}/{base}":
            raise ValueError(f"{path} is not in {source}'s normalized directory {attrs['dir']}")
        groups.append(
            {
                "path": path,
                "ids": ids,
                "decon": f"{attrs['decon']}/{base}",
                "exact": f"{attrs['exact']}/{base}",
                "fuzzy": f"{attrs['fuzzy']}/{base}" if attrs["fuzzy"] else None,
            }
        )
    resources = ResourceConfig(cpu=1, ram=args.worker_ram)
    ctx = ZephyrContext(name="contam-kept", resources=resources, max_workers=args.max_workers)
    ctx.execute(Dataset.from_list(groups).map_shard(make_checker(args.out_prefix)))


if __name__ == "__main__":
    main()
