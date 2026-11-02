# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run `aggregate.py` in an Iris job, next to the scan output (Gate 5).

The full scan's hits are ~300k small files and the aggregation needs more memory
than a dev worker has. This job copies the hits and the Bloom index to local
disk, aggregates once per common-feature limit, and uploads each result to
``OUT_PREFIX/limit_<N>/``. With ``--drop-url`` it also drops the decon drop sets
there, as the decon item rule does (see `aggregate.py`).

Usage: aggregate_job.py BLOOM_URL OUT_PREFIX HITS_URL... --common-doc-limit N [N ...] [--drop-url URL]
"""

import argparse
import os
import tempfile
import time

from rigging.filesystem.buckets import filesystem_for

from experiments.contam_report.aggregate import aggregate

FETCH_BATCH = 256
FETCH_CHUNK = 20_000
OUTPUTS = ("items.parquet", "doc_item.parquet", "hash_docs.parquet")
INDEX = "_bloom/eval_hash_index.parquet"


def fetch_tree(url: str, local_dir: str) -> None:
    fs, root = filesystem_for(url)
    remote = fs.find(root)
    pairs = [(r, os.path.join(local_dir, r[len(root) + 1 :])) for r in remote]
    for _, local in pairs:
        os.makedirs(os.path.dirname(local), exist_ok=True)
    start = time.time()
    for i in range(0, len(pairs), FETCH_CHUNK):
        chunk = pairs[i : i + FETCH_CHUNK]
        fs.get([r for r, _ in chunk], [lp for _, lp in chunk], batch_size=FETCH_BATCH)
        print(f"{url}: {i + len(chunk)}/{len(pairs)} files after {time.time() - start:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bloom_url")
    parser.add_argument("out_prefix")
    parser.add_argument("hits_urls", nargs="+")
    parser.add_argument("--common-doc-limit", type=int, nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="40GB")
    parser.add_argument("--drop-url")
    args = parser.parse_args()
    work = tempfile.mkdtemp(prefix="contam-aggregate-")
    bloom_dir = os.path.join(work, "bloom")
    os.makedirs(os.path.dirname(os.path.join(bloom_dir, INDEX)))
    fs, path = filesystem_for(f"{args.bloom_url.rstrip('/')}/{INDEX}")
    fs.get(path, os.path.join(bloom_dir, INDEX))
    hits_dirs = []
    for url in args.hits_urls:
        local = os.path.join(work, "hits", os.path.basename(url.rstrip("/")))
        fetch_tree(url, local)
        hits_dirs.append(local)
    drop_dir = None
    if args.drop_url:
        drop_dir = os.path.join(work, "drop")
        fetch_tree(args.drop_url, drop_dir)
    for limit in args.common_doc_limit:
        out_dir = os.path.join(work, f"limit_{limit}")
        os.makedirs(out_dir)
        start = time.time()
        summary = aggregate(bloom_dir, out_dir, hits_dirs, limit, args.threads, args.memory_limit, drop_dir)
        print(f"limit {limit} after {time.time() - start:.0f}s\n{summary}", flush=True)
        out_fs, out_path = filesystem_for(f"{args.out_prefix.rstrip('/')}/limit_{limit}")
        for name in OUTPUTS:
            out_fs.put(os.path.join(out_dir, name), f"{out_path}/{name}")
        with out_fs.open(f"{out_path}/summary.txt", "w") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
