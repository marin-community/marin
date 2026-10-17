# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split a family's pool tasks (held-out included) into shards for the #212 full run.

Each shard is one ``harbor_glm.py`` job with its own output prefix, so a failed job loses only its
shard. Shards are contiguous slices of the sorted task ids. Writes ``<out-dir>/<family>-shard-NN.json``
with the sha256 of the pool tasks.jsonl, the shard's tasks and their held-out flags. Paths may be remote; each
shard file is a ``harbor_glm.py --task-list``. ``--added-in`` keeps only tasks a pool version added.

    python -m experiments.post_training.sft_shootout.shard_tasks --family tmax \
      --manifest s3://.../pool/tasks.jsonl --shard-size 100 --out-dir s3://.../traces/shards
"""

import argparse
import hashlib
import json

from rigging.filesystem.storage_path import StoragePath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=["tmax", "swesmith"], required=True)
    parser.add_argument("--manifest", required=True, help="The #206 pool tasks.jsonl.")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default=None, help="Shard file name prefix (default: the family).")
    parser.add_argument("--added-in", type=int, nargs="*", help="Keep only tasks with these added_in versions.")
    args = parser.parse_args()

    text = StoragePath(args.manifest).read_text()
    rows = [json.loads(line) for line in text.splitlines()]
    rows = [row for row in rows if row["family"] == args.family]
    if args.added_in:
        rows = [row for row in rows if row.get("added_in") in args.added_in]
    rows = sorted(rows, key=lambda row: row["task_id"])
    digest = hashlib.sha256(text.encode()).hexdigest()
    shards = [rows[i : i + args.shard_size] for i in range(0, len(rows), args.shard_size)]
    for index, shard in enumerate(shards):
        record = {
            "family": args.family,
            "tasks_jsonl_sha256": digest,
            "package_revision": sorted({row["package_revision"] for row in shard}),
            "shard": index,
            "shards": len(shards),
            "tasks": [row["task_id"] for row in shard],
            "heldout": [row["task_id"] for row in shard if row["heldout"]],
        }
        name = f"{args.prefix or args.family}-shard-{index:02d}.json"
        StoragePath(f"{args.out_dir.rstrip('/')}/{name}").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"family": args.family, "tasks": len(rows), "shards": len(shards), "tasks_jsonl_sha256": digest}))


if __name__ == "__main__":
    main()
