# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split a family's pool tasks (held-out included) into shards for the #212 full run.

Each shard is one ``harbor_glm.py`` job with its own output prefix, so a failed job loses only its
shard. Shards are contiguous slices of the sorted task ids. Writes ``<out-dir>/<family>-shard-NN.json``
with the manifest's sha256, the shard's tasks and their held-out flags.

    python -m experiments.post_training.sft_shootout.shard_tasks --family tmax \
      --manifest pool/tasks.jsonl --shard-size 100 --out-dir experiments/post_training/sft_shootout/full_run
"""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=["tmax", "swesmith"], required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="The #206 pool tasks.jsonl.")
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    rows = sorted((row for row in rows if row["family"] == args.family), key=lambda row: row["task_id"])
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = [rows[i : i + args.shard_size] for i in range(0, len(rows), args.shard_size)]
    for index, shard in enumerate(shards):
        record = {
            "family": args.family,
            "manifest_sha256": digest,
            "package_revision": sorted({row["package_revision"] for row in shard}),
            "shard": index,
            "shards": len(shards),
            "tasks": [row["task_id"] for row in shard],
            "heldout": [row["task_id"] for row in shard if row["heldout"]],
        }
        (args.out_dir / f"{args.family}-shard-{index:02d}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"family": args.family, "tasks": len(rows), "shards": len(shards), "manifest_sha256": digest}))


if __name__ == "__main__":
    main()
