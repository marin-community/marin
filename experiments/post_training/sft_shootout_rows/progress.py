# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dry-build plain, RS-multi and relay over every finished generation shard and print one progress summary.

A shard is finished when its rebuilt.jsonl sits at a shard or round path (build.SHARD_PATH) with a _SUCCESS marker
beside it. All finished shards of both families (plain and relay runs) go to validate_pilot.py, which builds and
re-checks each arm under --output (a dry-run prefix with DRY_RUN_NOT_FOR_TRAINING markers; never an arm's input).
The summary line holds, per arm, the manifest fields the progress table needs.

Usage:
    python -m experiments.post_training.sft_shootout_rows.progress --traces-root <s3 .../traces/> --pool <tasks.jsonl> \\
        --pool-manifest <manifest.json> --replay <replay prefix> --output <fresh dry-run prefix>
"""

import argparse
import json
import subprocess
import sys

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_rows.build import finished_shards

ARMS = ("plain", "rs-multi", "relay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces-root", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--pool-manifest", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = finished_shards(args.traces_root, relay=False) + finished_shards(args.traces_root, relay=True)
    if not shards:
        raise SystemExit("No finished shards")
    names = [s.removeprefix(args.traces_root.rstrip("/") + "/") for s in shards]
    print("PROGRESS_SHARDS " + json.dumps(names), flush=True)
    command = [
        sys.executable,
        "-m",
        "experiments.post_training.sft_shootout_rows.validate_pilot",
        "--traces",
        *shards,
        "--arms",
        *ARMS,
        "--pool",
        args.pool,
        "--pool-manifest",
        args.pool_manifest,
        "--replay",
        args.replay,
        "--output",
        args.output,
    ]
    subprocess.run(command, check=True)
    summary = {"shards": names, "arms": {}}
    for arm in ARMS:
        manifest = json.loads((StoragePath(args.output) / arm / "manifest.json").read_text())
        summary["arms"][arm] = {
            key: manifest.get(key)
            for key in (
                "pool_manifest_sha256",
                "candidates",
                "eligibility",
                "over_context_by_family",
                "B_supervised_tokens",
                "splits",
                "replay",
                "relay_supervised_tokens",
                "relay_pool",
            )
        }
    print("PROGRESS_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
