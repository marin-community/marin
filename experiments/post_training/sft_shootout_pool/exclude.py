# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove tasks from a built pool without resampling, keeping the previous manifest.

Exclusions decided after the 2026-10-16 build:
- agronholm/typeguard SWE-smith tasks: setuptools vendors an older typeguard in the agent image
  (setuptools/_vendor/typeguard), a readable copy of clean code (leak_risk=vendored_clean_copy).
- swesmith-01094: its verifier scores the unpatched tree 1, also in the upstream package (leak_risk=broken_verifier).

Excluded tasks leave tasks.jsonl and swesmith-packages.parquet and join excluded.jsonl. Nothing is backfilled, so the
held-out fraction can drop below 5%. The old manifest is kept as manifest-v1.json; manifest.json records it.

Usage: python -m experiments.post_training.sft_shootout_pool.exclude --pool <prefix>
"""

import argparse
import hashlib
import io
import json

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_pool.pool_io import (
    MANIFEST,
    SWESMITH_PACKAGES,
    jsonl_bytes,
    kept_manifest,
    parquet_bytes,
    read_jsonl,
    read_pool,
    swesmith_revision,
    write_pool,
)

EXCLUDED_REPOS = {"agronholm/typeguard": "vendored_clean_copy"}
EXCLUDED_TASKS = {"swesmith-01094": "broken_verifier"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    args = parser.parse_args()
    pool = StoragePath(args.pool)
    if (pool / kept_manifest(1)).exists():
        raise ValueError("Exclusions were already applied to this pool")
    previous_bytes, manifest, files = read_pool(pool, "the build")

    rows = read_jsonl(files["tasks.jsonl"])
    removed = {}
    for row in rows:
        reason = EXCLUDED_TASKS.get(row["task_id"]) or EXCLUDED_REPOS.get(row["repo"])
        if reason:
            removed[row["task_id"]] = (row, reason)
    if len([task for task in EXCLUDED_TASKS if task in removed]) != len(EXCLUDED_TASKS):
        raise ValueError("An excluded task is not in the pool")
    kept = [row for row in rows if row["task_id"] not in removed]

    packages = [
        row for row in pq.read_table(io.BytesIO(files[SWESMITH_PACKAGES])).to_pylist() if row["path"] not in removed
    ]
    files[SWESMITH_PACKAGES] = parquet_bytes(packages)
    revision = swesmith_revision(args.pool, files)
    for row in kept:
        if row["family"] == "swesmith":
            row["package_revision"] = revision
    records = read_jsonl(files["swesmith-packages.jsonl"])
    files["swesmith-packages.jsonl"] = jsonl_bytes([record for record in records if record["task_id"] not in removed])
    files["tasks.jsonl"] = jsonl_bytes(kept)
    excluded = read_jsonl(files["excluded.jsonl"])
    excluded += [
        {"task_id": task, "family": row["family"], "repo": row["repo"], "heldout": row["heldout"], "leak_risk": reason}
        for task, (row, reason) in sorted(removed.items())
    ]
    files["excluded.jsonl"] = jsonl_bytes(excluded)

    swesmith = manifest["families"]["swesmith"]
    swesmith["pool_tasks"] = sum(row["family"] == "swesmith" for row in kept)
    swesmith["heldout_tasks"] = sum(row["family"] == "swesmith" and row["heldout"] for row in kept)
    swesmith["package_revision"] = revision
    swesmith["excluded_after_build"] = {
        "repos": EXCLUDED_REPOS,
        "tasks": EXCLUDED_TASKS,
        "removed": len(removed),
        "removed_heldout": sum(row["heldout"] for row, _ in removed.values()),
    }
    manifest["supersedes"] = {
        "manifest": kept_manifest(1),
        "sha256": hashlib.sha256(previous_bytes).hexdigest(),
        "change": "post-build exclusions, no resampling",
    }
    manifest["decon"]["near_threshold_tasks"] = {
        task: value for task, value in manifest["decon"]["near_threshold_tasks"].items() if task not in removed
    }
    manifest["file_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

    (pool / kept_manifest(1)).write_bytes(previous_bytes)
    write_pool(pool, files, manifest)
    print(
        "POOL_EXCLUDED "
        + json.dumps(
            {
                "manifest_sha256": hashlib.sha256((pool / MANIFEST).read_bytes()).hexdigest(),
                "families": {
                    f: {k: v for k, v in s.items() if k != "repo_allocation"} for f, s in manifest["families"].items()
                },
                "file_sha256": manifest["file_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
