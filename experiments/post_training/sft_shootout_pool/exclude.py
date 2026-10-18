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

from experiments.post_training.sft_shootout_pool.pool import parquet_bytes

EXCLUDED_REPOS = {"agronholm/typeguard": "vendored_clean_copy"}
EXCLUDED_TASKS = {"swesmith-01094": "broken_verifier"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    args = parser.parse_args()
    pool = StoragePath(args.pool)
    if (pool / "manifest-v1.json").exists():
        raise ValueError("Exclusions were already applied to this pool")
    previous_bytes = (pool / "manifest.json").read_bytes()
    manifest = json.loads(previous_bytes)
    files = {name: (pool / name).read_bytes() for name in manifest["file_sha256"]}
    for name, data in files.items():
        if hashlib.sha256(data).hexdigest() != manifest["file_sha256"][name]:
            raise ValueError(f"{name} changed since the build")

    rows = [json.loads(line) for line in files["tasks.jsonl"].decode().splitlines()]
    removed = {}
    for row in rows:
        reason = EXCLUDED_TASKS.get(row["task_id"]) or EXCLUDED_REPOS.get(row["repo"])
        if reason:
            removed[row["task_id"]] = (row, reason)
    if len([task for task in EXCLUDED_TASKS if task in removed]) != len(EXCLUDED_TASKS):
        raise ValueError("An excluded task is not in the pool")
    kept = [row for row in rows if row["task_id"] not in removed]

    packages = [
        row
        for row in pq.read_table(io.BytesIO(files["swesmith-packages.parquet"])).to_pylist()
        if row["path"] not in removed
    ]
    files["swesmith-packages.parquet"] = parquet_bytes(packages)
    revision = (
        f"{args.pool.rstrip('/')}/swesmith-packages.parquet@sha256:"
        f"{hashlib.sha256(files['swesmith-packages.parquet']).hexdigest()}"
    )
    for row in kept:
        if row["family"] == "swesmith":
            row["package_revision"] = revision
    records = [json.loads(line) for line in files["swesmith-packages.jsonl"].decode().splitlines()]
    files["swesmith-packages.jsonl"] = "".join(
        json.dumps(record) + "\n" for record in records if record["task_id"] not in removed
    ).encode()
    files["tasks.jsonl"] = "".join(json.dumps(row) + "\n" for row in kept).encode()
    excluded = [json.loads(line) for line in files["excluded.jsonl"].decode().splitlines()]
    excluded += [
        {"task_id": task, "family": row["family"], "repo": row["repo"], "heldout": row["heldout"], "leak_risk": reason}
        for task, (row, reason) in sorted(removed.items())
    ]
    files["excluded.jsonl"] = "".join(json.dumps(row) + "\n" for row in excluded).encode()

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
        "manifest": "manifest-v1.json",
        "sha256": hashlib.sha256(previous_bytes).hexdigest(),
        "change": "post-build exclusions, no resampling",
    }
    manifest["decon"]["near_threshold_tasks"] = {
        task: value for task, value in manifest["decon"]["near_threshold_tasks"].items() if task not in removed
    }
    manifest["file_sha256"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}

    (pool / "manifest-v1.json").write_bytes(previous_bytes)
    for name, data in files.items():
        (pool / name).write_bytes(data)
        if hashlib.sha256((pool / name).read_bytes()).hexdigest() != manifest["file_sha256"][name]:
            raise ValueError(f"Upload hash mismatch: {name}")
    (pool / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        "POOL_EXCLUDED "
        + json.dumps(
            {
                "manifest_sha256": hashlib.sha256((pool / "manifest.json").read_bytes()).hexdigest(),
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
