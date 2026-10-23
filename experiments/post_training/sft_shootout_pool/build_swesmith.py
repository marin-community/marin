# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build history-stripped SWE-smith Harbor packages for the selected tasks (issue #10212).

Reads the pinned DCAgent package parquet, clones each needed swesmith/<repo> mirror once (bare), computes every task's
fail-to-pass restore payload, rewrites Dockerfile and test.sh (swesmith_packages), and writes a parquet in the upstream
schema (path, task_binary) plus a JSONL record per task.

Usage:
    python -m experiments.post_training.sft_shootout_pool.build_swesmith --tasks <ids> --output <dir or s3 prefix>
"""

import argparse
import hashlib
import io
import json
import re
import tempfile
from pathlib import Path
from typing import NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_pool.pool_io import MANIFEST, jsonl_bytes
from experiments.post_training.sft_shootout_pool.sample import upstream_repo
from experiments.post_training.sft_shootout_pool.swesmith_packages import (
    fixed_package,
    instance_id,
    read_package,
    write_package,
)
from experiments.post_training.sft_shootout_pool.swesmith_restore import mirror, restore_payload

PACKAGES = ("DCAgent/swesmith-sandboxes-with_tests-oracle_verified_120s", "78b0d8791f9036b90ea303561254c9d617024258")
MIRROR_NAME = re.compile(r"^(.+?__.+?\.[0-9a-f]{8})\.")


def packages_parquet() -> Path:
    """Download the pinned upstream package parquet (columns path, task_binary)."""
    return Path(hf_hub_download(PACKAGES[0], "tasks.parquet", repo_type="dataset", revision=PACKAGES[1]))


class UpstreamPackages(NamedTuple):
    binaries: dict[str, bytes]
    configs: dict[str, dict]
    # {task: upstream 'owner/name'}
    repo_of: dict[str, str]


def upstream_packages() -> UpstreamPackages:
    """Every pinned upstream package with its tests/config.json and upstream repo."""
    binaries = {row["path"]: row["task_binary"] for row in pq.read_table(packages_parquet()).to_pylist()}
    configs = {task: json.loads(read_package(binary)["tests/config.json"]) for task, binary in binaries.items()}
    return UpstreamPackages(binaries, configs, {task: upstream_repo(config["repo"]) for task, config in configs.items()})


class FixedPackage(NamedTuple):
    upstream: dict[str, bytes]
    binary: bytes
    # The restore record: instance_id, upstream and fixed sha256, restored and deleted files.
    record: dict


def fix_package(task_binary: bytes, mirrors: Path, prebuilt: bytes | None = None) -> FixedPackage:
    """Rebuild one upstream package without git history, cloning its mirror into `mirrors` if needed.

    `prebuilt` is an already-built fixed package to use byte for byte; the restore record is still computed.
    """
    package = read_package(task_binary)
    instance = instance_id(package)
    restored, deleted = restore_payload(mirror(MIRROR_NAME.match(instance)[1], mirrors), instance)
    binary = prebuilt or write_package(fixed_package(package, restored, deleted))
    record = {
        "instance_id": instance,
        "upstream_sha256": hashlib.sha256(task_binary).hexdigest(),
        "fixed_sha256": hashlib.sha256(binary).hexdigest(),
        "restored_files": sorted(restored),
        "deleted_files": deleted,
    }
    return FixedPackage(package, binary, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="file with one task id (swesmith-NNNNN) per line")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    wanted = set(args.tasks.read_text().split())
    source = packages_parquet()
    rows = [row for row in pq.read_table(source).to_pylist() if row["path"] in wanted]
    if {row["path"] for row in rows} != wanted:
        raise ValueError(f"Missing packages: {sorted(wanted - {row['path'] for row in rows})[:5]}")
    built, records = [], []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for row in sorted(rows, key=lambda row: row["path"]):
            fixed = fix_package(row["task_binary"], root)
            built.append({"path": row["path"], "task_binary": fixed.binary})
            records.append({"task": row["path"]} | fixed.record)
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(built, schema=pq.read_schema(source)), buffer)
        parquet = buffer.getvalue()
        jsonl = jsonl_bytes(records)
        manifest = {
            "source": f"{PACKAGES[0]}@{PACKAGES[1]}/tasks.parquet",
            "tasks": len(records),
            "fix": "issue #10212: one-commit repository in the agent image; verifier-only fail-to-pass restore",
            "files": {
                "tasks.parquet": hashlib.sha256(parquet).hexdigest(),
                "packages.jsonl": hashlib.sha256(jsonl).hexdigest(),
            },
        }
        output.mkdirs()
        (output / "tasks.parquet").write_bytes(parquet)
        (output / "packages.jsonl").write_bytes(jsonl)
        (output / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        print("SWESMITH_BUILT " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
