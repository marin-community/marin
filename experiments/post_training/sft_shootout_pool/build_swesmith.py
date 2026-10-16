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

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sft_shootout_pool.swesmith_packages import (
    fixed_package,
    instance_id,
    read_package,
    write_package,
)
from experiments.post_training.sft_shootout_pool.swesmith_restore import mirror, restore_payload

PACKAGES = ("DCAgent/swesmith-sandboxes-with_tests-oracle_verified_120s", "78b0d8791f9036b90ea303561254c9d617024258")
MIRROR_NAME = re.compile(r"^(.+?__.+?\.[0-9a-f]{8})\.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="file with one task id (swesmith-NNNNN) per line")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a fresh output prefix")
    wanted = set(args.tasks.read_text().split())
    source = hf_hub_download(PACKAGES[0], "tasks.parquet", repo_type="dataset", revision=PACKAGES[1])
    rows = [row for row in pq.read_table(source).to_pylist() if row["path"] in wanted]
    if {row["path"] for row in rows} != wanted:
        raise ValueError(f"Missing packages: {sorted(wanted - {row['path'] for row in rows})[:5]}")
    built, records = [], []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for row in sorted(rows, key=lambda row: row["path"]):
            package = read_package(row["task_binary"])
            instance = instance_id(package)
            repository = mirror(MIRROR_NAME.match(instance)[1], root)
            files, deleted = restore_payload(repository, instance)
            binary = write_package(fixed_package(package, files, deleted))
            built.append({"path": row["path"], "task_binary": binary})
            records.append(
                {
                    "task": row["path"],
                    "instance_id": instance,
                    "upstream_sha256": hashlib.sha256(row["task_binary"]).hexdigest(),
                    "fixed_sha256": hashlib.sha256(binary).hexdigest(),
                    "restored_files": sorted(files),
                    "deleted_files": deleted,
                }
            )
        buffer = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(built, schema=pq.read_schema(source)), buffer)
        parquet = buffer.getvalue()
        jsonl = "".join(json.dumps(record) + "\n" for record in records).encode()
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
        (output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print("SWESMITH_BUILT " + json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
