# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package qualified fixture recipes as pending-admission Harbor tasks."""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tomlkit
from taskcompendium.pipeline.inputs import ConversionContext
from tomlkit.items import Table

from experiments.post_training.task_curation.calibforge_fixtures import FIXTURE_ARCHIVE, SETUP_SCRIPT
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    archive_files,
    unpack_task_binary,
)
from experiments.post_training.task_curation.tasktrove.harbor_export import archive_bytes

ENVIRONMENT_PREFIX = "environment/"


def transformed_task(row: dict, record: dict, bundles: Path, source: Path, timeout: float) -> bytes:
    """Keep private grader bytes unchanged and move only public fixtures to setup files."""
    if record["workdir"] != "/app":
        raise ValueError(f"Shared snapshot packaging requires the qualified /app workspace: {row['path']}")
    decoded = unpack_task_binary(row, ConversionContext(inputs={}, grader_environment=None))
    if decoded["archive_links"]:
        raise ValueError(f"Archive links need an explicit preservation contract: {row['path']}")
    files = archive_files(decoded).files
    modes = {name: metadata["mode"] for name, metadata in decoded["file_metadata"].items()}
    if any(name.startswith("setup_files/") for name in files):
        raise ValueError(f"Existing setup files need ordering qualification: {row['path']}")
    for name, content in files.items():
        if name.startswith(ENVIRONMENT_PREFIX):
            original = source / row["path"] / name
            if original.read_bytes() != content:
                raise ValueError(f"Source/release mismatch: {row['path']}/{name}")
            if original.stat().st_mode & 0o7777 != int(modes[name], 8):
                raise ValueError(f"Source/release mode mismatch: {row['path']}/{name}")
    if hashlib.sha256(files["environment/Dockerfile"]).hexdigest() != record["source_dockerfile_sha256"]:
        raise ValueError(f"Inventory/release Dockerfile mismatch: {row['path']}")
    config = tomlkit.parse(files["task.toml"].decode())
    environment = config["environment"]
    assert isinstance(environment, Table)
    if "setup" in environment:
        raise ValueError(f"Existing setup hook: {row['path']}")
    environment["docker_image"] = record["base_image"]
    environment["workdir"] = record["workdir"]
    environment["setup"] = {
        "command": f"sh /setup_files/{SETUP_SCRIPT}",
        "timeout_sec": timeout,
    }
    output = {name: content for name, content in files.items() if not name.startswith(ENVIRONMENT_PREFIX)}
    output["task.toml"] = tomlkit.dumps(config).encode()
    output["environment/Dockerfile"] = f"FROM {record['base_image']}\nWORKDIR /app\n".encode()
    for filename, digest_key in (
        (FIXTURE_ARCHIVE, "payload_sha256"),
        (SETUP_SCRIPT, "setup_sha256"),
    ):
        content = (bundles / "tasks" / row["path"] / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != record[digest_key]:
            raise ValueError(f"Bundle digest mismatch: {row['path']}/{filename}")
        output["setup_files/" + filename] = content
    return archive_bytes(output, modes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup-timeout", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    table = pq.read_table(args.approved)
    bundle_manifest = json.loads((args.bundles / "manifest.json").read_text())
    records = {row["task_id"]: row for row in bundle_manifest["tasks"]}
    if set(table.column("path").to_pylist()) != set(records):
        raise ValueError("Approved cohort and compiler cohort differ")
    candidates = []
    evidence = []
    for row in table.to_pylist():
        record = records[row["path"]]
        if record["status"] == "deferred":
            evidence.append({"task_id": row["path"], "status": "deferred", "reason": record["reason"]})
            continue
        transformed = transformed_task(row, record, args.bundles, args.source, args.setup_timeout)
        candidates.append({**row, "task_binary": transformed})
        evidence.append(
            {
                "task_id": row["path"],
                "status": "packaged_pending_admission",
                "source_archive_sha256": hashlib.sha256(row["task_binary"]).hexdigest(),
                "archive_sha256": hashlib.sha256(transformed).hexdigest(),
            }
        )
    pq.write_table(pa.Table.from_pylist(candidates, schema=table.schema), args.output / "pending-tasks.parquet")
    manifest = {
        "approved_sha256": hashlib.sha256(args.approved.read_bytes()).hexdigest(),
        "bundle_manifest_sha256": hashlib.sha256((args.bundles / "manifest.json").read_bytes()).hexdigest(),
        "candidate_count": len(evidence),
        "packaged_count": len(candidates),
        "admission": "not_run",
        "tasks": evidence,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
