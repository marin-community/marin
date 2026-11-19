# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish admitted SWE TaskSpecs with their declared Harbor bootstrap."""

import argparse
import hashlib
import io
import json
import tomllib

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

_SOURCE = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop/rl-data/"
    "swe-verified-20261112-v4/release/agentic-train.parquet"
)
_SOURCE_SHA256 = "453552f9d4e1fa27710f9434a303f7ab5ff390c22deb54572f9be885a2a80f5d"
_SOURCE_ROWS = 5608
_OUTPUT = "s3://marin-us-east-08a/marin/users/ahmad/rl-data/swe-verified-task-spec-setup/2026.11.19.1"
_TASK_COLUMN = "task_spec"


def _bind_bootstrap(serialized: str) -> tuple[str, str]:
    task = TaskSpec.model_validate_json(serialized)
    configs = [resource for resource in task.resources.verifier if resource.path == "task.toml"]
    if len(configs) != 1:
        raise ValueError(f"{task.id} must contain exactly one archived task.toml")
    config = tomllib.loads(resource_bytes(configs[0]).decode())
    command = config.get("environment", {}).get("setup", {}).get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{task.id} has no declared Harbor bootstrap command")
    row = json.loads(serialized)
    row["environment_requirements"]["setup_commands"] = [command]
    TaskSpec.model_validate(row)
    return task.id, json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def convert_release(source: StoragePath, sha256: str, rows: int, output: StoragePath, code_commit: str) -> dict:
    """Publish the verified task file and its derivation manifest."""
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != sha256:
        raise ValueError(f"Source SHA-256 mismatch: {source}")
    table = pq.read_table(io.BytesIO(source_bytes))
    if table.num_rows != rows:
        raise ValueError(f"Expected {rows} source rows, found {table.num_rows}: {source}")
    converted = []
    identities = []
    for original in table[_TASK_COLUMN].to_pylist():
        task_id, updated = _bind_bootstrap(original)
        converted.append(updated)
        identities.append(
            {
                "task_id": task_id,
                "input_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "output_sha256": hashlib.sha256(updated.encode()).hexdigest(),
            }
        )
    index = table.schema.get_field_index(_TASK_COLUMN)
    field = table.schema.field(index)
    result = table.set_column(index, field, pa.array(converted, type=field.type))
    buffer = io.BytesIO()
    pq.write_table(result, buffer)
    parquet = buffer.getvalue()
    manifest = {
        "input_uri": str(source),
        "input_sha256": sha256,
        "output_sha256": hashlib.sha256(parquet).hexdigest(),
        "code_commit": code_commit,
        "rows": rows,
        "tasks": identities,
    }
    output.mkdirs()
    for name, contents in (
        ("agentic-train.parquet", parquet),
        ("manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()),
    ):
        target = conditional_object(str(output / name))
        try:
            target.write(contents, expected_version=None)
        except ConditionalWriteError as error:
            existing = target.read()
            if existing is None or existing.data != contents:
                raise FileExistsError(f"Published conversion differs from {target.path}") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    provenance = Provenance.capture()
    if provenance.dirty or not provenance.base_commit:
        parser.error("Publish from a clean commit so the manifest identifies the converter")
    configure_coreweave_s3()
    manifest = convert_release(
        StoragePath(_SOURCE), _SOURCE_SHA256, _SOURCE_ROWS, StoragePath(_OUTPUT), provenance.base_commit
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
