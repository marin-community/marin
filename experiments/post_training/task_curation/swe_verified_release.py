# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish admitted SWE TaskSpecs with their declared Harbor bootstrap."""

import hashlib
import io
import json
import tomllib

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation import CANONICAL_ARTIFACT_ROOT, publish_once

_SOURCE_SHA256 = "453552f9d4e1fa27710f9434a303f7ab5ff390c22deb54572f9be885a2a80f5d"
_SOURCE_ROWS = 5608
_OUTPUT = StoragePath(CANONICAL_ARTIFACT_ROOT) / "rl-data" / "inputs" / "swe-verified-task-spec-setup" / "2026.11.19.1"
_TASK_COLUMN = "task_spec"


def _bind_bootstrap(serialized: str) -> tuple[str, str]:
    task = TaskSpec.model_validate_json(serialized)
    if task.environment_requirements.setup_commands:
        raise ValueError(f"{task.id} already declares setup_commands")
    configs = [resource for resource in task.resources.verifier if resource.path == "task.toml"]
    if len(configs) != 1:
        raise ValueError(f"{task.id} must contain exactly one archived task.toml")
    config = tomllib.loads(resource_bytes(configs[0]).decode())
    setup = config.get("environment", {}).get("setup", {})
    if unknown := setup.keys() - {"command", "timeout_sec"}:
        raise ValueError(f"{task.id} has unsupported Harbor setup keys: {', '.join(sorted(unknown))}")
    command = setup.get("command")
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
    publish_once(
        output, {"agentic-train.parquet": parquet, "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    )
    return manifest


@click.command()
@click.option("--source", required=True, help="Storage URI of the sealed v4 training Parquet.")
def main(source: str) -> None:
    provenance = Provenance.capture()
    if provenance.dirty or not provenance.base_commit:
        raise SystemExit("Publish from a clean commit so the manifest identifies the converter")
    configure_coreweave_s3()
    manifest = convert_release(StoragePath(source), _SOURCE_SHA256, _SOURCE_ROWS, _OUTPUT, provenance.base_commit)
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
