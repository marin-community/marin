# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version a single-turn Smartie cohort with a larger model deadline."""

import hashlib
import json

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.spec import LoweredTaskSpec

from experiments.post_training.smartie.validation_panel import file_hash


def extend_model_deadline(payload: str, model_timeout: float) -> str:
    """Extend enclosing deadlines equally, preserving tool and grader allowances."""
    spec = LoweredTaskSpec.model_validate_json(payload)
    session = spec.session
    if session.max_turns != 1:
        raise ValueError("Deadline transformation requires single-turn tasks")
    if session.model_turn_timeout is None or session.total_turn_timeout is None or session.attempt_timeout is None:
        raise ValueError("Deadline transformation requires finite existing deadlines")
    delta = model_timeout - session.model_turn_timeout
    if delta < 0:
        raise ValueError("Model deadline must not decrease")
    data = json.loads(payload)
    for field in ("model_turn_timeout", "total_turn_timeout", "attempt_timeout"):
        data["session"][field] += delta
    result = json.dumps(data, separators=(",", ":"))
    LoweredTaskSpec.model_validate_json(result)
    return result


def extend_dataset_deadlines(
    source: StoragePath, source_manifest: StoragePath, output: StoragePath, model_timeout: float
) -> dict:
    """Write a fresh cohort with unchanged rows and hash-linked source provenance."""
    if output.exists():
        raise FileExistsError(f"Deadline output must be new: {output}")
    manifest_bytes = source_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    tables = {}
    for split in ("train", "validation"):
        name = f"{split}.parquet"
        path = source / name
        if file_hash(path) != manifest["output_sha256"][name]:
            raise ValueError(f"Source hash mismatch: {name}")
        with path.open("rb") as stream:
            table = pq.read_table(stream)
        column = table.schema.get_field_index("lowered_task_spec")
        payloads = [extend_model_deadline(value, model_timeout) for value in table.column(column).to_pylist()]
        tables[name] = table.set_column(column, table.schema.field(column), pa.array(payloads))
    output.mkdirs(exist_ok=False)
    for name, table in tables.items():
        with (output / name).open("wb") as stream:
            pq.write_table(table, stream)
    report = {
        "source": str(source),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": manifest["output_sha256"],
        "model_turn_timeout": model_timeout,
        "transformation": "Extend model, total-turn and attempt deadlines by the same per-row delta",
        "rows": {name: table.num_rows for name, table in tables.items()},
        "output_sha256": {name: file_hash(output / name) for name in tables},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@click.command()
@click.option("--source", required=True)
@click.option("--source-manifest", required=True)
@click.option("--output", required=True)
@click.option("--model-timeout", type=float, required=True)
def main(source: str, source_manifest: str, output: str, model_timeout: float) -> None:
    report = extend_dataset_deadlines(
        StoragePath(source), StoragePath(source_manifest), StoragePath(output), model_timeout
    )
    click.echo(json.dumps(report))


if __name__ == "__main__":
    main()
