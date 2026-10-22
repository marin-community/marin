# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the small development panel from a verified, current full panel locally."""

import json
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_curriculum.bundle import file_sha256, write_manifest
from experiments.post_training.snowball_inventory_panel import (
    PANEL_SELECTION_VERSION,
    DevelopmentPanel,
    pairing_identity,
    select_development_rows,
)


def freeze_development_panel(source: Path, source_sha256: str, output: Path) -> dict:
    """Write a schema-preserving subset without changing the original panel."""
    source_path = StoragePath(str(source))
    if file_sha256(source_path) != source_sha256:
        raise ValueError("Development source differs from its frozen SHA256")
    if output.exists():
        raise FileExistsError(output)
    table = pq.read_table(source)
    rows = table.to_pylist()
    selected = select_development_rows(rows, DevelopmentPanel.FAST)
    index = {row["task_id"]: i for i, row in enumerate(rows)}
    subset = table.take(pa.array([index[row["task_id"]] for row in selected], type=pa.int64()))
    output.mkdir(parents=True)
    destination = output / "validation.parquet"
    pq.write_table(subset, destination, compression="zstd")
    written = pq.read_table(destination)
    if written.schema != table.schema or written.to_pylist() != selected:
        raise ValueError("Development serialization changed selected task records")
    if file_sha256(source_path) != source_sha256:
        raise ValueError("Development source changed during selection")
    report = {
        "selection_version": PANEL_SELECTION_VERSION,
        "source_sha256": source_sha256,
        "sha256": file_sha256(StoragePath(str(destination))),
        "family_counts": DevelopmentPanel.FAST.family_counts,
        "pairing_identity": pairing_identity(
            [(row["task_id"], row["evaluation_family"], row["source_contract_id"]) for row in selected]
        ),
        "task_ids": [row["task_id"] for row in selected],
        "original_records_preserved": True,
        "limitations": ["Reused development subset; not an independent final test"],
    }
    (output / "provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_manifest(StoragePath(str(output)))
    return report


@click.command(help=__doc__)
@click.option("--source", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--source-sha256", required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(source: Path, source_sha256: str, output: Path) -> None:
    click.echo(json.dumps(freeze_development_panel(source, source_sha256, output), indent=2))


if __name__ == "__main__":
    main()
