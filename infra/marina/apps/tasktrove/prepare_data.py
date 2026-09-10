# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the TaskTrove Clean Parquet viewer served by Marina.

The full clean dataset is too large to mirror into Marina. This script writes one Parquet file
containing a deterministic sample from every ``(mode, dockerfile_id)`` group. Rows retain the
clean dataset's schema, including the normalized task archive, so the app reads the generated
Parquet directly rather than maintaining a second catalog or task representation.

Usage:
    uv run python prepare_data.py s3://.../tasktrove/clean/<version>
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

DATA_DIR = Path(__file__).parents[2] / ".data" / "tasktrove"
VIEW_COLUMNS = (
    "source",
    "path",
    "family",
    "converter",
    "mode",
    "dockerfile_id",
    "language",
    "tags",
    "has_solution",
)


@dataclass(frozen=True, order=True)
class Sample:
    rank: str
    source: str
    path: str
    file: str
    row: int


def _rank(source: str, path: str) -> str:
    return hashlib.sha256(f"tasktrove-marina-v1\0{source}\0{path}".encode()).hexdigest()


def _sample_rows(files: list[StoragePath], samples_per_group: int) -> list[Sample]:
    selected: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for file in files:
        with file.open("rb") as handle:
            table = pq.read_table(handle, columns=list(VIEW_COLUMNS))
        for row_index, row in enumerate(table.to_pylist()):
            group = (row["mode"], row["dockerfile_id"])
            selected[group].append(
                Sample(_rank(row["source"], row["path"]), row["source"], row["path"], str(file), row_index)
            )
            selected[group] = sorted(selected[group])[:samples_per_group]
    return [sample for group in sorted(selected) for sample in selected[group]]


def _read_samples(samples: list[Sample]) -> pa.Table:
    by_file: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        by_file[sample.file].append(sample.row)

    tables = []
    for file, rows in sorted(by_file.items()):
        with StoragePath(file).open("rb") as handle:
            tables.append(pq.read_table(handle).take(pa.array(rows)))
    return pa.concat_tables(tables).sort_by(
        [("mode", "ascending"), ("dockerfile_id", "ascending"), ("source", "ascending"), ("path", "ascending")]
    )


def prepare_data(clean_output: StoragePath, output: Path, samples_per_group: int) -> dict:
    manifest = (clean_output / "manifest.json").read_text()
    files = sorted((clean_output / "tasks/*.parquet").glob(), key=str)
    samples = _sample_rows(files, samples_per_group)
    table = _read_samples(samples)

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(manifest)
    pq.write_table(table, output / "tasks.parquet", compression="snappy", row_group_size=1)
    return {"tasks": table.num_rows, "groups": len({(row["mode"], row["dockerfile_id"]) for row in table.to_pylist()})}


@click.command(help=__doc__)
@click.argument("clean_output")
@click.option("--output", type=click.Path(path_type=Path), default=DATA_DIR)
@click.option("--samples-per-group", type=click.IntRange(min=1), default=5, show_default=True)
def main(clean_output: str, output: Path, samples_per_group: int) -> None:
    configure_coreweave_s3()
    result = prepare_data(StoragePath(clean_output), output, samples_per_group)
    print(f"{result['tasks']} rows across {result['groups']} mode/environment groups -> {output / 'tasks.parquet'}")


if __name__ == "__main__":
    main()
