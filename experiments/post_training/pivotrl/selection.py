# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep the pivots the paper selects: mixed outcomes and a pass rate below the difficulty threshold (Eq. 5)."""

import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.rl.pass_rates import PASS_COUNT_SCHEMA, PASS_RATES_FILENAME
from rigging.filesystem.storage_path import StoragePath
from zephyr.writers import write_parquet_file

TRAIN_FILENAME = "train.parquet"


def head_rows(*, rows_path: str, rows_filename: str, output_path: str, limit: int) -> None:
    """Copy the first ``limit`` rows of ``rows_path/rows_filename`` to the same filename under ``output_path``."""
    with (StoragePath(rows_path) / rows_filename).open("rb") as source:
        head = pq.read_table(source).slice(0, limit)
    write_parquet_file(head.to_batches(), str(StoragePath(output_path) / rows_filename), schema=head.schema)


def pivot_filter(criterion: str, difficulty_threshold: float) -> pc.Expression:
    """Positive reward variance (0 < passed < total) and mean below lambda, counting ``criterion``.

    Excluded rows have ``total == 0``, so ``passed > 0`` already drops them.
    """
    passed, total = pc.field(criterion), pc.field("total")
    return (passed > 0) & (passed < total) & (pc.divide(passed.cast(pa.float64()), total) < difficulty_threshold)


def select_pivots(*, pass_rates_path: str, output_path: str, difficulty_threshold: float, criterion: str) -> None:
    """Write the selected candidates, in the candidate schema, plus counts.

    Args:
        pass_rates_path: A pass-rate artifact.
        output_path: Where ``train.parquet`` and ``manifest.json`` go.
        difficulty_threshold: Lambda; a pivot's pass rate must be below it.
        criterion: The count column a reply must pass: ``passed`` (the task's reward) or a
            ``passed_<component>`` column such as ``passed_tool_name``.
    """
    if not 0 < difficulty_threshold <= 1:
        raise ValueError("difficulty_threshold must be in (0, 1]")

    with (StoragePath(pass_rates_path) / PASS_RATES_FILENAME).open("rb") as source:
        table = pq.read_table(source)
    if criterion not in table.column_names:
        raise ValueError(f"no {criterion!r} column; pass-rate counts are {_count_columns(table)}")
    selected = table.filter(pivot_filter(criterion, difficulty_threshold)).drop_columns(_count_columns(table))
    if not selected.num_rows:
        raise ValueError("no candidate is a pivot")

    output = StoragePath(output_path)
    write_parquet_file(selected.to_batches(), str(output / TRAIN_FILENAME), schema=selected.schema)
    manifest = {
        "criterion": criterion,
        "difficulty_threshold": difficulty_threshold,
        "candidate_rows": table.num_rows,
        "selected_rows": selected.num_rows,
        "excluded_rows": table.num_rows - table["exclusion"].null_count,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _count_columns(table: pa.Table) -> list[str]:
    """The pass-rate count columns added to the candidate rows."""
    return [*PASS_COUNT_SCHEMA.names, *(name for name in table.column_names if name.startswith("passed_"))]
