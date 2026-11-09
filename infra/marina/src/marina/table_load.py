# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load small local data files into an applet through its SQL endpoint."""

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.json as pa_json
import pyarrow.parquet as pa_parquet
from fastapi.encoders import jsonable_encoder

from marina.client import AppletQuery

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
INSERT_BATCH_ROWS = 250
MAX_BATCH_BYTES = 1_000_000


def read_table(path: Path) -> pa.Table:
    """Read one supported local table format into Arrow."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pa_csv.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pa_json.read_json(path)
    if suffix == ".json":
        value = json.loads(path.read_text())
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("JSON table input must be an array of objects")
        return pa.Table.from_pylist(value)
    if suffix in {".parquet", ".pq"}:
        return pa_parquet.read_table(path)
    raise ValueError(f"unsupported table format {suffix!r}; use JSON, JSONL, CSV, or Parquet")


def checked_identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"invalid SQL identifier {value!r}")
    return f'"{value}"'


def postgres_type(data_type: pa.DataType) -> str:
    """Map the scalar Arrow types accepted by applet table loading to Postgres."""
    if pa.types.is_boolean(data_type):
        return "BOOLEAN"
    if pa.types.is_integer(data_type):
        return "BIGINT"
    if pa.types.is_floating(data_type) or pa.types.is_decimal(data_type):
        return "DOUBLE PRECISION"
    if pa.types.is_date(data_type):
        return "DATE"
    if pa.types.is_timestamp(data_type):
        return "TIMESTAMPTZ" if data_type.tz is not None else "TIMESTAMP"
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return "TEXT"
    raise ValueError(f"unsupported Arrow column type {data_type}")


def table_statements(
    table_name: str, table: pa.Table, *, replace: bool
) -> tuple[list[str], list[tuple[str, dict[str, object]]]]:
    """Return schema and parameterized inserts for one Arrow table."""
    target = checked_identifier(table_name)
    columns = [checked_identifier(field.name) for field in table.schema]
    if not columns:
        raise ValueError("table input has no columns")
    definitions = ", ".join(
        f"{column} {postgres_type(field.type)}" for column, field in zip(columns, table.schema, strict=True)
    )
    schema_sql = ([f"DROP TABLE IF EXISTS {target}"] if replace else []) + [
        f"CREATE TABLE IF NOT EXISTS {target} ({definitions})"
    ]
    rows = jsonable_encoder(table.to_pylist())
    inserts: list[tuple[str, dict[str, object]]] = []
    for batch_start in range(0, len(rows), INSERT_BATCH_ROWS):
        batch = rows[batch_start : batch_start + INSERT_BATCH_ROWS]
        parameters: dict[str, object] = {}
        values: list[str] = []
        for row_index, row in enumerate(batch):
            placeholders: list[str] = []
            for column_index, field in enumerate(table.schema):
                parameter = f"r{row_index}_c{column_index}"
                placeholders.append(f":{parameter}")
                parameters[parameter] = row[field.name]
            values.append("(" + ", ".join(placeholders) + ")")
        inserts.append((f"INSERT INTO {target} ({', '.join(columns)}) VALUES {', '.join(values)}", parameters))
    return schema_sql, inserts


def json_row_batches(rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    """Group rows into JSON arrays whose encoded SQL request stays under ``MAX_BATCH_BYTES``."""
    batch: list[str] = []
    size = 2
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        # The SQL API carries the row array as a JSON string, adding another escape layer.
        length = len(json.dumps(encoded).encode()) + 1
        if length + 2 > MAX_BATCH_BYTES:
            raise ValueError("One row exceeds the SQL batch budget")
        if size + length > MAX_BATCH_BYTES:
            yield "[" + ",".join(batch) + "]"
            batch, size = [], 2
        batch.append(encoded)
        size += length
    if batch:
        yield "[" + ",".join(batch) + "]"


def insert_load_rows(query: AppletQuery, table: str, columns: str, load_id: str, rows: Iterable[dict[str, Any]]) -> int:
    """Insert byte-bounded batches tagged with ``load_id`` and return the stored row count.

    ``columns`` is the ``name TYPE, ...`` list of the row fields; the table also has a ``load_id`` column.
    """
    names = ", ".join(column.strip().split()[0] for column in columns.split(","))
    count = 0
    for body in json_row_batches(rows):
        result = query(
            f"INSERT INTO {table} (load_id, {names}) SELECT :load_id, {names} "
            f"FROM jsonb_to_recordset(CAST(:rows AS JSONB)) AS row({columns})",
            {"load_id": load_id, "rows": body},
        )
        count += result["row_count"]
    return count
