# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded local projections shared by several Grafana panels."""

import math
from collections.abc import Callable, Hashable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
from finelog.errors import QueryResultTooLargeError


@dataclass(frozen=True)
class SourceQuery:
    """One bounded Finelog input to a dashboard dataset."""

    name: str
    sql: str
    max_rows: int
    max_samples: int | None = None


@dataclass(frozen=True)
class DashboardDataset:
    """Fixed source queries and local views for one cached dashboard dataset."""

    name: str
    cache_key: tuple[Hashable, ...]
    sources: tuple[SourceQuery, ...]
    setup_sql: tuple[str, ...]
    views: Mapping[str, str]
    max_result_rows: int


def validate_value(name: str, value: str, *, max_length: int) -> None:
    """Validate one dashboard identity value before interpolating it into SQL."""
    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > max_length or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains an invalid value")


def validate_values(name: str, values: tuple[str, ...], *, max_values: int, max_length: int) -> None:
    """Validate a required list of dashboard identity values."""
    if not values or len(values) > max_values:
        raise ValueError(f"{name} must contain between 1 and {max_values} values")
    for value in values:
        validate_value(name, value, max_length=max_length)


def validate_optional_values(name: str, values: tuple[str, ...], *, max_values: int, max_length: int) -> None:
    """Validate an optional list of dashboard identity values."""
    if len(values) > max_values:
        raise ValueError(f"{name} must contain at most {max_values} values")
    for value in values:
        validate_value(name, value, max_length=max_length)


def validate_time_window(
    start_ms: int,
    end_ms: int,
    *,
    max_window_ms: int,
    max_window_error: str,
) -> None:
    """Validate the common time bounds for a dashboard source query."""
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("to must be later than from and both times must be nonnegative")
    if end_ms - start_ms > max_window_ms:
        raise ValueError(max_window_error)


def bounded_bucket_ms(
    start_ms: int,
    end_ms: int,
    requested_bucket_ms: int,
    *,
    max_window_ms: int,
    max_window_error: str,
    min_bucket_ms: int,
    max_points: int,
) -> int:
    """Validate a dashboard window and return a bucket that caps point count."""
    validate_time_window(
        start_ms,
        end_ms,
        max_window_ms=max_window_ms,
        max_window_error=max_window_error,
    )
    if requested_bucket_ms <= 0:
        raise ValueError("bucket_ms must be positive")
    minimum_for_result_cap = math.ceil((end_ms - start_ms) / max_points)
    return min(end_ms - start_ms, max(requested_bucket_ms, min_bucket_ms, minimum_for_result_cap))


def packed_sample_count(table: pa.Table) -> int:
    """Count samples in every list column named ``points``."""
    if "points" not in table.column_names:
        return table.num_rows
    return pc.sum(pc.list_value_length(table["points"])).as_py() or 0


def validate_table_budget(name: str, table: pa.Table, *, max_rows: int, max_samples: int | None = None) -> None:
    """Reject an Arrow table that crossed its declared row or sample budget."""
    if table.num_rows > max_rows:
        raise QueryResultTooLargeError(f"{name} returned more than {max_rows} rows")
    if max_samples is not None and packed_sample_count(table) > max_samples:
        raise QueryResultTooLargeError(f"{name} exceeded its {max_samples}-sample budget")


def projection_database() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with the bridge projection budget."""
    return duckdb.connect(config={"threads": 1, "memory_limit": "512MB", "temp_directory": ""})


def project_dataset(
    dataset: DashboardDataset,
    source_tables: Mapping[str, pa.Table],
    projection_lock: AbstractContextManager[None],
    serialize: Callable[[pa.Table], list[dict[str, object]]],
    max_result_rows: int,
) -> list[dict[str, object]]:
    """Project fixed views from bounded Arrow inputs and tag each result row."""
    expected = {source.name for source in dataset.sources}
    if set(source_tables) != expected:
        raise ValueError(f"{dataset.name} expected sources {sorted(expected)}, got {sorted(source_tables)}")

    rows: list[dict[str, object]] = []
    with projection_lock, projection_database() as database:
        for name, table in source_tables.items():
            database.register(name, table)
        database.execute("CREATE MACRO json_get(d, f) AS json_extract_string(CAST(d AS VARCHAR), concat('$.', f))")
        try:
            for statement in dataset.setup_sql:
                database.execute(statement)
            for section, sql in dataset.views.items():
                table = database.execute(sql).to_arrow_table()
                projected = serialize(table)
                rows.extend({"section": section, **row} for row in projected)
                if len(rows) > max_result_rows:
                    raise QueryResultTooLargeError(f"{dataset.name} returned more than {max_result_rows} projected rows")
        except duckdb.OutOfMemoryException as err:
            raise QueryResultTooLargeError(f"{dataset.name} projection memory budget exceeded") from err
    return rows
