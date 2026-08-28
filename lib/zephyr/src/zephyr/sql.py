# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DataFusion SQL transforms over Arrow batches."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import pyarrow as pa
from datafusion import DataFrame, SessionContext, udf

from zephyr.batches import canonicalize_record_batch, iter_record_batches
from zephyr.parquet_scan import datafusion_context

SQL_INPUT_TABLE = "input"


def quote_identifier(name: str) -> str:
    """Quote one SQL identifier for DataFusion."""
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class SqlScalarFunction:
    """A Python scalar function registered before executing a SQL query."""

    name: str
    function: Callable
    input_types: tuple[pa.DataType, ...]
    return_type: pa.DataType
    volatility: str = "immutable"

    def register(self, context: SessionContext) -> None:
        context.register_udf(
            udf(
                self.function,
                list(self.input_types),
                self.return_type,
                self.volatility,
                self.name,
            )
        )


@dataclass(frozen=True)
class SqlQuery:
    """A query evaluated against the Arrow relation named ``input``."""

    text: str
    scalar_functions: tuple[SqlScalarFunction, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("SQL query must not be empty")


def sql(text: str, *, scalar_functions: tuple[SqlScalarFunction, ...] = ()) -> SqlQuery:
    """Create a SQL query for a group reducer."""
    return SqlQuery(text=text, scalar_functions=scalar_functions)


def apply_sql_query(context: SessionContext, input_frame: DataFrame, query: SqlQuery) -> DataFrame:
    """Plan ``query`` against ``input_frame`` in ``context``."""
    for scalar_function in query.scalar_functions:
        scalar_function.register(context)
    context.register_table(SQL_INPUT_TABLE, input_frame.into_view(temporary=True))
    return context.sql(query.text)


def execute_sql_batches(items: Iterable[object], query: SqlQuery) -> Iterator[pa.RecordBatch]:
    """Evaluate a batch-local SQL query over an Arrow item stream."""
    context = datafusion_context()
    for scalar_function in query.scalar_functions:
        scalar_function.register(context)

    input_schema: pa.Schema | None = None
    for batch in iter_record_batches(items):
        if input_schema is None:
            input_schema = batch.schema
        elif not input_schema.equals(batch.schema, check_metadata=True):
            raise ValueError(
                "SQL map input schema changed within one shard. "
                f"Expected:\n{input_schema}\nGot:\n{batch.schema}"
            )

        context.register_record_batches(SQL_INPUT_TABLE, [[batch]])
        try:
            output_frame = context.sql(query.text)
            output_schema = output_frame.schema()
            for output in output_frame.execute_stream():
                yield canonicalize_record_batch(output.to_pyarrow(), output_schema)
        finally:
            context.deregister_table(SQL_INPUT_TABLE)
