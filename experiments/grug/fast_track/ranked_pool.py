# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External ordering and token-prefix writes for quality pool records.

Records must carry a positive integer ``token_count`` column.
"""

from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import islice
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.writers import ensure_parent_dir

RANKED_BATCH_ROWS = 1024
RANGE_FILE_PREFIX = "range-"


@dataclass(frozen=True)
class RangeTokenTotal:
    range_key: int
    path: str
    documents: int
    tokens: int


@dataclass(frozen=True)
class RankedPool:
    shards: tuple[str, ...]
    range_totals: tuple[RangeTokenTotal, ...]
    total_documents: int
    total_tokens: int


@dataclass(frozen=True)
class PrefixResult:
    shards: tuple[str, ...]
    total_documents: int
    total_tokens: int


def write_parquet_tables(path: str, tables: Iterable[pa.Table], *, schema: pa.Schema | None = None) -> int:
    """Stream tables into one Parquet file and return its row count.

    The first table sets the file schema unless ``schema`` is given.
    """
    ensure_parent_dir(path)
    rows = 0
    with ExitStack() as stack:
        stream = stack.enter_context(StoragePath(path).open("wb"))
        writer = None
        for table in tables:
            if writer is None:
                writer = stack.enter_context(pq.ParquetWriter(stream, schema or table.schema))
            writer.write_table(table)
            rows += table.num_rows
    return rows


def _write_range(output_path: str) -> Callable[[int, Iterator[dict[str, Any]]], RangeTokenTotal]:
    def write(range_key: int, rows: Iterator[dict[str, Any]]) -> RangeTokenTotal:
        path = prefix_join(output_path, f"{RANGE_FILE_PREFIX}{range_key:05d}.parquet")
        tokens = 0

        def tables() -> Iterator[pa.Table]:
            nonlocal tokens
            while batch := list(islice(rows, RANKED_BATCH_ROWS)):
                tokens += sum(row["token_count"] for row in batch)
                yield pa.Table.from_pylist(
                    [{key: value for key, value in row.items() if not key.startswith("_ranked_")} for row in batch]
                )

        documents = write_parquet_tables(path, tables())
        if documents == 0:
            raise ValueError("rank range must contain at least one document")
        return RangeTokenTotal(range_key, path, documents, tokens)

    return write


def ranked_pool(
    dataset: Dataset[dict[str, Any]],
    *,
    ctx: ZephyrContext,
    output_path: str,
    rank_key: Callable[[dict[str, Any]], Any],
    range_key: Callable[[dict[str, Any]], int],
    num_ranges: int = 256,
) -> RankedPool:
    """Sort records within ordered ranges and write one bounded Parquet shard per range."""
    if num_ranges <= 0:
        raise ValueError("num_ranges must be positive")

    def add_rank_fields(row: dict[str, Any]) -> dict[str, Any]:
        key = range_key(row)
        count = row["token_count"]
        if not isinstance(key, int) or not 0 <= key < num_ranges:
            raise ValueError("rank range must be an integer in [0, num_ranges)")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("ranked pool records require positive integer token counts")
        if any(name.startswith("_ranked_") for name in row):
            raise ValueError("ranked pool input uses a reserved field")
        return {**row, "_ranked_range": key, "_ranked_rank": rank_key(row)}

    ranked = dataset.map(add_rank_fields)

    # The reducer streams each ordered range directly to its own file. Its small
    # return value carries the range totals to the coordinator.
    ranges = ctx.execute(
        ranked.group_by(
            key=lambda row: row["_ranked_range"],
            reducer=_write_range(output_path),
            sort_by=lambda row: row["_ranked_rank"],
            num_output_shards=num_ranges,
        )
    ).results
    ranges = tuple(sorted(ranges, key=lambda item: item.range_key))
    return RankedPool(
        tuple(item.path for item in ranges),
        ranges,
        sum(item.documents for item in ranges),
        sum(item.tokens for item in ranges),
    )


def _write_prefix(source_path: str, output_path: str, *, token_budget: int) -> tuple[int, int]:
    """Write the whole-document prefix that reaches ``token_budget`` and return its documents and tokens."""
    documents = 0
    tokens = 0

    def tables(parquet: pq.ParquetFile) -> Iterator[pa.Table]:
        nonlocal documents, tokens
        for batch in parquet.iter_batches(batch_size=RANKED_BATCH_ROWS):
            end = 0
            for count in batch.column("token_count").to_pylist():
                end += 1
                tokens += count
                if tokens >= token_budget:
                    break
            documents += end
            yield pa.Table.from_batches([batch.slice(0, end)])
            if tokens >= token_budget:
                return

    with StoragePath(source_path).open("rb") as source:
        write_parquet_tables(output_path, tables(pq.ParquetFile(source)))
    return documents, tokens


def take_token_prefix(
    pool: RankedPool,
    *,
    ctx: ZephyrContext,
    output_path: str,
    token_budget: int,
) -> PrefixResult:
    """Write the ordered whole-document prefix for a token budget."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if pool.total_tokens < token_budget:
        raise ValueError(f"ranked pool has {pool.total_tokens} usable tokens, below the {token_budget} token budget")

    preceding_tokens = 0
    preceding_documents = 0
    boundary = None
    for item in pool.range_totals:
        if preceding_tokens + item.tokens >= token_budget:
            boundary = item
            break
        preceding_tokens += item.tokens
        preceding_documents += item.documents
    if boundary is None:
        raise ValueError("ranked pool totals do not contain the requested token budget")

    remaining = token_budget - preceding_tokens
    paths = [item.path for item in pool.range_totals if item.range_key < boundary.range_key]
    if remaining == boundary.tokens:
        paths.append(boundary.path)
        boundary_documents, boundary_tokens = boundary.documents, boundary.tokens
    else:
        boundary_output = prefix_join(output_path, "boundary.parquet")
        boundary_documents, boundary_tokens = ctx.execute(
            Dataset.from_list([boundary.path]).map(
                lambda path: _write_prefix(path, boundary_output, token_budget=remaining)
            )
        ).results[0]
        paths.append(boundary_output)
    return PrefixResult(tuple(paths), preceding_documents + boundary_documents, preceding_tokens + boundary_tokens)
