# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External ordering and token-prefix writes for quality pool records."""

from collections.abc import Callable, Iterator
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
    requested_token_budget: int
    total_documents: int
    total_tokens: int
    overshoot_tokens: int

    @property
    def usable_tokens(self) -> int:
        """Return selected token mass without whole-document overshoot."""
        return self.total_tokens - self.overshoot_tokens


def _write_range(output_path: str) -> Callable[[int, Iterator[dict[str, Any]]], RangeTokenTotal]:
    def write(range_key: int, rows: Iterator[dict[str, Any]]) -> RangeTokenTotal:
        path = prefix_join(output_path, f"range-{range_key:05d}.parquet")
        ensure_parent_dir(path)
        documents = 0
        tokens = 0
        writer = None
        with StoragePath(path).open("wb") as stream:
            try:
                while batch := list(islice(rows, RANKED_BATCH_ROWS)):
                    clean_rows = [
                        {key: value for key, value in row.items() if not key.startswith("_ranked_")} for row in batch
                    ]
                    table = pa.Table.from_pylist(clean_rows)
                    if writer is None:
                        writer = pq.ParquetWriter(stream, table.schema)
                    writer.write_table(table)
                    documents += len(batch)
                    tokens += sum(row["_ranked_token_count"] for row in batch)
            finally:
                if writer is not None:
                    writer.close()
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
    token_count: Callable[[dict[str, Any]], int],
    range_key: Callable[[dict[str, Any]], int],
    num_ranges: int = 256,
) -> RankedPool:
    """Sort records within ordered ranges and write one bounded Parquet shard per range."""
    if num_ranges <= 0:
        raise ValueError("num_ranges must be positive")

    def add_rank_fields(row: dict[str, Any]) -> dict[str, Any]:
        key = range_key(row)
        rank = rank_key(row)
        count = token_count(row)
        if not isinstance(key, int) or not 0 <= key < num_ranges:
            raise ValueError("rank range must be an integer in [0, num_ranges)")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("ranked pool records require positive integer token counts")
        if any(name.startswith("_ranked_") for name in row):
            raise ValueError("ranked pool input uses a reserved field")
        return {**row, "_ranked_range": key, "_ranked_rank": rank, "_ranked_token_count": count}

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


def _write_prefix(
    source_path: str,
    output_path: str,
    *,
    token_budget: int,
) -> RangeTokenTotal:
    documents = 0
    tokens = 0
    writer = None
    ensure_parent_dir(output_path)
    with StoragePath(source_path).open("rb") as source, StoragePath(output_path).open("wb") as target:
        parquet = pq.ParquetFile(source)
        try:
            for batch in parquet.iter_batches(batch_size=RANKED_BATCH_ROWS):
                counts = batch.column(batch.schema.get_field_index("token_count")).to_pylist()
                end = 0
                for count in counts:
                    end += 1
                    documents += 1
                    tokens += count
                    if tokens >= token_budget:
                        break
                output = pa.Table.from_batches([batch.slice(0, end)])
                if writer is None:
                    writer = pq.ParquetWriter(target, output.schema)
                writer.write_table(output)
                if tokens >= token_budget:
                    break
        finally:
            if writer is not None:
                writer.close()
    return RangeTokenTotal(-1, output_path, documents, tokens)


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
        total_tokens = preceding_tokens + boundary.tokens
        total_documents = preceding_documents + boundary.documents
    else:
        boundary_output = prefix_join(output_path, "boundary.parquet")
        result = ctx.execute(
            Dataset.from_list([boundary.path]).map(
                lambda path: _write_prefix(path, boundary_output, token_budget=remaining)
            )
        ).results[0]
        paths.append(boundary_output)
        total_tokens = preceding_tokens + result.tokens
        total_documents = preceding_documents + result.documents
    return PrefixResult(
        tuple(paths),
        token_budget,
        total_documents,
        total_tokens,
        total_tokens - token_budget,
    )
