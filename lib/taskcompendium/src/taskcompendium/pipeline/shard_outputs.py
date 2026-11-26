# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write several Parquet projections of one Zephyr shard from a single pass over its rows."""

import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from tempfile import SpooledTemporaryFile
from typing import IO, Any

import pyarrow as pa
from zephyr.dataset import ShardInfo, format_shard_path
from zephyr.writers import write_parquet_file

SPOOL_MEMORY_BYTES = 64 * 1024 * 1024
"""Rows of one shard held in memory before the spool moves to local disk."""


@dataclass(frozen=True)
class ShardOutput:
    """One Parquet file per shard: the rows ``project`` keeps, in ``schema``.

    ``template`` names the file with ``{shard}`` and ``{total}`` placeholders, as in
    ``Dataset.write_parquet``; ``project`` returns ``None`` for rows outside the file.
    """

    template: str
    schema: pa.Schema
    project: Callable[[dict[str, Any]], dict[str, Any] | None]


def _spooled(spool: IO[bytes]) -> Iterator[dict[str, Any]]:
    spool.seek(0)
    while True:
        try:
            yield pickle.load(spool)
        except EOFError:
            return


def write_shard_outputs(rows: Iterable[dict[str, Any]], shard: ShardInfo, outputs: Sequence[ShardOutput]) -> list[str]:
    """Write every output of one shard with Zephyr's Parquet writer and return their paths.

    ``rows`` is consumed once into a local spool, so a lazy upstream transform, such as a
    model review, runs once however many files the shard writes. Every output is written,
    empty when it keeps no row, as ``Dataset.write_parquet`` does for an empty shard.
    """
    with SpooledTemporaryFile(max_size=SPOOL_MEMORY_BYTES) as spool:
        for row in rows:
            pickle.dump(row, spool, protocol=pickle.HIGHEST_PROTOCOL)
        paths = []
        for output in outputs:
            path = format_shard_path(output.template, shard.shard_idx, shard.total_shards)
            projected = (kept for row in _spooled(spool) if (kept := output.project(row)) is not None)
            write_parquet_file(projected, path, schema=output.schema)
            paths.append(path)
        return paths
