# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming writers shared by the pinned benchmark extractors."""

import json
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

_PARQUET_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])

_PARQUET_BATCH = 1000


def _write_parquet(path: str, records: Iterator[dict]) -> int:
    """Write ``records`` ({id, text}) to a single parquet file at ``path``.

    Streams in ``_PARQUET_BATCH``-row chunks so memory stays bounded for tasks
    with tens of thousands of docs (bbq=58k, swag=20k, babi=20k, ...).
    Compression: zstd. zephyr's parquet reader picks up the file regardless
    of the compression codec.
    """
    parent = StoragePath(path).parent
    if parent.key:
        parent.mkdirs()
    n = 0
    batch_ids: list[str] = []
    batch_texts: list[str] = []
    with StoragePath(path).open("wb") as raw:
        writer = pq.ParquetWriter(raw, _PARQUET_SCHEMA, compression="zstd")
        try:
            for rec in records:
                batch_ids.append(rec["id"])
                batch_texts.append(rec["text"])
                if len(batch_ids) >= _PARQUET_BATCH:
                    writer.write_table(pa.table({"id": batch_ids, "text": batch_texts}, schema=_PARQUET_SCHEMA))
                    n += len(batch_ids)
                    batch_ids, batch_texts = [], []
            if batch_ids:
                writer.write_table(pa.table({"id": batch_ids, "text": batch_texts}, schema=_PARQUET_SCHEMA))
                n += len(batch_ids)
        finally:
            writer.close()
    return n


def _write_json(path: str, value: dict[str, Any]) -> None:
    parent = StoragePath(path).parent
    if parent.key:
        parent.mkdirs()
    with StoragePath(path).open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
