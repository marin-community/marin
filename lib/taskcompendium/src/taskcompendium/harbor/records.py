# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed source/path lookup with a bounded normalized-Parquet read buffer."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


class NormalizedIndex:
    """Index row locations, retaining at most one batch of task payloads in memory."""

    def __init__(self, root: Path, database: Path):
        self.connection = sqlite3.connect(database)
        self.connection.execute(
            "CREATE TABLE locations (path TEXT PRIMARY KEY, file INTEGER, "
            "row_group INTEGER, offset INTEGER, seen INTEGER DEFAULT 0)"
        )
        self.files = sorted(root.glob("*.parquet"))
        self.group: tuple[int, int] | None = None
        self.batch: list[dict[str, Any]] = []
        self.start = 0
        self.batches: Iterator[list[dict[str, Any]]] = iter(())
        for file_index, path in enumerate(self.files):
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                offset = 0
                for batch in parquet.iter_batches(batch_size=4096, columns=["original_path"], row_groups=[group]):
                    values = batch.column(0).to_pylist()
                    if any(value is None for value in values):
                        raise ValueError(f"Missing original source path in {path}")
                    self.connection.executemany(
                        "INSERT INTO locations(path,file,row_group,offset) VALUES (?,?,?,?)",
                        [(value, file_index, group, offset + index) for index, value in enumerate(values)],
                    )
                    offset += len(values)
        self.connection.commit()

    def __enter__(self) -> "NormalizedIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.connection.close()

    def _row(self, file: int, group: int, offset: int) -> dict[str, Any]:
        if self.group != (file, group) or offset < self.start:
            self.group = (file, group)
            self.start = 0
            self.batch = []
            self.batches = (
                batch.to_pylist()
                for batch in pq.ParquetFile(self.files[file]).iter_batches(batch_size=16, row_groups=[group])
            )
        while offset >= self.start + len(self.batch):
            self.start += len(self.batch)
            self.batch = next(self.batches)
        return self.batch[offset - self.start]

    def get(self, path: str) -> dict[str, Any] | None:
        location = self.connection.execute(
            "SELECT file,row_group,offset,seen FROM locations WHERE path=?", (path,)
        ).fetchone()
        if location is None:
            return None
        file, group, offset, seen = location
        if seen:
            raise ValueError(f"Duplicate baseline source path: {path}")
        self.connection.execute("UPDATE locations SET seen=1 WHERE path=?", (path,))
        return self._row(file, group, offset)

    def unmatched(self) -> Iterator[dict[str, Any]]:
        for file, group, offset in self.connection.execute(
            "SELECT file,row_group,offset FROM locations WHERE seen=0 ORDER BY file,row_group,offset"
        ):
            yield self._row(file, group, offset)
