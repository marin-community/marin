# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Server-side Parquet reader for the TaskTrove browser."""

import bisect
import json
import threading
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException, Query, Response, status
from iris.cluster.controller.lru_cache import LRUCache
from marina.apps import RegisteredApi, Services, registered_api
from pyarrow.fs import FSSpecHandler, PyFileSystem
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from starlette.responses import StreamingResponse

TASKS_FILE = "tasks/part-00000.parquet"
MANIFEST_FILE = "manifest.json"
PAGE_LIMIT = 100
ROW_GROUP_CACHE_SIZE = 8
FILTER_CACHE_SIZE = 32
PAGE_CACHE_SIZE = 128
METADATA_COLUMNS = (
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


class TaskRow(BaseModel):
    """One TaskTrove row without its binary task archive."""

    model_config = ConfigDict(frozen=True)

    row: int
    source: str
    path: str
    family: str
    converter: str
    mode: str
    dockerfile_id: str
    environment: str
    language: str
    tags: tuple[str, ...]
    has_solution: bool


class TaskPage(BaseModel):
    """One filtered page and the total number of matching rows."""

    model_config = ConfigDict(frozen=True)

    rows: tuple[TaskRow, ...]
    total: int
    offset: int
    limit: int


@dataclass(frozen=True)
class TaskFilters:
    source: str = ""
    converter: str = ""
    mode: str = ""
    tag: str = ""
    environment: str = ""
    query: str = ""


class TaskStore:
    """A Parquet task table with process-local column, row-group, and query caches."""

    def __init__(self, data_url: str) -> None:
        self._fs, self._root = filesystem_for(data_url)
        self._pa_fs = PyFileSystem(FSSpecHandler(self._fs))
        self._tasks_path = prefix_join(self._root, TASKS_FILE)
        self._manifest_path = prefix_join(self._root, MANIFEST_FILE)
        self._load_lock = threading.Lock()
        self._row_group_ends: tuple[int, ...] | None = None
        self._environments: dict[str, str] | None = None
        self._columns: dict[str, pa.Array] = {}
        self._row_groups = LRUCache[int, pa.Table](ROW_GROUP_CACHE_SIZE)
        self._matches = LRUCache[TaskFilters, pa.Array](FILTER_CACHE_SIZE)
        self._pages = LRUCache[tuple[int, int, TaskFilters], TaskPage](PAGE_CACHE_SIZE)

    def _load_index(self) -> None:
        if self._row_group_ends is not None:
            return
        with self._load_lock:
            if self._row_group_ends is not None:
                return
            with self._fs.open(self._tasks_path, "rb") as handle:
                parquet = pq.ParquetFile(handle)
                row_group_ends: list[int] = []
                rows = 0
                for index in range(parquet.num_row_groups):
                    rows += parquet.metadata.row_group(index).num_rows
                    row_group_ends.append(rows)
            with self._fs.open(self._manifest_path, "rb") as handle:
                manifest = json.load(handle)
            self._row_group_ends = tuple(row_group_ends)
            self._environments = {
                dockerfile_id: details["base_image"] for dockerfile_id, details in manifest["dockerfiles"].items()
            }

    @property
    def num_rows(self) -> int:
        self._load_index()
        assert self._row_group_ends is not None
        return self._row_group_ends[-1] if self._row_group_ends else 0

    def _column(self, name: str) -> pa.Array:
        cached = self._columns.get(name)
        if cached is not None:
            return cached
        with self._load_lock:
            cached = self._columns.get(name)
            if cached is not None:
                return cached
            column = pq.read_table(
                self._tasks_path,
                filesystem=self._pa_fs,
                columns=[name],
                use_threads=True,
            )[name].combine_chunks()
            self._columns[name] = column
            return column

    def _environment_ids(self, environment: str) -> list[str]:
        self._load_index()
        assert self._environments is not None
        return [dockerfile_id for dockerfile_id, image in self._environments.items() if image == environment]

    def _row_group_metadata(self, row_group: int) -> pa.Table:
        cached = self._row_groups.get(row_group)
        if cached is not None:
            return cached
        with self._fs.open(self._tasks_path, "rb") as handle:
            table = pq.ParquetFile(handle).read_row_group(row_group, columns=list(METADATA_COLUMNS))
        return self._row_groups.put(row_group, table)

    def _position(self, row: int) -> tuple[int, int]:
        self._load_index()
        assert self._row_group_ends is not None
        if row < 0 or not self._row_group_ends or row >= self._row_group_ends[-1]:
            raise KeyError(row)
        row_group = bisect.bisect_right(self._row_group_ends, row)
        start = 0 if row_group == 0 else self._row_group_ends[row_group - 1]
        return row_group, row - start

    def _row(self, row: int, table: pa.Table, local_row: int) -> TaskRow:
        record: dict[str, Any] = table.slice(local_row, 1).to_pylist()[0]
        self._load_index()
        assert self._environments is not None
        dockerfile_id = record["dockerfile_id"]
        return TaskRow(
            row=row,
            source=record["source"],
            path=record["path"],
            family=record["family"],
            converter=record["converter"],
            mode=record["mode"],
            dockerfile_id=dockerfile_id,
            environment=self._environments.get(dockerfile_id, dockerfile_id),
            language=record["language"] or "",
            tags=tuple(record["tags"] or ()),
            has_solution=record["has_solution"],
        )

    def _rows(self, row_numbers: list[int]) -> tuple[TaskRow, ...]:
        located = [(row, *self._position(row)) for row in row_numbers]
        tables = {row_group: self._row_group_metadata(row_group) for _, row_group, _ in located}
        return tuple(self._row(row, tables[row_group], local_row) for row, row_group, local_row in located)

    def _matching_rows(
        self,
        filters: TaskFilters,
    ) -> pa.Array | None:
        cached = self._matches.get(filters)
        if cached is not None:
            return cached
        mask: pa.Array | None = None

        def add(next_mask: pa.Array) -> None:
            nonlocal mask
            mask = next_mask if mask is None else pc.and_(mask, next_mask)

        for column, value in (
            ("source", filters.source),
            ("converter", filters.converter),
            ("mode", filters.mode),
        ):
            if value:
                add(pc.equal(self._column(column), value))
        if filters.environment:
            environment_ids = self._environment_ids(filters.environment)
            if not environment_ids:
                return self._matches.put(filters, pa.array([], type=pa.int64()))
            add(pc.is_in(self._column("dockerfile_id"), value_set=pa.array(environment_ids)))
        if filters.tag:
            tags = self._column("tags")
            tag_rows = pc.filter(pc.list_parent_indices(tags), pc.equal(pc.list_flatten(tags), filters.tag))
            add(pc.is_in(pa.array(range(self.num_rows), type=pa.int64()), value_set=tag_rows))
        if filters.query:
            add(pc.match_substring(self._column("path"), filters.query, ignore_case=True))
        matches = None if mask is None else pc.indices_nonzero(mask)
        return matches if matches is None else self._matches.put(filters, matches)

    def page(
        self,
        offset: int,
        limit: int,
        filters: TaskFilters,
    ) -> TaskPage:
        """Return one page after applying the exact API filter tuple."""
        key = (offset, limit, filters)
        cached = self._pages.get(key)
        if cached is not None:
            return cached
        matches = self._matching_rows(filters)
        if matches is None:
            total = self.num_rows
            row_numbers = list(range(offset, min(offset + limit, total)))
        else:
            total = len(matches)
            row_numbers = matches.slice(offset, limit).to_pylist()
        result = TaskPage(rows=self._rows(row_numbers), total=total, offset=offset, limit=limit)
        return self._pages.put(key, result)

    def task(self, row: int) -> TaskRow:
        """Return one metadata row by its stable Parquet position."""
        row_group, local_row = self._position(row)
        return self._row(row, self._row_group_metadata(row_group), local_row)

    def archive(self, row: int) -> bytes:
        """Read one task archive directly from object storage."""
        row_group, local_row = self._position(row)
        with self._fs.open(self._tasks_path, "rb") as handle:
            column = pq.ParquetFile(handle).read_row_group(row_group, columns=["task_binary"])["task_binary"]
        value = column[local_row].as_py()
        if not isinstance(value, bytes):
            raise ValueError(f"Parquet row {row} has no binary task archive")
        return value


def create_api(services: Services) -> RegisteredApi:
    """Create the authenticated row-oriented TaskTrove API."""
    api = FastAPI(title="TaskTrove API", docs_url=None, redoc_url=None, openapi_url=None)
    store = TaskStore(services.data_url)

    @api.get("/tasks", response_model=TaskPage)
    def tasks(
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=PAGE_LIMIT),
        source: str = "",
        converter: str = "",
        mode: str = "",
        tag: str = "",
        environment: str = "",
        query: str = "",
    ) -> TaskPage:
        return store.page(
            offset,
            limit,
            TaskFilters(
                source=source.strip(),
                converter=converter.strip(),
                mode=mode.strip(),
                tag=tag.strip(),
                environment=environment.strip(),
                query=query.strip(),
            ),
        )

    @api.get("/tasks/{row}", response_model=TaskRow)
    def task(row: int) -> TaskRow:
        try:
            return store.task(row)
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No Parquet row {row}") from None

    @api.get("/tasks/{row}/archive", response_class=Response)
    def archive(row: int) -> StreamingResponse:
        try:
            body = store.archive(row)
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No Parquet row {row}") from None
        return StreamingResponse(
            iter((body,)),
            media_type="application/gzip",
            headers={"Content-Disposition": f'attachment; filename="task-{row}.tar.gz"'},
        )

    return registered_api(api)
