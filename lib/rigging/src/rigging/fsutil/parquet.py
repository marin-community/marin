# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parquet previews for the CLI and the browser.

Parquet keeps its schema and its row-group index in a footer, so the bounded head read
that serves every other format returns nothing a reader can parse. These helpers seek
instead: the footer gives the schema and the statistics, and rows decode a row group at
a time — :func:`parquet_lines` from the first group alone for ``cat`` and ``head``, and
:class:`ParquetViewSource` batch by batch as the interactive viewer pages forward. A
row group above the preview limit is reported instead of decoded either way.
"""

from rigging.filesystem.buckets import filesystem_for
from rigging.fsutil.listing import MAX_PREVIEW_BYTES
from rigging.fsutil.render import (
    aligned_lines,
    cell,
    column_widths,
    format_size,
    header_lines,
    record_lines,
    row_line,
    table_lines,
)

# Rows rendered when the caller asks for no particular count.
PREVIEW_ROWS = 20

# Rows decoded per fetch when the interactive viewer pages through a file.
VIEW_BATCH_ROWS = 100


class MissingParquetReader(RuntimeError):
    """Raised when a preview runs where no parquet reader is installed."""


def _pyarrow_parquet():
    # pyarrow is deliberately absent from marin-rigging's dependencies: rigging sits under
    # every other package, and one more requirement there re-resolves the workspace lock.
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415  # optional dep
    except ImportError as exc:
        raise MissingParquetReader("reading parquet requires pyarrow; install it with `pip install pyarrow`") from exc
    return pq


def is_parquet(name: str) -> bool:
    """Whether *name* is a parquet file.

    Parquet compresses its own column chunks, so a name is matched as stored rather than
    through :func:`rigging.fsutil.compression.uncompressed_name`.
    """
    return name.lower().endswith(".parquet")


def parquet_lines(url: str, rows: int = PREVIEW_ROWS) -> list[str]:
    """Render *url* as its schema, its footer statistics, and its first *rows* rows."""
    pq = _pyarrow_parquet()
    fs, path = filesystem_for(url)
    file_size = fs.size(path)
    with fs.open(path, "rb") as file:
        parquet_file = pq.ParquetFile(file)
        metadata = parquet_file.metadata
        lines = ["schema:", *_schema_lines(parquet_file.schema_arrow), "", *_summary_lines(metadata, file_size)]
        rendered_rows = _row_lines(parquet_file, metadata, rows)
    return [*lines, "", *rendered_rows] if rendered_rows else lines


def _schema_lines(schema) -> list[str]:
    return table_lines(["column", "type"], [[field.name, str(field.type)] for field in schema])


def _summary_lines(metadata, file_size: int) -> list[str]:
    return aligned_lines(
        [
            ["rows", str(metadata.num_rows)],
            ["columns", str(metadata.num_columns)],
            ["row groups", str(metadata.num_row_groups)],
            ["file size", format_size(file_size)],
            ["created by", metadata.created_by or "-"],
        ]
    )


def _row_lines(parquet_file, metadata, rows: int) -> list[str]:
    """The first *rows* rows of row group 0, or why they were not read.

    A short first group yields fewer rows than asked, and the count below the table says
    so. Row group 0 is the whole budget because parquet's smallest readable unit is a
    column chunk.
    """
    if rows <= 0:
        return []
    if metadata.num_rows == 0:
        return ["(no rows)"]

    group = metadata.row_group(0)
    if group.total_byte_size > MAX_PREVIEW_BYTES:
        return [
            f"[rows not read: row group 0 holds {format_size(group.total_byte_size)} uncompressed, "
            f"above the {format_size(MAX_PREVIEW_BYTES)} preview limit — copy the file to read it]"
        ]

    batch = next(parquet_file.iter_batches(batch_size=rows, row_groups=[0]), None)
    if batch is None:
        return ["(no rows)"]
    lines = record_lines(batch.to_pylist())
    if metadata.num_rows > batch.num_rows:
        lines.append(f"[showing {batch.num_rows} of {metadata.num_rows} rows]")
    return lines


class ParquetViewSource:
    """Feeds the interactive viewer one batch of parquet rows at a time.

    The viewer scans forward, so rows decode row group by row group and never hold more
    than one batch. A row group above the preview limit is reported and skipped rather
    than decoded, because parquet's smallest readable unit is a column chunk. Column
    widths freeze on the first batch so later batches align with the rendered header.
    """

    def __init__(self, url: str, batch_rows: int = VIEW_BATCH_ROWS):
        pq = _pyarrow_parquet()
        fs, path = filesystem_for(url)
        self._file_size = fs.size(path)
        self._file = fs.open(path, "rb")
        self._parquet = pq.ParquetFile(self._file)
        self._batch_rows = batch_rows
        self._headers = [field.name for field in self._parquet.schema_arrow]
        self._widths: list[int] | None = None
        self._group = 0
        self._batches = None
        self._rows_rendered = 0
        self._finished = False

    def __enter__(self) -> "ParquetViewSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    def head_lines(self) -> list[str]:
        """The schema and footer statistics shown above the rows."""
        metadata = self._parquet.metadata
        return [
            "schema:",
            *_schema_lines(self._parquet.schema_arrow),
            "",
            *_summary_lines(metadata, self._file_size),
            "",
        ]

    def more_lines(self) -> list[str]:
        """Render the next batch of rows, then a closing marker, then nothing."""
        metadata = self._parquet.metadata
        while not self._finished:
            if self._batches is not None:
                batch = next(self._batches, None)
                if batch is not None:
                    return self._render(batch)
                self._batches = None
                self._group += 1
            if self._group >= metadata.num_row_groups:
                self._finished = True
                if self._rows_rendered == metadata.num_rows:
                    return [f"[end of {metadata.num_rows} rows]"]
                return [f"[end: showed {self._rows_rendered} of {metadata.num_rows} rows]"]
            group = metadata.row_group(self._group)
            if group.total_byte_size > MAX_PREVIEW_BYTES:
                skipped = self._group
                self._group += 1
                return [
                    f"[row group {skipped} not read: {format_size(group.total_byte_size)} uncompressed, "
                    f"above the {format_size(MAX_PREVIEW_BYTES)} preview limit]"
                ]
            self._batches = self._parquet.iter_batches(batch_size=self._batch_rows, row_groups=[self._group])
        return []

    def _render(self, batch) -> list[str]:
        rows = [[cell(record.get(header)) for header in self._headers] for record in batch.to_pylist()]
        self._rows_rendered += len(rows)
        if self._widths is None:
            self._widths = column_widths(self._headers, rows)
            return [*header_lines(self._headers, self._widths), *(row_line(row, self._widths) for row in rows)]
        return [row_line(row, self._widths) for row in rows]
