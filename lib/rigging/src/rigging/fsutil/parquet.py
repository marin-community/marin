# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parquet previews for the CLI and the browser.

Parquet keeps its schema and its row-group index in a footer, so the bounded head read
that serves every other format returns nothing a reader can parse. These helpers seek
instead: the footer gives the schema and the statistics, and the rows come out of the
first row group, which is decoded only when it stays inside the preview limit.
"""

from rigging.filesystem.buckets import filesystem_for
from rigging.fsutil.listing import MAX_PREVIEW_BYTES
from rigging.fsutil.render import aligned_lines, format_size, record_lines, table_lines

# Rows rendered when the caller asks for no particular count.
PREVIEW_ROWS = 20


class MissingParquetReader(RuntimeError):
    """Raised when a preview runs where no parquet reader is installed."""


def is_parquet(name: str) -> bool:
    """Whether *name* is a parquet file.

    Parquet compresses its own column chunks, so a name is matched as stored rather than
    through :func:`rigging.fsutil.compression.uncompressed_name`.
    """
    return name.lower().endswith(".parquet")


def parquet_lines(url: str, rows: int = PREVIEW_ROWS) -> list[str]:
    """Render *url* as its schema, its footer statistics, and its first *rows* rows."""
    # pyarrow is deliberately absent from marin-rigging's dependencies: rigging sits under
    # every other package, and one more requirement there re-resolves the workspace lock.
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415  # optional dep
    except ImportError as exc:
        raise MissingParquetReader("reading parquet requires pyarrow; install it with `pip install pyarrow`") from exc

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
