# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parquet previews for the CLI and the browser.

Parquet keeps its schema and its row-group index in a footer, so the bounded head read
that serves every other format returns nothing a reader can parse. These helpers open
the object and seek instead: the footer costs one small range request, and the rows come
from the first row group, which is decoded only when it is small enough to be worth the
transfer.
"""

from rigging.filesystem.buckets import filesystem_for
from rigging.fsutil.listing import MAX_PREVIEW_BYTES
from rigging.fsutil.render import aligned_lines, format_size, record_lines, table_lines

# Rows rendered when the caller asks for no particular count.
PREVIEW_ROWS = 20


def is_parquet(name: str) -> bool:
    """Whether *name* is a parquet file.

    Parquet compresses its own column chunks, so a name is matched as stored rather than
    through :func:`rigging.fsutil.compression.uncompressed_name`.
    """
    return name.lower().endswith(".parquet")


def parquet_lines(url: str, rows: int = PREVIEW_ROWS) -> list[str]:
    """Render *url* as its schema, its footer statistics, and its first *rows* rows."""
    # pyarrow is deliberately absent from marin-rigging's dependencies: rigging sits under
    # every other package, and one more requirement there re-resolves the whole workspace
    # lock. Every environment that holds parquet files already installs pyarrow.
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415  # optional dep
    except ImportError as exc:
        raise RuntimeError("reading parquet requires pyarrow; install it with `pip install pyarrow`") from exc

    fs, path = filesystem_for(url)
    file_size = fs.size(path)
    with fs.open(path, "rb") as file:
        parquet_file = pq.ParquetFile(file)
        metadata = parquet_file.metadata
        lines = ["schema:", *_schema_lines(parquet_file.schema_arrow), "", *_summary_lines(metadata, file_size), ""]
        lines.extend(_row_lines(parquet_file, metadata, rows))
    return lines


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
    """The first *rows* rows, or why they were not read.

    A batch is served out of the first row group, and parquet's smallest readable unit is
    a whole column chunk. So the cost of one row is the cost of the row group that holds
    it, and a row group above the preview limit is reported rather than pulled down.
    """
    if metadata.num_rows == 0:
        return ["(no rows)"]

    group = metadata.row_group(0)
    if group.total_byte_size > MAX_PREVIEW_BYTES:
        return [
            f"[rows not read: row group 0 holds {format_size(group.total_byte_size)} uncompressed, "
            f"above the {format_size(MAX_PREVIEW_BYTES)} preview limit — copy the file to read it]"
        ]

    # A batch may otherwise run on into the next row group, which would put the read
    # above the limit that the check just cleared.
    batch = next(parquet_file.iter_batches(batch_size=max(1, min(rows, group.num_rows))), None)
    if batch is None:
        return ["(no rows)"]
    lines = record_lines(batch.to_pylist())
    if metadata.num_rows > batch.num_rows:
        lines.append(f"[showing {batch.num_rows} of {metadata.num_rows} rows]")
    return lines
