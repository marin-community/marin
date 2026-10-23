# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chunk sizing for turning Parquet row groups into Python records with bounded memory."""

# Target Python-object bytes per read chunk. ``zephyr.readers.load_parquet`` does
# ``yield from batch.to_pylist()``, converting an ENTIRE Parquet row group into Python
# dicts before yielding the first record -- roughly 3-4x the Arrow size. Truncation
# happens after the read, so it cannot bound this: a 944 MB row group still materializes
# in full. Slice the batch first; ``RecordBatch.slice`` is zero-copy, so only the slice
# becomes Python objects.
READ_CHUNK_TARGET_BYTES = 64 * 1024 * 1024


def rows_per_chunk(num_rows: int, nbytes: int, target_bytes: int = READ_CHUNK_TARGET_BYTES) -> int:
    """Rows to materialize at once, from a batch's own average row size.

    Megabyte-document sources get small chunks and small-document sources get large
    ones, without either needing to know the other's shape.
    """
    if num_rows <= 0:
        return 1
    avg_row_bytes = max(1, nbytes // num_rows)
    return max(1, target_bytes // avg_row_bytes)
