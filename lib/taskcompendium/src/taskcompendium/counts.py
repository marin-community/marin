# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the rows in pinned source payloads, per format the pipeline accepts.

The parquet counter reads footer metadata only, never data pages, and reads
through :class:`StoragePath`, so remote payloads — a pinned
``hf://datasets/<repo>@<revision>/...`` path among them — count identically to
local ones. Other formats gain counters when a consumer needs them.
"""

from pyarrow import parquet as pq
from rigging.filesystem.storage_path import StoragePath


def count_parquet_rows(path: StoragePath) -> int:
    """The file's row count from footer metadata, without reading data pages."""
    with path.open("rb") as handle:
        return pq.ParquetFile(handle).metadata.num_rows
