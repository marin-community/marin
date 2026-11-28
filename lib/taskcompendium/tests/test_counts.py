# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.counts import count_parquet_rows


def test_parquet_footer_counts_rows_across_row_groups(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"n": list(range(7))}), path, row_group_size=3)
    assert count_parquet_rows(StoragePath(str(path))) == 7


def test_memory_filesystem_counts_like_a_local_one():
    filesystem = StoragePath("memory://counts-bucket/rows.parquet")
    with filesystem.open("wb") as handle:
        pq.write_table(pa.table({"n": list(range(5))}), handle)
    assert count_parquet_rows(filesystem) == 5
