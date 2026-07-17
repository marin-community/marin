# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the smoke fixtures: a fact table and a small dim table, as Parquet.

Run host-side (``uv run --with pyarrow python gen_fixture.py``); the Parquet files are
bind-mounted read-only into the Ballista containers at /data. Kept tiny — the point is
to exercise a scan, a high-cardinality GROUP BY (shuffle), and a join, not scale.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

N = 100_000
OUT = Path(__file__).parent / "fixture"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Fact table: id, grp in [0,1000) (high cardinality for the shuffle), value.
    ids = list(range(N))
    nums = pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "grp": pa.array([i % 1000 for i in ids], pa.int64()),
            "value": pa.array([i * 2 for i in ids], pa.int64()),
        }
    )
    pq.write_table(nums, OUT / "nums.parquet")

    # Dim table for the join: one label per grp.
    dims = pa.table(
        {
            "grp": pa.array(list(range(1000)), pa.int64()),
            "label": pa.array([f"g{g:04d}" for g in range(1000)], pa.string()),
        }
    )
    pq.write_table(dims, OUT / "dims.parquet")

    print(f"wrote {OUT/'nums.parquet'} ({N} rows) and {OUT/'dims.parquet'} (1000 rows)")


if __name__ == "__main__":
    main()
