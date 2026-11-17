# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete causal attribution preserves every reference across index batches."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.decontam.viewer import export_reference_run


@pytest.mark.parametrize("batch_rows", [2, 100])
@pytest.mark.parametrize(
    ("hash_type", "edge"),
    [(pa.uint64(), 2**64 - 1), (pa.int64(), -(2**63))],
)
def test_complete_causal_join_preserves_large_hashes_and_all_eval_ids(
    tmp_path, monkeypatch, batch_rows, hash_type, edge
):
    path = tmp_path / "eval-index.parquet"
    pq.write_table(
        pa.table(
            {
                "hash": pa.array([7, 99, edge, None, 7, edge, 7, 100], type=hash_type),
                "eval_id": ["first", "unrelated", "edge-first", "null", "second", "edge-second", "first", "other"],
            }
        ),
        path,
        row_group_size=2,
    )
    monkeypatch.setattr(export_reference_run, "INDEX_BATCH_ROWS", batch_rows)

    assert export_reference_run._hash_to_evals(str(path), {7, edge, 101}) == {
        7: {"first", "second"},
        edge: {"edge-first", "edge-second"},
    }
