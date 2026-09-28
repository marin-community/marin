# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-shard driver's pieces: shard pairing, aligned reads and rebatching."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.cluster.quality.fast_transformer.shards import AlignedColumn, paired_basenames, rebatch


def test_rebatch_yields_full_batches_and_one_tail():
    batches = [
        pa.RecordBatch.from_pydict({"id": [str(i) for i in range(start, start + n)]})
        for start, n in ((0, 3), (3, 5), (8, 1))
    ]

    out = list(rebatch(iter(batches), 4))

    assert [b.num_rows for b in out] == [4, 4, 1]
    assert [i for b in out for i in b.column("id").to_pylist()] == [str(i) for i in range(9)]


def aligned(ids: list[str]) -> AlignedColumn:
    return AlignedColumn(np.array(ids, dtype=object), np.arange(len(ids), dtype=np.float32), "shard")


def test_aligned_column_takes_rows_in_order_across_batches():
    # A duplicate id is two documents at two positions, not one row to share.
    side = aligned(["b", "a", "dup", "dup"])

    first = side.take(np.array(["b", "a"], dtype=object))
    second = side.take(np.array(["dup", "dup"], dtype=object))

    assert first.tolist() == [0.0, 1.0] and second.tolist() == [2.0, 3.0]
    side.require_consumed()


def test_aligned_column_refuses_ids_out_of_step():
    side = aligned(["a", "b"])
    with pytest.raises(ValueError, match=r"rows 0\.\.2 do not carry"):
        side.take(np.array(["b", "a"], dtype=object))


def test_aligned_column_refuses_more_documents_than_rows():
    side = aligned(["a"])
    with pytest.raises(ValueError, match="do not carry"):
        side.take(np.array(["a", "b"], dtype=object))


def test_aligned_column_refuses_rows_left_over():
    side = aligned(["a", "b"])
    side.take(np.array(["a"], dtype=object))
    with pytest.raises(ValueError, match="2 rows against 1 documents"):
        side.require_consumed()


def write_parquet(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        pq.write_table(pa.table({"id": pa.array([], pa.string())}), directory / name)


def test_paired_basenames_refuses_an_asymmetric_leaf(tmp_path):
    """A basename one side lacks is a document set that would leave no trace in the output."""
    names = ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]
    write_parquet(tmp_path / "text", names)
    write_parquet(tmp_path / "embed", names)
    write_parquet(tmp_path / "other", [names[0], "part-00002-of-00003.parquet"])

    assert paired_basenames(str(tmp_path / "text"), str(tmp_path / "embed")) == names
    with pytest.raises(ValueError, match="not co-partitioned"):
        paired_basenames(str(tmp_path / "text"), str(tmp_path / "other"))
    with pytest.raises(FileNotFoundError):
        paired_basenames(str(tmp_path / "empty"), str(tmp_path / "embed"))
