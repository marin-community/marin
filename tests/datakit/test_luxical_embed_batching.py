# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory bounds for the Luxical embed step.

Embedding a source whose documents are megabytes each exhausts a worker, because
``batch_size`` bounds document *count* rather than bytes. Two bounds fix that, and both
live in the read path so nothing downstream can retain oversized text:

* :func:`sample_document` caps one document. Head truncation matches the Harrier
  pipeline (#7998), which truncates at 8,192 tokens.
* :func:`rows_per_chunk` caps how much of a Parquet row group becomes Python objects at
  once. ``zephyr.readers.load_parquet`` converts a whole row group via ``to_pylist()``
  before yielding one record, which no downstream bound can undo.

Truncation happens in :func:`_load_records_bounded` rather than in the map, because the
map runs after ``window``: truncating there would still let a window retain
``batch_size`` untruncated documents.
"""

import pyarrow as pa
import pyarrow.parquet as pq
from zephyr.readers import load_parquet

from experiments.datakit.embeddings.luxical import pipeline as luxical_pipeline
from experiments.datakit.embeddings.luxical.pipeline import (
    _load_records_bounded,
    rows_per_chunk,
    sample_document,
)

# ---- per-document truncation -------------------------------------------------


def test_document_at_or_below_cap_is_unchanged():
    """The common case must be byte-identical, or ordinary sources would shift."""
    assert sample_document("abc", 100) == "abc"
    assert sample_document("x" * 100, 100) == "x" * 100


def test_document_above_cap_is_truncated_to_its_head():
    assert sample_document("abcdefghij", 4) == "abcd"


def test_truncation_never_exceeds_the_cap():
    for n in (1_000, 10_000, 5_000_000):
        for cap in (90, 1_000, 32 * 1024):
            assert len(sample_document("z" * n, cap)) <= cap


def test_truncation_bounds_a_pathological_document():
    """The case that defeats every batch-level bound: a multi-MB single document."""
    assert len(sample_document("q" * 20_000_000, 32 * 1024)) == 32 * 1024


# ---- chunk sizing ------------------------------------------------------------


def test_rows_per_chunk_shrinks_as_rows_grow():
    """A megabyte-document source must get far fewer rows per chunk than a small one."""
    target = 64 * 1024 * 1024
    small = rows_per_chunk(num_rows=10_000, nbytes=10_000 * 2_000, target_bytes=target)
    large = rows_per_chunk(num_rows=10_000, nbytes=10_000 * 5_000_000, target_bytes=target)
    assert large < small
    assert large >= 1, "never zero: a single oversized row must still be materializable"


def test_rows_per_chunk_stays_within_the_byte_target():
    target = 1_000_000
    avg = 250_000
    n = rows_per_chunk(num_rows=100, nbytes=100 * avg, target_bytes=target)
    assert n * avg <= target


def test_rows_per_chunk_handles_an_empty_batch():
    assert rows_per_chunk(num_rows=0, nbytes=0, target_bytes=1_000) == 1


# ---- bounded reader ----------------------------------------------------------


def _write_shard(path, n_rows, text_len, row_group_size):
    table = pa.table(
        {
            "id": [f"d{i}" for i in range(n_rows)],
            "text": [f"{i}-" + "x" * text_len for i in range(n_rows)],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


class _CountingBatch:
    """Wraps a RecordBatch and records the size of every ``to_pylist`` the reader performs.

    The oracle for the memory bound is *how much is materialized at once*, which record
    equality cannot see: a reader that regressed to one whole-row-group ``to_pylist()``
    would still return identical records.
    """

    def __init__(self, batch, sizes=None):
        self._batch = batch
        self.pylist_sizes = sizes if sizes is not None else []

    @property
    def num_rows(self):
        return self._batch.num_rows

    @property
    def nbytes(self):
        return self._batch.nbytes

    def slice(self, offset, length):
        return _CountingBatch(self._batch.slice(offset, length), self.pylist_sizes)

    def to_pylist(self):
        rows = self._batch.to_pylist()
        self.pylist_sizes.append(len(rows))
        return rows


def test_reader_materializes_a_row_group_in_several_bounded_pieces(tmp_path, monkeypatch):
    """Regressing to a single whole-row-group to_pylist() must fail this test."""
    path = str(tmp_path / "big.parquet")
    _write_shard(path, n_rows=40, text_len=10_000, row_group_size=40)

    batch = next(iter(pq.ParquetFile(path).iter_batches(batch_size=40)))
    wrapper = _CountingBatch(batch)
    monkeypatch.setattr(luxical_pipeline, "load_parquet_batch", lambda _source: iter([wrapper]))
    monkeypatch.setattr(luxical_pipeline, "_READ_CHUNK_TARGET_BYTES", 20_000)

    records = list(_load_records_bounded(path, doc_sample_chars=10_000_000))

    assert len(wrapper.pylist_sizes) > 1, f"materialized in one piece: {wrapper.pylist_sizes}"
    assert max(wrapper.pylist_sizes) < 40, "no single materialization may cover the whole row group"
    assert sum(wrapper.pylist_sizes) == 40, "every row materialized exactly once"
    assert [r["id"] for r in records] == [f"d{i}" for i in range(40)]


def test_reader_truncates_before_yielding(tmp_path):
    """Truncation must happen in the reader, so windowing downstream cannot retain raw text."""
    path = str(tmp_path / "long.parquet")
    _write_shard(path, n_rows=4, text_len=50_000, row_group_size=4)

    records = list(_load_records_bounded(path, doc_sample_chars=100))

    assert [len(r["text"]) for r in records] == [100, 100, 100, 100]


def test_reader_preserves_records_when_nothing_is_truncated(tmp_path):
    """With the cap above every document, output must equal zephyr's reader exactly."""
    path = str(tmp_path / "shard.parquet")
    _write_shard(path, n_rows=500, text_len=200, row_group_size=64)

    assert list(_load_records_bounded(path, doc_sample_chars=10_000_000)) == list(load_parquet(path))


def test_reader_handles_empty_shard(tmp_path):
    path = str(tmp_path / "empty.parquet")
    pq.write_table(pa.table({"id": pa.array([], type=pa.string()), "text": pa.array([], type=pa.string())}), path)
    assert list(_load_records_bounded(path)) == []
