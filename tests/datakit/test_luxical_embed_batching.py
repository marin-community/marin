# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Memory bounds for the Luxical embed step.

Two bounds, each covering a distinct blowup, both needed:

* :func:`sample_document` caps one document. Code sources carry documents of many
  megabytes, and a single one must be embedded in one call however small the window is.
  Head truncation matches the Harrier pipeline (marin#7998), which truncates at 8,192
  tokens. With each document bounded, a whole ``batch_size`` window is bounded too.
* ``_load_records_bounded`` caps how much of a Parquet row group becomes Python objects
  at once. Truncation cannot do this -- it runs after the read, so a 944 MB row group
  would still materialize in full.
"""

import pyarrow as pa
import pyarrow.parquet as pq
from zephyr.readers import load_parquet

from experiments.datakit.embeddings.luxical import pipeline as luxical_pipeline
from experiments.datakit.embeddings.luxical.pipeline import _load_records_bounded, sample_document

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
    """The case that defeated every batch-level bound: a multi-MB single document."""
    assert len(sample_document("q" * 20_000_000, 32 * 1024)) == 32 * 1024


def test_truncation_is_deterministic():
    text = "abcdefghij" * 5_000
    assert sample_document(text, 300) == sample_document(text, 300)


def test_default_cap_bounds_a_full_window():
    """A batch_size window of capped documents must stay small enough to embed."""
    window_bytes = 4096 * luxical_pipeline.EMBED_DOC_SAMPLE_CHARS
    assert window_bytes <= 128 * 1024 * 1024


# ---- bounded Parquet reader ---------------------------------------------------


def _write_shard(path, n_rows, text_len, row_group_size):
    table = pa.table(
        {
            "id": [f"d{i}" for i in range(n_rows)],
            "text": [f"{i}-" + "x" * text_len for i in range(n_rows)],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def test_bounded_reader_matches_zephyr_reader_exactly(tmp_path):
    """Equivalence is the whole justification for a local reader: same records, same order."""
    path = str(tmp_path / "shard.parquet")
    _write_shard(path, n_rows=500, text_len=200, row_group_size=64)

    assert list(_load_records_bounded(path)) == list(load_parquet(path))


def test_bounded_reader_slices_large_rows_into_several_chunks(tmp_path, monkeypatch):
    """With a tiny byte target a row group must be materialized in pieces, not whole."""
    path = str(tmp_path / "big.parquet")
    _write_shard(path, n_rows=40, text_len=10_000, row_group_size=40)
    monkeypatch.setattr(luxical_pipeline, "_READ_CHUNK_TARGET_BYTES", 20_000)

    records = list(luxical_pipeline._load_records_bounded(path))
    assert [r["id"] for r in records] == [f"d{i}" for i in range(40)]
    assert records == list(load_parquet(path)), "chunking must not change records"


def test_bounded_reader_handles_empty_shard(tmp_path):
    path = str(tmp_path / "empty.parquet")
    pq.write_table(pa.table({"id": pa.array([], type=pa.string()), "text": pa.array([], type=pa.string())}), path)
    assert list(_load_records_bounded(path)) == []
