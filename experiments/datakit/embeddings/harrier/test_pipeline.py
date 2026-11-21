# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData

from experiments.datakit.embeddings.harrier import pipeline as harrier_pipeline
from experiments.datakit.embeddings.harrier.pipeline import HARRIER_DIM, HARRIER_MAX_RAW_TEXT_CHARS, _truncated_records

_CAP = HARRIER_MAX_RAW_TEXT_CHARS
_TEXTS = ["short", "é" + "x" * (_CAP + 6), "a" * _CAP, "b" * _CAP + "TAIL", "🙂" + "y" * (2 * _CAP)]


class _RecordingEmbedder:
    def __init__(self):
        self.texts = []

    def embed(self, texts):
        self.texts.extend(texts)
        return np.ones((len(texts), HARRIER_DIM), dtype=np.float32)


def test_embed_source_sends_tei_each_documents_first_chars_in_order(tmp_path, monkeypatch):
    source = tmp_path / "normalized"
    source.mkdir()
    ids = [f"d{i}" for i in range(len(_TEXTS))]
    pq.write_table(pa.table({"id": ids, "text": _TEXTS}), str(source / "part-0.parquet"), row_group_size=2)
    dedup = tmp_path / "dedup"
    dedup.mkdir()
    pq.write_table(
        pa.table({"id": ids, "is_cluster_canonical": [True, False, True, True, True]}),
        str(dedup / "part-0.parquet"),
    )
    embedder = _RecordingEmbedder()
    monkeypatch.setattr(harrier_pipeline, "_load_tei_embedder", lambda _name: embedder)

    artifact = harrier_pipeline.embed_source(
        str(tmp_path / "out"),
        NormalizedData(main_output_dir=str(source), dup_output_dir=str(source), counters={}),
        endpoint_name="tei",
        batch_size=2,
        dedup_attr_dir=str(dedup),
    )

    kept = [text for i, text in enumerate(_TEXTS) if i != 1]
    assert embedder.texts == [text[:_CAP] for text in kept]
    assert pq.read_table(artifact.shard_paths()[0]).column("id").to_pylist() == ["d0", "d2", "d3", "d4"]
    assert artifact.counters["embed/bytes_in"] == sum(len(text[:_CAP].encode()) for text in kept)


class _CountingBatch:
    """RecordBatch stand-in reporting an Arrow size far above the read target, recording each to_pylist size."""

    def __init__(self, batch, sizes, nbytes):
        self._batch = batch
        self.pylist_sizes = sizes
        self._nbytes = nbytes

    @property
    def num_rows(self):
        return self._batch.num_rows

    @property
    def nbytes(self):
        return self._nbytes

    def slice(self, offset, length):
        return _CountingBatch(self._batch.slice(offset, length), self.pylist_sizes, self._nbytes)

    def to_pylist(self):
        rows = self._batch.to_pylist()
        self.pylist_sizes.append(len(rows))
        return rows


def test_reader_yields_truncated_records_from_an_oversized_row_group_in_bounded_pieces():
    ids = [f"d{i}" for i in range(40)]
    texts = [f"{i}:" + "x" * _CAP for i in range(40)]
    sizes = []
    batch = _CountingBatch(pa.record_batch({"id": ids, "text": texts}), sizes, nbytes=40 * 64 * 1024 * 1024)

    records = list(_truncated_records(batch))

    assert max(sizes) < 40
    assert sum(sizes) == 40
    assert [record["id"] for record in records] == ids
    assert [record["text"] for record in records] == [text[:_CAP] for text in texts]
