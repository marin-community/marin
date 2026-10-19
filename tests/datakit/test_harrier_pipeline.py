# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import http.client
import io
import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import NormalizedData

from experiments.datakit.embeddings.harrier import pipeline as harrier_pipeline
from experiments.datakit.embeddings.harrier import tei, tei_client
from experiments.datakit.embeddings.harrier.pipeline import (
    HARRIER_DIM,
    HARRIER_MAX_RAW_TEXT_CHARS,
    EmbeddingAttrData,
    _truncated_records,
)

EMBEDDING_DIM = 4


def test_embedding_artifact_uses_calver():
    artifact = EmbeddingAttrData(
        output_dir="embeddings",
        source_key="source",
        model_name="harrier",
        embedding_dim=1_024,
        quantization_scale=0.3 / 127,
        quantization_range=0.3,
        batch_size=4_096,
    )

    assert artifact.model_dump()["version"] == "2026.08.11"


class _Resolver:
    def __init__(self, responses: list[list[str]]):
        self.responses = iter(responses)

    def resolve(self, _name: str):
        return SimpleNamespace(endpoints=[SimpleNamespace(url=url) for url in next(self.responses)])


def _embedding(value: float) -> list[float]:
    return [value, *([0.0] * (EMBEDDING_DIM - 1))]


def test_tei_ports_follow_gpu_pci_bus(monkeypatch):
    def check_output(*_args, **_kwargs):
        return "00000000:3B:00.0\n"

    monkeypatch.setattr(tei.subprocess, "check_output", check_output)

    assert tei._tei_ports() == (25_059, 26_059)


@pytest.mark.parametrize(
    "transient_error",
    [
        urllib.error.URLError("connection refused"),
        TimeoutError("timed out"),
        http.client.RemoteDisconnected("connection closed"),
    ],
)
def test_tei_client_retries_against_refreshed_endpoints(monkeypatch, transient_error):
    resolver = _Resolver([["http://dead"], ["http://live"]])
    monkeypatch.setattr(tei_client, "iris_ctx", lambda: SimpleNamespace(resolver=resolver))
    monkeypatch.setattr(tei_client.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(tei_client.random, "shuffle", lambda _items: None)

    def urlopen(request, *, timeout):
        assert timeout == 300
        if request.full_url.startswith("http://dead"):
            raise transient_error
        texts = json.loads(request.data)["inputs"]
        return io.BytesIO(json.dumps([_embedding(float(index)) for index, _text in enumerate(texts)]).encode())

    monkeypatch.setattr(tei_client.urllib.request, "urlopen", urlopen)

    embeddings = tei_client.TeiEmbeddingClient("tei", EMBEDDING_DIM).embed(["document"])

    np.testing.assert_array_equal(embeddings, np.asarray([_embedding(0)], dtype=np.float32))


def test_tei_client_splits_large_requests_without_reordering(monkeypatch):
    resolver = _Resolver([["http://live"]])
    monkeypatch.setattr(tei_client, "iris_ctx", lambda: SimpleNamespace(resolver=resolver))
    monkeypatch.setattr(tei_client.random, "shuffle", lambda _items: None)

    def urlopen(request, *, timeout):
        assert timeout == 300
        texts = json.loads(request.data)["inputs"]
        if len(texts) > 2:
            raise urllib.error.HTTPError(request.full_url, 413, "payload too large", {}, None)
        return io.BytesIO(json.dumps([_embedding(float(text)) for text in texts]).encode())

    monkeypatch.setattr(tei_client.urllib.request, "urlopen", urlopen)

    embeddings = tei_client.TeiEmbeddingClient("tei", EMBEDDING_DIM).embed(["0", "1", "2", "3", "4"])

    np.testing.assert_array_equal(embeddings[:, 0], np.arange(5, dtype=np.float32))


_CAP = HARRIER_MAX_RAW_TEXT_CHARS
_TEXTS = ["short", "é" * (_CAP + 7), "a" * _CAP, "b" * (_CAP + 1), "🙂x" * _CAP]


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
    assert artifact.counters["zephyr/records_in"] == 2 * len(_TEXTS)
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


def test_reader_materializes_an_oversized_row_group_in_bounded_pieces():
    ids = [f"d{i}" for i in range(40)]
    sizes = []
    batch = _CountingBatch(pa.record_batch({"id": ids, "text": ["x"] * 40}), sizes, nbytes=40 * 64 * 1024 * 1024)

    records = list(_truncated_records(batch))

    assert max(sizes) < 40
    assert sum(sizes) == 40
    assert [record["id"] for record in records] == ids
