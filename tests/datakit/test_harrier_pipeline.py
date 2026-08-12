# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import http.client
import io
import json
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.datakit.embeddings.harrier import tei, tei_client
from experiments.datakit.embeddings.harrier.pipeline import EmbeddingAttrData

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
    monkeypatch.setattr(tei.subprocess, "check_output", lambda *args, **kwargs: "00000000:3B:00.0\n")

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

    def urlopen(request, timeout):
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

    def urlopen(request, timeout):
        texts = json.loads(request.data)["inputs"]
        if len(texts) > 2:
            raise urllib.error.HTTPError(request.full_url, 413, "payload too large", {}, None)
        return io.BytesIO(json.dumps([_embedding(float(text)) for text in texts]).encode())

    monkeypatch.setattr(tei_client.urllib.request, "urlopen", urlopen)

    embeddings = tei_client.TeiEmbeddingClient("tei", EMBEDDING_DIM).embed(["0", "1", "2", "3", "4"])

    np.testing.assert_array_equal(embeddings[:, 0], np.arange(5, dtype=np.float32))
