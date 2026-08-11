# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import urllib.error
from types import SimpleNamespace

import numpy as np

from experiments.datakit.embeddings.harrier import tei_client

EMBEDDING_DIM = 4


class _Resolver:
    def __init__(self, responses: list[list[str]]):
        self.responses = iter(responses)

    def resolve(self, _name: str):
        return SimpleNamespace(endpoints=[SimpleNamespace(url=url) for url in next(self.responses)])


def _embedding(value: float) -> list[float]:
    return [value, *([0.0] * (EMBEDDING_DIM - 1))]


def test_tei_client_retries_against_refreshed_endpoints(monkeypatch):
    resolver = _Resolver([["http://dead"], ["http://live"]])
    monkeypatch.setattr(tei_client, "iris_ctx", lambda: SimpleNamespace(resolver=resolver))
    monkeypatch.setattr(tei_client.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(tei_client.random, "shuffle", lambda _items: None)

    def urlopen(request, timeout):
        assert timeout == 300
        if request.full_url.startswith("http://dead"):
            raise urllib.error.URLError("connection refused")
        texts = json.loads(request.data)["inputs"]
        return io.BytesIO(json.dumps([_embedding(float(index)) for index, _text in enumerate(texts)]).encode())

    monkeypatch.setattr(tei_client.urllib.request, "urlopen", urlopen)

    embeddings = tei_client.TeiEmbeddingClient("tei", EMBEDDING_DIM).embed(["document"])

    np.testing.assert_array_equal(embeddings, np.asarray([_embedding(0)], dtype=np.float32))


def test_tei_client_splits_large_requests_without_reordering(monkeypatch):
    resolver = _Resolver([["http://live"]])
    monkeypatch.setattr(tei_client, "iris_ctx", lambda: SimpleNamespace(resolver=resolver))
    monkeypatch.setattr(tei_client.random, "shuffle", lambda _items: None)
    request_sizes = []

    def urlopen(request, timeout):
        assert timeout == 300
        texts = json.loads(request.data)["inputs"]
        request_sizes.append(len(texts))
        if len(texts) > 2:
            raise urllib.error.HTTPError(request.full_url, 413, "payload too large", {}, None)
        return io.BytesIO(json.dumps([_embedding(float(text)) for text in texts]).encode())

    monkeypatch.setattr(tei_client.urllib.request, "urlopen", urlopen)

    embeddings = tei_client.TeiEmbeddingClient("tei", EMBEDDING_DIM).embed(["0", "1", "2", "3", "4"])

    assert request_sizes == [5, 2, 3, 1, 2]
    np.testing.assert_array_equal(embeddings[:, 0], np.arange(5, dtype=np.float32))
