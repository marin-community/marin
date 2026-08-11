# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from experiments.datakit.embeddings.harrier import tei, tei_client
from experiments.datakit.embeddings.harrier.pipeline import EmbeddingDocumentSet, select_document

EMBEDDING_DIM = 4


class _Resolver:
    def __init__(self, responses: list[list[str]]):
        self.responses = iter(responses)

    def resolve(self, _name: str):
        return SimpleNamespace(endpoints=[SimpleNamespace(url=url) for url in next(self.responses)])


def _embedding(value: float) -> list[float]:
    return [value, *([0.0] * (EMBEDDING_DIM - 1))]


def test_select_document_separates_fuzzy_duplicates_from_retained_documents():
    singleton = {"id": "singleton"}
    canonical = {"id": "canonical"}
    duplicate = {"id": "duplicate"}

    assert select_document(duplicate, {"is_cluster_canonical": False}, EmbeddingDocumentSet.ALL) == duplicate
    assert select_document(singleton, None, EmbeddingDocumentSet.DEDUPLICATED) == singleton
    assert select_document(canonical, {"is_cluster_canonical": True}, EmbeddingDocumentSet.DEDUPLICATED) == canonical
    assert select_document(duplicate, {"is_cluster_canonical": False}, EmbeddingDocumentSet.DEDUPLICATED) is None

    assert select_document(singleton, None, EmbeddingDocumentSet.FUZZY_DUPLICATES) is None
    assert select_document(canonical, {"is_cluster_canonical": True}, EmbeddingDocumentSet.FUZZY_DUPLICATES) is None
    assert (
        select_document(duplicate, {"is_cluster_canonical": False}, EmbeddingDocumentSet.FUZZY_DUPLICATES) == duplicate
    )


def test_tei_service_uses_iris_allocated_ports(monkeypatch):
    process = MagicMock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    registry = MagicMock()
    registry.registered.return_value = contextlib.nullcontext()

    monkeypatch.setattr(
        tei,
        "get_job_info",
        lambda: SimpleNamespace(advertise_host="worker.example", ports={"http": 12_001, "metrics": 12_002}),
    )
    monkeypatch.setattr(tei, "iris_ctx", lambda: SimpleNamespace(registry=registry))
    monkeypatch.setattr(tei, "configure_logging", lambda: None)
    monkeypatch.setattr(tei, "_download_model", lambda _config, _root: Path("/model"))
    monkeypatch.setattr(tei.subprocess, "Popen", MagicMock(return_value=process))
    wait_until_ready = MagicMock()
    monkeypatch.setattr(tei, "_wait_until_ready", wait_until_ready)

    config = tei.TeiServiceConfig(endpoint_name="tei-endpoint", model_archive="model.tar", max_input_tokens=8_192)
    with pytest.raises(RuntimeError, match="TEI exited with code 0"):
        tei.run_tei_service(config)

    command = tei.subprocess.Popen.call_args.args[0]
    assert command[command.index("--port") + 1] == "12001"
    assert command[command.index("--prometheus-port") + 1] == "12002"
    wait_until_ready.assert_called_once_with(process, 12_001)
    registry.registered.assert_called_once_with("tei-endpoint", "http://worker.example:12001", {"backend": "tei"})


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

    def urlopen(request, timeout):
        assert timeout == 300
        texts = json.loads(request.data)["inputs"]
        if len(texts) > 2:
            raise urllib.error.HTTPError(request.full_url, 413, "payload too large", {}, None)
        return io.BytesIO(json.dumps([_embedding(float(text)) for text in texts]).encode())

    monkeypatch.setattr(tei_client.urllib.request, "urlopen", urlopen)

    embeddings = tei_client.TeiEmbeddingClient("tei", EMBEDDING_DIM).embed(["0", "1", "2", "3", "4"])

    np.testing.assert_array_equal(embeddings[:, 0], np.arange(5, dtype=np.float32))
