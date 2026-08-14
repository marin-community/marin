# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import http.client
import io
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import ArtifactRecord, write_artifact, write_record
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from experiments.datakit.embeddings.harrier import tei, tei_client
from experiments.datakit.embeddings.harrier.merge import (
    MERGE_WORKER_RESOURCES,
    EmbeddingSourcePair,
    discover_source_pairs,
    merge_embedding_source,
    verify_merged_output,
)
from experiments.datakit.embeddings.harrier.pipeline import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_SCHEMA,
    HARRIER_DIM,
    HARRIER_REPO,
    HARRIER_REVISION,
    QUANT_RANGE,
    QUANT_SCALE,
    EmbeddingAttrData,
    EmbeddingDocumentSet,
    select_document,
)

EMBEDDING_DIM = 4


class _Resolver:
    def __init__(self, responses: list[list[str]]):
        self.responses = iter(responses)

    def resolve(self, _name: str):
        return SimpleNamespace(endpoints=[SimpleNamespace(url=url) for url in next(self.responses)])


def _embedding(value: float) -> list[float]:
    return [value, *([0.0] * (EMBEDDING_DIM - 1))]


def _int8_embeddings(values: list[int]) -> pa.FixedSizeListArray:
    flat_values = np.repeat(np.asarray(values, dtype=np.int8), HARRIER_DIM)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat_values), HARRIER_DIM)


def _write_embedding_shard(path: Path, ids: list[str], values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays([pa.array(ids), _int8_embeddings(values)], schema=EMBEDDING_SCHEMA)
    pq.write_table(table, path, row_group_size=1)


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


def test_merge_embedding_source_preserves_repeated_ids_and_prefers_old_overlap(tmp_path):
    deduplicated_dir = tmp_path / "deduplicated"
    fuzzy_duplicate_dir = tmp_path / "fuzzy-duplicates"
    normalized_dir = tmp_path / "normalized"
    output_dir = tmp_path / "merged"

    _write_embedding_shard(
        deduplicated_dir / "part-00000-of-00002.parquet",
        ["a", "c", "c"],
        [1, 3, 4],
    )
    _write_embedding_shard(
        fuzzy_duplicate_dir / "part-00000-of-00002.parquet",
        ["b", "c", "c"],
        [2, 30, 31],
    )
    _write_embedding_shard(deduplicated_dir / "part-00001-of-00002.parquet", ["d"], [5])
    _write_embedding_shard(fuzzy_duplicate_dir / "part-00001-of-00002.parquet", ["e", "f"], [6, 7])
    normalized_dir.mkdir()
    pq.write_table(
        pa.table({"id": ["a", "b", "c", "c", "c"]}),
        normalized_dir / "part-00000-of-00002.parquet",
        row_group_size=2,
    )
    pq.write_table(
        pa.table({"id": ["d", "e", "f"]}),
        normalized_dir / "part-00001-of-00002.parquet",
        row_group_size=2,
    )

    context = ZephyrContext(
        resources=MERGE_WORKER_RESOURCES,
        max_workers=2,
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="merge-harrier-test",
        stage_runner_factory=InlineRunner,
    )
    with context:
        artifact = merge_embedding_source(
            output_path=str(output_dir),
            source_key="normalized/source/outputs/main",
            normalized_path=str(normalized_dir),
            deduplicated_path=str(deduplicated_dir),
            fuzzy_duplicate_path=str(fuzzy_duplicate_dir),
            zephyr_context=context,
        )

    output_paths = sorted(output_dir.glob("*.parquet"))
    assert [path.name for path in output_paths] == [
        "part-00000-of-00002.parquet",
        "part-00001-of-00002.parquet",
    ]
    tables = [pq.read_table(path) for path in output_paths]
    normalized_tables = [pq.read_table(normalized_dir / path.name, columns=["id"]) for path in output_paths]
    assert all(table.schema.equals(EMBEDDING_SCHEMA) for table in tables)
    assert [table.column("id").to_pylist() for table in tables] == [
        table.column("id").to_pylist() for table in normalized_tables
    ]
    assert [[embedding[0].as_py() for embedding in table.column("embedding")] for table in tables] == [
        [1, 2, 3, 4, 30],
        [5, 6, 7],
    ]
    assert artifact.source_key == "normalized/source/outputs/main"
    assert artifact.counters["merge/deduplicated_docs"] == 4
    assert artifact.counters["merge/fuzzy_duplicate_docs"] == 5
    assert artifact.counters["merge/overlapping_docs"] == 1
    assert artifact.counters["merge/verified_shards"] == 2
    assert artifact.counters["merge/verified_rows"] == 8


def test_merge_embedding_sources_share_one_zephyr_pool(tmp_path, monkeypatch):
    deduplicated_dir = tmp_path / "deduplicated"
    fuzzy_duplicate_dir = tmp_path / "fuzzy-duplicates"
    normalized_dir = tmp_path / "normalized"
    _write_embedding_shard(deduplicated_dir / "part-00000.parquet", ["a"], [1])
    _write_embedding_shard(fuzzy_duplicate_dir / "part-00000.parquet", ["b"], [2])
    normalized_dir.mkdir()
    pq.write_table(pa.table({"id": ["a", "b"]}), normalized_dir / "part-00000.parquet")

    context = ZephyrContext(
        resources=MERGE_WORKER_RESOURCES,
        max_workers=1,
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="merge-harrier-test",
        stage_runner_factory=InlineRunner,
    )
    started_contexts = []
    original_start_pool = ZephyrContext._start_pool

    def track_start_pool(self, worker_count, idle_policy):
        started_contexts.append(self)
        return original_start_pool(self, worker_count, idle_policy)

    monkeypatch.setattr(ZephyrContext, "_start_pool", track_start_pool)
    with context:
        for source_name in ("first", "second"):
            artifact = merge_embedding_source(
                output_path=str(tmp_path / source_name),
                source_key=f"normalized/{source_name}/outputs/main",
                normalized_path=str(normalized_dir),
                deduplicated_path=str(deduplicated_dir),
                fuzzy_duplicate_path=str(fuzzy_duplicate_dir),
                zephyr_context=context,
            )
            assert artifact.counters["merge/verified_shards"] == 1

    assert started_contexts == [context]


def test_discover_source_pairs_accepts_fuzzy_artifact_without_result(tmp_path):
    source_name = "nested/source"
    source_key = str(tmp_path / "normalized" / source_name / "outputs" / "main")
    normalized_path = tmp_path / "normalized-artifacts" / source_name
    deduplicated_path = tmp_path / "deduplicated" / "nested" / "source_deadbeef"
    fuzzy_duplicate_path = tmp_path / "fuzzy-duplicates" / "nested" / "source_cafebabe"
    for path in (normalized_path, deduplicated_path, fuzzy_duplicate_path):
        path.mkdir(parents=True)

    write_artifact(
        NormalizedData(main_output_dir=source_key, dup_output_dir=f"{source_key}-dups", counters={}),
        str(normalized_path),
    )
    write_artifact(
        EmbeddingAttrData(
            output_dir=str(deduplicated_path),
            source_key=source_key,
            model_name=HARRIER_REPO,
            model_revision=HARRIER_REVISION,
            embedding_dim=HARRIER_DIM,
            quantization_scale=QUANT_SCALE,
            quantization_range=QUANT_RANGE,
            batch_size=DEFAULT_BATCH_SIZE,
        ),
        str(deduplicated_path),
    )
    write_record(
        ArtifactRecord(
            output_path=str(fuzzy_duplicate_path),
            dep_paths=[str(normalized_path)],
            config={
                "model": HARRIER_REPO,
                "revision": HARRIER_REVISION,
                "batch_size": DEFAULT_BATCH_SIZE,
                "document_set": EmbeddingDocumentSet.FUZZY_DUPLICATES.value,
            },
        )
    )
    (deduplicated_path / ".executor_status").write_text("SUCCESS")
    (fuzzy_duplicate_path / ".executor_status").write_text("SUCCESS")

    pairs = discover_source_pairs(
        deduplicated_prefix=str(tmp_path / "deduplicated"),
        fuzzy_duplicate_prefix=str(tmp_path / "fuzzy-duplicates"),
        source_names=[source_name],
    )

    assert pairs == [
        EmbeddingSourcePair(
            source_name=source_name,
            source_key=source_key,
            normalized_path=source_key,
            deduplicated_path=str(deduplicated_path),
            fuzzy_duplicate_path=str(fuzzy_duplicate_path),
        )
    ]


def test_verify_merged_output_rejects_per_shard_row_count_mismatch(tmp_path):
    normalized_dir = tmp_path / "normalized"
    output_dir = tmp_path / "output"
    normalized_dir.mkdir()
    _write_embedding_shard(output_dir / "part-00000.parquet", ["a"], [1])
    pq.write_table(pa.table({"id": ["a", "b"]}), normalized_dir / "part-00000.parquet")

    with pytest.raises(ValueError, match="row count does not match"):
        verify_merged_output(str(output_dir), str(normalized_dir))


def test_verify_merged_output_rejects_id_order_mismatch(tmp_path):
    normalized_dir = tmp_path / "normalized"
    output_dir = tmp_path / "output"
    normalized_dir.mkdir()
    _write_embedding_shard(output_dir / "part-00000.parquet", ["a", "c"], [1, 2])
    pq.write_table(pa.table({"id": ["a", "b"]}), normalized_dir / "part-00000.parquet")

    with pytest.raises(ValueError, match="ID order does not match"):
        verify_merged_output(str(output_dir), str(normalized_dir))


@pytest.mark.parametrize(
    ("allocated_ports", "fallback_ports", "expected_ports"),
    [
        ({"http": 12_001, "metrics": 12_002}, (13_001, 13_002), (12_001, 12_002)),
        ({"http": 0, "metrics": 0}, (13_001, 13_002), (13_001, 13_002)),
    ],
)
def test_tei_service_selects_usable_ports(monkeypatch, allocated_ports, fallback_ports, expected_ports):
    process = MagicMock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    registry = MagicMock()
    registry.registered.return_value = contextlib.nullcontext()

    monkeypatch.setattr(
        tei,
        "get_job_info",
        lambda: SimpleNamespace(advertise_host="worker.example", ports=allocated_ports),
    )
    monkeypatch.setattr(tei, "iris_ctx", lambda: SimpleNamespace(registry=registry))
    monkeypatch.setattr(tei, "configure_logging", lambda: None)
    monkeypatch.setattr(tei, "_download_model", lambda _config, _root: Path("/model"))
    monkeypatch.setattr(tei.subprocess, "Popen", MagicMock(return_value=process))
    wait_until_ready = MagicMock()
    monkeypatch.setattr(tei, "_wait_until_ready", wait_until_ready)

    config = tei.TeiServiceConfig(
        endpoint_name="tei-endpoint",
        model_archive="model.tar",
        max_input_tokens=8_192,
        fallback_port=fallback_ports[0],
        fallback_prometheus_port=fallback_ports[1],
    )
    with pytest.raises(RuntimeError, match="TEI exited with code 0"):
        tei.run_tei_service(config)

    command = tei.subprocess.Popen.call_args.args[0]
    assert command[command.index("--port") + 1] == str(expected_ports[0])
    assert command[command.index("--prometheus-port") + 1] == str(expected_ports[1])
    wait_until_ready.assert_called_once_with(process, expected_ports[0])
    registry.registered.assert_called_once_with(
        "tei-endpoint", f"http://worker.example:{expected_ports[0]}", {"backend": "tei"}
    )


def test_tei_fallback_ports_are_unique_within_a_pool():
    ports = {
        port for index in range(96) for port in tei._fallback_port_pair(run_id="1234abcd", instances=96, index=index)
    }

    assert len(ports) == 192
    assert min(ports) >= tei.TEI_FALLBACK_PORT_START
    assert max(ports) < tei.TEI_FALLBACK_PORT_END


@pytest.mark.parametrize(
    "request_error",
    [
        pytest.param(lambda: urllib.error.URLError("connection refused"), id="url-error"),
        pytest.param(lambda: http.client.RemoteDisconnected("remote end closed connection"), id="remote-disconnected"),
        pytest.param(lambda: TimeoutError("request timed out"), id="timeout"),
        pytest.param(lambda: http.client.IncompleteRead(b"", 1), id="incomplete-read"),
    ],
)
def test_tei_client_retries_against_refreshed_endpoints(monkeypatch, request_error):
    resolver = _Resolver([["http://dead"], ["http://live"]])
    monkeypatch.setattr(tei_client, "iris_ctx", lambda: SimpleNamespace(resolver=resolver))
    monkeypatch.setattr(tei_client.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(tei_client.random, "shuffle", lambda _items: None)

    def urlopen(request, timeout):
        assert timeout == 300
        if request.full_url.startswith("http://dead"):
            raise request_error()
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
