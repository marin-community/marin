# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-6 cross-backend cutover parity gate (THE phase gate).

This is the data-safety proof for the production cutover: the PYTHON server
writes a corpus to a ``log_dir`` (registering a non-trivial schema, writing rows
across namespaces + the privileged ``log`` namespace, and forcing a compaction
so L>=1 segments exist), is stopped, and then the RUST server boots on the SAME
``log_dir`` with NO Rust catalog sidecar and NO adoption sentinel. The Rust
server's boot-time ``ensure_catalog_adopted`` must rebuild its catalog purely by
scanning the on-disk parquet layout + footers (never reading Python's DuckDB
sidecar), so that the namespaces, segments, schemas, queryable contents, and log
entries are byte-identical to what the Python server reported.

All assertions are on STRUCTURED RPC responses (NamespaceInfo fields, decoded
Arrow tables, GetTableSchema columns, LogEntry fields) — never on log strings.

The harness (:class:`tests.parity.conftest.CutoverHarness`) is intentionally
NOT parametrized over ``server_backend``: the cutover identity is specifically
Python-writer to Rust-reader. The whole module is skipped when the Rust binary
is not built (the reader cannot run).
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync
from finelog.rpc.logging_connect import LogServiceClientSync

from tests.parity.conftest import CutoverHarness, maintain

pytestmark = pytest.mark.timeout(60)

# A namespace registered with an EMPTY key_column + a `timestamp_ms` column. The
# default key-resolution rule (resolve_key_column) maps an empty key_column to
# `timestamp_ms`, and Rust schema recovery reproduces an empty key_column for any
# adopted schema that carries `timestamp_ms` — so this namespace's GetTableSchema
# round-trips EXACTLY across the cutover (the recoverable subset). The schema is
# non-trivial: a nullable column exercises nullability recovery from the footer.
_WORKER_NS = "iris.worker"
_METRICS_NS = "iris.metrics"

_LOG_KEY = "/job/cutover/0:0"


def _worker_schema() -> stats_pb2.Schema:
    return stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            # A nullable column: nullability must survive footer recovery.
            stats_pb2.Column(name="note", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True),
        ],
        key_column="",
    )


def _worker_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("worker_id", pa.string(), nullable=False),
            pa.field("mem_bytes", pa.int64(), nullable=False),
            pa.field("timestamp_ms", pa.int64(), nullable=False),
            pa.field("note", pa.string(), nullable=True),
        ]
    )


def _metrics_schema() -> stats_pb2.Schema:
    return stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="metric", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="value", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=False),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
        key_column="",
    )


def _metrics_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("metric", pa.string(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("timestamp_ms", pa.int64(), nullable=False),
        ]
    )


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _stats(url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=url)


def _logs(url: str) -> LogServiceClientSync:
    return LogServiceClientSync(address=url)


def _decode(resp: stats_pb2.QueryResponse) -> pa.Table:
    return paipc.open_stream(io.BytesIO(resp.arrow_ipc)).read_all()


def _column_types(schema: stats_pb2.Schema) -> list[tuple[str, int]]:
    """(name, type) per wire column — the cutover-recoverable subset.

    The wire schema strips the implicit ``seq`` column on both backends. We
    compare name + type but NOT ``nullable``: schema recovery reads the parquet
    footer, and a DuckDB-compacted segment marks every column nullable in its
    parquet output (DuckDB's COPY does not carry Arrow non-nullability), so the
    non-nullable flag of a compacted namespace is not faithfully recoverable
    from disk. This is documented lossiness (see ``store/adopt.rs``); a
    nullable-superset never changes queryable contents — assertion (2) proves
    the data + result schema are identical."""
    return [(c.name, c.type) for c in schema.columns]


def _namespace_infos(client: StatsServiceClientSync) -> dict[str, stats_pb2.NamespaceInfo]:
    listed = client.list_namespaces(stats_pb2.ListNamespacesRequest())
    return {n.namespace: n for n in listed.namespaces}


def _write_corpus(url: str) -> None:
    """Register two namespaces + push logs + write rows, forcing >=1 L0 segment
    per namespace and at least one L>=1 segment via a forced compaction."""
    stats = _stats(url)
    logs = _logs(url)

    stats.register_table(stats_pb2.RegisterTableRequest(namespace=_WORKER_NS, schema=_worker_schema()))
    stats.register_table(stats_pb2.RegisterTableRequest(namespace=_METRICS_NS, schema=_metrics_schema()))

    # Worker rows across several WriteRows calls. Each WriteRows acks only after
    # the rows are on a sealed L0 segment, so N calls => N L0 segments.
    for i in range(4):
        batch = pa.RecordBatch.from_pydict(
            {
                "worker_id": [f"w-{i}"],
                "mem_bytes": [100 + i],
                "timestamp_ms": [1_000 + i],
                "note": [None if i % 2 == 0 else f"note-{i}"],
            },
            schema=_worker_arrow_schema(),
        )
        stats.write_rows(stats_pb2.WriteRowsRequest(namespace=_WORKER_NS, arrow_ipc=_ipc_bytes(batch)))

    for i in range(3):
        batch = pa.RecordBatch.from_pydict(
            {"metric": [f"m-{i}"], "value": [float(i) + 0.5], "timestamp_ms": [2_000 + i]},
            schema=_metrics_arrow_schema(),
        )
        stats.write_rows(stats_pb2.WriteRowsRequest(namespace=_METRICS_NS, arrow_ipc=_ipc_bytes(batch)))

    # Push logs to the privileged `log` namespace.
    entries = [
        logging_pb2.LogEntry(
            source=_LOG_KEY,
            data=f"line-{i}",
            timestamp=logging_pb2.Timestamp(epoch_ms=i),
            level=logging_pb2.LOG_LEVEL_INFO,
        )
        for i in range(6)
    ]
    logs.push_logs(logging_pb2.PushLogsRequest(key=_LOG_KEY, entries=entries))

    # Force a compaction on the worker namespace so it has an L>=1 segment (the
    # cutover must adopt segments at multiple levels, with level recovered from
    # the filename).
    maintain(url, _WORKER_NS, force_compact_l0=True)


def _query_all(url: str, namespace: str) -> pa.Table:
    return _decode(_stats(url).query(stats_pb2.QueryRequest(sql=f'SELECT * FROM "{namespace}" ORDER BY seq')))


def _fetch_log_lines(url: str) -> tuple[list[str], int]:
    resp = _logs(url).fetch_logs(
        logging_pb2.FetchLogsRequest(
            source=_LOG_KEY,
            match_scope=logging_pb2.MATCH_SCOPE_EXACT,
        )
    )
    return [e.data for e in resp.entries], resp.cursor


def test_cutover_python_writes_rust_reads_identically(cutover_harness: CutoverHarness) -> None:
    """The full cutover gate: a Python-populated log_dir is adopted by the Rust
    server and queried back identically (stats, query Arrow, schema, logs)."""
    h = cutover_harness

    # --- Phase 1: Python writes the corpus, we snapshot its observable state.
    py_url = h.writer.start()
    _write_corpus(py_url)

    py_stats = _stats(py_url)
    py_infos = _namespace_infos(py_stats)
    py_worker_table = _query_all(py_url, _WORKER_NS)
    py_metrics_table = _query_all(py_url, _METRICS_NS)
    py_worker_schema = py_stats.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace=_WORKER_NS)).schema
    py_metrics_schema = py_stats.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace=_METRICS_NS)).schema
    py_log_lines, py_log_cursor = _fetch_log_lines(py_url)

    h.writer.stop()

    # The Python DuckDB sidecar may be present; the Rust server MUST ignore it.
    # The Rust catalog sidecar + adoption sentinel must NOT exist yet.
    assert not h.rust_sidecar().exists(), "Rust must boot on a fresh (un-adopted) dir"
    assert not h.rust_sentinel().exists()

    # --- Phase 2: Rust boots on the SAME dir and rebuilds the catalog from disk.
    rust_url = h.reader.start()
    # Adoption ran before bind and stamped the sentinel + wrote the sidecar.
    assert h.rust_sentinel().exists(), "adoption sentinel must be written at boot"
    assert h.rust_sidecar().exists(), "Rust catalog sidecar must be rebuilt at boot"

    rust_stats = _stats(rust_url)
    rust_infos = _namespace_infos(rust_stats)

    # (1) Identical NamespaceInfo: exact row_count/min_seq/max_seq/segment_count;
    #     byte_size > 0 but NOT asserted equal (compression differs across writers).
    for ns in (_WORKER_NS, _METRICS_NS, "log"):
        assert ns in rust_infos, f"adopted catalog missing {ns!r}"
        py = py_infos[ns]
        rust = rust_infos[ns]
        assert rust.row_count == py.row_count, f"{ns}: row_count"
        assert rust.min_seq == py.min_seq, f"{ns}: min_seq"
        assert rust.max_seq == py.max_seq, f"{ns}: max_seq"
        assert rust.segment_count == py.segment_count, f"{ns}: segment_count"
        assert rust.byte_size > 0, f"{ns}: byte_size present"

    # (2) Identical Query Arrow results (decoded-table equality, ORDER BY seq).
    assert _query_all(rust_url, _WORKER_NS).equals(py_worker_table)
    assert _query_all(rust_url, _METRICS_NS).equals(py_metrics_table)

    # (3) GetTableSchema matches on the recoverable subset (name + type per
    #     column + key_column). Both namespaces use an empty key_column with a
    #     `timestamp_ms` present, which Rust recovers exactly. Nullability is
    #     documented-lossy for compacted segments (see _column_types).
    rust_worker_schema = rust_stats.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace=_WORKER_NS)).schema
    rust_metrics_schema = rust_stats.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace=_METRICS_NS)).schema
    assert _column_types(rust_worker_schema) == _column_types(py_worker_schema)
    assert rust_worker_schema.key_column == py_worker_schema.key_column == ""
    assert _column_types(rust_metrics_schema) == _column_types(py_metrics_schema)
    assert rust_metrics_schema.key_column == py_metrics_schema.key_column == ""

    # The metrics namespace is NOT compacted (its segments are pyarrow-written
    # L0 parquet), and pyarrow preserves Arrow non-nullability in the footer, so
    # nullability IS faithfully recovered here — pinning the lossiness boundary
    # to DuckDB-compacted output (the worker namespace, asserted by-type above).
    assert [(c.name, c.nullable) for c in rust_metrics_schema.columns] == [
        (c.name, c.nullable) for c in py_metrics_schema.columns
    ]

    # (4) FetchLogs round-trip on the adopted `log` namespace: same entries +
    #     same cursor (= max seq), proving the implicit log ns was adopted.
    rust_log_lines, rust_log_cursor = _fetch_log_lines(rust_url)
    assert rust_log_lines == py_log_lines == [f"line-{i}" for i in range(6)]
    assert rust_log_cursor == py_log_cursor


def test_cutover_idempotent_second_rust_boot(cutover_harness: CutoverHarness) -> None:
    """A second Rust boot on the already-adopted dir (done sentinel) returns the
    same stats without rescanning — the sentinel fast path is idempotent."""
    h = cutover_harness

    py_url = h.writer.start()
    _write_corpus(py_url)
    h.writer.stop()

    # First Rust boot: cold adoption.
    rust_url = h.reader.start()
    first = _namespace_infos(_stats(rust_url))
    assert h.rust_sentinel().exists()
    h.reader.stop()

    # Second Rust boot: done sentinel -> fast path, sidecar authoritative.
    rust_url2 = h.reader.start()
    second = _namespace_infos(_stats(rust_url2))

    for ns in (_WORKER_NS, _METRICS_NS, "log"):
        assert ns in first and ns in second
        a, b = first[ns], second[ns]
        assert (a.row_count, a.min_seq, a.max_seq, a.segment_count) == (
            b.row_count,
            b.min_seq,
            b.max_seq,
            b.segment_count,
        ), f"{ns}: stats drifted across idempotent reboot"
