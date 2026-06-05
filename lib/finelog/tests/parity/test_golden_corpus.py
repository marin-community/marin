# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-3 golden-query corpus parity (the SQL-dialect gate).

The TOP risk of the DuckDB -> DataFusion rewrite is silent SQL-dialect
divergence. This file pins it with an executable gate: seed a FIXED dataset over
RPC, run each real-corpus query, and assert the decoded Arrow table matches a
canonical expectation. Because the SAME assertion body runs against BOTH the
Python (DuckDB) and Rust (DataFusion) backends (parametrized by ``server_backend``),
a passing run on both backends IS the cross-engine Arrow diff.

Corpus sources (the real query shapes the codebase issues through
``StatsService.Query``):

- ``lib/iris/scripts/job_profile_summary.py:158`` — ``SELECT source, profile_data
  FROM "<ns>" WHERE (source = '..' OR source LIKE '..%') AND type = '..'
  ORDER BY source, captured_at`` (=, OR, LIKE %, AND, multi-col ORDER BY, quoted
  dotted identifier, bytes-column projection).
- ``lib/finelog/deploy/cli.py`` query_cmd — arbitrary operator SQL incl.
  ``COUNT(*) AS n`` (server enforces NO row cap).
- ``tests/test_query.py`` shapes — projection, WHERE >=, JOIN USING, ORDER BY.

Each query is matched against a canonical-ordered expectation; type checks are
modulo the Utf8/Utf8View normalization pinned via
``map_string_types_to_utf8view=false`` (so string columns are Utf8 on both).
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

from tests.parity.conftest import Backend

pytestmark = pytest.mark.timeout(60)

PROFILE_NS = "iris.profile"
WORKER_NS = "iris.worker"


# ---------------------------------------------------------------------------
# Wire helpers.
# ---------------------------------------------------------------------------


def _stats_client(url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=url)


def _ipc_bytes(batch: pa.RecordBatch) -> bytes:
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _query(client: StatsServiceClientSync, sql: str) -> pa.Table:
    resp = client.query(stats_pb2.QueryRequest(sql=sql))
    return paipc.open_stream(io.BytesIO(resp.arrow_ipc)).read_all()


def _register(client: StatsServiceClientSync, namespace: str, schema: stats_pb2.Schema) -> None:
    client.register_table(stats_pb2.RegisterTableRequest(namespace=namespace, schema=schema))


def _write(client: StatsServiceClientSync, namespace: str, batch: pa.RecordBatch) -> None:
    client.write_rows(stats_pb2.WriteRowsRequest(namespace=namespace, arrow_ipc=_ipc_bytes(batch)))


# ---------------------------------------------------------------------------
# Fixed seed dataset (identical on both backends).
# ---------------------------------------------------------------------------

_PROFILE_PROTO_SCHEMA = stats_pb2.Schema(
    columns=[
        stats_pb2.Column(name="source", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        stats_pb2.Column(name="type", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        stats_pb2.Column(name="profile_data", type=stats_pb2.COLUMN_TYPE_BYTES, nullable=False),
        stats_pb2.Column(name="captured_at", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
    ],
    key_column="",
)

_WORKER_PROTO_SCHEMA = stats_pb2.Schema(
    columns=[
        stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
    ],
    key_column="",
)


def _profile_batch() -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {
            "source": ["/job/a/0:0", "/job/a/1:0", "/job/b/0:0"],
            "type": ["cpu", "cpu", "mem"],
            "profile_data": [b"\x01\x02", b"\x03", b"\x04\x05\x06"],
            "captured_at": [30, 10, 20],
            "timestamp_ms": [1, 2, 3],
        },
        schema=pa.schema(
            [
                pa.field("source", pa.string(), nullable=False),
                pa.field("type", pa.string(), nullable=False),
                pa.field("profile_data", pa.binary(), nullable=False),
                pa.field("captured_at", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )


def _worker_batch() -> pa.RecordBatch:
    return pa.RecordBatch.from_pydict(
        {
            "worker_id": ["w-1", "w-2", "w-3"],
            "mem_bytes": [100, 200, 300],
            "timestamp_ms": [1, 2, 3],
        },
        schema=pa.schema(
            [
                pa.field("worker_id", pa.string(), nullable=False),
                pa.field("mem_bytes", pa.int64(), nullable=False),
                pa.field("timestamp_ms", pa.int64(), nullable=False),
            ]
        ),
    )


def _seed(client: StatsServiceClientSync) -> None:
    _register(client, PROFILE_NS, _PROFILE_PROTO_SCHEMA)
    _write(client, PROFILE_NS, _profile_batch())
    _register(client, WORKER_NS, _WORKER_PROTO_SCHEMA)
    _write(client, WORKER_NS, _worker_batch())


# ---------------------------------------------------------------------------
# The corpus: (id, sql, expected-as-pydict, sort-keys for canonicalization).
#
# `expected` is a column-name -> list-of-values mapping; the test sorts both the
# actual and expected rows by `sort_keys` before comparing, so result row order
# from the engine is irrelevant (only the SET of rows + projected columns is the
# contract). This is the cross-engine Arrow diff.
# ---------------------------------------------------------------------------

_CORPUS = [
    pytest.param(
        # The real job_profile_summary.py:158 shape.
        'SELECT source, profile_data FROM "iris.profile" '
        "WHERE (source = '/job/a/0:0' OR source LIKE '/job/a/%') AND type = 'cpu' "
        "ORDER BY source, captured_at",
        {
            "source": ["/job/a/0:0", "/job/a/1:0"],
            "profile_data": [b"\x01\x02", b"\x03"],
        },
        ["source"],
        id="profile_summary_or_like_and_order",
    ),
    pytest.param(
        'SELECT worker_id, mem_bytes FROM "iris.worker" ORDER BY worker_id',
        {"worker_id": ["w-1", "w-2", "w-3"], "mem_bytes": [100, 200, 300]},
        ["worker_id"],
        id="projection_order",
    ),
    pytest.param(
        'SELECT worker_id FROM "iris.worker" WHERE mem_bytes >= 200',
        {"worker_id": ["w-2", "w-3"]},
        ["worker_id"],
        id="where_gte",
    ),
    pytest.param(
        # COUNT(*) AS n — deploy/cli.py free-form shape; server has no row cap.
        'SELECT COUNT(*) AS n FROM "iris.worker"',
        {"n": [3]},
        ["n"],
        id="count_star_alias",
    ),
    pytest.param(
        # lowercase count(*) (DuckDB accepts both cases).
        "SELECT count(*) AS n FROM \"iris.profile\" WHERE type = 'cpu'",
        {"n": [2]},
        ["n"],
        id="count_star_lower_where",
    ),
    pytest.param(
        "SELECT source FROM \"iris.profile\" WHERE source LIKE '/job/a/%' ORDER BY source",
        {"source": ["/job/a/0:0", "/job/a/1:0"]},
        ["source"],
        id="like_prefix",
    ),
    pytest.param(
        # SELECT * carries the registered columns incl. implicit seq.
        "SELECT * FROM \"iris.worker\" WHERE worker_id = 'w-2'",
        {
            "seq": [2],
            "worker_id": ["w-2"],
            "mem_bytes": [200],
            "timestamp_ms": [2],
        },
        ["worker_id"],
        id="select_star_with_seq",
    ),
]


def _canonicalize(table: pa.Table, sort_keys: list[str]) -> dict:
    """Return a column-name -> sorted-row-values dict for order-independent
    comparison."""
    sort_idx = sorted(
        range(table.num_rows),
        key=lambda i: tuple(table.column(k)[i].as_py() for k in sort_keys),
    )
    return {name: [table.column(name)[i].as_py() for i in sort_idx] for name in table.column_names}


@pytest.mark.parametrize("sql, expected, sort_keys", _CORPUS)
def test_golden_corpus_arrow_parity(
    finelog_url: str,
    server_backend: Backend,
    sql: str,
    expected: dict,
    sort_keys: list[str],
) -> None:
    client = _stats_client(finelog_url)
    _seed(client)
    table = _query(client, sql)

    assert set(table.column_names) == set(
        expected.keys()
    ), f"column set mismatch: got {table.column_names}, want {list(expected.keys())}"
    got = _canonicalize(table, sort_keys)
    want = _canonicalize(pa.table({k: pa.array(v) for k, v in expected.items()}), sort_keys)
    for name in expected:
        assert got[name] == want[name], f"column {name!r} mismatch: {got[name]} != {want[name]}"

    # String columns must be Utf8 (not Utf8View) so the wire schema matches
    # DuckDB/pyarrow output on both backends.
    for field in table.schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            assert field.type == pa.string(), f"{field.name} is {field.type}, want utf8"

    # The decoded result schema must be ALL-NULLABLE on both backends. DuckDB
    # returns every result column nullable; the Rust/DataFusion backend must
    # normalize source non-nullability away to match (relax_result_nullability).
    # `select_star_with_seq` projects the store-form non-nullable `seq`, so this
    # assertion fails if that normalization regresses — it is the wire-schema
    # half of the cross-engine diff that value comparison alone cannot see.
    for field in table.schema:
        assert field.nullable, f"result column {field.name!r} is non-nullable; DuckDB returns all columns nullable"
