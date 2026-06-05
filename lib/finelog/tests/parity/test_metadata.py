# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 metadata-RPC parity tests.

Drive RegisterTable / GetTableSchema / ListNamespaces / DropTable over real
HTTP/RPC against both the Python and Rust servers. These never import
``DuckDBLogStore`` or any store internals — the seam is the RPC socket. The
register/merge/name-validation cases are re-expressed from ``test_register.py``
and the drop cases from ``test_drop.py``, asserting on STRUCTURED wire output
(effective_schema columns, NamespaceInfo fields, ConnectError codes), never on
log strings.

Key parity adjustment vs the store-level tests: the wire ``effective_schema``
has the implicit ``seq`` column STRIPPED, so the worker schema's effective
columns are ``("worker_id", "mem_bytes", "timestamp_ms")``.
"""

from __future__ import annotations

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

from tests.parity.conftest import Backend, worker_schema

# Spawning a server subprocess can exceed the global per-test timeout under
# load, so give parity tests more room.
pytestmark = pytest.mark.timeout(40)


def _stats_client(finelog_url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=finelog_url)


def _worker_columns() -> list[stats_pb2.Column]:
    return [
        stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
        stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
    ]


def _column_names(schema: stats_pb2.Schema) -> tuple[str, ...]:
    return tuple(c.name for c in schema.columns)


def _register(
    client: StatsServiceClientSync, namespace: str, schema: stats_pb2.Schema
) -> stats_pb2.RegisterTableResponse:
    return client.register_table(stats_pb2.RegisterTableRequest(namespace=namespace, schema=schema))


@pytest.mark.parametrize("name", ["iris.worker", "iris.worker.v2", "a", "a-b", "abc.def_ghi", "x" * 64])
def test_register_accepts_valid_names(finelog_url: str, server_backend: Backend, name: str) -> None:
    client = _stats_client(finelog_url)
    resp = _register(client, name, worker_schema())
    # Wire form strips the implicit ``seq`` column.
    assert _column_names(resp.effective_schema) == ("worker_id", "mem_bytes", "timestamp_ms")
    assert resp.effective_schema.key_column == ""


@pytest.mark.parametrize(
    "name",
    ["", "Iris.Worker", ".starts-dot", "1starts-digit", "x" * 65, "has space", "has/slash", "..", "../escape"],
)
def test_register_rejects_invalid_names(finelog_url: str, server_backend: Backend, name: str) -> None:
    client = _stats_client(finelog_url)
    with pytest.raises(ConnectError) as exc:
        _register(client, name, worker_schema())
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_register_rejects_schema_without_ordering_key(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    schema = stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
        key_column="",
    )
    with pytest.raises(ConnectError) as exc:
        _register(client, "iris.worker", schema)
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_register_rejects_explicit_key_missing_from_columns(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    schema = stats_pb2.Schema(
        columns=[stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False)],
        key_column="ts",
    )
    with pytest.raises(ConnectError) as exc:
        _register(client, "iris.worker", schema)
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_register_implicit_timestamp_ms_accepted(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    resp = _register(client, "iris.worker", worker_schema())
    assert _column_names(resp.effective_schema) == ("worker_id", "mem_bytes", "timestamp_ms")


def test_register_explicit_key_column_echoed(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    schema = stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="ts", type=stats_pb2.COLUMN_TYPE_TIMESTAMP_MS, nullable=False),
        ],
        key_column="ts",
    )
    resp = _register(client, "iris.worker", schema)
    assert resp.effective_schema.key_column == "ts"
    assert _column_names(resp.effective_schema) == ("worker_id", "ts")


def test_register_idempotent_and_subset_return_full(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    full = stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            stats_pb2.Column(name="cpu_pct", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=True),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
    )
    first = _register(client, "iris.worker", full)
    again = _register(client, "iris.worker", full)
    assert _column_names(first.effective_schema) == _column_names(again.effective_schema)

    subset = stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
    )
    eff = _register(client, "iris.worker", subset)
    assert _column_names(eff.effective_schema) == ("worker_id", "mem_bytes", "cpu_pct", "timestamp_ms")


def test_register_additive_nullable_extension_merges(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    extended = stats_pb2.Schema(
        columns=[*_worker_columns(), stats_pb2.Column(name="note", type=stats_pb2.COLUMN_TYPE_STRING, nullable=True)],
    )
    eff = _register(client, "iris.worker", extended)
    assert _column_names(eff.effective_schema) == ("worker_id", "mem_bytes", "timestamp_ms", "note")
    # A subsequent register of the base returns the merged schema.
    again = _register(client, "iris.worker", worker_schema())
    assert _column_names(again.effective_schema) == ("worker_id", "mem_bytes", "timestamp_ms", "note")


def test_register_type_change_rejects(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    bad = stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=False),  # was INT64
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
    )
    with pytest.raises(ConnectError) as exc:
        _register(client, "iris.worker", bad)
    assert exc.value.code == Code.FAILED_PRECONDITION


def test_register_non_additive_new_non_nullable_rejects(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    bad = stats_pb2.Schema(
        columns=[
            *_worker_columns(),
            stats_pb2.Column(name="cpu_pct", type=stats_pb2.COLUMN_TYPE_FLOAT64, nullable=False),
        ],
    )
    with pytest.raises(ConnectError) as exc:
        _register(client, "iris.worker", bad)
    assert exc.value.code == Code.FAILED_PRECONDITION


def test_register_key_column_hint_coerced_to_registered(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    # base has key_column="" (implicit timestamp_ms).
    _register(client, "iris.worker", worker_schema())
    # Re-register with a differing key_column hint succeeds and is coerced.
    eff = _register(client, "iris.worker", worker_schema(key_column="timestamp_ms"))
    assert eff.effective_schema.key_column == ""


def test_get_table_schema_round_trip_and_unknown(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    registered = _register(client, "iris.worker", worker_schema())
    fetched = client.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace="iris.worker"))
    assert _column_names(fetched.schema) == _column_names(registered.effective_schema)
    assert fetched.schema.key_column == registered.effective_schema.key_column

    with pytest.raises(ConnectError) as exc:
        client.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace="nope.unknown"))
    assert exc.value.code == Code.NOT_FOUND


def test_list_namespaces_includes_log_and_zero_stats(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    resp = client.list_namespaces(stats_pb2.ListNamespacesRequest())
    by_name = {info.namespace: info for info in resp.namespaces}
    assert "log" in by_name
    assert "iris.worker" in by_name

    worker = by_name["iris.worker"]
    assert _column_names(worker.schema) == ("worker_id", "mem_bytes", "timestamp_ms")
    assert worker.row_count == 0
    assert worker.byte_size == 0
    assert worker.min_seq == 0
    assert worker.max_seq == 0
    assert worker.segment_count == 0


def test_drop_registered_empty_namespace(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    _register(client, "iris.worker", worker_schema())
    client.drop_table(stats_pb2.DropTableRequest(namespace="iris.worker"))

    # After drop, list no longer includes it and get-schema is NOT_FOUND.
    resp = client.list_namespaces(stats_pb2.ListNamespacesRequest())
    assert "iris.worker" not in {info.namespace for info in resp.namespaces}
    with pytest.raises(ConnectError) as exc:
        client.get_table_schema(stats_pb2.GetTableSchemaRequest(namespace="iris.worker"))
    assert exc.value.code == Code.NOT_FOUND

    # Re-registering the same name succeeds.
    re = _register(client, "iris.worker", worker_schema())
    assert _column_names(re.effective_schema) == ("worker_id", "mem_bytes", "timestamp_ms")


def test_drop_unknown_namespace_not_found(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    with pytest.raises(ConnectError) as exc:
        client.drop_table(stats_pb2.DropTableRequest(namespace="nope.unknown"))
    assert exc.value.code == Code.NOT_FOUND


def test_drop_log_namespace_rejected(finelog_url: str, server_backend: Backend) -> None:
    client = _stats_client(finelog_url)
    with pytest.raises(ConnectError) as exc:
        client.drop_table(stats_pb2.DropTableRequest(namespace="log"))
    assert exc.value.code == Code.INVALID_ARGUMENT
    # log is still listed.
    resp = client.list_namespaces(stats_pb2.ListNamespacesRequest())
    assert "log" in {info.namespace for info in resp.namespaces}


def _ns_info(client: StatsServiceClientSync, namespace: str) -> stats_pb2.NamespaceInfo:
    listed = {n.namespace: n for n in client.list_namespaces(stats_pb2.ListNamespacesRequest()).namespaces}
    return listed[namespace]


def test_effective_policy_round_trips_and_empty_keeps_existing(finelog_url: str, server_backend: Backend) -> None:
    """The storage-policy plumbing is load-bearing and only observable on the
    wire: a handler regression returning the request policy, or dropping the
    empty-keeps-existing rule, would pass every schema-only test. Exercise the
    full RegisterTable.effective_policy contract end to end."""
    client = _stats_client(finelog_url)
    ns = "iris.policy"

    # Fresh register with a non-empty policy -> echoed back verbatim.
    resp = client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=ns,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_segments=5, max_bytes=1024),
        )
    )
    assert resp.effective_policy.max_segments == 5
    assert resp.effective_policy.max_bytes == 1024
    assert resp.effective_policy.max_age_seconds == 0
    # ListNamespaces surfaces the same policy.
    assert _ns_info(client, ns).storage_policy.max_segments == 5

    # Re-register with an EMPTY policy -> existing policy preserved (an old
    # client must not be able to wipe a tighter policy a newer client set).
    kept = client.register_table(stats_pb2.RegisterTableRequest(namespace=ns, schema=worker_schema()))
    assert kept.effective_policy.max_segments == 5
    assert kept.effective_policy.max_bytes == 1024

    # Re-register with a NON-empty policy -> full replace (last-write-wins);
    # unset fields fall back to inherit (proto3 zero).
    replaced = client.register_table(
        stats_pb2.RegisterTableRequest(
            namespace=ns,
            schema=worker_schema(),
            storage_policy=stats_pb2.StoragePolicy(max_age_seconds=99),
        )
    )
    assert replaced.effective_policy.max_age_seconds == 99
    assert replaced.effective_policy.max_segments == 0
    assert replaced.effective_policy.max_bytes == 0
