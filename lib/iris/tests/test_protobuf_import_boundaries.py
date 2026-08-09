# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ratchet RPC dependencies toward explicit transport boundaries."""

import ast
from pathlib import Path

from iris.rpc import resource_pb2

_SOURCE_ROOT = Path(__file__).parents[1] / "src" / "iris"
_GENERATED_SOURCE_SUFFIXES = ("_pb2.py", "_connect.py")
_GENERATED_IMPORT_SUFFIXES = ("_pb2", "_connect")

# These modules already are transport clients, servers, or codecs. Entries are
# deliberately per imported module so unrelated protobufs cannot leak into an
# otherwise valid adapter unnoticed.
_TRANSPORT_ALLOWLIST = frozenset(
    {
        ("actor/client.py", "iris.rpc.actor_connect"),
        ("actor/client.py", "iris.rpc.actor_pb2"),
        ("actor/client.py", "iris.rpc.compression"),
        ("actor/client.py", "iris.rpc.errors"),
        ("actor/pool.py", "iris.rpc.actor_connect"),
        ("actor/pool.py", "iris.rpc.actor_pb2"),
        ("actor/pool.py", "iris.rpc.compression"),
        ("actor/pool.py", "iris.rpc.errors"),
        ("actor/server.py", "iris.rpc.actor_connect"),
        ("actor/server.py", "iris.rpc.actor_pb2"),
        ("actor/server.py", "iris.rpc.compression"),
        ("cli/connect.py", "iris.rpc.compression"),
        ("cli/connect.py", "iris.rpc.controller_connect"),
        ("cli/cluster.py", "iris.rpc.worker_codec"),
        ("cli/query.py", "iris.rpc.query_pb2"),
        ("cli/rpc.py", "iris.rpc.actor_connect"),
        ("cli/rpc.py", "iris.rpc.controller_connect"),
        ("cli/rpc.py", "iris.rpc.worker_connect"),
        ("cluster/backends/k8s/logship.py", "finelog.rpc.logging_pb2"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.legacy_job_codec"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.worker_codec"),
        ("cli/process_status.py", "iris.rpc.worker_codec"),
        ("cluster/client/endpoint_client.py", "iris.rpc.controller_pb2"),
        ("cluster/client/endpoint_client.py", "iris.rpc.errors"),
        ("cluster/client/resource_client.py", "iris.rpc.compression"),
        ("cluster/client/resource_client.py", "iris.rpc.errors"),
        ("cluster/client/resource_client.py", "iris.rpc.resource_connect"),
        ("cluster/client/resource_client.py", "iris.rpc.resource_pb2"),
        ("cluster/client/resource_client.py", "iris.rpc.resource_codec"),
        ("cluster/client/resource_conversion.py", "iris.rpc.resource_pb2"),
        ("cluster/client/resource_conversion.py", "iris.rpc.resource_codec"),
        ("cluster/controller/dashboard.py", "iris.rpc.async_adapter"),
        ("cluster/controller/dashboard.py", "iris.rpc.auth"),
        ("cluster/controller/dashboard.py", "iris.rpc.compression"),
        ("cluster/controller/dashboard.py", "iris.rpc.controller_connect"),
        ("cluster/controller/dashboard.py", "iris.rpc.interceptors"),
        ("cluster/controller/dashboard.py", "iris.rpc.resource_connect"),
        ("cluster/controller/resources/legacy_rpc.py", "iris.rpc.controller_pb2"),
        ("cluster/controller/resources/legacy_rpc.py", "iris.rpc.job_pb2"),
        ("cluster/controller/resources/legacy_rpc.py", "iris.rpc.legacy_job_codec"),
        ("cluster/controller/resources/legacy_rpc.py", "iris.rpc.proto_display"),
        ("cluster/controller/resources/logs.py", "finelog.rpc.logging_pb2"),
        ("cluster/controller/resources/rpc.py", "iris.rpc.auth"),
        ("cluster/controller/resources/rpc.py", "iris.rpc.iris_logging_pb2"),
        ("cluster/controller/resources/rpc.py", "iris.rpc.resource_codec"),
        ("cluster/controller/resources/rpc.py", "iris.rpc.resource_pb2"),
        ("cluster/controller/service.py", "iris.rpc.worker_codec"),
        ("cluster/federation/legacy_rpc.py", "iris.rpc.controller_pb2"),
        ("cluster/federation/legacy_rpc.py", "iris.rpc.job_pb2"),
        ("cluster/federation/legacy_rpc.py", "iris.rpc.legacy_job_codec"),
        ("cluster/federation/peer.py", "iris.rpc.worker_codec"),
        ("cluster/worker/dashboard.py", "iris.rpc.async_adapter"),
        ("cluster/worker/dashboard.py", "iris.rpc.compression"),
        ("cluster/worker/dashboard.py", "iris.rpc.worker_connect"),
        ("cluster/worker/worker.py", "iris.rpc.worker_codec"),
        ("cluster/worker/service.py", "iris.rpc.errors"),
        ("cluster/worker/service.py", "iris.rpc.job_pb2"),
        ("cluster/worker/service.py", "iris.rpc.worker_codec"),
        ("cluster/worker/service.py", "iris.rpc.worker_pb2"),
        ("rpc/__init__.py", "iris.rpc.controller_pb2"),
        ("rpc/__init__.py", "iris.rpc.job_pb2"),
        ("rpc/__init__.py", "iris.rpc.query_pb2"),
        ("rpc/__init__.py", "iris.rpc.time_pb2"),
        ("rpc/__init__.py", "iris.rpc.vm_pb2"),
        ("rpc/__init__.py", "iris.rpc.worker_pb2"),
        ("rpc/compression.py", "iris.rpc.codecs"),
        ("rpc/errors.py", "google.protobuf.any_pb2"),
        ("rpc/errors.py", "iris.rpc.errors_pb2"),
        ("rpc/interceptors.py", "iris.rpc.errors"),
        ("rpc/legacy_job_codec.py", "iris.rpc.job_pb2"),
        ("rpc/profile_codec.py", "iris.rpc.job_pb2"),
        ("rpc/proto_display.py", "iris.rpc.vm_pb2"),
        ("rpc/resource_codec.py", "iris.rpc.resource_pb2"),
        ("rpc/worker_codec.py", "iris.rpc.job_pb2"),
        ("rpc/worker_codec.py", "iris.rpc.legacy_job_codec"),
        ("rpc/worker_codec.py", "iris.rpc.profile_codec"),
        ("rpc/worker_codec.py", "iris.rpc.worker_pb2"),
        ("time_proto.py", "iris.rpc.time_pb2"),
    }
)

# This is migration debt, not an exemption. Remove an entry in the same change
# that moves its conversion to a transport adapter. Exact equality below makes
# both new leakage and stale debt visible.
_PROTOBUF_DEBT = frozenset(
    {
        ("cli/cluster.py", "iris.rpc.controller_pb2"),
        ("cli/cluster.py", "iris.rpc.job_pb2"),
        ("cli/cluster.py", "iris.rpc.proto_display"),
        ("cli/cluster.py", "iris.rpc.query_pb2"),
        ("cli/cluster.py", "iris.rpc.vm_pb2"),
        ("cli/main.py", "iris.rpc.controller_pb2"),
        ("cli/main.py", "iris.rpc.proto_display"),
        ("cli/process_status.py", "finelog.rpc.logging_pb2"),
        ("cli/process_status.py", "iris.rpc.job_pb2"),
        ("cli/process_status.py", "iris.rpc.profile_codec"),
        ("cli/resource_commands.py", "iris.rpc.proto_display"),
        ("cluster/backends/k8s/tasks.py", "iris.rpc.controller_pb2"),
        ("cluster/backends/k8s/tasks.py", "iris.rpc.vm_pb2"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.compression"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.controller_pb2"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.job_pb2"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.profile_codec"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.vm_pb2"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.worker_connect"),
        ("cluster/backends/rpc/backend.py", "iris.rpc.worker_pb2"),
        ("cluster/client/job_info.py", "iris.rpc.job_pb2"),
        ("cluster/client/job_info.py", "iris.rpc.legacy_job_codec"),
        ("cluster/client/remote_client.py", "iris.rpc.compression"),
        ("cluster/client/remote_client.py", "iris.rpc.controller_connect"),
        ("cluster/controller/auth.py", "iris.rpc.auth"),
        ("cluster/controller/autoscaler/operations.py", "iris.rpc.vm_pb2"),
        ("cluster/controller/autoscaler/runtime.py", "iris.rpc.vm_pb2"),
        ("cluster/controller/autoscaler/scaling_group.py", "iris.rpc.time_pb2"),
        ("cluster/controller/autoscaler/scaling_group.py", "iris.rpc.vm_pb2"),
        ("cluster/controller/autoscaler/status.py", "iris.rpc.vm_pb2"),
        ("cluster/controller/autoscaler/worker_registry.py", "iris.rpc.vm_pb2"),
        ("cluster/controller/backend.py", "iris.rpc.controller_pb2"),
        ("cluster/controller/backend.py", "iris.rpc.vm_pb2"),
        ("cluster/controller/controller.py", "iris.rpc.auth"),
        ("cluster/controller/endpoint_service.py", "iris.rpc.controller_pb2"),
        ("cluster/controller/endpoint_service.py", "iris.rpc.job_pb2"),
        ("cluster/controller/resources/jobs.py", "iris.rpc.auth"),
        ("cluster/controller/service.py", "iris.rpc.auth"),
        ("cluster/controller/service.py", "iris.rpc.controller_pb2"),
        ("cluster/controller/service.py", "iris.rpc.job_pb2"),
        ("cluster/controller/service.py", "iris.rpc.legacy_job_codec"),
        ("cluster/controller/service.py", "iris.rpc.profile_codec"),
        ("cluster/controller/service.py", "iris.rpc.proto_display"),
        ("cluster/controller/service.py", "iris.rpc.query_pb2"),
        ("cluster/controller/service.py", "iris.rpc.vm_pb2"),
        ("cluster/federation/manager.py", "iris.rpc.controller_pb2"),
        ("cluster/federation/peer.py", "iris.rpc.controller_connect"),
        ("cluster/federation/peer.py", "iris.rpc.controller_pb2"),
        ("cluster/federation/peer.py", "iris.rpc.job_pb2"),
        ("cluster/federation/peer.py", "iris.rpc.legacy_job_codec"),
        ("cluster/federation/peer.py", "iris.rpc.profile_codec"),
        ("cluster/federation/peer.py", "iris.rpc.resource_connect"),
        ("cluster/federation/peer.py", "iris.rpc.resource_pb2"),
        ("cluster/log_keys.py", "finelog.rpc.logging_pb2"),
        ("cluster/types.py", "iris.rpc.controller_pb2"),
        ("cluster/worker/task_attempt.py", "finelog.rpc.logging_pb2"),
        ("cluster/worker/task_attempt.py", "iris.rpc.errors"),
        ("cluster/worker/task_attempt.py", "iris.rpc.proto_display"),
        ("cluster/worker/worker.py", "iris.rpc.compression"),
        ("cluster/worker/worker.py", "iris.rpc.controller_connect"),
        ("cluster/worker/worker.py", "iris.rpc.controller_pb2"),
        ("cluster/worker/worker_types.py", "finelog.rpc.logging_pb2"),
    }
)


def _tracked_module(module: str) -> bool:
    return module == "iris.rpc" or module.startswith("iris.rpc.") or module.endswith(_GENERATED_IMPORT_SUFFIXES)


def _tracked_imports(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names if _tracked_module(alias.name))
    if not isinstance(node, ast.ImportFrom):
        return ()

    parent = node.module or ""
    if parent == "iris.rpc":
        return tuple(f"{parent}.{alias.name}" for alias in node.names)
    if parent.startswith("iris.rpc.") or _tracked_module(parent):
        return (parent,)

    joined = tuple(f"{parent}.{alias.name}" if parent else alias.name for alias in node.names)
    return tuple(module for module in joined if _tracked_module(module))


def _rpc_imports() -> frozenset[tuple[str, str]]:
    imports: set[tuple[str, str]] = set()
    generated_root = _SOURCE_ROOT / "rpc"
    for path in _SOURCE_ROOT.rglob("*.py"):
        if path.is_relative_to(generated_root) and path.name.endswith(_GENERATED_SOURCE_SUFFIXES):
            continue
        relative_path = path.relative_to(_SOURCE_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            imports.update((relative_path, module) for module in _tracked_imports(node))
    return frozenset(imports)


def _format_entries(entries: set[tuple[str, str]] | frozenset[tuple[str, str]]) -> str:
    if not entries:
        return "  (none)"
    return "\n".join(f"  {path}: {module}" for path, module in sorted(entries))


def test_rpc_imports_match_transport_boundaries_and_debt_manifest() -> None:
    observed = _rpc_imports()
    expected = _TRANSPORT_ALLOWLIST | _PROTOBUF_DEBT
    introduced = observed - expected
    resolved = _PROTOBUF_DEBT - observed
    stale_transport_entries = _TRANSPORT_ALLOWLIST - observed

    assert observed == expected, (
        "RPC imports crossed the resource-model boundary.\n"
        "Move new imports to a transport adapter; remove resolved entries from _PROTOBUF_DEBT.\n"
        f"New dependency debt:\n{_format_entries(introduced)}\n"
        f"Resolved debt still listed:\n{_format_entries(resolved)}\n"
        f"Stale transport allowlist entries:\n{_format_entries(stale_transport_entries)}"
    )


def test_resource_service_wire_is_independent_of_the_retired_job_wire() -> None:
    assert "job.proto" not in {dependency.name for dependency in resource_pb2.DESCRIPTOR.dependencies}
