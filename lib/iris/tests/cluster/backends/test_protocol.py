# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
from collections.abc import Sequence
from pathlib import Path

import pytest
from iris.cluster.backends.protocol import (
    AttemptRuntime,
    AutoscaleRequest,
    AutoscaleResult,
    BackendBinding,
    BackendCapability,
    ExactAttemptTarget,
    NodeReader,
    ReconcileRequest,
    ReconcileResult,
    ScheduleRequest,
    ScheduleResult,
    SliceReader,
    SourceSnapshot,
)
from iris.cluster.backends.resolver import BackendResolver
from iris.cluster.resources.attempt import AttemptDetail
from iris.cluster.resources.endpoint import ExecResult, ProfileResult
from iris.cluster.resources.errors import BackendIdentityUnknown
from iris.cluster.resources.identity import NodeIdentity, SliceIdentity
from iris.cluster.resources.node import NodeDetail, NodeSummary
from iris.cluster.resources.slice import SliceDetail, SliceSummary
from iris.cluster.resources.source import Freshness, ResourceSourceStatus, SourceState
from iris.rpc import job_pb2
from rigging.timing import Deadline, Duration, Timestamp

NOW = Timestamp.from_ms(1)


def _status(backend_id: str) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=f"backend:{backend_id}",
        backend_id=backend_id,
        state=SourceState.AVAILABLE,
        freshness=Freshness.CURRENT,
        observed_at=NOW,
        error_code="",
        error_message="",
    )


class _Attempts:
    def describe_attempt(self, target: ExactAttemptTarget, *, deadline: Deadline) -> AttemptDetail:
        raise NotImplementedError

    def exec_attempt(
        self,
        target: ExactAttemptTarget,
        command: Sequence[str],
        *,
        deadline: Deadline,
    ) -> ExecResult:
        raise NotImplementedError

    def profile_attempt(
        self,
        target: ExactAttemptTarget,
        profile: job_pb2.ProfileType,
        *,
        duration: Duration,
        deadline: Deadline,
    ) -> ProfileResult:
        raise NotImplementedError


class _Nodes:
    def __init__(self, backend_id: str) -> None:
        self._snapshot = SourceSnapshot[NodeSummary]((), _status(backend_id))

    def snapshot_nodes(self) -> SourceSnapshot[NodeSummary]:
        return self._snapshot

    def describe_node(self, identity: NodeIdentity, *, deadline: Deadline) -> NodeDetail:
        raise NotImplementedError


class _Slices:
    def __init__(self, backend_id: str) -> None:
        self._snapshot = SourceSnapshot[SliceSummary]((), _status(backend_id))

    def snapshot_slices(self) -> SourceSnapshot[SliceSummary]:
        return self._snapshot

    def describe_slice(self, identity: SliceIdentity, *, deadline: Deadline) -> SliceDetail:
        raise NotImplementedError


class _Tasks:
    name = "fake"
    capabilities = frozenset({BackendCapability.WORKER_DAEMON})

    def __init__(self, backend_id: str) -> None:
        self.backend_id = backend_id
        self.attempts: AttemptRuntime = _Attempts()

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return ScheduleResult()

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        return ReconcileResult()

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        return AutoscaleResult(nodes=request.nodes, slices=request.slices)

    def close(self) -> None:
        pass


def _binding(backend_id: str) -> BackendBinding:
    tasks = _Tasks(backend_id)
    nodes: NodeReader = _Nodes(backend_id)
    slices: SliceReader = _Slices(backend_id)
    return BackendBinding(tasks=tasks, nodes=nodes, slices=slices)


def test_backend_resolver_requires_the_exact_configured_identity() -> None:
    gpu = _binding("gpu")
    resolver = BackendResolver({"gpu": gpu})

    assert resolver.require("gpu") is gpu
    for unknown in ("", "default", "GPU"):
        with pytest.raises(BackendIdentityUnknown, match="unknown backend identity"):
            resolver.require(unknown)


def test_backend_resolver_rejects_mismatched_composition() -> None:
    with pytest.raises(ValueError, match="does not match task backend"):
        BackendResolver({"gpu": _binding("cpu")})


def test_backend_resolver_owns_an_immutable_view_of_composition() -> None:
    gpu = _binding("gpu")
    configured = {"gpu": gpu}
    resolver = BackendResolver(configured)
    configured.clear()

    assert resolver.require("gpu") is gpu


def test_backend_protocol_modules_do_not_import_controller_state() -> None:
    backends_dir = Path(__file__).parents[3] / "src" / "iris" / "cluster" / "backends"
    imported: set[str] = set()
    for filename in ("protocol.py", "resolver.py"):
        module = ast.parse((backends_dir / filename).read_text())
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)

    forbidden = ("iris.cluster.controller", "iris.cluster.persistence", "sqlalchemy", "sqlite3")
    assert not sorted(name for name in imported if name.startswith(forbidden))
