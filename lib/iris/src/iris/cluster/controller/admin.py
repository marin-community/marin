# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations for health probes, bundles, checkpoints, and raw queries."""

import json
from dataclasses import dataclass
from typing import Any, Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import require_identity
from sqlalchemy import select, text

from iris.cluster.bundle import BundleStore
from iris.cluster.controller.checkpoint import CHECKPOINT_EPOCH_META_KEY, CheckpointResult
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import (
    federation_changelog_table,
    federation_sync_state_table,
    meta_table,
    task_attempts_table,
)
from iris.rpc import controller_pb2, query_pb2
from iris.time_proto import timestamp_to_proto


class CheckpointRuntime(Protocol):
    def begin_checkpoint(self) -> tuple[str, CheckpointResult]: ...


@dataclass(frozen=True, slots=True)
class AdminDependencies:
    db: ControllerDB
    bundles: BundleStore
    runtime: CheckpointRuntime


def bundle_zip(dependencies: AdminDependencies, bundle_id: str) -> bytes:
    return dependencies.bundles.get(bundle_id)


def blob_data(dependencies: AdminDependencies, blob_id: str) -> bytes:
    return dependencies.bundles.get(blob_id)


def probe_database(dependencies: AdminDependencies) -> int | None:
    """Return checkpoint ancestry after verifying controller state is readable."""
    with dependencies.db.read_snapshot() as tx:
        checkpoint_epoch_ms = tx.execute(
            select(meta_table.c.value).where(meta_table.c.key == CHECKPOINT_EPOCH_META_KEY)
        ).scalar()
        tx.execute(select(task_attempts_table.c.attempt_uid).limit(1)).first()
        tx.execute(select(federation_changelog_table.c.seq).limit(1)).first()
        tx.execute(select(federation_sync_state_table.c.peer_id).limit(1)).first()
    return int(checkpoint_epoch_ms) if checkpoint_epoch_ms is not None else None


def begin_checkpoint(
    dependencies: AdminDependencies,
    request: controller_pb2.Controller.BeginCheckpointRequest,
    context: Any,
) -> controller_pb2.Controller.BeginCheckpointResponse:
    del request, context
    path, result = dependencies.runtime.begin_checkpoint()
    response = controller_pb2.Controller.BeginCheckpointResponse(
        checkpoint_path=path,
        job_count=result.job_count,
        task_count=result.task_count,
        worker_count=result.worker_count,
    )
    response.created_at.CopyFrom(timestamp_to_proto(result.created_at))
    return response


def execute_raw_query(
    dependencies: AdminDependencies,
    request: query_pb2.RawQueryRequest,
    context: Any,
) -> query_pb2.RawQueryResponse:
    del context
    identity = require_identity()
    if identity.role != "admin":
        raise ConnectError(Code.PERMISSION_DENIED, "admin role required for raw queries")

    # A read snapshot sets PRAGMA query_only, but a compound PRAGMA could disable
    # it before updating state. Restrict the administrative endpoint to SELECT.
    if request.sql.lstrip()[:6].upper() != "SELECT":
        raise ConnectError(Code.INVALID_ARGUMENT, "only SELECT statements are allowed")

    with dependencies.db.read_snapshot() as tx:
        result = tx.execute(text(request.sql))
        columns = [query_pb2.ColumnMeta(name=name, type="unknown") for name in result.keys()]
        rows = [json.dumps([_encode_query_cell(value) for value in row]) for row in result.all()]
    return query_pb2.RawQueryResponse(columns=columns, rows=rows)


def _encode_query_cell(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<blob:{len(value)} bytes>"
    return value
