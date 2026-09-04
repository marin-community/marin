# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller database diagnostics and administrative queries."""

import json
from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import require_identity
from sqlalchemy import select, text

from iris.cluster.controller.checkpoint import CHECKPOINT_EPOCH_META_KEY
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import (
    federation_changelog_table,
    federation_sync_state_table,
    meta_table,
    task_attempts_table,
)
from iris.rpc import query_pb2


@dataclass(frozen=True, slots=True)
class DiagnosticDependencies:
    db: ControllerDB


def probe_database(dependencies: DiagnosticDependencies) -> int | None:
    """Return checkpoint ancestry after verifying controller state is readable."""
    with dependencies.db.read_snapshot() as tx:
        checkpoint_epoch_ms = tx.execute(
            select(meta_table.c.value).where(meta_table.c.key == CHECKPOINT_EPOCH_META_KEY)
        ).scalar()
        tx.execute(select(task_attempts_table.c.attempt_uid).limit(1)).first()
        tx.execute(select(federation_changelog_table.c.seq).limit(1)).first()
        tx.execute(select(federation_sync_state_table.c.peer_id).limit(1)).first()
    return int(checkpoint_epoch_ms) if checkpoint_epoch_ms is not None else None


def execute_raw_query(
    dependencies: DiagnosticDependencies,
    request: query_pb2.RawQueryRequest,
    context: RequestContext,
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
