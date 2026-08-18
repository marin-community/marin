# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistence for durable resource mutations and their runtime targets."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from connectrpc.code import Code
from google.protobuf import any_pb2
from rigging.timing import Timestamp
from sqlalchemy import insert, select, update

from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import (
    resource_operation_targets_table,
    resource_operations_table,
)
from iris.cluster.types import WorkerId
from iris.rpc import resource_pb2
from iris.time_proto import timestamp_to_proto

OPERATION_TYPE = "iris/operation"

# Completed operations are permanent replay fences for logical-current
# selectors. Deleting one would let a delayed retry bind to a replacement Job or
# Task with the same name.


class OperationPhase(StrEnum):
    ACCEPTED = "accepted"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeTarget:
    ref: resource_pb2.ResourceRef
    backend_id: str
    worker_id: WorkerId | None


class OperationRow(Protocol):
    operation_id: str
    principal_id: str
    request_id: str
    request_hash: str
    verb: str
    requested_authority: str
    requested_type: str
    requested_id: str
    requested_uid: str | None
    resolved_authority: str
    resolved_type: str
    resolved_id: str
    resolved_uid: str
    update_type_url: str
    update_payload: bytes
    reason: str
    phase: str
    peer_id: str
    remote_operation_id: str
    error_code: str | None
    error_message: str
    accepted_at_ms: Timestamp
    applied_at_ms: Timestamp | None
    completed_at_ms: Timestamp | None


def by_request(tx: Tx, principal_id: str, request_id: str) -> OperationRow | None:
    row = tx.execute(
        select(resource_operations_table).where(
            resource_operations_table.c.principal_id == principal_id,
            resource_operations_table.c.request_id == request_id,
        )
    ).first()
    return cast(OperationRow | None, row)


def by_id(tx: Tx, operation_id: str) -> OperationRow | None:
    row = tx.execute(
        select(resource_operations_table).where(resource_operations_table.c.operation_id == operation_id)
    ).first()
    return cast(OperationRow | None, row)


def pending(tx: Tx) -> list[OperationRow]:
    rows = tx.execute(
        select(resource_operations_table)
        .where(resource_operations_table.c.phase.in_((OperationPhase.ACCEPTED, OperationPhase.VERIFYING)))
        .order_by(resource_operations_table.c.accepted_at_ms, resource_operations_table.c.operation_id)
    ).all()
    return cast(list[OperationRow], rows)


def insert_operation(
    tx: Tx,
    *,
    operation_id: str,
    principal_id: str,
    request_id: str,
    request_hash: str,
    requested_ref: resource_pb2.ResourceRef,
    resolved_ref: resource_pb2.ResourceRef,
    update: any_pb2.Any,
    reason: str,
    phase: OperationPhase,
    peer_id: str,
    targets: tuple[RuntimeTarget, ...],
    now: Timestamp,
) -> None:
    tx.execute(
        insert(resource_operations_table).values(
            operation_id=operation_id,
            principal_id=principal_id,
            request_id=request_id,
            request_hash=request_hash,
            verb="update",
            requested_authority=requested_ref.authority_cluster_id,
            requested_type=requested_ref.type,
            requested_id=requested_ref.id,
            requested_uid=requested_ref.uid if requested_ref.HasField("uid") else None,
            resolved_authority=resolved_ref.authority_cluster_id,
            resolved_type=resolved_ref.type,
            resolved_id=resolved_ref.id,
            resolved_uid=resolved_ref.uid,
            update_type_url=update.type_url,
            update_payload=update.value,
            reason=reason,
            phase=phase,
            peer_id=peer_id,
            accepted_at_ms=now,
            applied_at_ms=now if phase != OperationPhase.ACCEPTED else None,
            completed_at_ms=now if phase == OperationPhase.VERIFIED else None,
        )
    )
    if targets:
        tx.execute(
            insert(resource_operation_targets_table),
            [
                {
                    "operation_id": operation_id,
                    "ordinal": ordinal,
                    "authority_cluster_id": target.ref.authority_cluster_id,
                    "resource_type": target.ref.type,
                    "resource_id": target.ref.id,
                    "resource_uid": target.ref.uid,
                    "backend_id": target.backend_id,
                    "worker_id": target.worker_id,
                }
                for ordinal, target in enumerate(targets)
            ],
        )


def mark_verifying(tx: Tx, operation_id: str, *, remote_operation_id: str = "") -> None:
    values: dict[str, object] = {
        "phase": OperationPhase.VERIFYING,
        "applied_at_ms": Timestamp.now(),
    }
    if remote_operation_id:
        values["remote_operation_id"] = remote_operation_id
    tx.execute(
        update(resource_operations_table)
        .where(resource_operations_table.c.operation_id == operation_id)
        .values(**values)
    )


def mark_verified(tx: Tx, operation_id: str) -> None:
    now = Timestamp.now()
    tx.execute(
        update(resource_operations_table)
        .where(resource_operations_table.c.operation_id == operation_id)
        .values(phase=OperationPhase.VERIFIED, completed_at_ms=now)
    )


def mark_failed(tx: Tx, operation_id: str, *, code: Code, message: str) -> None:
    now = Timestamp.now()
    tx.execute(
        update(resource_operations_table)
        .where(resource_operations_table.c.operation_id == operation_id)
        .values(
            phase=OperationPhase.FAILED,
            error_code=code.value,
            error_message=message,
            completed_at_ms=now,
        )
    )


def verification_targets(tx: Tx) -> dict[str, tuple[RuntimeTarget, ...]]:
    """Return exact runtime targets for local operations awaiting verification."""
    rows = tx.execute(
        select(
            resource_operations_table.c.operation_id,
            resource_operation_targets_table.c.authority_cluster_id,
            resource_operation_targets_table.c.resource_type,
            resource_operation_targets_table.c.resource_id,
            resource_operation_targets_table.c.resource_uid,
            resource_operation_targets_table.c.backend_id,
            resource_operation_targets_table.c.worker_id,
        )
        .join(
            resource_operation_targets_table,
            resource_operation_targets_table.c.operation_id == resource_operations_table.c.operation_id,
        )
        .where(
            resource_operations_table.c.phase == OperationPhase.VERIFYING,
            resource_operations_table.c.peer_id == "",
        )
        .order_by(resource_operations_table.c.operation_id, resource_operation_targets_table.c.ordinal)
    ).all()
    targets: dict[str, list[RuntimeTarget]] = {}
    for row in rows:
        targets.setdefault(row.operation_id, []).append(
            RuntimeTarget(
                ref=resource_pb2.ResourceRef(
                    authority_cluster_id=row.authority_cluster_id,
                    type=row.resource_type,
                    id=row.resource_id,
                    uid=row.resource_uid,
                ),
                backend_id=row.backend_id,
                worker_id=row.worker_id,
            )
        )
    return {operation_id: tuple(operation_targets) for operation_id, operation_targets in targets.items()}


def to_proto(tx: Tx, row: OperationRow) -> resource_pb2.Operation:
    operation = resource_pb2.Operation(
        ref=resource_pb2.ResourceRef(
            authority_cluster_id=row.resolved_authority,
            type=OPERATION_TYPE,
            id=row.operation_id,
            uid=row.operation_id,
        ),
        phase=_phase_to_proto(row.phase),
        verb=row.verb,
        requested_ref=_requested_ref(row),
        resolved_ref=resolved_ref(row),
        accepted_at=timestamp_to_proto(row.accepted_at_ms),
    )
    target_rows = tx.execute(
        select(resource_operation_targets_table)
        .where(resource_operation_targets_table.c.operation_id == row.operation_id)
        .order_by(resource_operation_targets_table.c.ordinal)
    ).all()
    operation.affected.extend(
        resource_pb2.ResourceRef(
            authority_cluster_id=target.authority_cluster_id,
            type=target.resource_type,
            id=target.resource_id,
            uid=target.resource_uid,
        )
        for target in target_rows
    )
    if row.applied_at_ms is not None:
        operation.applied_at.CopyFrom(timestamp_to_proto(row.applied_at_ms))
    if row.completed_at_ms is not None:
        operation.completed_at.CopyFrom(timestamp_to_proto(row.completed_at_ms))
    if row.error_code is not None:
        operation.error.CopyFrom(resource_pb2.OperationError(code=row.error_code, message=row.error_message))
    return operation


def forwarded_request(row: OperationRow) -> resource_pb2.UpdateResourceRequest:
    return resource_pb2.UpdateResourceRequest(
        mutation=resource_pb2.MutationMetadata(request_id=row.operation_id, reason=row.reason),
        ref=resolved_ref(row),
        update=any_pb2.Any(type_url=row.update_type_url, value=row.update_payload),
    )


def _requested_ref(row: OperationRow) -> resource_pb2.ResourceRef:
    ref = resource_pb2.ResourceRef(
        authority_cluster_id=row.requested_authority,
        type=row.requested_type,
        id=row.requested_id,
    )
    if row.requested_uid is not None:
        ref.uid = row.requested_uid
    return ref


def resolved_ref(row: OperationRow) -> resource_pb2.ResourceRef:
    return resource_pb2.ResourceRef(
        authority_cluster_id=row.resolved_authority,
        type=row.resolved_type,
        id=row.resolved_id,
        uid=row.resolved_uid,
    )


def _phase_to_proto(phase: str) -> int:
    match OperationPhase(phase):
        case OperationPhase.ACCEPTED:
            return resource_pb2.OPERATION_PHASE_ACCEPTED
        case OperationPhase.VERIFYING:
            return resource_pb2.OPERATION_PHASE_VERIFYING
        case OperationPhase.VERIFIED:
            return resource_pb2.OPERATION_PHASE_VERIFIED
        case OperationPhase.FAILED:
            return resource_pb2.OPERATION_PHASE_FAILED
    raise ValueError(f"unknown operation phase: {phase!r}")
