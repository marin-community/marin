# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed persistence for durable resource-action receipts."""

from rigging.timing import Timestamp
from sqlalchemy import and_, insert, literal, or_, select

from iris.cluster.controller.persistence.database import Tx
from iris.cluster.controller.persistence.schema import action_receipts_table
from iris.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.resources.identity import ResourceKey, ResourceKind


def action_by_idempotency_key(
    tx: Tx,
    *,
    principal_id: str,
    kind: ActionKind,
    idempotency_key: str,
) -> tuple[ActionReceipt, str] | None:
    row = tx.execute(
        select(action_receipts_table).where(
            action_receipts_table.c.principal_id == principal_id,
            action_receipts_table.c.action_kind == kind.value,
            action_receipts_table.c.client_idempotency_key == idempotency_key,
        )
    ).first()
    return (_receipt(row), row.payload_hash) if row is not None else None


def action_by_id(tx: Tx, action_id: str) -> ActionReceipt | None:
    row = tx.execute(select(action_receipts_table).where(action_receipts_table.c.action_id == action_id)).first()
    return _receipt(row) if row is not None else None


def actions_for_target(
    tx: Tx,
    target: ResourceKey,
    *,
    after: Timestamp | None,
    before: tuple[Timestamp, str] | None = None,
    limit: int,
) -> tuple[ActionReceipt, ...]:
    target_id = target.resource_id
    if target.kind is ResourceKind.JOB:
        scope = or_(
            action_receipts_table.c.target_id == target_id,
            action_receipts_table.c.target_id.like(f"{_escaped_like(target_id)}/%", escape="\\"),
        )
    elif target.kind is ResourceKind.TASK:
        scope = or_(
            action_receipts_table.c.target_id == target_id,
            action_receipts_table.c.target_id.like(f"{_escaped_like(target_id)}:%", escape="\\"),
        )
    else:
        scope = action_receipts_table.c.target_id == target_id
    stmt = select(action_receipts_table).where(
        action_receipts_table.c.authority_cluster_id == target.cluster_id,
        scope,
    )
    if after is not None:
        stmt = stmt.where(action_receipts_table.c.updated_at_ms > after.epoch_ms())
    if before is not None:
        before_time, before_entry_id = before
        entry_id = literal("action:").concat(action_receipts_table.c.action_id)
        stmt = stmt.where(
            or_(
                action_receipts_table.c.updated_at_ms < before_time.epoch_ms(),
                and_(
                    action_receipts_table.c.updated_at_ms == before_time.epoch_ms(),
                    entry_id < before_entry_id,
                ),
            )
        )
    rows = tx.execute(
        stmt.order_by(action_receipts_table.c.updated_at_ms.desc(), action_receipts_table.c.action_id.desc()).limit(
            limit
        )
    ).all()
    return tuple(_receipt(row) for row in rows)


def insert_action(
    tx: Tx,
    receipt: ActionReceipt,
    *,
    authority_cluster_id: str,
    authority_action_id: str,
    backend_id: str,
    execution_cluster_id: str,
    principal_id: str,
    idempotency_key: str,
    payload_hash: str,
) -> None:
    tx.execute(
        insert(action_receipts_table).values(
            action_id=receipt.action_id,
            authority_cluster_id=authority_cluster_id,
            authority_action_id=authority_action_id,
            action_kind=receipt.kind.value,
            target_kind=receipt.target.kind.value,
            target_id=receipt.target.resource_id,
            expected_target_uid=receipt.expected_target_uid,
            expected_attempt_uid=receipt.expected_attempt_uid or "",
            backend_id=backend_id,
            execution_cluster_id=execution_cluster_id,
            principal_id=principal_id,
            client_idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            state=receipt.state.value,
            result_code=receipt.result_code.value,
            result_message=receipt.result_message,
            created_at_ms=receipt.created_at.epoch_ms(),
            updated_at_ms=receipt.updated_at.epoch_ms(),
            completed_at_ms=receipt.completed_at.epoch_ms() if receipt.completed_at is not None else None,
        )
    )


def _receipt(row) -> ActionReceipt:
    return ActionReceipt(
        action_id=row.action_id,
        kind=ActionKind(row.action_kind),
        target=ResourceKey(row.authority_cluster_id, ResourceKind(row.target_kind), row.target_id),
        expected_target_uid=row.expected_target_uid,
        expected_attempt_uid=row.expected_attempt_uid or None,
        state=ActionState(row.state),
        result_code=ActionResult(row.result_code),
        result_message=row.result_message,
        created_at=Timestamp.from_ms(row.created_at_ms),
        updated_at=Timestamp.from_ms(row.updated_at_ms),
        completed_at=Timestamp.from_ms(row.completed_at_ms) if row.completed_at_ms is not None else None,
    )


def _escaped_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
