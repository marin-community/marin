# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

import hashlib
import json
import uuid
from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.persistence import action as action_persistence
from iris.cluster.controller.persistence.database import Tx
from iris.cluster.federation.protocol import CancelTarget
from iris.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.resources.errors import (
    ActionIdempotencyConflict,
)
from iris.resources.identity import (
    AttemptIdentity,
    ResourceKey,
)


@dataclass(frozen=True, slots=True)
class _RemoteActionContext:
    peer_id: str
    authority_cluster_id: str
    backend_id: str
    execution_cluster_id: str


@dataclass(frozen=True, slots=True)
class _CompletedCancel:
    receipt: ActionReceipt
    cancel_target: CancelTarget | None


@dataclass(frozen=True, slots=True)
class _CompletedAction:
    receipt: ActionReceipt


@dataclass(frozen=True, slots=True)
class _RemoteTerminalAction:
    context: _RemoteActionContext
    attempt: AttemptIdentity


def _require_idempotency_key(value: str) -> str:
    if not value.strip():
        raise ValueError("idempotency_key is required")
    return value


def _action_payload_hash(
    kind: ActionKind,
    target_uid: str,
    attempt_uid: str | None,
    reason: str,
) -> str:
    encoded = json.dumps(
        {
            "kind": kind.value,
            "target_uid": target_uid,
            "attempt_uid": attempt_uid,
            "reason": reason,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _duplicate_action(
    tx: Tx,
    *,
    principal_id: str,
    kind: ActionKind,
    idempotency_key: str,
    payload_hash: str,
) -> ActionReceipt | None:
    existing = action_persistence.action_by_idempotency_key(
        tx,
        principal_id=principal_id,
        idempotency_key=_require_idempotency_key(idempotency_key),
    )
    if existing is None:
        return None
    receipt, stored_hash = existing
    if receipt.kind is not kind or stored_hash != payload_hash:
        raise ActionIdempotencyConflict("idempotency key was already used for a different action")
    return receipt


def _completed_action(
    *,
    kind: ActionKind,
    target: ResourceKey,
    expected_target_uid: str,
    expected_attempt_uid: str | None,
    expected_attempt_number: int | None,
    result: ActionResult,
) -> ActionReceipt:
    now = Timestamp.now()
    return ActionReceipt(
        action_id=uuid.uuid4().hex,
        kind=kind,
        target=target,
        expected_target_uid=expected_target_uid,
        expected_attempt_uid=expected_attempt_uid,
        state=ActionState.SUCCEEDED,
        result_code=result,
        result_message="",
        created_at=now,
        updated_at=now,
        completed_at=now,
        expected_attempt_number=expected_attempt_number,
    )
