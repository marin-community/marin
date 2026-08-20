# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable public resource-action receipts."""

from dataclasses import dataclass
from enum import StrEnum

from rigging.timing import Timestamp

from iris.resources.errors import InvalidResourceKey
from iris.resources.identity import ResourceKey, ResourceKind


class ActionKind(StrEnum):
    CANCEL_JOB = "cancel_job"
    RETRY_TASK = "retry_task"
    TERMINATE_ATTEMPT = "terminate_attempt"
    FAIL_ATTEMPT = "fail_attempt"


class ActionState(StrEnum):
    ACCEPTED = "accepted"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ActionResult(StrEnum):
    NONE = "none"
    SATISFIED = "satisfied"
    TARGET_ABSENT = "target_absent"
    PROVIDER_REJECTED = "provider_rejected"
    INTERNAL_ERROR = "internal_error"


_ACTION_TARGET_KIND = {
    ActionKind.CANCEL_JOB: ResourceKind.JOB,
    ActionKind.RETRY_TASK: ResourceKind.TASK,
    ActionKind.TERMINATE_ATTEMPT: ResourceKind.ATTEMPT,
    ActionKind.FAIL_ATTEMPT: ResourceKind.ATTEMPT,
}


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    action_id: str
    kind: ActionKind
    target: ResourceKey
    expected_target_uid: str
    expected_attempt_uid: str | None
    state: ActionState
    result_code: ActionResult
    result_message: str
    created_at: Timestamp
    updated_at: Timestamp
    completed_at: Timestamp | None
    expected_attempt_number: int | None = None

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise InvalidResourceKey("action_id must be non-empty")
        if self.target.kind is not _ACTION_TARGET_KIND[self.kind]:
            raise InvalidResourceKey(f"{self.kind.value} cannot target {self.target.kind.value}")
        if not self.expected_target_uid.strip():
            raise InvalidResourceKey("expected_target_uid must be non-empty")
        if self.expected_attempt_uid is not None and not self.expected_attempt_uid.strip():
            raise InvalidResourceKey("expected_attempt_uid must be non-empty when present")
        if self.kind is ActionKind.CANCEL_JOB:
            if self.expected_attempt_uid is not None or self.expected_attempt_number is not None:
                raise InvalidResourceKey("cancel_job cannot target an Attempt")
        elif self.expected_attempt_uid is None or self.expected_attempt_number is None:
            raise InvalidResourceKey(f"{self.kind.value} requires an exact Attempt")
        elif self.expected_attempt_number < 0:
            raise InvalidResourceKey("expected_attempt_number must be non-negative")
