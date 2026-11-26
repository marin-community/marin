# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execution evidence kept separately from task definitions and review findings."""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from taskcompendium.models import (
    ConversationEvent,
    ConversationTrace,
    GradingAttempt,
    StateSubmission,
    TaskResource,
    decode_json_value,
)
from taskcompendium.runtime.resources import resource_bytes


@dataclass(frozen=True)
class RuntimeEvidence:
    files: dict[str, bytes]
    state_json: str


def grading_attempt(conversation: ConversationTrace, evidence: RuntimeEvidence) -> GradingAttempt:
    """Pair a conversation with the files and decoded state an environment captured."""
    return GradingAttempt(conversation, evidence.files, StateSubmission(decode_json_value(evidence.state_json)))


class Termination(StrEnum):
    FINAL_MESSAGE = "final_message"
    STEP_LIMIT = "step_limit"
    INFRA_ERROR = "infra_error"


class RolloutRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    control: str
    events: tuple[ConversationEvent, ...]
    termination: Termination
    artifacts: tuple[TaskResource, ...]
    state_json: str
    detail: str

    def evidence(self) -> RuntimeEvidence:
        return RuntimeEvidence(
            {f"/{resource.path}": resource_bytes(resource) for resource in self.artifacts}, self.state_json
        )
