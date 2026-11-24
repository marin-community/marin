# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage conventions for compact task health receipts emitted by RL workers."""

from dataclasses import dataclass
from enum import StrEnum

ATTEMPT_RECEIPTS_DIRECTORY = "task-health"


class Phase(StrEnum):
    NONE = "none"
    ENVIRONMENT = "environment"
    TASK_SETUP = "task_setup"
    AGENT = "agent"
    VERIFIER = "verifier"


@dataclass(frozen=True)
class AttemptReceipt:
    attempt_id: str
    task_id: str
    task_digest: str
    outcome: str
    grading_failure: str | None
    exception_type: str | None
    error: str | None
    phase: Phase
    docker_image: str | None
    setup_sha256: str | None
    setup_completed: bool | None
    setup_stdout: str | None
    setup_stderr: str | None
    verifier_stdout: str | None
    verifier_stderr: str | None
