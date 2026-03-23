# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RL run lifecycle actor.

Tracks whether the training run is running, completed, or failed.
Rollout workers poll this to know when to shut down. The trainer
signals completion or failure. This is separate from weight transfer
coordination — lifecycle is its own concern.
"""

import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RLRunState:
    """Lightweight actor that tracks RL run lifecycle.

    Created by the coordinator, shared by trainer and rollout workers.
    """

    def __init__(self):
        self._status: RunStatus = RunStatus.RUNNING
        self._failure_message: str | None = None

    def get_status(self) -> str:
        return self._status.value

    def is_terminal(self) -> bool:
        return self._status != RunStatus.RUNNING

    def mark_completed(self) -> None:
        if self._status == RunStatus.RUNNING:
            self._status = RunStatus.COMPLETED
            logger.info("RL run marked as completed")

    def mark_failed(self, message: str = "") -> None:
        if self._status == RunStatus.RUNNING:
            self._status = RunStatus.FAILED
            self._failure_message = message
            logger.info("RL run marked as failed: %s", message)

    def get_failure_message(self) -> str | None:
        return self._failure_message
