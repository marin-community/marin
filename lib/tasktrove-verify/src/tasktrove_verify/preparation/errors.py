# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Terminal preparation failures, independent of any partially prepared value."""

from dataclasses import dataclass

from harbor_config.errors import ErrorCategory

from tasktrove_verify.grade import InvalidTask, Reward, Status


@dataclass(frozen=True)
class PreparationFailure:
    status: Status
    category: ErrorCategory
    error_type: str
    message: str
    stage: str


class PreparationError(RuntimeError):
    """Exception API carrying the original failure and finalized minimum verdict."""

    def __init__(self, failure: PreparationFailure, verdict: Reward):
        super().__init__(failure.message)
        self.failure = failure
        self.verdict = verdict


class InvalidPreparation(PreparationError, InvalidTask):
    """Preparation error compatible with the existing InvalidTask exception API."""
