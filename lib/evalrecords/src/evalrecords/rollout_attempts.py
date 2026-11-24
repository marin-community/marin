# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stored attempt rows and object names of a TaskSpec evaluation archive."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict
from taskcompendium.grading_result import Outcome

ATTEMPTS_TABLE = "evaluation_attempts"
RUN_OBJECT = "evaluation/run.json"


class AttemptPhase(StrEnum):
    STARTED = "started"
    FINISHED = "finished"


class Attempt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    dataset: str
    task_id: str
    source_row: str
    trial: int
    retry: int
    phase: AttemptPhase
    seed: int | None
    outcome: Outcome | None
    error: str | None
    result_blob: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    stop_reason: str | None = None
    """The rollout's stop reason, such as ``"stop"`` or ``"length"``.

    ``None`` on a started row, for likelihood scoring, and in result roots written before this column existed.
    """


def attempt_from_row(row: Mapping[str, Any]) -> Attempt:
    """Read a stored attempt row, including one that predates a column with a default."""
    return Attempt.model_validate({name: row[name] for name in Attempt.model_fields if name in row})
