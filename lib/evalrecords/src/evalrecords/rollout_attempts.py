# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stored attempt rows and object names of a TaskSpec evaluation archive."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from taskcompendium.grading_result import Outcome

ATTEMPTS_TABLE = "evaluation_attempts"
RUN_OBJECT = "evaluation/run.json"


class RunModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class AttemptPhase(StrEnum):
    STARTED = "started"
    FINISHED = "finished"


class Attempt(RunModel):
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
