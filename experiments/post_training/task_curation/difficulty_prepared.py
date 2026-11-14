# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared source contracts shared with Harbor's isolated trial process."""

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import DifficultySample
from pydantic import BaseModel, ConfigDict


class PreparedTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    path: str | None
    workload: str
    backend: str
    reason: str | None
    judge_model: str | None


class PreparedSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sample: DifficultySample
    sample_receipt: ContentEvidenceReference
    tasks: list[PreparedTask]
