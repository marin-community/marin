# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared source contracts shared with Harbor's isolated trial process."""

import base64
import tomllib
from collections.abc import Mapping
from pathlib import Path

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import DifficultySample
from pydantic import BaseModel, ConfigDict, Field

from experiments.post_training.task_curation.difficulty_config import (
    DECLARED_JUDGE_ENV_VAR,
    DECLARED_JUDGE_PINS,
    DECLARED_JUDGE_PROVIDER_PREFIX,
)

GRADER_PRIVILEGE_DROP_UID = "user=65534"


def grader_requires_multi_uid(spec: Mapping) -> bool:
    """Check whether the frozen verifier requests a fixed-UID privilege drop.

    LOCAL maps one UID, so these graders need Daytona's full UID map.
    """
    for resource in (spec.get("resources") or {}).get("verifier") or ():
        content = base64.b64decode(resource["source"]["content_base64"]).decode("utf-8", errors="replace")
        if GRADER_PRIVILEGE_DROP_UID in content:
            return True
    return False


class PreparedTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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


def task_declared_judge_model(task: PreparedTask) -> str | None:
    """The grader model a frozen native bundle declares for its own verifier, if any.

    Parses the frozen task.toml directly with stdlib tomllib: this contract
    helper must stay importable where Harbor (and marin) are not installed - the
    controller, the runtime, and Harbor's isolated trial process. Only qualified
    declared judges are served, so an unqualified declaration fails closed
    instead of being silently re-judged by the shared route's model.
    """
    if task.path is None or task.workload != "native":
        return None
    document = tomllib.loads((Path(task.path) / "task.toml").read_text())
    declared = ((document.get("verifier") or {}).get("env") or {}).get(DECLARED_JUDGE_ENV_VAR)
    if declared is None:
        return None
    if not declared.startswith(DECLARED_JUDGE_PROVIDER_PREFIX):
        raise ValueError(
            f"Task {task.task_id} declares judge {declared!r} outside the "
            f"{DECLARED_JUDGE_PROVIDER_PREFIX} route this lane serves"
        )
    model = declared.removeprefix(DECLARED_JUDGE_PROVIDER_PREFIX)
    if model not in DECLARED_JUDGE_PINS:
        raise ValueError(f"Task {task.task_id} declares judge {model!r} this lane has not qualified")
    return model


def attempt_judge_model(task: PreparedTask) -> str | None:
    """The judge model one attempt's grading uses: the prepared judge or the bundle's declared one."""
    return task.judge_model or task_declared_judge_model(task)
