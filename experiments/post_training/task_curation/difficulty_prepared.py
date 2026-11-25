# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared source contracts shared with Harbor's isolated trial process."""

import base64
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import DifficultySample
from pydantic import BaseModel, ConfigDict, Field

from experiments.post_training.task_curation.difficulty.resource_deviations import ResourceDeviation
from experiments.post_training.task_curation.difficulty_config import (
    DECLARED_JUDGE_ENV_VAR,
    DECLARED_JUDGE_PINS,
    DECLARED_JUDGE_PROVIDER_PREFIX,
)

GRADER_PRIVILEGE_DROP_UID = "user=65534"

ENV_TEMPLATE_DEFAULT = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*):-(?P<default>[^}]*)\}")
"""Harbor's `${VAR:-default}` template with a fallback default."""


def resolve_judge_template(declared: str) -> str | None:
    """The model a frozen verifier env value names, with Harbor's `${VAR:-default}` resolved.

    A template resolves to its default — the guest may still override the variable
    at serving time, which the judge-route repair relies on. A template without a
    default resolves to nothing this contract can name; plain values return
    unchanged.
    """
    resolved = ENV_TEMPLATE_DEFAULT.sub(lambda match: match.group("default"), declared.strip())
    if "${" in resolved:
        return None
    return resolved or None


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
    resource_deviation: ResourceDeviation | None = Field(default=None, exclude_if=lambda value: value is None)


class PreparedSource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    sample: DifficultySample
    sample_receipt: ContentEvidenceReference
    tasks: list[PreparedTask]


def bundle_declared_judge(task: PreparedTask) -> str | None:
    """The judge model a frozen native bundle's verifier env declares, with templates resolved.

    Returns None when the bundle declares no judge or its template carries no
    default; the runtime's serving wiring then has no statically nameable judge.
    """
    if task.path is None or task.workload != "native":
        return None
    document = tomllib.loads((Path(task.path) / "task.toml").read_text())
    declared = ((document.get("verifier") or {}).get("env") or {}).get(DECLARED_JUDGE_ENV_VAR)
    if declared is None:
        return None
    return resolve_judge_template(declared)


def bundle_judge_template_var(task: PreparedTask) -> str | None:
    """The bundle's own judge template variable, when its declaration is one.

    The judge-route repair pins the re-declared judge through this variable, so
    the bundle's own `${VAR:-default}` declaration names the served route.
    """
    if task.path is None or task.workload != "native":
        return None
    document = tomllib.loads((Path(task.path) / "task.toml").read_text())
    declared = ((document.get("verifier") or {}).get("env") or {}).get(DECLARED_JUDGE_ENV_VAR)
    match = ENV_TEMPLATE_DEFAULT.fullmatch(declared.strip()) if declared else None
    return match.group("name") if match else None


def task_declared_judge_model(task: PreparedTask) -> str | None:
    """The grader model a frozen native bundle declares for its own verifier, if any.

    Parses the frozen task.toml directly with stdlib tomllib: this contract
    helper must stay importable where Harbor (and marin) are not installed - the
    controller, the runtime, and Harbor's isolated trial process. Env-template
    declarations resolve to their default first, so the fail-closed errors name
    the model the verifier would actually call. Only qualified declared judges
    are served, so an unqualified declaration fails closed instead of being
    silently re-judged by the shared route's model.
    """
    resolved = bundle_declared_judge(task)
    if resolved is None:
        return None
    if not resolved.startswith(DECLARED_JUDGE_PROVIDER_PREFIX):
        raise ValueError(
            f"Task {task.task_id} declares judge {resolved!r} outside the "
            f"{DECLARED_JUDGE_PROVIDER_PREFIX} route this lane serves"
        )
    model = resolved.removeprefix(DECLARED_JUDGE_PROVIDER_PREFIX)
    if model not in DECLARED_JUDGE_PINS:
        raise ValueError(f"Task {task.task_id} declares judge {model!r} this lane has not qualified")
    return model


def attempt_judge_model(task: PreparedTask) -> str | None:
    """The judge model one attempt's grading uses: the prepared judge or the bundle's declared one.

    A task whose preparation typed its judge carries the prepared answer - either
    the source's re-declared judge-route repair or the typed unqualified gap - so
    the bundle's declaration is never re-parsed against a decided contract.
    """
    if task.reason is not None:
        return task.judge_model
    return task.judge_model or task_declared_judge_model(task)
