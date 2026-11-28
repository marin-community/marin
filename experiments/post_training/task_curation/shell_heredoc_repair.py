# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair Docker build contexts whose shell heredoc introducers carry whitespace before the delimiter.

The source packages of RST rts-1k and Lego-X Terminal-Lego-15k write heredocs as
``RUN cat << 'EOF' > file`` — whitespace between the ``<<`` operator and its quoted
delimiter. POSIX shells accept that text, but the BuildKit Dockerfile lexer (the
built-in frontend and docker/dockerfile:1.x alike) only recognizes a heredoc whose
delimiter directly follows the operator, so the build fails to parse: the first body
line surfaces as "unknown instruction". The legacy Kaniko build silently dropped the
heredoc body instead, so every previously built image of these contexts carries empty
files and must not be bound. The repair is mechanical and semantics-preserving: it
removes only the whitespace between the operator and a quoted delimiter inside the
context's Dockerfile. Unquoted delimiters, ``<<-`` forms and already-tight
introducers are left untouched, and every non-Dockerfile context file passes through
unchanged. The per-file change counts are recorded in the task's lineage resource.
"""

import json
import re
from pathlib import PurePosixPath

from pydantic import BaseModel
from taskcompendium.identity import task_spec_sha256
from taskcompendium.models import DockerBuildContext, ScriptGrader, TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes

REPAIR_VERSION = "shell-heredoc-repair-v1"
LINEAGE_PATH = "taskcompendium/shell-heredoc-repair.json"
DOCKERFILE_NAME = "Dockerfile"
# Whitespace between the operator and a quoted delimiter: `<< 'EOF'` -> `<<'EOF'`.
# The dash form (`<<- 'X'`), unquoted delimiters and tight introducers do not match.
HEREDOC_INTRODUCER = re.compile(r"<<[ \t]+(?=['\"])")

LINEAGE_REASON = (
    "BuildKit's Dockerfile lexer rejects a heredoc introducer written with whitespace before its "
    "quoted delimiter, and the legacy Kaniko build silently dropped the heredoc body; the "
    "introducer is normalized in place"
)
LINEAGE_CHANGE = "<< 'DELIM' -> <<'DELIM': whitespace between the operator and a quoted delimiter removed"


class HeredocFileChange(BaseModel):
    """One repaired Dockerfile inside a build context."""

    file: str
    introducers: int


class HeredocTaskLineage(BaseModel):
    """The recorded repair of one task's build contexts."""

    version: str
    reason: str
    change: str
    files: list[HeredocFileChange]
    input_task_sha256: str


def repair_dockerfile(text: str) -> tuple[str, int]:
    """Collapse whitespace between a heredoc operator and its quoted delimiter.

    Returns the repaired text and the number of normalized introducers.
    """
    return HEREDOC_INTRODUCER.subn("<<", text)


def _repaired_build(build: DockerBuildContext, role: str, changes: list[HeredocFileChange]) -> DockerBuildContext:
    role_changes: list[HeredocFileChange] = []
    files = []
    for file in build.files:
        if PurePosixPath(file.path).name != DOCKERFILE_NAME:
            files.append(file)
            continue
        text = resource_bytes(file).decode()
        fixed, introducers = repair_dockerfile(text)
        if not introducers:
            files.append(file)
            continue
        role_changes.append(HeredocFileChange(file=f"{role}:{file.path}", introducers=introducers))
        files.append(
            inline_resource(file.path, fixed.encode()).model_copy(update={"mode": file.mode, "mtime_ns": file.mtime_ns})
        )
    if not role_changes:
        return build
    changes.extend(role_changes)
    return build.model_copy(update={"files": tuple(files)})


def repair_task(task: TaskSpec) -> TaskSpec:
    """Normalize the heredoc introducers of every Dockerfile build context; record the lineage.

    The task passes through unchanged when no Dockerfile carries the defect. The actor and
    grader environments are repaired independently: they usually share one context, and the
    recorded change names the owning role.
    """
    changes: list[HeredocFileChange] = []
    environment = task.environment_requirements
    updated_actor = (
        _repaired_build(environment.docker_build, "actor", changes) if environment.docker_build is not None else None
    )
    updated_grader = None
    if isinstance(task.grader, ScriptGrader) and task.grader.environment is not None:
        build = task.grader.environment.docker_build
        if build is not None:
            repaired_build = _repaired_build(build, "grader", changes)
            updated_grader = repaired_build if repaired_build is not build else None
    if not changes:
        return task
    lineage = HeredocTaskLineage(
        version=REPAIR_VERSION,
        reason=LINEAGE_REASON,
        change=LINEAGE_CHANGE,
        files=changes,
        input_task_sha256=task_spec_sha256(task),
    )
    actor = environment.model_copy(update={"docker_build": updated_actor})
    grader = (
        task.grader.model_copy(
            update={"environment": task.grader.environment.model_copy(update={"docker_build": updated_grader})}
        )
        if updated_grader is not None
        else task.grader
    )
    resources = task.resources
    repaired = task.model_copy(
        update={
            "environment_requirements": actor,
            "grader": grader,
            "resources": resources.model_copy(
                update={
                    "oracle": (
                        *(resource for resource in resources.oracle if resource.path != LINEAGE_PATH),
                        inline_resource(LINEAGE_PATH, lineage.model_dump_json().encode()),
                    ),
                }
            ),
            "labels": {**task.labels, "heredoc_repair_version": REPAIR_VERSION},
        }
    )
    return TaskSpec.model_validate_json(repaired.model_dump_json())


def lineage_of(task: TaskSpec) -> HeredocTaskLineage | None:
    """Read a task's recorded heredoc-repair lineage, or None when the task was not repaired."""
    for resource in task.resources.oracle:
        if resource.path == LINEAGE_PATH:
            return HeredocTaskLineage.model_validate_json(resource_bytes(resource))
    return None


def lineage_json(lineage: HeredocTaskLineage) -> str:
    """The release-level lineage document entry for one repaired task."""
    return json.dumps(lineage.model_dump(), indent=1)
