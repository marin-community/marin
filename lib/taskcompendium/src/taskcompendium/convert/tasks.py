# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct conversation and shell workspace tasks with packaged graders."""

from collections.abc import Mapping, Sequence
from typing import Any

from verifyit.spec import Spec

from taskcompendium.convert.answers import evidence_resource
from taskcompendium.grader import GraderPackage, verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationEvent,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import RawRow

SHELL_CAPABILITIES = ("shell", "filesystem")


def shell_environment(environment: EnvironmentRequirements) -> EnvironmentRequirements:
    """Add the shell contract while retaining the required software and workspace."""
    return environment.model_copy(
        update={
            "capabilities": tuple(dict.fromkeys((*environment.capabilities, *SHELL_CAPABILITIES))),
        }
    )


def workspace_task(
    row: RawRow,
    *,
    instruction: str,
    spec: Spec,
    environment: EnvironmentRequirements,
    grader_environment: EnvironmentRequirements,
    output_paths: tuple[str, ...],
    verifier: tuple[TaskResource, ...] = (),
    worker: tuple[TaskResource, ...] = (),
    oracle: tuple[TaskResource, ...] = (),
    tags: tuple[str, ...] = (),
) -> TaskSpec:
    """A shell task in the declared environment whose ``spec`` grades the declared files and directories.

    The grader runs in a fresh machine of ``grader_environment``. ``verifier`` files are installed
    under ``/tests`` for the grader only; ``worker`` files are mounted for the agent and ``oracle``
    files for the grader controls.
    """
    package = verifyit_package(spec, verifier, environment=grader_environment)
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=shell_environment(environment),
        resources=ResourceGroups(worker=worker, oracle=oracle, verifier=package.resources),
        output_paths=output_paths,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=package.grader,
        tags=tags,
    )


def conversation_task(
    row: RawRow,
    *,
    events: Sequence[ConversationEvent],
    package: GraderPackage,
    evidence: Mapping[str, Any] | None = None,
) -> TaskSpec:
    """A task whose plain-text reply to ``events`` the packaged grader scores.

    ``evidence`` is source material kept beside the grader for review, never shown to the solver.
    """
    verifier = (*package.resources, *((evidence_resource(evidence),) if evidence else ()))
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=tuple(events)),
        environment_requirements=EnvironmentRequirements(),
        resources=ResourceGroups(verifier=verifier),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
    )
