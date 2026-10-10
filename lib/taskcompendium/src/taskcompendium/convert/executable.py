# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shell workspace tasks graded against captured output files."""

from verifyit.spec import Spec

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ProviderRequirement,
    ResourceGroups,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.shell import BASH, INTERFACE

SHELL_CAPABILITIES = ("shell", "filesystem")


def shell_environment(environment: EnvironmentRequirements) -> EnvironmentRequirements:
    """Add the shell contract while retaining the required software and workspace."""
    return environment.model_copy(
        update={
            "capabilities": tuple(dict.fromkeys((*environment.capabilities, *SHELL_CAPABILITIES))),
            "tool_providers": {"shell": ProviderRequirement(action_interface=INTERFACE, initial_state={})},
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
    """A shell task in the ``environment`` image whose ``spec`` grades the agent's ``output_paths``.

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
        interaction_tools=(BASH,),
        resources=ResourceGroups(worker=worker, oracle=oracle, verifier=package.resources),
        output_paths=output_paths,
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=package.grader,
        tags=tags,
    )
