# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-Bench and SWE-Gym tasks with patch grading in a fresh machine."""

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
    VerifierSpec,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.shell_verifier import (
    ArtifactKind,
    ExitCodeReward,
    ShellVerifierSpec,
    VerifierArtifact,
    VerifierCommand,
)

PATCH_PATH = "/tmp/taskcompendium/model.patch"
GRADER_PATH = "/tests/evaluate.sh"
BASE_REF = "refs/taskcompendium/base"


class SWEInstance(BaseModel):
    """Source fields that define the problem and private grader."""

    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    eval_script: str = Field(min_length=1)


def swe_task(instance: SWEInstance, *, source: Source, environment: EnvironmentRequirements) -> TaskSpec:
    """Keep evaluation private and grade the submitted Git patch in a fresh image."""
    if environment.docker_image is None or environment.tool_providers:
        raise ValueError("SWE tasks require a prebuilt, digest-pinned shell image")
    task_environment = environment.model_copy(
        update={
            "capabilities": ("shell", "filesystem"),
            "setup_commands": (*environment.setup_commands, f"git update-ref {BASE_REF} HEAD"),
        }
    )
    verifier = ShellVerifierSpec(
        collect=(
            VerifierCommand(
                argv=(
                    "sh",
                    "-c",
                    f'mkdir -p "$(dirname "$1")" && git add -A && git diff --cached --binary {BASE_REF} > "$1"',
                    "collect-patch",
                    PATCH_PATH,
                )
            ),
        ),
        artifacts=(VerifierArtifact(source=PATCH_PATH, target=PATCH_PATH, kind=ArtifactKind.FILE),),
        argv=("sh", "-c", 'git apply --binary "$1" && bash "$2"', "evaluate-patch", PATCH_PATH, GRADER_PATH),
        reward=ExitCodeReward(),
    )
    return TaskSpec(
        id=f"{source.dataset}:{instance.instance_id}",
        context=ConversationInput(
            events=(
                TextMessage(
                    role="system",
                    content=(
                        "Use the shell tool to inspect and change the repository. "
                        "Submit a final response when the changes are complete."
                    ),
                ),
                TextMessage(role="user", content=instance.problem_statement),
            )
        ),
        environment_requirements=task_environment,
        answer_type=AnswerType.WORKSPACE_STATE,
        verifier=VerifierSpec(
            kind="shell",
            parameters_json=verifier.model_dump_json(),
            environment_requirements=environment,
        ),
        resources=ResourceGroups(verifier=(inline_resource("evaluate.sh", instance.eval_script.encode()),)),
        source=source,
    )
