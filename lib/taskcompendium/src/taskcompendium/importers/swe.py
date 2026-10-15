# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-Bench and SWE-Gym tasks with patch grading in a fresh machine."""

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    ConversationInput,
    EnvironmentRequirements,
    ExitCodeReward,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.runtime.resources import inline_resource

PATCH_PATH = "/tmp/taskcompendium/model.patch"
GRADER_PATH = "/tests/evaluate.sh"
BASE_REF = "refs/taskcompendium/base"


class SWEInstance(BaseModel):
    """Source fields that define the problem and its evaluation script."""

    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    eval_script: str = Field(min_length=1)


def swe_task(instance: SWEInstance, *, source: Source, environment: EnvironmentRequirements) -> TaskSpec:
    """Grade the submitted Git patch in a fresh machine from the same image."""
    if environment.docker_image is None or environment.tool_providers:
        raise ValueError("SWE tasks require a prebuilt, digest-pinned shell image")
    task_environment = environment.model_copy(
        update={
            "capabilities": ("shell", "filesystem"),
            "setup_commands": (*environment.setup_commands, f"git update-ref {BASE_REF} HEAD"),
        }
    )
    grader = ScriptGrader(
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
        cwd=environment.working_directory or "/",
        environment=environment,
        answer_path=None,
        reward=ExitCodeReward(),
    )
    return TaskSpec(
        id=f"{source.dataset}:{instance.instance_id}",
        context=ConversationInput(events=(TextMessage(role="user", content=instance.problem_statement),)),
        environment_requirements=task_environment,
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=grader,
        resources=ResourceGroups(verifier=(inline_resource("evaluate.sh", instance.eval_script.encode()),)),
        source=source,
    )
