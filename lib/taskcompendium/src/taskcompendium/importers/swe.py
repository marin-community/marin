# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-Bench and SWE-Gym tasks with patch grading in a fresh machine."""

from pydantic import BaseModel, ConfigDict, Field

from taskcompendium.environment import (
    ArtifactKind,
    EnvironmentCommand,
    EnvironmentFile,
    EnvironmentKind,
    EnvironmentSpec,
    ExitCodeReward,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.models import (
    FILESYSTEM_CAPABILITY,
    SHELL_CAPABILITY,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)

PATCH_PATH = "/tmp/taskcompendium/model.patch"
GRADER_PATH = "/tmp/taskcompendium/evaluate.sh"
BASE_REF = "refs/taskcompendium/base"
BASE_REF_TIMEOUT = 30


class SWEInstance(BaseModel):
    """Source fields that define the problem, task image, and private grader."""

    model_config = ConfigDict(frozen=True)

    instance_id: str = Field(min_length=1)
    problem_statement: str = Field(min_length=1)
    eval_script: str = Field(min_length=1)
    image_name: str | None = None


def swe_image(instance: SWEInstance, dataset: str) -> str:
    """Resolve the image name stored in the source row or its dataset convention."""
    if instance.image_name:
        return instance.image_name
    instance_id = instance.instance_id
    if "swe-gym" in dataset.lower():
        return f"docker.io/xingyaoww/sweb.eval.x86_64.{instance_id.replace('__', '_s_')}:latest".lower()
    if "swe-bench" in dataset.lower():
        return f"docker.io/swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()
    raise ValueError(f"SWE tasks require image_name for dataset {dataset!r}")


def swe_task(
    instance: SWEInstance, *, source: Source, environment: EnvironmentSpec, verifier_timeout: float
) -> TaskSpec:
    """Keep the evaluation script private and grade only the submitted Git patch."""
    if environment.kind == EnvironmentKind.NULL or environment.interaction is not None:
        raise ValueError("SWE tasks require an executable shell environment")
    task_environment = environment.model_copy(
        update={
            "setup": (
                *environment.setup,
                EnvironmentCommand(argv=("git", "update-ref", BASE_REF, "HEAD"), timeout=BASE_REF_TIMEOUT),
            )
        }
    )
    verifier = ShellVerifierSpec(
        collect=(
            EnvironmentCommand(
                argv=(
                    "sh",
                    "-c",
                    f'mkdir -p "$(dirname "$1")" && git add -A && git diff --cached --binary {BASE_REF} > "$1"',
                    "collect-patch",
                    PATCH_PATH,
                ),
                timeout=verifier_timeout,
            ),
        ),
        artifacts=(VerifierArtifact(source=PATCH_PATH, target=PATCH_PATH, kind=ArtifactKind.FILE),),
        argv=("sh", "-c", 'git apply --binary "$1" && bash "$2"', "evaluate-patch", PATCH_PATH, GRADER_PATH),
        timeout=verifier_timeout,
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
        environment_requirements=EnvironmentRequirements(capabilities=(SHELL_CAPABILITY, FILESYSTEM_CAPABILITY)),
        answer_type=AnswerType.STATE,
        environment=task_environment,
        verifier=VerifierSpec(
            kind=VerifierKind.SHELL,
            parameters_json=verifier.model_dump_json(),
            environment=environment,
            files=(EnvironmentFile(path=GRADER_PATH, content=instance.eval_script.encode()),),
        ),
        source=source,
    )
