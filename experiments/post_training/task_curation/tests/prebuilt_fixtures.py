# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published cohort fixtures shared by importer and oracle receipt tests."""

import json

from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)

IMAGE = "ubuntu:24.04@sha256:" + "a" * 64


def prebuilt_task(task_id: str, *, image: str | None = IMAGE, capabilities: tuple[str, ...] = ()) -> dict:
    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image=image,
        working_directory="/app",
        setup_commands=("true",),
        capabilities=capabilities,
    )
    grader = ScriptGrader(
        argv=("true",),
        environment=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE),
        reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    spec = TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="Do the task."),)),
        environment_requirements=requirements,
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=grader,
        source=Source(dataset="d", revision="r", row=task_id, importer_revision="1"),
        resources=ResourceGroups(),
    )
    return {
        "path": task_id,
        "lowered_task_spec": json.dumps({"task": spec.model_dump(mode="json"), "runtime": {"task_machine": {}}}),
    }
