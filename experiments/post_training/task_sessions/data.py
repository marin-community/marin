# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare procedural answer tasks for SkyRL session graders."""

from marin.execution.fingerprint import canonical_json
from taskcompendium.chat import chat_input
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    SessionGrader,
    Source,
    TaskSpec,
)
from taskcompendium.runtime.resources import inline_resource


def answer_task_record(
    prompt: list[dict[str, str]], *, dataset: str, split: str, index: int, extras: dict, config: dict
) -> dict[str, str]:
    """Return a bare TaskSpec row with private session-grader parameters."""
    parameters = {"extras": extras, "config": config}
    row = f"{split}:{index}"
    task = TaskSpec(
        id=f"{dataset}:{row}",
        context=chat_input(prompt),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=SessionGrader(),
        resources=ResourceGroups(verifier=(inline_resource("config.json", canonical_json(parameters).encode()),)),
        source=Source(
            dataset=dataset,
            revision=canonical_sha256({"prompt": prompt, "parameters": parameters}),
            row=row,
            importer_revision="procedural-session-v1",
        ),
    )
    return {"task_spec": task.model_dump_json()}
