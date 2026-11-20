# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The new judge changes task-owned transport while preserving task semantics."""

import ast
import json

from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210_inputs import task_with_judge


def test_task_rebinding_preserves_solver_prompt_and_reference():
    source = b"from eval.execution.judge import JudgeBackend\nbackend = JudgeBackend(judge, session)\n"
    task = TaskSpec(
        id="fixture/equivalence",
        context=ConversationInput(events=(TextMessage(role="user", content="What is 2+2?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="fixture"),
        source=Source(dataset="fixture", revision="pinned", row="0", importer_revision="fixture-v1"),
        resources=ResourceGroups(
            verifier=(
                inline_resource("grade.py", source),
                inline_resource(
                    "config.json", json.dumps({"judge_model": "old/judge", "spec": {"target": "4"}}).encode()
                ),
            )
        ),
    )
    rebound = task_with_judge(task, MICRO210_JUDGE)
    assert rebound.model_copy(update={"resources": task.resources}) == task
    files = {resource.path: resource_bytes(resource) for resource in rebound.resources.verifier}
    config = json.loads(files["config.json"])
    assert config["judge_model"] == config["judge"]["model"] == "zai-org/GLM-5.3-Flash"
    assert config["spec"] == {"target": "4"}
    for name in ("grade.py", "micro210_together_judge.py", "micro210_judge_config.py"):
        ast.parse(files[name])
    parameters = MICRO210_JUDGE.completion_parameters(
        {"max_tokens": 512, "structured_outputs": {"choice": ["correct", "incorrect", "not_attempted"]}}
    )
    assert parameters == {
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_effort": "low",
        "response_format": {"type": "regex", "pattern": "(correct|incorrect|not_attempted)"},
    }
