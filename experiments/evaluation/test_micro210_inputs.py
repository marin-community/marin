# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The new judge changes task-owned transport while preserving task semantics."""

import ast
import json
import tomllib

import httpx
import pytest
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
from experiments.evaluation.micro210_inputs import (
    choice_seed,
    reject_fixed_choice_positions,
    task_with_judge,
    truthfulqa_choice_task,
)


@pytest.fixture
def task():
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
    return task


def test_task_rebinding_preserves_solver_prompt_and_reference(task):
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
        "top_p": 0.95,
        "reasoning_effort": "low",
        "response_format": {"type": "regex", "pattern": "(correct|incorrect|not_attempted)"},
    }


def test_bixbench_rebinding_applies_model_and_sampling_to_http_request(task, monkeypatch):
    source = b"""import os
from openai import OpenAI

def complete():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": "fixture"}],
        max_tokens=1024,
    )
"""
    task = task.model_copy(
        update={
            "id": "bixbench/fixture",
            "resources": ResourceGroups(
                verifier=(
                    inline_resource("task.toml", b'[verifier.env]\nMODEL_NAME="old/judge"\n'),
                    inline_resource("tests/llm_judge.py", source),
                )
            ),
        }
    )
    rebound = task_with_judge(task, MICRO210_JUDGE)
    files = {resource.path: resource_bytes(resource) for resource in rebound.resources.verifier}
    environment = tomllib.loads(files["task.toml"].decode())["verifier"]["env"]
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    for key in ("OPENAI_BASE_URL", "MODEL_NAME"):
        monkeypatch.setenv(key, environment[key])
    requests = []

    def send(_client, request, **kwargs):
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "zai-org/GLM-5.3-Flash",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "fixture-verdict"}, "finish_reason": "stop"}
                ],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    namespace = {}
    exec(files["tests/llm_judge.py"], namespace)
    assert namespace["complete"]().choices[0].message.content == "fixture-verdict"
    request = requests[0]
    assert str(request.url) == "https://api.together.ai/v1/chat/completions"
    body = json.loads(request.content)
    assert {key: body[key] for key in ("model", "temperature", "top_p", "reasoning_effort", "max_tokens")} == {
        "model": "zai-org/GLM-5.3-Flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "reasoning_effort": "low",
        "max_tokens": 1024,
    }
    assert request.extensions["timeout"]["read"] == 600.0


def truthfulqa_fixture(task, task_id: str):
    choices = ["a true answer", "a false answer", "another true answer", "another false answer"]
    prompt = "Question?\n\n" + "\n".join(f"{chr(65 + i)}. {answer}" for i, answer in enumerate(choices))
    prompt += "\n\nChoose one truthful answer. End with its letter in \\boxed{...}."
    config = {
        "spec": {
            "doc": {"mc2_targets": {"choices": choices, "labels": [1, 0, 1, 0]}},
            "grader": {"name": "lm_eval:truthfulqa_mc2"},
            "requests": [{"continuation": answer} for answer in choices],
            "target": 0,
        }
    }
    context = task.context.model_copy(update={"events": (TextMessage(role="user", content=prompt),)})
    resources = task.resources.model_copy(
        update={"verifier": (inline_resource("config.json", json.dumps(config).encode()),)}
    )
    return task.model_copy(update={"id": task_id, "context": context, "resources": resources})


def test_truthfulqa_shuffle_preserves_accepted_answer_text_and_records_seed(task):
    source = truthfulqa_fixture(task, "truthfulqa/example")
    shuffled = truthfulqa_choice_task(source)
    assert shuffled == truthfulqa_choice_task(source)
    assert shuffled.labels["micro210_choice_seed"] == str(choice_seed(source.id))
    order = [int(index) for index in shuffled.labels["micro210_choice_order"].split(",")]
    assert sorted(order) == list(range(4))
    config = json.loads(resource_bytes(shuffled.resources.verifier[0]))["spec"]
    expected = {"a true answer", "another true answer"}
    assert {
        choice
        for choice, label in zip(
            config["doc"]["mc2_targets"]["choices"], config["doc"]["mc2_targets"]["labels"], strict=True
        )
        if label
    } == expected
    assert config["target"] == order.index(0)
    assert [request["continuation"] for request in config["requests"]] == [
        config["doc"]["mc2_targets"]["choices"][i] for i in range(4)
    ]
    prompt = shuffled.context.events[-1].content
    assert prompt is not None
    for index, choice in enumerate(config["doc"]["mc2_targets"]["choices"]):
        assert f"{chr(65 + index)}. {choice}" in prompt


def test_micro210_rejects_benchmark_with_one_correct_letter_in_every_row(task):
    rows = []
    for index in range(3):
        fixture = truthfulqa_fixture(task, f"truthfulqa/example-{index}")
        rows.append({"task_id": fixture.id, "benchmark": "truthfulqa", "task_spec": fixture.model_dump_json()})
    with pytest.raises(ValueError, match="truthfulqa accepts fixed choice position"):
        reject_fixed_choice_positions(rows)
