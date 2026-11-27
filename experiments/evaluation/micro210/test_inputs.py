# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The new judge changes task-owned transport while preserving task semantics."""

import ast
import json
import tomllib

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import rollout_spec
from taskcompendium.models import (
    AgentMachine,
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.evaluation.judge_config import MICRO210_JUDGE
from experiments.evaluation.micro210.inputs import JUDGE_COMMAND, native_inputs, task_with_judge
from tests.rl.micro210_fixture import policy_fixture_reference


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
                    "config.json",
                    json.dumps(
                        {
                            "judge_model": "old/judge",
                            "spec": {
                                "target": "4",
                                "grader": {"name": "math_equivalence", "reference": {"judge": True, "answers": ["4"]}},
                            },
                        }
                    ).encode(),
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
    assert config["spec"]["target"] == "4"
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


def script_row_task(task_id: str, grader_name: str, reference: dict) -> TaskSpec:
    """One single-turn task whose frozen config names the registry grader that grades it."""
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content="What is 2+2?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("python3", "grade.py"),
            placement=AgentMachine(),
        ),
        source=Source(dataset="fixture", revision="pinned", row="0", importer_revision="fixture-v1"),
        resources=ResourceGroups(
            verifier=(
                inline_resource("grade.py", b"print(1)\n"),
                inline_resource(
                    "config.json",
                    json.dumps(
                        {"spec": {"target": "4", "grader": {"name": grader_name, "reference": reference}}}
                    ).encode(),
                ),
            )
        ),
    )


def test_native_inputs_bind_the_judge_only_to_judged_rows(tmp_path):
    judged = script_row_task("fixture/olympiad", "math_equivalence", {"judge": True, "answers": ["4"]})
    unjudged = script_row_task("fixture/humaneval", "program_suite", {"language": "python"})
    rollouts = [rollout_spec(unjudged), rollout_spec(judged)]
    original = pa.table(
        {
            "task_id": [task.id for task in (unjudged, judged)],
            "repetition_index": [0, 0],
            "task_spec": [unjudged.model_dump_json(), judged.model_dump_json()],
            "task_rollout_spec": [rollout.model_dump_json() for rollout in rollouts],
        }
    )
    source, destination = tmp_path / "frozen.parquet", tmp_path / "native.parquet"
    pq.write_table(original, source)
    policy = json.loads(StoragePath(policy_fixture_reference().uri).read_text())
    native_inputs(str(source), str(destination), policy, {"fixture:sealed": "sha256:" + "a" * 64})
    converted = pq.read_table(destination)
    graders = []
    for row in converted.to_pylist():
        native = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        graders.append((row["task_id"], native.task.grader))
    unjudged_grader = next(grader for task_id, grader in graders if task_id == "fixture/humaneval")
    judged_grader = next(grader for task_id, grader in graders if task_id == "fixture/olympiad")
    assert unjudged_grader.judge is None
    assert unjudged_grader.argv == ("python3", "grade.py")
    assert judged_grader.judge.model == MICRO210_JUDGE.model
    assert judged_grader.argv[:4] == JUDGE_COMMAND
    assert judged_grader.argv[4:] == ("python3", "grade.py")


def test_native_table_preserves_repeated_membership_and_package_provisioning(tmp_path, task):
    package = {
        "instruction.md": b"What is 2+2?",
        "task.toml": (
            b"""[agent]
timeout_sec = 900
[verifier]
timeout_sec = 600
[verifier.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
OPENAI_BASE_URL = "https://api.together.ai/v1"
MODEL_NAME = "zai-org/GLM-5.3-Flash"
[environment]
cpus = 2
memory_mb = 8192
storage_mb = 10240
allow_internet = true
"""
        ),
        "environment/Dockerfile": b"FROM fixture:sealed\nWORKDIR /workspace\n",
        "tests/test.sh": b"echo 0.5 > /logs/verifier/reward.txt\n",
    }
    frozen_task = task.model_copy(
        update={
            "id": "bixbench-pi/fixture",
            "answer_type": AnswerType.WORKSPACE_STATE,
            "grader": NoGrader(reason="Harbor grades the completed trial"),
            "resources": ResourceGroups(verifier=tuple(inline_resource(path, data) for path, data in package.items())),
        }
    )
    frozen_rollout = rollout_spec(frozen_task, harness="harbor_trial")
    original = pa.table(
        {
            "task_id": [frozen_task.id, frozen_task.id],
            "repetition_index": [0, 1],
            "task_spec": [frozen_task.model_dump_json()] * 2,
            "task_rollout_spec": [frozen_rollout.model_dump_json()] * 2,
        }
    )
    source, destination = tmp_path / "frozen.parquet", tmp_path / "native.parquet"
    pq.write_table(original, source)
    policy = json.loads(StoragePath(policy_fixture_reference().uri).read_text())
    native_inputs(str(source), str(destination), policy, {"fixture:sealed": "sha256:" + "a" * 64})
    converted = pq.read_table(destination)
    assert converted.select(["task_id", "repetition_index"]) == original.select(["task_id", "repetition_index"])
    for row in converted.to_pylist():
        native = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        assert native.task.id == frozen_task.id
        assert native.task.context == frozen_task.context
        assert native.task.source == frozen_task.source
        assert (native.task_machine.cpus, native.task_machine.memory_mb, native.task_machine.storage_mb) == (
            2,
            8192,
            10240,
        )
        assert native.task_machine.network == "allow"
        assert native.harness == "pi"
        assert native.task.verifies_stopped_workspace is True
        assert native.limits.total_turn_timeout == 1800
        assert native.limits.verifier_timeout == native.task.grader.timeout == 600
        assert {r.path: resource_bytes(r) for r in native.task.resources.verifier} == {
            "test.sh": package["tests/test.sh"]
        }
        assert native.task.grader.judge.model == MICRO210_JUDGE.model
        assert "${OPENAI_API_KEY}" not in native.task.model_dump_json()
