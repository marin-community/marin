# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute pinned original Ultra/LCB behavior fixtures, not source-row goldens."""

import asyncio
from dataclasses import asdict
from functools import partial

import pytest
from shellbox.machine import DockerImage, MachineSpec
from taskcompendium.datasets.nemotron_ultra import normalization
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    PlainText,
    ResourceGroups,
    Source,
    StateSubmission,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import ImportRejection, NormalizedTask, RawRow
from taskcompendium.runtime.grading import grade_in_sandbox

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import code_binding
from lib.taskcompendium.tests.test_runtime import LocalGradingMachines


def run_original(tmp_path, tests, answer, message=None):
    pytest.importorskip("skyrl_gym.envs.nemotron_ultra.code_gen", reason="Requires installed original scorer assets")
    image = "fixture@sha256:" + "a" * 64
    package = code_binding.original_package({"contract": {"verifier_metadata": {"unit_tests": tests}}}, image)
    task = TaskSpec(
        id="code-fixture",
        context=ConversationInput(events=(TextMessage(role="user", content="Code fixture"),)),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=package.grader,
        resources=ResourceGroups(verifier=package.resources),
    )
    attempt = GradingAttempt(
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer))),
        state=None if message is None else StateSubmission({"assistant_message": message}),
    )
    result = asyncio.run(
        grade_in_sandbox(task, attempt, LocalGradingMachines(tmp_path), MachineSpec(DockerImage(image)))
    )
    return asdict(result)


@pytest.mark.parametrize(
    "answer,reward",
    [
        ("```python\nprint(3)\n```", 1.0),
        ("print(3)", 0.0),
        ("```python\nprint(0)\n```", 0.0),
        ("```python\nprint(0)\n``` then ```python\nprint(3)\n```", 1.0),
        ("<think>hidden</think>```python\nprint(3)\n```", 1.0),
        ("<think>unfinished```python\nprint(3)\n```", 0.0),
    ],
)
def test_original_final_answer_and_last_fenced_program(tmp_path, answer, reward):
    result = run_original(tmp_path, {"inputs": [""], "outputs": ["3\n"]}, answer)
    assert (result["status"], result["reward"]) == ("graded", reward)


def test_original_binary_reward_and_stop_on_failure(tmp_path):
    result = run_original(
        tmp_path, {"inputs": ["1\n", "2\n"], "outputs": ["999\n", "2\n"]}, "```python\nprint(input())\n```"
    )
    assert result["reward"] == 0.0
    assert result["detail"]["executed_tests"] == 1
    assert result["detail"]["total_tests"] == 2


def test_original_function_call_and_tuple_result(tmp_path):
    tests = {"fn_name": "solve", "inputs": [[1, 2]], "outputs": [[1, 2]]}
    result = run_original(tmp_path, tests, "```python\nclass Solution:\n def solve(self,a,b): return (a,b)\n```")
    assert (result["status"], result["reward"]) == ("graded", 1.0)


def test_original_d8_stdin_buffer_limitation_is_not_replaced_by_544_fix(tmp_path):
    result = run_original(
        tmp_path,
        {"inputs": ["3\n"], "outputs": ["3\n"]},
        "```python\nimport sys\nprint(sys.stdin.buffer.read().decode().strip())\n```",
    )
    assert (result["status"], result["reward"]) == ("graded", 0.0)


def test_captured_provider_reasoning_preserves_original_malformed_think_penalty(tmp_path):
    result = run_original(
        tmp_path,
        {"inputs": [""], "outputs": ["3\n"]},
        "```python\nprint(3)\n```",
        {"role": "assistant", "content": "stale", "reasoning_content": "<think><think>hidden</think></think>"},
    )
    assert (result["status"], result["reward"]) == ("graded", 0.0)
    assert result["detail"]["result"] == "pass"
    assert result["detail"]["reasoning_format_violation_rate"] == 1.0


def test_malformed_source_tests_write_no_reward(tmp_path):
    result = run_original(tmp_path, {"inputs": [], "outputs": []}, "```python\nprint(3)\n```")
    assert (result["status"], result["failure"]) == ("infra_error", "missing_reward")


def test_original_per_test_timeout_returns_failed_rollout(tmp_path):
    result = run_original(tmp_path, {"inputs": [""], "outputs": ["3\n"]}, "```python\nwhile True: pass\n```")
    assert (result["status"], result["reward"]) == ("graded", 0.0)
    assert result["detail"]["result"] == "failed_tests"


def test_native_code_binding_keeps_public_context_and_private_tests():
    row = RawRow(
        "code",
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        {
            "dataset": "fixture",
            "agent_ref": {"name": "code_gen_simple_agent"},
            "responses_create_params": {"input": [{"role": "user", "content": "Print3."}]},
            "verifier_metadata": {"unit_tests": {"inputs": [""], "outputs": ["3\n"]}},
        },
    )
    original = normalization.normalize(row, selector="fixture", family="competitive-programming")
    normalized = code_binding.normalize_isolated(
        row,
        image="fixture@sha256:" + "a" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="competitive-programming"),
    )
    assert isinstance(normalized, NormalizedTask)
    assert isinstance(original, NormalizedTask)
    assert normalized.task.context == original.task.context
    assert normalized.task.resources.worker == original.task.resources.worker
    assert {r.path for r in normalized.task.resources.verifier} == {
        "source_callable.py",
        "invocation.json",
        "config.json",
    }
    row.data["verifier_metadata"]["unit_tests"] = {}
    invalid = code_binding.normalize_isolated(
        row,
        image="fixture@sha256:" + "a" * 64,
        normalize_task=partial(normalization.normalize, selector="fixture", family="competitive-programming"),
    )
    assert isinstance(invalid, ImportRejection)
    assert invalid.reason == "invalid_original_code_tests"
