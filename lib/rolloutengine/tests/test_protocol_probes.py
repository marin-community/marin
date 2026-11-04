# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend
from taskcompendium.examples.protocol_probes import (
    FETCH_REQUIREMENT,
    NONCE_FILE,
    PROBE_URL,
    probe_check,
    probe_nonce,
    probe_tasks,
)
from taskcompendium.fetch import FETCH_ACTION_INTERFACE, HttpPage, NetworkFailure, fetch_requirement
from taskcompendium.models import (
    CHAT_TEMPLATE_KWARGS,
    ENABLE_THINKING,
    CommandSemantics,
    EnvironmentRequirements,
    TaskSpec,
)
from taskcompendium.protocol import PROTOCOL_DIAGNOSTICS_KEY, ProtocolCheck

from rolloutengine.harness.fetch import FetchProvider
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.models import ModelRequest, ModelTurn, Prompt
from rolloutengine.test_support import SHELL, ReplayModel, engine, machine_runtime, rollout_spec

FIXTURE_IMAGE = "fixture@sha256:" + "0" * 64
ENVIRONMENT = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=FIXTURE_IMAGE)
TASKS = {probe_check(task): task for task in probe_tasks(grader_environment=ENVIRONMENT, variants=1)}
FETCH_HARNESS = {SHELL: ShellHarness(command_timeout=None, tools={FETCH_ACTION_INTERFACE: FetchProvider()})}
THINKING_OFF = {CHAT_TEMPLATE_KWARGS: {ENABLE_THINKING: False}}


def tool_call(name: str, arguments: dict, call_id: str = "call-0") -> dict:
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": function}]}


def reply(content: str) -> dict:
    return {"role": "assistant", "content": content}


def protocol(graded) -> dict:
    return graded.grade.diagnostics.get(PROTOCOL_DIAGNOSTICS_KEY, {})


async def test_tool_probe_grades_a_real_shell_rollout():
    task = TASKS[ProtocolCheck.TOOL_EXECUTION]
    model = ReplayModel([tool_call("run_command", {"command": f"cat /{NONCE_FILE}"}), reply(probe_nonce(task.id))])

    graded = await engine(model, {Backend.SHELLSIM: ShellSimMachineFactory()}).run_task(
        rollout_spec(task, machine=machine_runtime(backend=Backend.SHELLSIM))
    )

    assessment = protocol(graded)["tool_execution"]
    assert (assessment["label"], assessment["protocol_compliance"], assessment["answer_correctness"]) == (
        "executed",
        True,
        True,
    )


@pytest.mark.parametrize(
    "page,label,correct",
    [
        pytest.param(None, "retrieved", True, id="retrieved"),
        pytest.param(HttpPage(status=503, body="Unavailable"), "http_error", False, id="http-error"),
        pytest.param(NetworkFailure(error="Connection reset"), "network_error", False, id="network-error"),
    ],
)
async def test_url_probe_serves_declared_pages_and_labels_failed_retrievals(page, label, correct):
    task = TASKS[ProtocolCheck.URL_LOADING]
    if page is not None:
        environment = task.environment_requirements.model_copy(
            update={"tool_providers": {FETCH_REQUIREMENT: fetch_requirement({PROBE_URL: page})}}
        )
        task = task.model_copy(update={"environment_requirements": environment})
    answer = probe_nonce(task.id) if page is None else "The page could not be retrieved."
    model = ReplayModel([tool_call("fetch_url", {"url": PROBE_URL}), reply(answer)])

    graded = await engine(model, harnesses=FETCH_HARNESS).run_task(rollout_spec(task))

    [_, observation] = graded.rollout.conversations[0].messages[-3:-1]
    assert json.loads(observation["content"])["url"] == PROBE_URL
    assessment = protocol(graded)["url_loading"]
    assert (assessment["label"], assessment["protocol_compliance"], graded.grade.reward == 1.0) == (
        label,
        True,
        correct,
    )


async def test_fetch_provider_answers_unknown_urls_and_bad_arguments_as_observations():
    task = TASKS[ProtocolCheck.URL_LOADING]
    model = ReplayModel(
        [
            tool_call("fetch_url", {"address": PROBE_URL}, "bad"),
            tool_call("fetch_url", {"url": "https://elsewhere.invalid"}, "unknown"),
            reply("no token"),
        ]
    )

    graded = await engine(model, harnesses=FETCH_HARNESS).run_task(rollout_spec(task))

    tool_messages = [message for message in graded.rollout.conversations[0].messages if message["role"] == "tool"]
    assert [json.loads(message["content"]).get("status") for message in tool_messages] == [None, 404]
    assert protocol(graded)["url_loading"]["label"] == "not_attempted"


@dataclass
class ServedModel:
    """Answer with a fixed raw response, recording each request."""

    raw_response: str
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        visible = self.raw_response.rsplit("</think>", 1)[-1]
        return ModelTurn(reply(visible), (10, 11), (20,), (-0.5,), "stop", raw_response=self.raw_response)


class ThinkingOffHarness:
    """Ask the model with thinking turned off, as a launch that disables thinking does."""

    def check(self, task: TaskSpec, limits) -> None:
        pass

    async def run(self, context, session) -> None:
        [instruction] = context.task.context.events
        await session.start(Prompt(({"role": "user", "content": instruction.content},), THINKING_OFF))


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param("{nonce}", ("respected", True), id="respected"),
        pytest.param("<think>easy</think>{nonce}", ("reasoning_emitted", False), id="emitted"),
    ],
)
async def test_thinking_probe_grades_the_raw_response_when_the_request_turned_thinking_off(raw, expected):
    task = TASKS[ProtocolCheck.THINKING_DISABLED]
    model = ServedModel(raw.format(nonce=probe_nonce(task.id)))

    graded = await engine(model, harnesses={"off": ThinkingOffHarness()}).run_task(rollout_spec(task, harness="off"))

    assessment = protocol(graded)["thinking_disabled"]
    assert (assessment["label"], assessment["protocol_compliance"], graded.grade.reward) == (*expected, 1.0)


async def test_thinking_is_not_assessed_without_raw_responses_or_when_thinking_was_not_turned_off():
    task = TASKS[ProtocolCheck.THINKING_DISABLED]
    model = ReplayModel([reply(probe_nonce(task.id))])

    off = await engine(model, harnesses={"off": ThinkingOffHarness()}).run_task(rollout_spec(task, harness="off"))
    default = await engine(ServedModel(probe_nonce(task.id))).run_task(rollout_spec(task))

    assert "not_evaluable" in protocol(off)["thinking_disabled"]
    assert protocol(default) == {}
