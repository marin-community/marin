# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from contextlib import nullcontext

import httpx
import pytest
from rolloutengine.contracts import ModelTurn, RolloutContractError
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineStartupError
from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.russell_rsi.rollout_eval import completion_message, rollout_evidence
from experiments.post_training.russell_rsi.token_preflight import preflight_task, run_token_preflight


@pytest.mark.parametrize(
    "modes,expected_error",
    [
        (("direct", "tool"), None),
        (("tool", "direct"), None),
        (("direct", "direct"), ValueError),
        (("prefix", "tool"), RolloutContractError),
        (("adapter", "tool"), RolloutContractError),
        (("tool", "prefix"), RolloutContractError),
        (("bad_wire", "bad_wire"), ValueError),
    ],
)
def test_preflight_suite_gate(tmp_path, modes, expected_error):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"tokens": [1]}))
        ) as client:
            probe = -1

            async def turn(request):
                nonlocal probe
                manifest = json.loads((tmp_path / "token-preflight-suite.json").read_text())
                assert len(manifest["fixtures"]) == 2
                first_turn = not request.prefix_token_ids
                if first_turn:
                    probe += 1
                value = ("48213", "73961")[probe]
                mode = modes[probe]
                await client.post("https://model.test/v1/completions", json={"prompt": list(request.prefix_token_ids)})
                if mode == "adapter":
                    raise RolloutContractError("Renderer cannot retain the sampled prefix")
                if first_turn and mode != "direct":
                    assert all(value not in message["content"] for message in request.messages)
                    message = completion_message(
                        '<tool_call>{"name":"shell","arguments":'
                        '{"command":"cat /workspace/preflight-value.txt"}}</tool_call>',
                        request.options["tools"],
                        len(request.messages),
                    )
                    if mode == "bad_wire":
                        function = message["tool_calls"][0]["function"]
                        function["arguments"] = json.loads(function["arguments"])
                    return ModelTurn(message, (1,), (2,), None, "stop")
                if not first_turn:
                    observations = [message for message in request.messages if message.get("role") == "tool"]
                    assert value in observations[0]["content"]
                prompt = (1,) if first_turn else (1, 2, 3)
                if mode == "prefix":
                    prompt = (9, 3)
                return ModelTurn(
                    completion_message(value, request.options["tools"], len(request.messages)),
                    prompt,
                    (4,),
                    None,
                    "stop",
                )

            await run_token_preflight(turn, client, str(tmp_path))

    error_message = "contract failure" if expected_error is RolloutContractError else "Neither preflight probe"
    with pytest.raises(expected_error, match=error_message) if expected_error else nullcontext():
        asyncio.run(run())
    evidence = json.loads((tmp_path / "token-preflight.json").read_text())
    assert evidence["status"] == ("failed" if expected_error else "passed")
    assert len(evidence["attempts"]) == 2
    for index, mode in enumerate(modes, 1):
        attempt = json.loads((tmp_path / f"token-preflight-probe-{index}.json").read_text())
        assert attempt == evidence["attempts"][index - 1]
        assert attempt["status"] == ("passed" if mode == "tool" else "failed")
        assert len(attempt["requests"]) == (2 if mode in ("tool", "prefix") else 1)
        if mode == "bad_wire":
            assert attempt["rollout"]["metrics"]["invalid_assistant_message"] == 1
            assert len(attempt["rollout"]["steps"]) == 1
            assert attempt["interrupted_operation"] == "grade"
            assert attempt["cause"]["type"] == "ValidationError"
            assert "function.arguments" in attempt["cause"]["message"]


def test_completion_message_supports_two_shell_calls_before_grading():
    task = TaskSpec(
        id="two-shell-calls",
        context=ConversationInput(events=(TextMessage(role="user", content="Read the unit file twice."),)),
        environment=EnvironmentSpec(
            kind=EnvironmentKind.SHELLSIM,
            files=(EnvironmentFile(path="/workspace/unit-value.txt", content=b"89\n"),),
        ),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(89, tolerance_abs=0, tolerance_rel=0),
        source=Source(dataset="unit", revision="1", row="0", importer_revision="1"),
    )
    turns = 0

    async def turn(request):
        nonlocal turns
        turns += 1
        text = (
            '<tool_call>{"name":"shell","arguments":{"command":"cat /workspace/unit-value.txt"}}</tool_call>'
            if turns < 3
            else "89"
        )
        message = completion_message(text, request.options["tools"], len(request.messages))
        prompt = (*request.prefix_token_ids, 99)
        return ModelTurn(message, prompt, (4,), None, "stop")

    engine = ShellboxRolloutEngine(
        turn,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=4,
        command_timeout=10,
        convention=SubmissionConvention(id="unit", answer_format=AnswerFormat.PLAIN),
    )
    rollout = asyncio.run(engine.run(task))
    assert rollout.grade.reward == 1
    observations = [message for message in rollout.messages if message["role"] == "tool"]
    assert len(observations) == 2
    assert all(json.loads(message["content"])["stdout"] == "89\n" for message in observations)


@pytest.mark.parametrize("stage", ["start", "grade", "success"])
def test_rollout_evidence_preserves_execution_failure_and_partial_tokens(stage):
    class UnavailableMachineFactory:
        async def create(self, spec):
            raise OSError("Guest transport is unavailable")

    async def turn(request):
        message = {"role": "assistant", "content": "89"}
        if stage == "grade":
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "bad-wire",
                        "type": "function",
                        "function": {"name": "shell", "arguments": {"command": "true"}},
                    }
                ],
            }
        return ModelTurn(
            message=message, prompt_token_ids=(1,), response_token_ids=(2,), logprobs=None, stop_reason="stop"
        )

    task = preflight_task(1, "Return the file value.", 89)
    engine = ShellboxRolloutEngine(
        turn,
        {EnvironmentKind.SHELLSIM: UnavailableMachineFactory() if stage == "start" else ShellSimMachineFactory()},
        max_turns=4,
        command_timeout=10,
        convention=SubmissionConvention(id="unit", answer_format=AnswerFormat.PLAIN),
    )
    rollout, evidence = asyncio.run(rollout_evidence(engine, task))
    record = json.loads(json.dumps(evidence))
    assert record["task_id"] == task.id
    assert record["response_token_ids"] == ([] if stage == "start" else [2])
    if stage == "success":
        assert record["grade"]["reward"] == 1.0
        assert record["interrupted_operation"] is None
        assert record["execution_error"] is None
        return
    assert record["grade"]["reward"] is None
    assert rollout.grade.reward is None
    assert record["interrupted_operation"] == stage
    cause = record["execution_error"]
    assert cause["type"] == ("OSError" if stage == "start" else "ValidationError")
    assert cause["message"] in record["execution_error"]["traceback"]
    if stage == "start":
        assert cause["message"] == "Guest transport is unavailable"


@pytest.mark.parametrize("failed_starts,fail_command", [(1, False), (3, False), (0, True)])
def test_startup_retry_preserves_failures_without_replacing_model_samples(failed_starts, fail_command):
    failures = []
    model_requests = []

    class Machine:
        def __init__(self, machine):
            self.machine = machine

        async def run(self, command):
            if command.argv[-1] == "fail-after-turn":
                raise MachineStartupError("Failure after inference")
            return await self.machine.run(command)

        async def upload(self, source, target):
            await self.machine.upload(source, target)

        async def download(self, source, target):
            await self.machine.download(source, target)

        async def close(self):
            await self.machine.close()

    class Factory:
        attempts = 0

        async def create(self, spec):
            self.attempts += 1
            # Each failed start must be saved before the next machine starts.
            assert len(failures) == min(self.attempts - 1, failed_starts)
            if self.attempts <= failed_starts:
                raise MachineStartupError(f"Failed start {self.attempts}")
            return Machine(await ShellSimMachineFactory().create(spec))

    async def turn(request):
        model_requests.append(request)
        message = {"role": "assistant", "content": "89"}
        if fail_command:
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "command",
                        "type": "function",
                        "function": {"name": "shell", "arguments": json.dumps({"command": "fail-after-turn"})},
                    }
                ],
            }
        return ModelTurn(message, (1,), (2,), None, "stop")

    factory = Factory()
    engine = ShellboxRolloutEngine(
        turn,
        {EnvironmentKind.SHELLSIM: factory},
        max_turns=4,
        command_timeout=10,
        convention=SubmissionConvention(id="unit", answer_format=AnswerFormat.PLAIN),
    )
    task = preflight_task(1, "Return the file value.", 89)
    rollout, evidence = asyncio.run(
        rollout_evidence(
            engine,
            task,
            startup_attempts=3,
            record_startup_failure=lambda attempt, record: failures.append((attempt, json.loads(json.dumps(record)))),
        )
    )
    assert factory.attempts == (3 if failed_starts == 3 else failed_starts + 1)
    assert len(model_requests) == (0 if failed_starts == 3 else 1)
    assert [record[0] for record in failures] == list(range(1, failed_starts + 1))
    for attempt, record in failures:
        assert record["response_token_ids"] == []
        assert record["execution_error"]["message"] == f"Failed start {attempt}"
    if failed_starts == 3:
        assert rollout.grade.reward is None
        assert evidence["startup_attempt"] == 3
    elif fail_command:
        assert evidence["interrupted_operation"] == "advance"
        assert evidence["response_token_ids"] == (2,)
    else:
        assert rollout.grade.reward == 1.0
        assert evidence["interrupted_operation"] is None
