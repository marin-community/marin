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
from taskcompendium.environment import EnvironmentFile, EnvironmentKind, EnvironmentSpec
from taskcompendium.grading import numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.russell_rsi.rollout_eval import completion_message
from experiments.post_training.russell_rsi.token_preflight import run_token_preflight


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
