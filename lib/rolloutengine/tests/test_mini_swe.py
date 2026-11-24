# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
from shellbox.machine import ExitReason, Result

from rolloutengine.agents.mini_swe.agent import MiniSweModelFormat, MiniSweOptions, mini_swe_harness
from rolloutengine.agents.model_relay import VllmTokenizer
from rolloutengine.models import Rollout
from rolloutengine.test_support import (
    REPLAY_MODEL,
    ReplayEndpoint,
    ReplayMachine,
    recorded_commands,
    recorded_requests,
    replay_rollout,
    replay_scenarios,
    without_stream,
)

REPLAY = replay_scenarios(Path(__file__).parent / "fixtures" / "mini_swe" / "harbor_replay.json")
WRAPPER = ("/bin/sh", "-c", 'exec /bin/sh -c "$1" 2>&1', "mini-swe-command")
UNAME_ARGV = ("/bin/sh", "-c", "uname -s && uname -r && uname -v && uname -m")


def replay_machine(recorded: dict[str, Any]) -> ReplayMachine:
    """Answer each command with the result that mini-swe-agent's command got, and the system information probe."""
    system, _node, release, version, machine = recorded["uname"]
    uname = Result(0, f"{system}\n{release}\n{version}\n{machine}\n".encode(), b"", False, False, ExitReason.EXITED)
    return ReplayMachine(recorded["machine"], WRAPPER, fixed_results={UNAME_ARGV: uname})


def sent_request(request: dict[str, Any]) -> dict[str, Any]:
    """Harbor's request as vLLM reads it.

    Harbor sends back the SDK's dump of each reply, whose unset fields are null; vLLM ignores null message fields.
    """
    body = dict(request["request"])
    body["messages"] = [
        {key: value for key, value in message.items() if value is not None} for message in body["messages"]
    ]
    return {**request, "request": body}


async def mini_swe_rollout(recorded: dict[str, Any], model: ReplayEndpoint, machine: ReplayMachine) -> Rollout:
    options = recorded["options"]

    def harness(client):
        return mini_swe_harness(
            MiniSweOptions(
                max_input_tokens=options["max_input_tokens"],
                max_output_tokens=options["max_output_tokens"],
                model_format=MiniSweModelFormat(options["model_format"]),
            ),
            VllmTokenizer(client, REPLAY_MODEL),
        )

    return await replay_rollout(harness, recorded["instruction"], model, machine)


@pytest.mark.parametrize("scenario", sorted(REPLAY))
async def test_mini_swe_sends_harbors_requests_and_commands(scenario):
    recorded = REPLAY[scenario]
    model = ReplayEndpoint(recorded["model"])
    machine = replay_machine(recorded)

    rollout = await mini_swe_rollout(recorded, model, machine)

    assert without_stream(model.requests) == [sent_request(request) for request in recorded_requests(recorded["model"])]
    assert machine.commands == recorded_commands(recorded["machine"])
    # Submitting, running out of context and a context rejection all end the episode for grading.
    assert rollout.failure is None


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ('["ls"]', "Missing 'command' argument in bash tool call."),
        ('{"command": ["ls"]}', "The 'command' argument of the bash tool call must be a string."),
    ],
)
async def test_a_bash_call_without_a_string_command_gets_the_format_error_prompt(arguments, error):
    recorded = REPLAY["format_errors"]
    exchanges = json.loads(json.dumps(recorded["model"]))
    exchanges[1]["response"]["choices"][0]["message"]["tool_calls"] = [
        {"id": "call_0_0", "type": "function", "function": {"name": "bash", "arguments": arguments}}
    ]
    model = ReplayEndpoint(exchanges)
    machine = replay_machine(recorded)

    rollout = await mini_swe_rollout(recorded, model, machine)

    # mini-swe-agent itself fails on such a call; the port tells the model, which continues to its submission.
    assert f"<error>\n{error}\n</error>" in model.requests[2]["request"]["messages"][-1]["content"]
    assert [command["command"] for command in machine.commands] == [recorded["machine"][0]["command"]]
    assert rollout.failure is None
