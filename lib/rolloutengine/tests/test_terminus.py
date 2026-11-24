# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path
from typing import Any

import pytest
from taskcompendium.importers.harbor.package import HARBOR_SHELL_PREFIX

from rolloutengine.agents.terminus.agent import TerminusOptions, TerminusParser, terminus_harness
from rolloutengine.agents.terminus.terminus_json_plain_parser import TerminusJSONPlainParser
from rolloutengine.agents.terminus.terminus_xml_plain_parser import TerminusXMLPlainParser
from rolloutengine.models import Rollout
from rolloutengine.test_support import (
    ReplayEndpoint,
    ReplayMachine,
    recorded_commands,
    recorded_requests,
    replay_rollout,
    replay_scenarios,
    without_stream,
)

REPLAY = replay_scenarios(Path(__file__).parent / "fixtures" / "terminus" / "harbor_replay.json")
# Each batch of keystrokes has a fresh marker, which the transcript records as a placeholder.
MARKER = re.compile(r"__HARBOR_TMUX_([0-9a-f]{32})_")
MARKER_PLACEHOLDER = "__HARBOR_TMUX_{marker}_"


def replay_machine(exchanges: list[dict[str, Any]]) -> ReplayMachine:
    return ReplayMachine(exchanges, HARBOR_SHELL_PREFIX, nonce=MARKER, placeholder=MARKER_PLACEHOLDER)


async def terminus_rollout(
    recorded: dict[str, Any], model: ReplayEndpoint, machine: ReplayMachine, **options: Any
) -> Rollout:
    """Run Terminus with the recorded options, overridden by ``options``, against the replayed model and machine."""
    options = {**recorded["options"], "parser": TerminusParser(recorded["options"]["parser"]), **options}
    harness = terminus_harness(TerminusOptions(**options))
    return await replay_rollout(lambda client: harness, recorded["instruction"], model, machine)


@pytest.mark.parametrize("scenario", sorted(REPLAY))
async def test_terminus_sends_harbors_requests_and_terminal_commands(scenario):
    recorded = REPLAY[scenario]
    model = ReplayEndpoint(recorded["model"])
    machine = replay_machine(recorded["machine"])
    await terminus_rollout(recorded, model, machine)

    assert without_stream(model.requests) == recorded_requests(recorded["model"])
    assert machine.commands == recorded_commands(recorded["machine"])


async def test_terminus_out_of_context_ends_its_attempt_with_the_turns_it_completed():
    recorded = REPLAY["context_rejection"]
    model = ReplayEndpoint(recorded["model"])

    rollout = await terminus_rollout(recorded, model, replay_machine(recorded["machine"]), enable_summarize=False)

    # The second request exceeds the context window, and without summarization Terminus stops there.
    assert [exchange["status"] for exchange in recorded["model"][: len(model.requests)]] == [200, 400]
    assert [len(conversation.steps) for conversation in rollout.conversations] == [1]


async def test_terminus_main_turn_limit_keeps_the_last_command():
    recorded = REPLAY["context_rejection"]
    model = ReplayEndpoint(recorded["model"])
    machine = replay_machine(recorded["machine"])

    rollout = await terminus_rollout(recorded, model, machine, max_main_turns=1)

    assert without_stream(model.requests) == recorded_requests(recorded["model"])[:1]
    assert [len(conversation.steps) for conversation in rollout.conversations] == [1]
    assert any("send-keys" in command.get("command", "") for command in machine.commands)


async def test_terminus_reply_cut_after_its_context_summary_ends_its_attempt():
    recorded = REPLAY["context_rejection"]
    exchanges = json.loads(json.dumps(recorded["model"]))
    handoff = next(
        index
        for index, exchange in enumerate(exchanges)
        if exchange["request"]["messages"][-1]["content"].startswith("Here are the answers")
    )
    # The model's first reply after the summary handoff reaches its output limit.
    exchanges[handoff]["response"]["choices"][0]["finish_reason"] = "length"
    model = ReplayEndpoint(exchanges)

    rollout = await terminus_rollout(recorded, model, replay_machine(recorded["machine"]))

    assert len(model.requests) == handoff + 1
    assert rollout.failure is None


async def test_terminus_model_that_ends_its_terminal_ends_its_attempt_with_the_turns_it_completed():
    recorded = REPLAY["session_exit"]
    exchanges = [dict(exchange) for exchange in recorded["machine"]]
    # The model's `exit` ends the session before the batch reaches its next terminal action.
    batch = next(exchange for exchange in exchanges if "DEAD_AFTER" in exchange["stdout"])
    batch["stdout"] = batch["stdout"].replace("DEAD_AFTER", "SESSION_ENDED")

    rollout = await terminus_rollout(recorded, ReplayEndpoint(recorded["model"]), replay_machine(exchanges))

    assert [len(conversation.steps) for conversation in rollout.conversations] == [1]
    assert rollout.failure is None


async def test_terminus_model_that_sends_unencodable_keystrokes_ends_its_attempt_with_the_turns_it_completed():
    recorded = REPLAY["session_exit"]
    exchanges = json.loads(json.dumps(recorded["model"]))
    message = exchanges[0]["response"]["choices"][0]["message"]
    # An unpaired surrogate parses as JSON but cannot be encoded for the terminal.
    message["content"] = message["content"].replace("exit\\n", "\\ud800")

    rollout = await terminus_rollout(recorded, ReplayEndpoint(exchanges), replay_machine(recorded["machine"]))

    assert [len(conversation.steps) for conversation in rollout.conversations] == [1]
    assert rollout.failure is None


@pytest.mark.parametrize(
    ("response", "error", "duration"),
    [
        ('{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n"}]}', "", 1.0),
        ('{"analysis": "a", "plan": "p"}', "Missing required fields: commands", None),
        (
            '{"analysis": "a", "plan": "p", "commands": [{"duration": 1}]}',
            "Command 1 missing required 'keystrokes'",
            None,
        ),
    ],
)
def test_json_parser_requires_commands_and_defaults_their_duration(response, error, duration):
    result = TerminusJSONPlainParser().parse_response(response)
    assert error in result.error if error else not result.error
    assert [command.duration for command in result.commands] == ([] if duration is None else [duration])


def test_xml_parser_keeps_keystrokes_verbatim_and_salvages_a_complete_response():
    parser = TerminusXMLPlainParser()
    response = (
        "<response><analysis>a</analysis><plan>p</plan><commands>"
        '<keystrokes duration="0.5">echo "a<b" && ls\n</keystrokes></commands>'
        "<task_complete>true</task_complete></response>"
    )
    result = parser.parse_response(response)
    assert [(command.keystrokes, command.duration) for command in result.commands] == [('echo "a<b" && ls\n', 0.5)]
    assert result.is_task_complete
    assert parser.salvage_truncated_response(response.removesuffix("</response>")) is None
    assert parser.salvage_truncated_response(response + "\ntrailing text") == response
