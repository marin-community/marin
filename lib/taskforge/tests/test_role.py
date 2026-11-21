# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The role runner: openings, machine-less sittings and the session it hands its caller."""

import json
from dataclasses import replace
from pathlib import PurePosixPath

import pytest
from lib.taskforge.tests.fake_resources import resources
from lib.taskforge.tests.resource_doubles import FakeInference
from lib.taskforge.tests.stage_fakes import Environments

from taskforge.effects.context import Context
from taskforge.effects.events import ScopeEntered
from taskforge.effects.log import EventLog
from taskforge.harness import role as roles
from taskforge.harness.protocol import Stopped
from taskforge.harness.role import Answer, Fresh, Visit, sitting
from taskforge.harness.tools.evidence import Bounded
from taskforge.model.llm import Completion, FinishReason, Message, Role, ToolCall
from taskforge.model.workpiece import Conversation
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.wiring import build_config


def finish(call_id: str) -> Completion:
    return Completion("", (ToolCall(call_id, "finish", json.dumps({"reason": "done"})),), FinishReason.TOOL_CALLS)


def machine_less(role: roles.Role) -> roles.Role:
    """``role`` with every tool's replies bounded, as a visit without a machine requires."""
    return replace(role, tools={name: Bounded(tool, 4000) for name, tool in role.tools.items()})


async def test_a_role_without_a_machine_runs_its_session_and_opens_no_sandbox() -> None:
    ctx = Context(log=EventLog())
    client = FakeInference((finish("f"),))
    bundle = replace(resources(ctx), role_inference={"adversary": client})
    role = machine_less(build_config(load_policy(DEFAULTS_DIR), {}).adversary)
    async with sitting(ctx, role, Visit("review", "unused", None), bundle) as session:
        outcome = await session(Fresh("system", "user"))
    assert isinstance(bundle.envs, Environments) and not bundle.envs.created
    assert [m.content for m in client.requests[0].conversation] == ["system", "user"]
    assert isinstance(outcome, Stopped) and outcome.detail == "done"
    with pytest.raises(ValueError, match="has no machine"):
        async with sitting(ctx, role, Visit("review", "unused", None, lambda: ((PurePosixPath("/x"), b""),)), bundle):
            pass


async def test_an_answer_replies_to_the_pending_call_in_the_same_conversation() -> None:
    log = EventLog()
    ctx = Context(log=log)
    client = FakeInference((finish("again"),))
    bundle = replace(resources(ctx), role_inference={"adversary": client})
    role = machine_less(build_config(load_policy(DEFAULTS_DIR), {}).adversary)
    pending = Message(Role.ASSISTANT, "", (ToolCall("report-1", "report", "{}"),))
    saved = Conversation((Message(Role.SYSTEM, "s"), Message(Role.USER, "u"), pending), ())
    visit = Visit("adversary:finder", "validate/0/0/adversary:finder/0/0", None)
    async with sitting(ctx, role, visit, bundle) as session:
        await session(Answer(saved, "did not reproduce"))
    (request,) = client.requests
    assert request.conversation[-1] == Message(Role.TOOL, "did not reproduce", tool_call_id="report-1")
    assert request.conversation[:3] == saved.messages
    assert [s.name for s in request.tools] == [*role.tools]
    assert any(isinstance(r.event, ScopeEntered) and r.event.name == "adversary:finder" for r in log.records)
    with pytest.raises(ValueError, match="pending tool call"):
        async with sitting(ctx, role, Visit("x", "y", None), bundle) as session:
            await session(Answer(Conversation(saved.messages[:2], ()), "late"))
