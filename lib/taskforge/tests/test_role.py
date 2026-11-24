# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The role runner: openings, machine-less sittings and the session it hands its caller."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.fake_resources import resources
from lib.taskforge.tests.resource_doubles import FakeInference
from lib.taskforge.tests.stage_fakes import Environments
from taskcompendium.models import EnvironmentRequirements

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved, ScopeEntered
from taskforge.effects.log import Billing, BufferSink, EventLog
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.protocol import Exit, Harness, Stopped, StopReason, Tool, ToolResult
from taskforge.harness.role import Answer, Fresh, Visit, sitting
from taskforge.harness.tools.base import ValidatedTool
from taskforge.llm.policy import LLMPolicy
from taskforge.model.llm import Completion, FinishReason, Message, Role, ToolCall
from taskforge.model.workpiece import Conversation
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.env import WriteFile
from taskforge.resources.inference import InferenceClient
from taskforge.wiring import build_config


def finish(call_id: str) -> Completion:
    return Completion("", (ToolCall(call_id, "finish", json.dumps({"reason": "done"})),), FinishReason.TOOL_CALLS)


# Regression: with the payer inserted as a positional field, ``Visit(scope, machine, files)`` took the files callback
# as its payer: the machine-less visit skipped hydration, paid for a model call and could not journal it;
# ``Visit(scope, machine)`` failed to construct.
async def test_a_role_without_a_machine_runs_its_session_and_opens_no_sandbox() -> None:
    sink = BufferSink()
    ctx = Context(log=EventLog(sink=sink))
    client = FakeInference((finish("f"),))
    bundle = replace(resources(ctx), role_inference={"adversary": client})
    role = build_config(load_policy(DEFAULTS_DIR), {}).adversary
    async with ctx.scope("validate", bills=Billing.SOLVER):  # a visit that names no payer does not take the stage's
        async with sitting(ctx, role, Visit("review", None), bundle) as session:
            outcome = await session(Fresh("system", "user"))
    assert isinstance(bundle.envs, Environments) and not bundle.envs.created
    assert [m.content for m in client.requests[0].conversation] == ["system", "user"]
    assert isinstance(outcome, Stopped) and outcome.detail == "done"
    with pytest.raises(ValueError, match="has no machine"):
        async with sitting(ctx, role, Visit("review", None, lambda: ((PurePosixPath("/x"), b""),)), bundle):
            pass
    assert len(client.requests) == 1  # refused before its first model call
    await ctx.drain()
    paid = [r.billed for r in local_records(sink) if isinstance(r.event, Resolved)]
    assert paid == [Billing.UNATTRIBUTED]


async def test_an_answer_replies_to_the_pending_call_in_the_same_conversation() -> None:
    log = EventLog()
    ctx = Context(log=log)
    client = FakeInference((finish("again"),))
    bundle = replace(resources(ctx), role_inference={"adversary": client})
    role = build_config(load_policy(DEFAULTS_DIR), {}).adversary
    pending = Message(Role.ASSISTANT, "", (ToolCall("report-1", "report", "{}"),))
    saved = Conversation((Message(Role.SYSTEM, "s"), Message(Role.USER, "u"), pending), ())
    visit = Visit("adversary:finder", None)
    async with sitting(ctx, role, visit, bundle) as session:
        await session(Answer(saved, "did not reproduce"))
    (request,) = client.requests
    assert request.conversation[-1] == Message(Role.TOOL, "did not reproduce", tool_call_id="report-1")
    assert request.conversation[:3] == saved.messages
    assert [s.name for s in request.tools] == [*role.tools]
    assert any(isinstance(r.event, ScopeEntered) and r.event.name == "adversary:finder" for r in log.records)
    with pytest.raises(ValueError, match="pending tool call"):
        async with sitting(ctx, role, Visit("x", None), bundle) as session:
            await session(Answer(Conversation(saved.messages[:2], ()), "late"))


@dataclass(frozen=True, slots=True)
class NoArgs:
    pass


@dataclass(frozen=True, slots=True)
class Loud(ValidatedTool[NoArgs]):
    name = "loud"
    description = "Reply with far more than a reply may hold."
    args_type = NoArgs

    @override
    async def run(self, ctx: Context, args: NoArgs) -> ToolResult:
        return ToolResult("x" * 50_000)


async def test_a_machine_less_session_gets_every_reply_clipped_and_never_a_spill() -> None:
    """Regression: the loop's own unknown-tool diagnostic spilled a reply to a file a
    machine-less reviewer could never read. Every reply, tool output and loop diagnostic alike, is clipped instead."""
    log = EventLog()
    ctx = Context(log=log)
    unknown = Completion("", (ToolCall("u", "x" * 50_000, "{}"),), FinishReason.TOOL_CALLS)
    loud = Completion("", (ToolCall("l", "loud", "{}"),), FinishReason.TOOL_CALLS)
    client = FakeInference((unknown, loud, finish("f")))
    bundle = replace(resources(ctx), role_inference={"adversary": client})
    role = build_config(load_policy(DEFAULTS_DIR), {}).adversary
    chars = load_policy(DEFAULTS_DIR).setting("tools.output_chars", int)
    async with sitting(ctx, role, Visit("review", None, tools={"loud": Loud()}), bundle) as session:
        await session(Fresh("system", "user"))
    replies = [m.content for m in client.requests[-1].conversation if m.role == "tool"]
    assert len(replies) == 2 and all(len(reply) <= chars for reply in replies)
    assert all("this session has no machine" in reply for reply in replies)
    assert not [record for record in log.records if isinstance(record.event, WriteFile)]


@dataclass
class FourArguments(Harness):
    """An injected harness written against the four-argument protocol: it records each session and finishes."""

    sessions: list[Conversation] = field(default_factory=list)

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        self.sessions.append(session)
        return Stopped(StopReason.FINISHED, session, "done")


@pytest.mark.parametrize("machine", [None, EnvironmentRequirements()], ids=["machine-less", "machine-backed"])
async def test_an_injected_four_argument_harness_runs_unchanged(machine: EnvironmentRequirements | None) -> None:
    """Regression: the role runner passed ``spill=`` to every harness, so one written
    against the four-argument protocol raised before it ran, after a machine-backed visit had taken its sandbox. The
    runner configures only its own harness per visit."""
    ctx = Context(log=EventLog())
    bundle = replace(resources(ctx), role_inference={"adversary": FakeInference(())})
    harness = FourArguments()
    role = replace(build_config(load_policy(DEFAULTS_DIR), {}).adversary, harness=harness)
    async with sitting(ctx, role, Visit("review", machine), bundle) as session:
        outcome = await session(Fresh("system", "user"))
    assert isinstance(outcome, Stopped) and outcome.detail == "done" and len(harness.sessions) == 1


class Constructed(AgenticHarness):
    """An injected stock-harness subclass with its own constructor, which ``dataclasses.replace`` cannot rebuild."""

    def __init__(self, detail: str) -> None:
        super().__init__(262144, LLMPolicy(), "continue", 20000, lambda arguments: arguments)
        object.__setattr__(self, "detail", detail)

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        return Stopped(StopReason.FINISHED, session, self.__dict__["detail"])


@pytest.mark.parametrize("machine", [None, EnvironmentRequirements()], ids=["machine-less", "machine-backed"])
async def test_an_injected_stock_harness_subclass_runs_unchanged(machine: EnvironmentRequirements | None) -> None:
    """Regression: configuring every ``AgenticHarness`` per visit rebuilt an injected subclass
    through ``dataclasses.replace``, which called its own constructor with the stock fields and raised. Only the exact
    stock class is configured."""
    ctx = Context(log=EventLog())
    bundle = replace(resources(ctx), role_inference={"adversary": FakeInference(())})
    role = replace(build_config(load_policy(DEFAULTS_DIR), {}).adversary, harness=Constructed("its own"))
    async with sitting(ctx, role, Visit("review", machine), bundle) as session:
        outcome = await session(Fresh("system", "user"))
    assert isinstance(outcome, Stopped) and outcome.detail == "its own"
