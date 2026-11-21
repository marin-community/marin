# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One runner for every model role: its machine, workspace hydration, scope, conversation and close."""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import PurePosixPath
from types import MappingProxyType

from taskcompendium.models import EnvironmentRequirements

from taskforge.effects.context import Context
from taskforge.effects.log import Opened
from taskforge.harness.protocol import Harness, Tool
from taskforge.lowering import host_requirements
from taskforge.model.llm import Message
from taskforge.model.llm import Role as Speaker
from taskforge.model.stage import Exit
from taskforge.model.workpiece import Conversation
from taskforge.resources.bundle import Resources
from taskforge.resources.env import (
    Env,
    ReadFile,
    RunCommand,
    WriteFile,
    WriteFiles,
    read_file,
    run_command,
    write_file,
    write_files,
)
from taskforge.resources.execution import close_resource


@dataclass(frozen=True, slots=True)
class Role:
    """A model role: ``name`` selects its model (``<name>.model``) and client, ``harness`` runs its session (model
    profile, sampling, prompt limit, nudge), and ``tools`` are its tools, whose schemas the model sees in this order."""

    name: str
    harness: Harness
    tools: Mapping[str, Tool[object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", MappingProxyType(dict(self.tools)))


@dataclass(frozen=True, slots=True)
class Fresh:
    """A new conversation: the role's system prompt and its first user message."""

    system: str
    user: str


@dataclass(frozen=True, slots=True)
class Resume:
    """A saved conversation, continued as it stands; an author repair resumes from its revision prompt."""

    session: Conversation


@dataclass(frozen=True, slots=True)
class Answer:
    """A saved conversation whose last assistant turn left a tool call pending, answered with ``content``."""

    session: Conversation
    content: str


type Opening = Fresh | Resume | Answer


@dataclass(frozen=True, slots=True)
class Visit:
    """One sitting of a role: its scope, its sandbox owner and machine (``None``: no machine), the files hydrated
    before its first model call, and tools bound to this visit alone."""

    scope: str
    owner: str
    machine: EnvironmentRequirements | None
    files: tuple[tuple[PurePosixPath, bytes], ...] = ()
    tools: Mapping[str, Tool[object]] = field(default_factory=dict)


def opened(opening: Opening, tools: Mapping[str, Tool[object]]) -> Conversation:
    schemas = tuple(tool.schema for tool in tools.values())
    match opening:
        case Fresh(system, user):
            return Conversation((Message(Speaker.SYSTEM, system), Message(Speaker.USER, user)), schemas)
        case Resume(session):
            return session
        case Answer(session, content):
            calls = session.messages[-1].tool_calls if session.messages else ()
            if not calls:
                raise ValueError("Answer needs a last assistant turn with a pending tool call")
            answer = Message(Speaker.TOOL, content, tool_call_id=calls[-1].id)
            return replace(session, messages=(*session.messages, answer))


def install_env(ctx: Context, env: Env) -> None:
    ctx.handle(RunCommand, partial(run_command, notify=ctx.notify), env)
    ctx.handle(ReadFile, partial(read_file, notify=ctx.notify), env)
    ctx.handle(WriteFile, partial(write_file, notify=ctx.notify), env)
    ctx.handle(WriteFiles, partial(write_files, notify=ctx.notify), env)


@asynccontextmanager
async def sitting(
    ctx: Context, role: Role, visit: Visit, resources: Resources
) -> AsyncIterator[Callable[[Opening], Awaitable[Exit]]]:
    """Open ``visit``'s machine and scope, hydrate its files and yield its session, ready to run from an opening.

    The caller runs the session and reads its exit inside the scope (an adversary reproduces its claims there, and may
    run it again with an ``Answer`` to its report); the machine closes after the scope, however it ends.
    """
    if visit.machine is None and visit.files:
        raise ValueError(f"{visit.scope} hydrates files but has no machine")
    env = None if visit.machine is None else await resources.envs.create(host_requirements(visit.machine))
    try:
        async with ctx.scope(visit.scope, None if env is None else Opened(visit.owner, env.fingerprint().digest)):
            if env is not None:
                install_env(ctx, env)
                await ctx.emit(WriteFiles(visit.files))
            tools = {**role.tools, **visit.tools}

            async def session(opening: Opening) -> Exit:
                return await role.harness.run(ctx, opened(opening, tools), resources.inference_for(role.name), tools)

            yield session
    finally:
        if env is not None:
            await close_resource(env.close, ctx.notify)
