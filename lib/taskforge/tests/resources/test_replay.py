# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import errno
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import PurePosixPath
from typing import override

from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resource_doubles import FakeCommand, FakeEnv, FakeInference
from rolloutengine.models import GradedRollout, Rollout
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import inline_resource

from taskforge.effects.context import Context
from taskforge.effects.events import Failed, Fingerprint, ReplayPolicy, Resolved, ScopeEntered
from taskforge.effects.log import BufferSink, EventLog, Record
from taskforge.harness.role import install_env
from taskforge.llm.policy import LLMPolicy
from taskforge.model.controls import Candidate
from taskforge.model.llm import Completion, Message, Role, ToolSchema
from taskforge.resources.env import (
    CommandResult,
    Env,
    ReadFile,
    RunCommand,
    WriteFile,
    run_command,
    write_file,
)
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.ids import Ids, NewId, new_id
from taskforge.resources.inference import InferenceClient, InferenceRequest, complete
from taskforge.resources.rollouts import SolverRollout


def configure(ctx: Context, env: Env, client: InferenceClient, ids: Ids) -> None:
    ctx.handle(RunCommand, partial(run_command, notify=ctx.notify), env)
    ctx.handle(WriteFile, partial(write_file, notify=ctx.notify), env)
    ctx.handle(InferenceRequest, partial(complete, notify=ctx.notify), client)
    ctx.handle(NewId, new_id, ids)


async def test_reload_reexecutes_environment_but_reuses_inference_and_ids() -> None:
    path = PurePosixPath("/workspace/binary")
    result = CommandResult(0, "done", "", 0.1, False)
    script = {"build": FakeCommand(result, {path: b"result"})}
    env = FakeEnv("image", script)
    client = FakeInference((Completion("answer"),))
    event = InferenceRequest(
        (Message(Role.USER, "solve"),),
        (ToolSchema("shell", "Run command", {"type": "object", "required": ("command",)}),),
        LLMPolicy(temperature=1),
    )
    sink = BufferSink()
    ctx = Context(log=EventLog(sink=sink))
    async with ctx.scope("author"):
        configure(ctx, env, client, Ids(lambda: "original"))
        await ctx.emit(WriteFile(path, b"\xff\x00"))
        assert (await ctx.emit(event)).content == "answer"
        assert await ctx.emit(RunCommand("build", 5)) == result
        assert await ctx.emit(NewId("author/0/id")) == "original"
    await ctx.drain()

    replay_env = FakeEnv("image", script)
    replay_client = FakeInference(())
    replay = Context(log=EventLog(replay_from=local_records(sink)))
    async with replay.scope("author"):
        configure(replay, replay_env, replay_client, Ids(lambda: "new"))
        await replay.emit(WriteFile(path, b"\xff\x00"))
        assert (await replay.emit(event)).content == "answer"
        await replay.emit(RunCommand("build", 5.0))
        assert await replay.emit(NewId("author/0/id")) == "original"
        assert await replay.emit(NewId("author/1/id")) == "new"
    await replay.drain()
    assert not replay_client.requests
    assert await replay_env.read(path) == b"result"


async def test_jsonl_round_trips_resource_results_binary_payloads_and_envelopes() -> None:
    path = PurePosixPath("/workspace/file")
    spec = rollout_spec(arithmetic_task())
    grade = GradeResult(Outcome.GRADED, 1, diagnostics={"binary": b"\xff"})
    rollout = GradedRollout(Rollout("trial", (), "stop", outputs={"file": b"\xff\x00"}), grade)
    env = Fingerprint("env", "definition", ReplayPolicy.REEXECUTE)
    records = (
        Record(1, 0.0, ScopeEntered("author")),
        Record(2, 0.0, ReadFile(path)),
        Record(3, 0.0, Resolved(ReadFile(path), (env,), b"\xff\x00"), answers=2),
        Record(4, 0.0, WriteFile(path, b"\xff\x00")),
        Record(5, 0.0, Failed("EACCES", f"[Errno 13] Permission denied: '{path}'"), answers=4),
        Record(6, 0.0, Resolved(SolverRollout(spec, 1, 0), (), rollout)),
        Record(
            7,
            0.0,
            Resolved(GradeCandidate(spec, Candidate("12", (inline_resource("workspace/file", b"\xff"),))), (), grade),
        ),
    )
    sink = BufferSink()
    await sink.write(records)
    assert local_records(sink) == records


class Tracked(FakeEnv):
    def __init__(self, commands: Mapping[str, FakeCommand]) -> None:
        super().__init__("image", commands)
        self.executed: list[str] = []

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        self.executed.append(cmd)
        return await super().run(cmd, timeout=timeout)


def scripted(**outputs: str) -> Tracked:
    """A machine whose commands print the given text."""
    return Tracked({name: FakeCommand(CommandResult(0, out, "", 0, False), {}) for name, out in outputs.items()})


async def attempt(log: EventLog, env: FakeEnv, commands: Sequence[str]) -> list[CommandResult]:
    """One session that runs ``commands`` in order on ``env``."""
    ctx = Context(log=log)
    try:
        async with ctx.scope("author"):
            install_env(ctx, env)
            return [await ctx.emit(RunCommand(command, 5)) for command in commands]
    finally:
        await ctx.drain()


async def test_a_resumed_session_runs_every_command_live_and_sees_its_fresh_output() -> None:
    """A recorded command and one the first launch never issued both run on the new machine; the recorded output of
    the first is not served back, and two identical commands in one session run twice."""
    log = EventLog()
    first = await attempt(log, scripted(x="pid 1\n", y="y"), ["x", "y"])
    assert [result.stdout for result in first] == ["pid 1\n", "y"]
    resumed = scripted(x="pid 2\n", z="z")
    again = await attempt(EventLog(replay_from=log.records), resumed, ["x", "z", "x"])
    assert [result.stdout for result in again] == ["pid 2\n", "z", "pid 2\n"]
    assert resumed.executed == ["x", "z", "x"]


async def test_a_handled_failure_is_journalled_against_its_request_and_a_resume_meets_the_live_outcome() -> None:
    path = PurePosixPath("/workspace/status")

    async def read(log: EventLog, env: FakeEnv) -> bytes | str:
        ctx = Context(log=log)
        try:
            async with ctx.scope("author"):
                install_env(ctx, env)
                return await ctx.emit(ReadFile(path))
        except OSError as error:
            return str(error)
        finally:
            await ctx.drain()

    log = EventLog()
    message = await read(log, FakeEnv("image", {}))
    request, outcome = log.records[1:3]
    assert isinstance(request.event, ReadFile) and isinstance(failed := outcome.event, Failed)
    assert (failed.category, failed.message, outcome.answers) == (
        errno.errorcode[errno.ENOENT],
        message,
        request.sequence,
    )
    present = FakeEnv("image", {})
    present.files[path] = b"B"
    assert await read(EventLog(replay_from=log.records), present) == b"B"
