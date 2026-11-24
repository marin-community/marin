# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resource_doubles import FakeCommand, FakeEnv, FakeInference
from rolloutengine.models import GradedRollout, Rollout
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import inline_resource

from taskforge.effects.context import Context
from taskforge.effects.events import Failed, Fingerprint, Note, ReplayPolicy, Resolved, ScopeEntered
from taskforge.effects.issues import LogFailure, Restart, Unreproducible
from taskforge.effects.log import BufferSink, EventLog, Opened, Owners, Record
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


def opened(env: Env, owner: str = "author/0") -> Opened:
    return Opened(owner, env.fingerprint().digest)


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
    async with ctx.scope("author", opened(env)):
        configure(ctx, env, client, Ids(lambda: "original"))
        await ctx.emit(WriteFile(path, b"\xff\x00"))
        assert (await ctx.emit(event)).content == "answer"
        assert await ctx.emit(RunCommand("build", 5)) == result
        assert await ctx.emit(NewId("author/0/id")) == "original"
    await ctx.drain()

    replay_env = FakeEnv("image", script)
    replay_client = FakeInference(())
    replay = Context(log=EventLog(replay_from=local_records(sink)))
    async with replay.scope("author", opened(replay_env)):
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
        Record(1, 0.0, ScopeEntered("author"), opened=Opened("author/0", "definition")),
        Record(2, 0.0, ReadFile(path), attempt=1),
        Record(3, 0.0, Resolved(ReadFile(path), (env,), b"\xff\x00"), answers=2),
        Record(4, 0.0, WriteFile(path, b"\xff\x00"), attempt=1),
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


RECORDED = CommandResult(0, "pid 1\n", "", 0.1, False)


@pytest.mark.parametrize(
    "fresh, served, diverged",
    [
        (RECORDED, RECORDED, False),
        (CommandResult(0, "pid 2\n", "", 0.3, False), RECORDED, True),
        (CommandResult(1, "", "disk full\n", 0.3, False), Unreproducible, False),
    ],
    ids=["same output", "different output", "re-execution fails"],
)
async def test_a_reexecuted_command_serves_its_recording_and_the_next_request_runs_live(
    fresh: CommandResult, served: CommandResult | type[Exception], diverged: bool
) -> None:
    path = PurePosixPath("/workspace/out")
    recording = Context(log=(log := EventLog()))
    original = FakeEnv("image", {"build": FakeCommand(RECORDED, {path: b"built"})})
    async with recording.scope("author", opened(original)):
        configure(recording, original, FakeInference(()), Ids())
        await recording.emit(RunCommand("build", 5))
    await recording.drain()

    replay = Context(log=(replayed := EventLog(replay_from=log.records)))
    env = FakeEnv("image", {"build": FakeCommand(fresh, {path: b"built"}), "new": FakeCommand(fresh, {})})
    if isinstance(served, type):
        with pytest.raises(served, match=r"env\.run_command"):
            async with replay.scope("author", opened(env)):
                configure(replay, env, FakeInference(()), Ids())
                await replay.emit(RunCommand("build", 5))
    else:
        async with replay.scope("author", opened(env)):
            configure(replay, env, FakeInference(()), Ids())
            assert await replay.emit(RunCommand("build", 5)) == served
            assert env.files[path] == b"built"
            # Beyond the recording, the command's own result is the result.
            assert await replay.emit(RunCommand("new", 5)) == fresh
    await replay.drain()
    notes = [record.event.msg for record in replayed.records if isinstance(record.event, Note)]
    assert len(notes) == diverged


class Tracked(FakeEnv):
    def __init__(self, commands: Mapping[str, FakeCommand | Restart]) -> None:
        super().__init__("image", commands)
        self.executed: list[str] = []

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        self.executed.append(cmd)
        return await super().run(cmd, timeout=timeout)


class Gated(Tracked):
    """Runs each command after its sibling has issued one too when given a barrier, so the two overlap."""

    def __init__(self, barrier: asyncio.Barrier | None, stdout: str) -> None:
        super().__init__({"ps": FakeCommand(CommandResult(0, stdout, "", 0, False), {})})
        self.barrier = barrier

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        if self.barrier is not None:
            await self.barrier.wait()
        return await super().run(cmd, timeout=timeout)


async def siblings(
    log: EventLog, outputs: Mapping[str, str], order: str, overlap: bool
) -> tuple[dict[str, CommandResult], dict[str, Gated]]:
    """One launch in which sibling adversaries on identical machines run the same command, in ``order``."""
    ctx = Context(log=log)
    barrier = asyncio.Barrier(2) if overlap else None
    envs = {name: Gated(barrier, stdout) for name, stdout in outputs.items()}

    async def adversary(name: str) -> CommandResult:
        async with ctx.scope(f"adversary:{name}", opened(envs[name], f"validate/0/0/adversary:{name}/0/0")):
            install_env(ctx, envs[name])
            return await ctx.emit(RunCommand("ps", 5))

    async with ctx.scope("workpiece"):
        if overlap:
            results = await asyncio.gather(*map(adversary, order))
        else:
            results = [await adversary(name) for name in order]
    await ctx.drain()
    return dict(zip(order, results, strict=True)), envs


# Regression: siblings whose readiness order reversed on resume received each other's output, and siblings
# whose identical commands overlapped held although their results agreed.
@pytest.mark.parametrize(
    ("recorded", "fresh", "overlap"),
    [
        ({"a": "a\n", "b": "b\n"}, {"a": "a 2\n", "b": "b 2\n"}, False),
        ({"a": "/workspace\n", "b": "/workspace\n"}, {"a": "/workspace\n", "b": "/workspace\n"}, True),
    ],
    ids=["different results, reversed order", "identical results, overlapping"],
)
async def test_each_owner_replays_its_own_recording_in_whatever_order_its_siblings_resume(
    recorded: dict[str, str], fresh: dict[str, str], overlap: bool
) -> None:
    first = EventLog()
    results, _ = await siblings(first, recorded, "ab", overlap)
    assert {name: result.stdout for name, result in results.items()} == recorded
    resumed, envs = await siblings(EventLog(replay_from=first.records), fresh, "ba", overlap)
    assert {name: result.stdout for name, result in resumed.items()} == recorded
    assert all(env.executed == ["ps"] for env in envs.values())


async def attempt(log: EventLog, env: FakeEnv, commands: Sequence[str]) -> list[CommandResult]:
    """One attempt at the owner ``author/0`` that runs ``commands`` in order and returns normally."""
    ctx = Context(log=log)
    try:
        async with ctx.scope("author", opened(env)):
            install_env(ctx, env)
            return [await ctx.emit(RunCommand(command, 5)) for command in commands]
    finally:
        await ctx.drain()


def scripted(**commands: int) -> Tracked:
    """A machine whose commands exit with the given statuses and report their own name."""
    return Tracked({name: FakeCommand(CommandResult(code, name, "", 0, False), {}) for name, code in commands.items()})


async def test_a_session_that_diverges_from_its_recording_holds_before_the_new_request_runs() -> None:
    log = EventLog()
    await attempt(log, scripted(x=0, y=0), ["x", "y"])
    resumed = scripted(x=0, y=0, z=0)
    with pytest.raises(Unreproducible, match="another request at position 1"):
        await attempt(EventLog(replay_from=log.records), resumed, ["x", "z"])
    assert resumed.executed == ["x"]


async def test_a_clean_return_before_the_recorded_history_ends_holds_but_an_interrupted_one_does_not() -> None:
    log = EventLog()
    await attempt(log, scripted(x=0, y=0), ["x", "y"])
    with pytest.raises(Unreproducible, match="returned before its recorded history ended"):
        await attempt(EventLog(replay_from=log.records), scripted(x=0, y=0), ["x"])
    interrupted = Tracked({"x": FakeCommand(CommandResult(0, "x", "", 0, False), {}), "y": Restart("lost")})
    with pytest.raises(Restart, match="lost"):
        await attempt(EventLog(replay_from=log.records), interrupted, ["x", "y"])


@pytest.mark.parametrize("y_exit", [0, 1], ids=["same", "changed"])
async def test_a_reconstruction_interrupted_inside_the_known_prefix_leaves_the_rest_to_verify(y_exit: int) -> None:
    """Attempt 1 established x, y; attempt 2 died re-applying x; attempt 3 must still verify y and serve it."""
    log = EventLog()
    await attempt(log, scripted(x=0, y=0), ["x", "y"])
    resumed = EventLog(replay_from=log.records)
    with pytest.raises(Restart):
        await attempt(resumed, FakeEnv("image", {"x": Restart("machine died")}), ["x", "y"])
    third = FakeEnv(
        "image",
        {
            "x": FakeCommand(CommandResult(0, "x", "", 0, False), {}),
            "y": FakeCommand(CommandResult(y_exit, "y 3", "", 0, False), {}),
        },
    )
    if y_exit:
        with pytest.raises(Unreproducible, match="exited 0 when recorded but 1 here"):
            await attempt(resumed, third, ["x", "y"])
    else:
        assert [result.stdout for result in await attempt(resumed, third, ["x", "y"])] == ["x", "y"]


PATH = PurePosixPath("/workspace/status")


async def reads(log: EventLog, env: FakeEnv, contents: Sequence[bytes | None]) -> list[bytes | str]:
    """Read ``PATH`` once per entry, which a guest process sets (or removes) just before; return what was served."""
    ctx = Context(log=log)
    served: list[bytes | str] = []
    async with ctx.scope("author", opened(env)):
        install_env(ctx, env)
        for content in contents:
            env.files.pop(PATH, None) if content is None else env.files.__setitem__(PATH, content)
            try:
                served.append(await ctx.emit(ReadFile(PATH)))
            except OSError as error:
                served.append(str(error))
    await ctx.drain()
    return served


async def test_a_handled_failure_is_an_outcome_that_replays_as_the_same_failure() -> None:
    log = EventLog()
    original = await reads(log, FakeEnv("image", {}), (None, b"B"))
    (failed,) = (record.event for record in log.records if isinstance(record.event, Failed))
    assert failed.category == errno.errorcode[errno.ENOENT] and original == [failed.message, b"B"]
    assert await reads(EventLog(replay_from=log.records), FakeEnv("image", {}), (None, b"B")) == original
    with pytest.raises(Unreproducible, match="failed with ENOENT when recorded but succeeded here"):
        await reads(EventLog(replay_from=log.records), FakeEnv("image", {}), (b"B", b"B"))


async def test_a_changed_sandbox_definition_holds_when_the_owner_opens() -> None:
    log = EventLog()
    await attempt(log, scripted(x=0), ["x"])
    changed = EventLog(replay_from=log.records)
    with pytest.raises(Unreproducible, match="another sandbox definition"):
        await attempt(changed, FakeEnv("other image", {}), ["x"])
    assert changed.records == log.records


ENV = (Fingerprint("env", "definition", ReplayPolicy.REEXECUTE),)
READ = ReadFile(PATH)
OPEN = Record(1, 0.0, ScopeEntered("author"), opened=Opened("author/0", "definition"))


@pytest.mark.parametrize(
    ("rows", "issue"),
    [
        ((Record(2, 0.0, READ, attempt=7),), LogFailure),
        ((Record(2, 0.0, READ, attempt=1), Record(3, 0.0, READ, attempt=1)), LogFailure),
        ((Record(2, 0.0, Resolved(READ, ENV, b"x"), answers=1),), LogFailure),
        (
            (Record(2, 0.0, READ, attempt=1), Record(3, 0.0, Resolved(RunCommand("x", 5), ENV, None), answers=2)),
            LogFailure,
        ),
        (
            (Record(2, 0.0, READ, attempt=1), *(Record(n, 0.0, Resolved(READ, ENV, b"x"), answers=2) for n in (3, 4))),
            LogFailure,
        ),
        ((Record(2, 0.0, ScopeEntered("author"), opened=Opened("author/0", "other")),), LogFailure),
    ],
    ids=[
        "no open row",
        "second request while one is unanswered",
        "outcome of no request",
        "outcome of another request",
        "two outcomes",
        "owner reopened on another definition",
    ],
)
def test_a_journal_whose_sandbox_rows_break_ownership_is_refused(
    rows: tuple[Record, ...], issue: type[Exception]
) -> None:
    with pytest.raises(issue):
        Owners((OPEN, *rows))
