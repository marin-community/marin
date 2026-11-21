# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from functools import partial
from pathlib import Path, PurePosixPath
from typing import override

import pytest
from lib.taskforge.tests.effect_fakes import local_records
from lib.taskforge.tests.resource_doubles import FakeCommand, FakeEnv, FakeInference
from rolloutengine.models import GradedRollout, Rollout
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import inline_resource

from taskforge.effects.context import Context
from taskforge.effects.events import LAUNCH_SCOPE, Event, Note, Resolved, event_digest
from taskforge.effects.issues import Unreproducible
from taskforge.effects.log import EventLog, JsonlSink, Record
from taskforge.llm.policy import LLMPolicy
from taskforge.model.controls import Candidate
from taskforge.model.llm import Completion, Message, Role, ToolSchema
from taskforge.resources.env import (
    CommandResult,
    Env,
    ReadFile,
    RunCommand,
    WriteFile,
    read_file,
    run_command,
    write_file,
)
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.ids import Ids, NewId, new_id
from taskforge.resources.inference import InferenceClient, InferenceRequest, complete
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import install_env


def configure(ctx: Context, env: Env, client: InferenceClient, ids: Ids) -> None:
    ctx.handle(RunCommand, partial(run_command, notify=ctx.notify), env)
    ctx.handle(WriteFile, partial(write_file, notify=ctx.notify), env)
    ctx.handle(InferenceRequest, partial(complete, notify=ctx.notify), client)
    ctx.handle(NewId, new_id, ids)


async def test_reload_reexecutes_environment_but_reuses_inference_and_ids(tmp_path: Path) -> None:
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
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    ctx = Context(log=EventLog(sink=sink))
    configure(ctx, env, client, Ids(lambda: "original"))
    await ctx.emit(WriteFile(path, b"\xff\x00"))
    assert (await ctx.emit(event)).content == "answer"
    assert await ctx.emit(RunCommand("build", 5)) == result
    assert await ctx.emit(NewId("author/0/id")) == "original"
    await ctx.drain()

    replay_env = FakeEnv("image", script)
    replay_client = FakeInference(())
    replay = Context(log=EventLog(replay_from=local_records(sink)))
    configure(replay, replay_env, replay_client, Ids(lambda: "new"))
    await replay.emit(WriteFile(path, b"\xff\x00"))
    assert (await replay.emit(event)).content == "answer"
    await replay.emit(RunCommand("build", 5.0))
    assert await replay.emit(NewId("author/0/id")) == "original"
    assert await replay.emit(NewId("author/1/id")) == "new"
    await replay.drain()
    assert not replay_client.requests
    assert replay_env.fingerprint() == env.fingerprint()
    assert await replay_env.read(path) == b"result"


async def test_jsonl_round_trips_resource_results_and_binary_payloads(tmp_path: Path) -> None:
    path = PurePosixPath("/workspace/file")
    spec = rollout_spec(arithmetic_task())
    grade = GradeResult(Outcome.GRADED, 1, diagnostics={"binary": b"\xff"})
    rollout = GradedRollout(Rollout("trial", (), "stop", outputs={"file": b"\xff\x00"}), grade)
    events: tuple[Event[object], ...] = (
        Resolved(ReadFile(path), (), b"\xff\x00"),
        Resolved(WriteFile(path, b"\xff\x00"), (), None),
        Resolved(SolverRollout(spec, 1, 0), (), rollout),
        Resolved(GradeCandidate(spec, Candidate("12", (inline_resource("workspace/file", b"\xff"),))), (), grade),
    )
    sink = JsonlSink(str(tmp_path / "results.jsonl"))
    await sink.write(tuple(Record(index, 0.0, event) for index, event in enumerate(events, 1)))
    restored = local_records(sink)
    assert tuple(record.event for record in restored) == events


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
async def test_a_reexecuted_command_serves_its_recording_and_chains_it(
    fresh: CommandResult, served: CommandResult | type[Exception], diverged: bool
) -> None:
    path = PurePosixPath("/workspace/out")
    recording = Context(log=(log := EventLog()))
    original = FakeEnv("image", {"build": FakeCommand(RECORDED, {path: b"built"})})
    configure(recording, original, FakeInference(()), Ids())
    await recording.emit(RunCommand("build", 5))
    await recording.drain()

    replay = Context(log=(replayed := EventLog(replay_from=log.records)))
    env = FakeEnv("image", {"build": FakeCommand(fresh, {path: b"built"}), "new": FakeCommand(fresh, {})})
    configure(replay, env, FakeInference(()), Ids())
    if isinstance(served, type):
        with pytest.raises(served, match=r"env\.run_command"):
            await replay.emit(RunCommand("build", 5))
    else:
        assert await replay.emit(RunCommand("build", 5)) == served
        assert env.fingerprint() == original.fingerprint()
        assert env.files[path] == b"built"
        # With no recording, the command's own result is the result.
        assert await replay.emit(RunCommand("new", 5)) == fresh
    await replay.drain()
    notes = [record.event.msg for record in replayed.records if isinstance(record.event, Note)]
    assert len(notes) == diverged


class Gated(FakeEnv):
    """Holds each command until its sibling has issued one too, so the two requests are in flight at once."""

    def __init__(self, barrier: asyncio.Barrier, stdout: str) -> None:
        super().__init__("image", {"ps": FakeCommand(CommandResult(0, stdout, "", 0, False), {})})
        self.barrier = barrier
        self.executed: list[str] = []

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        self.executed.append(cmd)
        await self.barrier.wait()
        return await super().run(cmd, timeout=timeout)


async def adversaries(
    log: EventLog, outputs: tuple[str, str]
) -> tuple[list[CommandResult | BaseException], list[Gated]]:
    """One launch in which two adversaries on identical machines run the same command at once."""
    ctx = Context(log=log)
    barrier = asyncio.Barrier(2)
    envs = [Gated(barrier, stdout) for stdout in outputs]

    async def adversary(name: str, env: Gated) -> CommandResult:
        async with ctx.scope(f"adversary:{name}"):
            install_env(ctx, env)
            return await ctx.emit(RunCommand("ps", 5))

    async with ctx.scope(LAUNCH_SCOPE):
        results = await asyncio.gather(*map(adversary, "ab", envs), return_exceptions=True)
    await ctx.drain()
    return results, envs


@pytest.mark.parametrize(
    ("outputs", "held"), [(("1\n", "1\n"), False), (("1\n", "2\n"), True)], ids=["same", "different"]
)
async def test_sibling_machines_with_one_history_never_take_each_others_results(
    outputs: tuple[str, str], held: bool
) -> None:
    """Sibling machines with one history share a replay key, and the record does not say whose result is whose. Equal
    results are served; different ones hold before either is used."""
    first = EventLog()
    recorded, _ = await adversaries(first, outputs)
    assert [result.stdout for result in recorded if isinstance(result, CommandResult)] == list(outputs)
    resolved = [r.event for r in first.records if isinstance(r.event, Resolved)]
    assert len({(event_digest(event.request), event.fingerprints) for event in resolved}) == 1
    resumed, envs = await adversaries(EventLog(replay_from=first.records), (outputs[1], outputs[0]))
    if held:
        assert all(isinstance(result, Unreproducible) for result in resumed)
        assert not any(env.executed for env in envs)
    else:
        assert [result.stdout for result in resumed if isinstance(result, CommandResult)] == list(outputs)
        assert all(env.executed == ["ps"] for env in envs)


async def test_each_launch_matches_the_occurrences_of_the_launches_before_it_in_order() -> None:
    """A guest process changes a file between identical reads. Each resumed launch re-dispatches the reads recorded
    before it, gets them in order, and reads the machine once beyond them."""
    path = PurePosixPath("/workspace/status")
    contents = (b"pending", b"done", b"failed")
    records: tuple[Record, ...] = ()
    reads = []
    for launch in range(len(contents)):
        ctx = Context(log=(log := EventLog(replay_from=records)))
        env = FakeEnv("image", {})
        ctx.handle(ReadFile, partial(read_file, notify=ctx.notify), env)
        async with ctx.scope(LAUNCH_SCOPE):
            served = []
            for content in contents[: launch + 1]:
                env.files[path] = content
                served.append(await ctx.emit(ReadFile(path)))
        await ctx.drain()
        reads.append(served)
        records = log.records
    assert reads == [list(contents[: launch + 1]) for launch in range(len(contents))]
