# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from pathlib import Path, PurePosixPath

from rolloutengine.models import GradedRollout, Rollout
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import inline_resource

from forge.effects.context import Context
from forge.effects.events import Event, Resolved
from forge.effects.log import EventLog, JsonlSink, Record
from forge.model.controls import Candidate
from forge.model.llm import Completion, Message, Sampling, ToolSchema
from forge.resources.clock import Clock, Now, now
from forge.resources.env import (
    CommandResult,
    Env,
    FakeCommand,
    FakeEnv,
    ReadFile,
    RunCommand,
    WriteFile,
    run_command,
    write_file,
)
from forge.resources.grading import GradeCandidate
from forge.resources.ids import Ids, NewId, new_id
from forge.resources.inference import FakeInference, InferenceClient, InferenceRequest, complete
from forge.resources.rollouts import SolverRollout


def note(event: Event[None]) -> None:
    pass


def configure(ctx: Context, env: Env, client: InferenceClient, clock: Clock, ids: Ids) -> None:
    for resource in (env, client, clock, ids):
        ctx.resource(resource, resource.fingerprint)
    ctx.handle(RunCommand, partial(run_command, notify=ctx.emit), uses=(env,))
    ctx.handle(WriteFile, partial(write_file, notify=ctx.emit), uses=(env,))
    ctx.handle(InferenceRequest, partial(complete, notify=ctx.emit), uses=(client,))
    ctx.handle(Now, now, uses=(clock,))
    ctx.handle(NewId, new_id, uses=(ids,))


async def test_reload_reexecutes_environment_but_reuses_inference_clock_and_ids(tmp_path: Path) -> None:
    path = PurePosixPath("/workspace/binary")
    result = CommandResult(0, "done", "", 0.1, False)
    script = {"build": FakeCommand(result, {path: b"result"})}
    env = FakeEnv("image", script, note)
    client = FakeInference((Completion("answer"),))
    event = InferenceRequest(
        (Message("user", "solve"),),
        (ToolSchema("shell", "Run command", {"type": "object", "required": ("command",)}),),
        Sampling(temperature=1),
    )
    sink = JsonlSink(str(tmp_path / "events.jsonl"))
    ctx = Context(log=EventLog(sink=sink))
    configure(ctx, env, client, Clock(lambda: 12.5), Ids(lambda: "original"))
    await ctx.emit(WriteFile(path, b"\xff\x00"))
    assert (await ctx.emit(event)).content == "answer"
    assert await ctx.emit(RunCommand("build", 5)) == result
    assert await ctx.emit(Now("author/0/start")) == 12.5
    assert await ctx.emit(NewId("author/0/id")) == "original"
    await ctx.drain()

    replay_env = FakeEnv("image", script, note)
    replay_client = FakeInference(())
    replay = Context(log=EventLog(replay_from=await sink.read()))
    configure(replay, replay_env, replay_client, Clock(lambda: 99), Ids(lambda: "new"))
    await replay.emit(WriteFile(path, b"\xff\x00"))
    assert (await replay.emit(event)).content == "answer"
    await replay.emit(RunCommand("build", 5.0))
    assert await replay.emit(Now("author/0/start")) == 12.5
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
    restored = await sink.read()
    assert tuple(record.event for record in restored) == events
