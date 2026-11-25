# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

from rolloutengine.models import GradedRollout, Rollout
from rolloutengine.test_support import arithmetic_task, rollout_spec
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.runtime.resources import inline_resource

from taskforge.effects.context import Context
from taskforge.effects.events import Event, Resolved
from taskforge.effects.log import EventLog, JsonlSink, Record
from taskforge.llm.policy import LLMPolicy
from taskforge.model.controls import Candidate
from taskforge.model.llm import Completion, Message, Role, ToolSchema
from taskforge.resources.clock import Clock, Now, now
from taskforge.resources.env import (
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
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.ids import Ids, NewId, new_id
from taskforge.resources.inference import FakeInference, InferenceClient, InferenceRequest, complete
from taskforge.resources.rollouts import SolverRollout


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
        (Message(Role.USER, "solve"),),
        (ToolSchema("shell", "Run command", {"type": "object", "required": ("command",)}),),
        LLMPolicy(temperature=1),
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


@dataclass(frozen=True)
class LegacyRollout:
    """``Rollout`` with its attempt ID under the field name that older event logs store."""

    run_id: str
    conversations: tuple[Any, ...]
    stop_reason: str
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, bytes] = field(default_factory=dict)
    failure: Any = None


@dataclass(frozen=True)
class LegacyGradedRollout:
    rollout: LegacyRollout
    grade: GradeResult


async def test_jsonl_reads_and_extends_a_log_that_stores_the_attempt_id_as_run_id(tmp_path: Path) -> None:
    spec, grade = rollout_spec(arithmetic_task()), GradeResult(Outcome.GRADED, 1)
    legacy = Resolved(
        SolverRollout(spec, 1, 0), (), LegacyGradedRollout(LegacyRollout("trial", (), "stop"), grade), True
    )
    path = tmp_path / "results.jsonl"
    # A resumed workpiece logs the same replayed rollout again, which the log stores as a reference.
    await JsonlSink(str(path)).write((Record(1, 0.0, legacy), Record(2, 0.0, legacy)))
    assert '"ref"' in path.read_text().splitlines()[1]

    resolved = Resolved(SolverRollout(spec, 1, 0), (), GradedRollout(Rollout("trial", (), "stop"), grade), True)
    sink = JsonlSink(str(path))
    assert [record.event for record in await sink.read()] == [resolved, resolved]
    await sink.write((Record(3, 0.0, resolved),))
    assert [record.event for record in await JsonlSink(str(path)).read()] == [resolved] * 3
