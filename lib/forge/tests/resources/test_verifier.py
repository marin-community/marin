# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import override

import pytest
from lib.forge.tests.resources.test_rollouts_grading import FakeEngine
from rolloutengine.models import GradedRollout, ModelRequest, ModelTurn, Rollout, TraceMode
from rolloutengine.test_support import arithmetic_task, machine_runtime, rollout_limits
from shellbox.machine import (
    Backend,
    Command,
    ExecutionRequirements,
    Machine,
    MachineFactory,
    MachineSpec,
    MachineTerminated,
    Result,
)
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ResourceGroups, ScriptGrader, StdoutReward
from taskcompendium.runtime.resources import inline_resource

from forge.effects.context import Context
from forge.effects.issues import Restart
from forge.effects.log import EventLog
from forge.harness.tools.adversary import AdversaryClaim
from forge.lowering import (
    GRADING_MACHINE,
    LoweringSettings,
    lower_with_settings,
    requirements,
    rollout_engine,
)
from forge.model.controls import Candidate, RewardBand
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.rollouts import RolloutResource, SolverRollout
from forge.resources.sandbox.factories import shellsim_factory
from forge.resources.verifier import RoundVerifier
from forge.stages.validate import claim_result

SCRIPT = b"""
from pathlib import Path
paths = [Path(root) / 'left-behind' for root in ['/app', '/workspace', '/tests', '/tmp']]
clean = all(not path.exists() for path in paths)
for path in paths:
    path.write_text('must not survive')
print(int(clean and Path('/app/answer.txt').read_text() == '12'))
"""


class CountedMachine(Machine):
    def __init__(self, machine: Machine, die: bool, failure: RuntimeError | None = None) -> None:
        self.machine, self.die, self.closed = machine, die, False
        self.failure = failure

    @override
    async def run(self, command: Command) -> Result:
        if self.die and command.argv == ("python3", "/tests/grade.py"):
            self.die = False
            raise self.failure or MachineTerminated("killed during grader") from RuntimeError(
                "Iris exec failed: 'NoneType' object has no attribute 'decode'"
            )
        return await self.machine.run(command)

    @override
    async def upload(self, source: Path, target: str) -> None:
        await self.machine.upload(source, target)

    @override
    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def reset(self) -> None:
        await self.machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    "rm -rf /app /workspace /tests /tmp /custom /rootfile; mkdir -p /app /workspace /tests /tmp",
                )
            )
        )

    @override
    async def close(self) -> None:
        self.closed = True
        await self.machine.close()


class CountedFactory(MachineFactory):
    backend = Backend.SHELLSIM

    def __init__(self, *, die: bool = False, failure: RuntimeError | None = None) -> None:
        self.factory = shellsim_factory()
        self.machines: list[CountedMachine] = []
        self.die = die
        self.failure = failure

    @override
    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.factory.check_environment(requirements, spec)

    @override
    async def create(self, spec: MachineSpec) -> Machine:
        # A controlled startup cost makes the avoided factory call measurable without cluster traffic.
        await asyncio.sleep(0.05)
        machine = CountedMachine(await self.factory.create(spec), self.die and not self.machines, self.failure)
        self.machines.append(machine)
        return machine


def setup(factory: CountedFactory) -> tuple[RoundVerifier, GradingResource]:
    engine = rollout_engine({"shellsim": factory}, command_timeout=1)
    holder = RoundVerifier(engine, 5)
    return holder, GradingResource(holder, "test", engine)


def event(script: bytes = SCRIPT) -> GradeCandidate:
    grader = ScriptGrader(
        argv=("python3", "/tests/grade.py"),
        reward=StdoutReward(),
        environment=requirements(image=None),
        answer_path="/app/answer.txt",
        timeout=5,
    )
    task = arithmetic_task(grader=grader, resources=ResourceGroups(verifier=(inline_resource("grade.py", script),)))
    spec = lower_with_settings(
        task,
        LoweringSettings(GRADING_MACHINE, GRADING_MACHINE, rollout_limits()),
        {"shellsim": shellsim_factory()},
    )
    return GradeCandidate(spec.model_copy(update={"trace": TraceMode.MESSAGES}), Candidate("12"))


async def twelve(request: ModelRequest) -> ModelTurn:
    return ModelTurn({"role": "assistant", "content": "12"}, (), (), None, "stop")


async def test_grades_and_solver_share_clean_machine_and_rehydrate_resources() -> None:
    factory = CountedFactory()
    pool, resource = setup(factory)
    durations = []
    try:
        for expected, candidate in [(1, event()), (1, event()), (0, event(SCRIPT.replace(b"== '12'", b"== '9'")))]:
            started = time.perf_counter()
            result = await resource.grade(candidate)
            durations.append(time.perf_counter() - started)
            assert result.reward == expected, result
        solver = RolloutResource(resource.engine, twelve, "test")
        result = await solver.run(SolverRollout(event().spec, 0, 0))
        assert result.grade.reward == 1
        assert result.grade.diagnostics["verifier_lease"] == 4
        assert len(factory.machines) == 1 and not factory.machines[0].closed
        print(
            f"grade seconds: cold={durations[0]:.6f}, warm={durations[1]:.6f}, "
            f"rehydrated={durations[2]:.6f}; startup=0.050"
        )
    finally:
        await pool.close()
    assert factory.machines[0].closed


@pytest.mark.parametrize(
    "failure",
    [
        None,
        RuntimeError(
            "Iris exec failed: invalid literal for int() with base 10: 'Internal error occurred: "
            "error executing command in container: stopped'"
        ),
    ],
)
async def test_dead_verifier_restarts_with_one_replacement(failure: RuntimeError | None) -> None:
    factory = CountedFactory(die=True, failure=failure)
    pool, resource = setup(factory)
    try:
        with pytest.raises(Restart):
            await resource.grade(event())
        assert factory.machines[0].closed
        assert (await resource.grade(event())).reward == 1
        assert (await resource.grade(event())).reward == 1
        assert len(factory.machines) == 2
    finally:
        await pool.close()


async def test_fresh_confirmation_and_replay_require_no_shared_machine_state() -> None:
    factory = CountedFactory()
    pool, resource = setup(factory)
    log = EventLog()
    candidate = event()
    claim = AdversaryClaim("passes", candidate.candidate, RewardBand(Outcome.GRADED, 1, 1))
    try:
        ctx = Context(log=log)
        ctx.resource(resource, resource.fingerprint)
        ctx.handle(GradeCandidate, lambda ev, res: grade(ev, res, notify=ctx.emit), uses=(resource,))
        first = await claim_result(ctx, candidate.spec, claim)
        await ctx.drain()
        assert first.met and first.grade is not None
        assert first.grade.diagnostics["verifier_fresh"] is True
        assert len(factory.machines) == 2 and factory.machines[1].closed
        await pool.close()
        replay = Context(log=EventLog(replay_from=log.records))
        replay.resource(resource, resource.fingerprint)
        replay.handle(GradeCandidate, lambda ev, res: grade(ev, res, notify=replay.emit), uses=(resource,))
        assert await claim_result(replay, candidate.spec, claim) == first
        await replay.drain()
        assert len(factory.machines) == 2
    finally:
        await pool.close()


async def test_spec_change_and_round_end_replace_machine() -> None:
    factory = CountedFactory()
    pool, resource = setup(factory)
    candidate = event()
    try:
        assert (await resource.grade(candidate)).reward == 1
        selection = candidate.spec.verifier_machine
        assert selection is not None
        changed = candidate.spec.model_copy(
            update={"verifier_machine": selection.model_copy(update={"memory_mb": 4096})}
        )
        assert (await resource.grade(replace(candidate, spec=changed))).reward == 1
        assert factory.machines[0].closed and len(factory.machines) == 2
        await pool.close()
        assert (await resource.grade(candidate)).reward == 1
        assert factory.machines[1].closed and len(factory.machines) == 3
    finally:
        await pool.close()


async def test_parallel_grades_are_serialized_on_one_machine() -> None:
    factory = CountedFactory()
    pool, resource = setup(factory)
    try:
        results = await asyncio.gather(*(resource.grade(event()) for _ in range(4)))
        assert all(result.reward == 1 for result in results)
        assert sorted(result.diagnostics["verifier_lease"] for result in results) == [1, 2, 3, 4]
        assert len(factory.machines) == 1
    finally:
        await pool.close()


async def test_fresh_adversary_result_overrides_shared_confirmation() -> None:
    shared = FakeEngine((GradedRollout(Rollout("shared", (), "stop"), GradeResult(Outcome.GRADED, 1)),))
    fresh = FakeEngine((GradedRollout(Rollout("fresh", (), "stop"), GradeResult(Outcome.GRADED, 0)),))
    resource = GradingResource(shared, "test", fresh)
    ctx = Context(log=EventLog())
    ctx.resource(resource, resource.fingerprint)
    ctx.handle(GradeCandidate, partial(grade, notify=ctx.emit), uses=(resource,))
    candidate = event()
    finding = await claim_result(
        ctx, candidate.spec, AdversaryClaim("passes", candidate.candidate, RewardBand(Outcome.GRADED, 1, 1))
    )
    await ctx.drain()
    assert not finding.met and finding.grade is not None and finding.grade.reward == 0
    assert len(shared.run_ids) == len(fresh.run_ids) == 1


async def test_candidate_files_and_staged_roots_are_replaced_between_grades() -> None:
    script = b"""
from pathlib import Path
answer = Path('/app/answer.txt').read_text()
expected = answer == 'first'
clean = not Path('/custom/leftover').exists()
Path('/custom').mkdir(exist_ok=True)
Path('/custom/leftover').write_text('must not survive')
valid = Path('/workspace/custom/output').read_text() == answer
rootfile = Path('/workspace/rootfile')
print(int(clean and valid and rootfile.exists() == expected and (not expected or rootfile.stat().st_size == 210000)))
"""
    candidate = event(script)
    task = candidate.spec.task.model_copy(
        update={
            "environment_requirements": requirements(image=None),
            "output_paths": ("/workspace/custom/output", "/workspace/rootfile"),
        }
    )
    spec = candidate.spec.model_copy(update={"task": task, "task_machine": machine_runtime(backend="shellsim")})
    factory = CountedFactory()
    pool, resource = setup(factory)
    try:
        first = Candidate(
            "first", (inline_resource("custom/output", b"first"), inline_resource("rootfile", b"present" * 30000))
        )
        second = Candidate("second", (inline_resource("custom/output", b"second"),))
        assert (await resource.grade(GradeCandidate(spec, first))).reward == 1
        assert (await resource.grade(GradeCandidate(spec, second))).reward == 1
        assert len(factory.machines) == 3  # two independent task machines, one shared verifier
        assert sum(not machine.closed for machine in factory.machines) == 1
    finally:
        await pool.close()


async def test_backend_without_reset_is_recreated() -> None:
    engine = rollout_engine({"shellsim": shellsim_factory()}, command_timeout=1)
    holder = RoundVerifier(engine, 5)
    resource = GradingResource(holder, "test")
    try:
        assert (await resource.grade(event())).reward == 1
        original = holder.machine
        assert original is not None
        assert (await resource.grade(event())).reward == 1
        assert holder.machine is not original
        with pytest.raises(RuntimeError):
            await original.run(Command(("true",)))
    finally:
        await holder.close()


async def test_dead_machine_during_reset_is_recreated(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = CountedFactory()
    holder, resource = setup(factory)
    try:
        assert (await resource.grade(event())).reward == 1

        async def lost() -> None:
            raise MachineTerminated("lost before reset")

        monkeypatch.setattr(factory.machines[0], "reset", lost)
        assert (await resource.grade(event())).reward == 1
        assert len(factory.machines) == 2 and factory.machines[0].closed
    finally:
        await holder.close()


async def test_solver_machines_run_concurrently_before_shared_grading() -> None:
    factory = CountedFactory()
    pool, _ = setup(factory)
    candidate = event()
    task = candidate.spec.task.model_copy(update={"environment_requirements": requirements(image=None)})
    spec = candidate.spec.model_copy(update={"task": task, "task_machine": machine_runtime(backend="shellsim")})
    started = 0
    ready = asyncio.Event()

    async def model(request: ModelRequest) -> ModelTurn:
        nonlocal started
        started += 1
        if started == 3:
            ready.set()
        await asyncio.wait_for(ready.wait(), 5)
        return await twelve(request)

    try:
        async with asyncio.TaskGroup() as group:
            trials = [group.create_task(pool.run_task(spec, model)) for _ in range(3)]
        assert all(trial.result().grade.reward == 1 for trial in trials)
        assert sorted(trial.result().grade.diagnostics["verifier_lease"] for trial in trials) == [1, 2, 3]
        assert len(factory.machines) == 4
        assert sum(not machine.closed for machine in factory.machines) == 1
    finally:
        await pool.close()
    assert all(machine.closed for machine in factory.machines)


async def test_shared_verifier_cleanup_warns_and_releases_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = CountedFactory()
    holder, resource = setup(factory)
    notes = []
    holder.notify = notes.append
    assert (await resource.grade(event())).reward == 1
    machine = holder.machine
    assert machine is not None
    close = machine.close

    async def failed_close() -> None:
        await close()
        raise RuntimeError("controller unavailable during close")

    monkeypatch.setattr(machine, "close", failed_close)
    await holder.close()
    assert holder.machine is None and holder.key is None
    assert len(notes) == 1
