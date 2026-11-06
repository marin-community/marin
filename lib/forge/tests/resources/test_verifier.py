# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import override

import pytest
from iris.rpc import controller_pb2
from lib.forge.tests.resources.test_rollouts_grading import FakeEngine
from lib.shellbox.tests.test_iris_machine import local_machine
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
from taskcompendium.models import ResourceGroups, StdoutReward
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
    script_grader,
)
from forge.model.controls import Candidate, RewardBand
from forge.resources.grading import GradeCandidate, GradingResource, grade
from forge.resources.rollouts import RolloutResource, SolverRollout
from forge.resources.sandbox.factories import MachineHost, shellsim_factory
from forge.resources.verifier import PROCESS_RESET, VerifierEngine, VerifierFactory, VerifierLease, VerifierPool
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


def setup(factory: CountedFactory) -> tuple[VerifierPool, GradingResource]:
    pool = VerifierPool(5)
    engine = rollout_engine(
        {"shellsim": factory, "verifier:shellsim": VerifierFactory(pool, factory)}, command_timeout=1
    )
    return pool, GradingResource(VerifierEngine(engine), "test", VerifierEngine(engine, fresh=True))


def event(script: bytes = SCRIPT) -> GradeCandidate:
    package = script_grader(
        ("python3", "/tests/grade.py"),
        StdoutReward(),
        environment=requirements(image=None),
        files=(inline_resource("grade.py", script),),
        answer_path="/app/answer.txt",
        timeout=5,
    )
    task = arithmetic_task(grader=package.grader, resources=ResourceGroups(verifier=package.resources))
    spec = lower_with_settings(
        task,
        LoweringSettings(MachineHost.IRIS, GRADING_MACHINE, GRADING_MACHINE, rollout_limits()),
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
        assert first.grade.diagnostics["verifier_lease"] == 2
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


def test_process_cleanup_preserves_supervisor_and_current_exec_only(monkeypatch: pytest.MonkeyPatch) -> None:
    # PID 2 was recycled: matching its old PID alone must not preserve the leaked process.
    # The active Python reset (92), timeout wrapper (91), and exec shell (90)
    # are all ancestors of the cleanup command and must survive it.
    processes = {1: (0, "100"), 2: (1, "999"), 3: (1, "300"), 90: (1, "900"), 91: (90, "910"), 92: (91, "920")}
    killed = []

    def stat(path: Path, *args: object, **kwargs: object) -> str:
        pid = int(path.parent.name)
        parent, started = processes[pid]
        fields = ["S", str(parent), *(["0"] * 17), started]
        return f"{pid} (command with spaces) " + " ".join(fields)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "glob", lambda path, pattern: iter(Path(f"/proc/{pid}/stat") for pid in processes))
        patch.setattr(Path, "read_text", stat)
        patch.setattr(os, "getpid", lambda: 92)
        patch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
        patch.setattr(sys, "argv", ["reset", json.dumps({"1": "100", "2": "200"})])
        exec(PROCESS_RESET, {})
    assert killed == [(2, signal.SIGKILL), (3, signal.SIGKILL)]


async def test_candidate_files_and_staged_roots_are_replaced_between_grades() -> None:
    script = b"""
from pathlib import Path
answer = Path('/app/answer.txt').read_text()
expected = answer == 'first'
clean = not Path('/custom/leftover').exists()
Path('/custom/leftover').write_text('must not survive')
print(int(clean and Path('/custom/output').read_text() == answer and Path('/rootfile').exists() == expected))
"""
    candidate = event(script)
    task = candidate.spec.task.model_copy(
        update={
            "environment_requirements": requirements(image=None),
            "output_paths": ("/custom/output", "/rootfile"),
        }
    )
    spec = candidate.spec.model_copy(update={"task": task, "task_machine": machine_runtime(backend="shellsim")})
    factory = CountedFactory()
    pool, resource = setup(factory)
    try:
        first = Candidate("first", (inline_resource("custom/output", b"first"), inline_resource("rootfile", b"present")))
        second = Candidate("second", (inline_resource("custom/output", b"second"),))
        assert (await resource.grade(GradeCandidate(spec, first))).reward == 1
        assert (await resource.grade(GradeCandidate(spec, second))).reward == 1
        assert len(factory.machines) == 3  # two independent task machines, one shared verifier
        assert sum(not machine.closed for machine in factory.machines) == 1
    finally:
        await pool.close()


async def test_reset_preserves_iris_capture_files_and_runtime_directory(tmp_path: Path) -> None:
    class SandboxRpc:
        def exec_in_container(
            self, request: controller_pb2.Controller.ExecInContainerRequest, timeout_ms: int
        ) -> controller_pb2.Controller.ExecInContainerResponse:
            # Execute the real Iris capture/readback scripts against isolated host
            # paths. No cleanup command may touch the host's actual /tmp or roots.
            if request.command[-1] == "id -u && id -un":
                return controller_pb2.Controller.ExecInContainerResponse(exit_code=0, stdout="0\nroot\n")
            command = [
                re.sub(r"/(tmp|workspace|tests|app)(?=/|[\s'\"]|$)", lambda match: str(tmp_path / match.group(1)), part)
                for part in request.command
            ]
            command[-1] = f"export PATH={shlex.quote(str(tmp_path / 'bin'))}:$PATH\n" + command[-1]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            return controller_pb2.Controller.ExecInContainerResponse(
                exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr
            )

    for name in ("tmp", "workspace", "tests", "app"):
        (tmp_path / name).mkdir()
    # Model mount points: removing their contents is allowed, unlinking a root is not.
    binary = tmp_path / "bin"
    binary.mkdir()
    protected = "|".join(shlex.quote(str(tmp_path / name)) for name in ("tmp", "workspace", "tests", "app"))
    (binary / "rm").write_text(
        f'#!/bin/sh\nfor target do\ncase "$target" in {protected}) '
        'echo "rm: Device or resource busy" >&2; exit 1;; esac\ndone\nexec /bin/rm "$@"\n'
    )
    (binary / "rm").chmod(0o755)
    inodes = {name: (tmp_path / name).stat().st_ino for name in ("tmp", "workspace", "tests", "app")}
    runtime = tmp_path / "tmp/.shellbox"
    runtime.mkdir()
    (runtime / "state").write_text("runtime")
    (tmp_path / "tmp/.shellbox-stale-out").write_text("capture")
    (tmp_path / "tmp/candidate-leftover").write_text("remove")
    (tmp_path / "workspace/previous").write_text("remove")
    machine, _ = local_machine(tmp_path, rpc=SandboxRpc())
    pool = VerifierPool(5)
    lease = VerifierLease(pool, machine, Backend.GVISOR, replace(machine.spec, workdir="/workspace"), False)
    try:
        await lease.reset()
        assert (await lease.run(Command(("printf", "after reset"), cwd="/workspace"))).stdout == b"after reset"
        assert (runtime / "state").read_text() == "runtime"
        assert (tmp_path / "tmp/.shellbox-stale-out").read_text() == "capture"
        assert not (tmp_path / "tmp/candidate-leftover").exists()
        assert not (tmp_path / "workspace/previous").exists()
        assert all((tmp_path / name).stat().st_ino == inode for name, inode in inodes.items())
        await lease.reset()
        assert (await lease.run(Command(("printf", "second reset"), cwd="/workspace"))).stdout == b"second reset"
    finally:
        await machine.close()
