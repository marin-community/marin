# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Iris verifier reuse and replacement after mid-command termination."""

import asyncio
import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest
from rigging.filesystem.factory import open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from shellbox.backends.iris.machine import ExecOutput, IrisMachine, IrisMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Command, ExitReason, MachineSpec
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    StdoutReward,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource

import forge
from forge.effects.issues import Restart
from forge.lowering import (
    lower_with_settings,
    lowering_settings,
    requirements,
    rollout_engine,
)
from forge.model.controls import Candidate
from forge.policy import GRADER_BASE_IMAGE, load_policy
from forge.resources.grading import GradeCandidate, GradingResource
from forge.resources.verifier import RoundVerifier

SCRIPT = b"""
from pathlib import Path
roots = ['/app', '/workspace', '/tests', '/tmp']
assert all(not (Path(root) / 'candidate-residue').exists() for root in roots)
assert Path('/app/answer.txt').read_text() == '12'
for root in roots:
    (Path(root) / 'candidate-residue').write_text('must be cleared')
print(1)
"""
KILL_SCRIPT = b"""
from pathlib import Path
import time
Path('/app/mid-command').write_text('running')
time.sleep(120)
print(1)
"""


class Factory(IrisMachineFactory):
    def __init__(self) -> None:
        super().__init__(controller_url=os.environ["IRIS_CONTROLLER_URL"], scheduling_timeout=1200)
        self.created: list[IrisMachine] = []

    @override
    async def create(self, spec: MachineSpec) -> IrisMachine:
        machine = await super().create(spec)
        self.created.append(machine)
        return machine


@pytest.mark.cluster
@pytest.mark.timeout(3300)
@pytest.mark.skipif(
    not all(os.environ.get(key) for key in ("IRIS_CONTROLLER_URL", "CW_KEY_ID", "CW_KEY_SECRET", "PROBE_OUTPUT")),
    reason="requires a live Iris job, storage credentials, and artifact destination",
)
async def test_verifier_reuse_and_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    configure_coreweave_s3()
    factory = Factory()
    engine = rollout_engine({"gvisor": factory}, command_timeout=30)
    pool = RoundVerifier(engine, 60)
    resource = GradingResource(pool, "live-probe")
    policy = load_policy(Path(forge.__file__).parent / "policy/defaults")
    settings = lowering_settings(policy)
    # A busy admission queue may take twenty minutes; the grader still gets 180 seconds.
    settings = replace(
        settings,
        verifier_machine=replace(settings.verifier_machine, startup_timeout=1500),
        task_machine=replace(settings.task_machine, startup_timeout=1500),
    )
    output = os.environ["PROBE_OUTPUT"]
    results: dict[str, object] = {"passed": False, "workdir": "/app", "grades": []}
    grades: list[dict[str, object]] = []
    results["grades"] = grades

    def save() -> None:
        results["machines"] = [str(machine.job.job_id) for machine in factory.created]
        with open_url(output, "wt") as stream:
            json.dump(results, stream)
        print(json.dumps(results), flush=True)

    def event(script: bytes) -> GradeCandidate:
        task = TaskSpec(
            id="live-lease-probe",
            context=ConversationInput(events=(TextMessage(role="user", content="Answer 12"),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=ScriptGrader(
                argv=("python3", "/tests/grade.py"),
                reward=StdoutReward(),
                environment=requirements(image=GRADER_BASE_IMAGE, workdir="/app"),
                answer_path="/app/answer.txt",
                cwd="/app",
                timeout=180,
            ),
            resources=ResourceGroups(verifier=(inline_resource("grade.py", script),)),
            source=Source(dataset="forge-probe", revision="1", row="1", importer_revision="1"),
        )
        spec = lower_with_settings(task, settings, {"gvisor": factory})
        spec = spec.model_copy(update={"limits": spec.limits.model_copy(update={"verifier_timeout": 1500})})
        return GradeCandidate(spec, Candidate("12"))

    try:
        for index in range(3):
            started = time.monotonic()
            grade = await resource.grade(event(SCRIPT))
            assert grade.reward == 1, grade
            grades.append(
                {
                    "iteration": index,
                    "seconds": time.monotonic() - started,
                    "reward": grade.reward,
                    "lease": grade.diagnostics["verifier_lease"],
                }
            )
            assert len(factory.created) == 1
            save()
        machine = factory.created[0]
        pending = asyncio.create_task(resource.grade(event(KILL_SCRIPT)))
        try:
            async with asyncio.timeout(180):
                while True:
                    try:
                        if await machine.read_file("/app/mid-command") == b"running":
                            break
                    except FileNotFoundError:
                        pass
                    await asyncio.sleep(1)
            print("KILLING_SANDBOX_MID_COMMAND " + str(machine.job.job_id), flush=True)
            await asyncio.to_thread(machine.job.cancel)
            try:
                await asyncio.wait_for(pending, 180)
            except Restart as error:
                results["restart"] = str(error)
            else:
                raise AssertionError("Killed verifier did not raise Restart")
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
        started = time.monotonic()
        recovered = await resource.grade(event(SCRIPT))
        assert recovered.reward == 1 and len(factory.created) == 2, recovered
        results["replacement_seconds"] = time.monotonic() - started
        results["replacement_lease"] = recovered.diagnostics["verifier_lease"]
        recovered_machine = factory.created[-1]
        calls = 0
        original_exec = IrisMachine._exec_sync

        def counted_exec(self: IrisMachine, argv: list[str], timeout: float | None = None) -> ExecOutput:
            nonlocal calls
            calls += 1
            return original_exec(self, argv, timeout)

        monkeypatch.setattr(IrisMachine, "_exec_sync", counted_exec)
        counts = []
        for command, expected in (("true", 4), ("printf out", 5), ("printf out; printf err >&2", 6)):
            calls = 0
            command_result = await recovered_machine.run(Command(("sh", "-c", command)))
            assert command_result.exit_code == 0 and calls == expected, (command_result, calls)
            counts.append(calls)
        results["exec_calls_before"] = [6, 7, 8]
        results["exec_calls_after"] = counts
        results["verifier_machine_count"] = len(factory.created)
        save()
        large = event(
            SCRIPT.replace(b"print(1)", b"assert Path('/workspace/large.bin').read_bytes() == b'x' * 210000\nprint(1)")
        )
        task = large.spec.task.model_copy(
            update={
                "environment_requirements": requirements(image=GRADER_BASE_IMAGE, workdir="/workspace"),
                "output_paths": ("/workspace/large.bin",),
            }
        )
        large_spec = lower_with_settings(task, settings, {"gvisor": factory})
        large_spec = large_spec.model_copy(
            update={"limits": large_spec.limits.model_copy(update={"verifier_timeout": 1500})}
        )
        large = GradeCandidate(large_spec, Candidate("12", (inline_resource("large.bin", b"x" * 210000),)))
        large_grade = await resource.grade(large)
        assert large_grade.reward == 1, large_grade
        results["candidate_bytes"] = 210000
        results["candidate_grade"] = large_grade.reward
        results["passed"] = True
        save()
    except Exception:
        results["error"] = traceback.format_exc()
        save()
        raise
    finally:
        await pool.close()


@pytest.mark.cluster
@pytest.mark.timeout(1800)
@pytest.mark.skipif(not os.environ.get("PROBE_OUTPUT"), reason="requires live Iris and storage")
async def test_nested_command_timeout_keeps_sandbox_alive() -> None:
    configure_coreweave_s3()
    factory = Factory()
    machine = await factory.create(MachineSpec(source=RegistryImage(GRADER_BASE_IMAGE), workdir="/workspace"))
    result: dict[str, object] = {"passed": False, "machine": str(machine.job.job_id)}
    try:
        command = await machine.run(Command(("sh", "-c", "printf before; timeout 1 sh -c 'sleep 5'"), timeout=3))
        result.update(exit_code=command.exit_code, reason=command.reason.value, stdout=command.stdout.decode())
        assert command.reason is ExitReason.TIMED_OUT and command.exit_code == 124
        assert command.stdout == b"before"
        assert (await machine.run(Command(("true",)))).exit_code == 0
        result["passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        await machine.close()
        with open_url(os.environ["PROBE_OUTPUT"], "wt") as stream:
            json.dump(result, stream)
        print(json.dumps(result), flush=True)
