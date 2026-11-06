# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Iris verifier reuse and replacement after mid-command termination."""

import asyncio
import json
import os
import time
import traceback
from pathlib import Path

import pytest
from rigging.filesystem.factory import open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.machine import MachineSpec
from taskcompendium.models import AnswerType, PlainText, Source, StdoutReward
from taskcompendium.runtime.resources import inline_resource

import forge
from forge.effects.issues import Restart
from forge.lowering import (
    NO_MACHINE,
    assemble,
    lower_with_settings,
    lowering_settings,
    requirements,
    rollout_engine,
    script_grader,
)
from forge.model.controls import Candidate
from forge.policy import load_policy
from forge.resources.grading import GradeCandidate, GradingResource
from forge.resources.sandbox.images import GRADER_BASE_IMAGE
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
        super().__init__(controller_url=os.environ["IRIS_CONTROLLER_URL"])
        self.created: list[IrisMachine] = []

    async def create(self, spec: MachineSpec) -> IrisMachine:
        machine = await super().create(spec)
        self.created.append(machine)
        return machine


@pytest.mark.cluster
@pytest.mark.timeout(1200)
@pytest.mark.skipif(
    not all(os.environ.get(key) for key in ("IRIS_CONTROLLER_URL", "CW_KEY_ID", "CW_KEY_SECRET", "PROBE_OUTPUT")),
    reason="requires a live Iris job, storage credentials, and artifact destination",
)
async def test_verifier_reuse_and_replacement() -> None:
    configure_coreweave_s3()
    factory = Factory()
    engine = rollout_engine({"gvisor": factory}, command_timeout=30)
    pool = RoundVerifier(engine, 60)
    resource = GradingResource(pool, "live-probe")
    policy = load_policy(Path(forge.__file__).parent / "policy/defaults")
    settings = lowering_settings(policy)
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
        package = script_grader(
            ("python3", "/tests/grade.py"),
            StdoutReward(),
            environment=requirements(image=GRADER_BASE_IMAGE, workdir="/app"),
            files=(inline_resource("grade.py", script),),
            answer_path="/app/answer.txt",
            cwd="/app",
            timeout=180,
        )
        task = assemble(
            "live-lease-probe",
            "Answer 12",
            AnswerType.TEXT,
            PlainText(),
            package,
            Source(dataset="forge-probe", revision="1", row="1", importer_revision="1"),
            environment=NO_MACHINE,
        )
        return GradeCandidate(lower_with_settings(task, settings, {"gvisor": factory}), Candidate("12"))

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
        results["passed"] = True
        save()
    except Exception:
        results["error"] = traceback.format_exc()
        save()
        raise
    finally:
        await pool.close()
