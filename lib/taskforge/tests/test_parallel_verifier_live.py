# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure one workpiece's concurrent solver machines and shared verifier on Iris."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from lib.taskforge.tests.test_verifier_live import Factory
from rigging.filesystem.factory import open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rolloutengine.models import ModelRequest, ModelTurn, TraceMode
from rolloutengine.test_support import arithmetic_task
from shellbox.backends.iris.machine import IrisMachine
from shellbox.machine import MachineSpec
from taskcompendium.models import ResourceGroups, ScriptGrader, StdoutReward
from taskcompendium.runtime.resources import inline_resource

import taskforge
from taskforge.lowering import lower_with_settings, lowering_settings, requirements, rollout_engine
from taskforge.policy import GRADER_BASE_IMAGE, load_policy
from taskforge.resources.verifier import RoundVerifier


@pytest.mark.cluster
@pytest.mark.timeout(2400)
@pytest.mark.skipif(not os.environ.get("PROBE_OUTPUT"), reason="requires live Iris and storage")
async def test_three_solver_machines_share_one_verifier() -> None:
    configure_coreweave_s3()
    factory = Factory()
    samples: list[int] = []
    original_create = factory.create

    async def counted_create(spec: MachineSpec) -> IrisMachine:
        machine = await original_create(spec)
        samples.append(sum(not item._closed for item in factory.created))
        return machine

    factory.create = counted_create
    settings = lowering_settings(load_policy(Path(taskforge.__file__).parent / "policy/defaults"))
    settings = replace(
        settings,
        task_machine=replace(settings.task_machine, startup_timeout=1500),
        verifier_machine=replace(settings.verifier_machine, startup_timeout=1500),
    )
    task = arithmetic_task(
        grader=ScriptGrader(
            argv=("python3", "/tests/grade.py"),
            reward=StdoutReward(),
            environment=requirements(image=GRADER_BASE_IMAGE),
            answer_path="/workspace/answer.txt",
            timeout=60,
        ),
        resources=ResourceGroups(verifier=(inline_resource("grade.py", b"print(1)\n"),)),
    ).model_copy(update={"environment_requirements": requirements(image=GRADER_BASE_IMAGE)})
    spec = lower_with_settings(task, settings, {"gvisor": factory}).model_copy(update={"trace": TraceMode.MESSAGES})
    pool = RoundVerifier(rollout_engine({"gvisor": factory}, command_timeout=30), 60)
    arrived = 0
    simultaneous = 0
    gate = asyncio.Event()
    result: dict[str, object] = {"passed": False}

    async def model(request: ModelRequest) -> ModelTurn:
        nonlocal arrived, simultaneous
        arrived += 1
        simultaneous = max(simultaneous, arrived)
        if arrived == 3:
            gate.set()
        try:
            await asyncio.wait_for(gate.wait(), 120)
        except TimeoutError:
            gate.set()
        arrived -= 1
        return ModelTurn({"role": "assistant", "content": "12"}, (), (), None, "stop")

    try:
        async with asyncio.TaskGroup() as group:
            runs = [group.create_task(pool.run_task(spec, model, run_id=f"parallel-{i}")) for i in range(3)]
        result.update(
            simultaneous_solvers=simultaneous,
            peak_live_machines=max(samples),
            created=len(factory.created),
            live_after_trials=sum(not machine._closed for machine in factory.created),
            leases=sorted(run.result().grade.diagnostics["verifier_lease"] for run in runs),
        )
        assert all(run.result().grade.reward == 1 for run in runs)
        assert len(factory.created) == 4 and result["live_after_trials"] == 1
        if os.environ.get("PROBE_EXPECT_PARALLEL") == "1":
            assert simultaneous == 3 and max(samples) == 4, result
        result["passed"] = True
    finally:
        await pool.close()
        result["live_after_close"] = sum(not machine._closed for machine in factory.created)
        result["machines"] = [str(machine.job.job_id) for machine in factory.created]
        with open_url(os.environ["PROBE_OUTPUT"], "wt") as stream:
            json.dump(result, stream)
        print(json.dumps(result), flush=True)
