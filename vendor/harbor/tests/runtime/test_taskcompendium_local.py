# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify real TaskSpec grader provisioning on a privileged LOCAL CPU worker."""

import json
import os
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory
from taskcompendium.harbor.adapter import write_harbor_task
from taskcompendium.models import TaskSpec, verifyit_spec
from taskcompendium.pipeline.resource_calibration import INITIAL_REQUIREMENTS

from experiments.post_training.task_curation.tests.review_server import JudgeReply, running_review_server
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial

pytestmark = pytest.mark.runtime
FIXTURES = Path(__file__).parents[4] / "experiments/post_training/task_curation/tests/fixtures/harbor_task_specs"


@pytest.mark.parametrize("name", ["tau-rlvr1-repaired", "tau-rlvr2-repaired", "language-mixing"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
@pytest.mark.timeout(600)
@pytest.mark.asyncio
async def test_original_judge_contract_uses_real_local_disk_and_cgroup_limits(
    tmp_path, monkeypatch, pytestconfig, name, reward
):
    task = TaskSpec.model_validate_json((FIXTURES / f"{name}.json").read_text())
    original = task.model_dump_json()
    evidence = Path(os.environ.get("IRIS_OUTPUT_DIR", str(tmp_path))) / f"{name}-{int(reward)}"
    evidence.mkdir(parents=True)
    directory = evidence / "task"
    write_harbor_task(task, directory, INITIAL_REQUIREMENTS)
    response = evidence / "response.json"
    response.write_text(json.dumps([{"role": "assistant", "content": "A provisioning qualification response."}]))
    create = LocalMachineFactory.create
    allocations = []

    async def measured_create(factory, spec):
        assert factory.cgroup_parent is not None
        before = set(factory.cgroup_parent.glob("shellbox-*"))
        machine = await create(factory, spec)
        try:
            groups = set(factory.cgroup_parent.glob("shellbox-*")) - before
            assert len(groups) == 1
            group = groups.pop()
            quota, period = map(int, (group / "cpu.max").read_text().split())
            assert quota == spec.cpus * period
            assert (group / "memory.max").read_text().strip() == str(spec.memory_mb * 1024 * 1024)
            images = list(factory.disk_pool.root.glob("sandbox-*/disk.img"))
            assert len(images) == 1
            assert images[0].stat().st_size == spec.storage_mb * 1024 * 1024
            allocations.append((spec.cpus, spec.memory_mb, spec.storage_mb))
        except BaseException:
            await machine.close()
            raise
        return machine

    monkeypatch.setattr(LocalMachineFactory, "create", measured_create)
    judge = verifyit_spec(task.grader)
    with running_review_server() as server:
        server.judge_replies.append(
            JudgeReply(next(label for label, score in judge.label_scores.items() if score == reward))
        )
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        worker = {"disk_root": str(tmp_path / "disks"), "model_endpoint": endpoint, "model_name": judge.model}
        parent = pytestconfig.getoption("--local-cgroup-parent")
        if parent is not None:
            worker["cgroup_parent"] = parent
        trial = await Trial.create(
            TrialConfig(
                task=TaskConfig(path=directory),
                trial_name=name,
                trials_dir=evidence / "trials",
                agent=AgentConfig(name="task-spec-response", kwargs={"response_path": str(response)}),
                environment=EnvironmentConfig(kwargs=worker),
            )
        )
        result = await trial.run()
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}
        assert len(server.requests) == 1
        assert server.requests[0]["model"] == judge.model
        assert server.requests[0]["temperature"] == judge.temperature
    assert allocations == [(1, 2048, 2048)]
    assert not list((tmp_path / "disks").glob("sandbox-*"))
    assert (trial.paths.verifier_dir / "verdict.json").is_file()
    assert task.model_dump_json() == original
