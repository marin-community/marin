import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.factory import AgentFactory
from harbor.agents.oracle import OracleAgent
from harbor.environments.local import LocalEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig, TaskOS
from harbor.models.trial.config import AgentConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.resource_usage import resource_counters, write_resource_usage
from tests.unit.environments.test_local import ProcessTaskFactory


@pytest.mark.parametrize("exit_code", [0, 7])
async def test_restored_oracle_runs_source_solution_through_local_environment(tmp_path, monkeypatch, exit_code):
    task_dir = tmp_path / "task"
    for name in ("environment", "tests", "solution"):
        (task_dir / name).mkdir(parents=True)
    image = "example.test/oracle@sha256:" + "a" * 64
    (task_dir / "task.toml").write_text("[environment]\nallow_internet=false\n")
    (task_dir / "instruction.md").write_text("Write the answer in the workspace.")
    (task_dir / "tests/test.sh").write_text("true\n")
    (task_dir / "solution/solve.sh").write_text(f"printf solved > answer.txt\nexit {exit_code}\n")
    (task_dir / "environment/shellbox-images.json").write_text(
        json.dumps({"main": {"image": image, "workdir": "/workspace"}})
    )
    factory = ProcessTaskFactory(tmp_path / "guest")
    monkeypatch.setattr("harbor.environments.local.LocalMachineFactory", lambda **kwargs: factory)
    paths = TrialPaths(tmp_path / "trial")
    paths.agent_dir.mkdir(parents=True)
    agent = AgentFactory.create_agent_from_config(
        AgentConfig(name="oracle"), logs_dir=paths.agent_dir, task_dir=task_dir, trial_paths=paths
    )
    assert AgentFactory.get_agent_class("oracle") is OracleAgent
    environment = LocalEnvironment(
        environment_dir=task_dir / "environment", environment_name="oracle", session_id="trial",
        task_env_config=EnvironmentConfig(allow_internet=False), trial_paths=paths,
    )
    async with asyncio.TaskGroup() as services:
        await environment.start(False, service_tasks=services)
        try:
            await environment.ensure_dirs([EnvironmentPaths.agent_dir])
            await agent.setup(environment)
            await agent.run("Write the answer.", environment, AgentContext())
        finally:
            await environment.stop(True)
    assert (factory.root / "workspace/answer.txt").read_text() == "solved"
    assert (paths.agent_dir / "oracle.txt").is_file()
    assert (paths.agent_dir / "exit-code.txt").read_text() == str(exit_code)


async def test_oracle_cgroup_evidence_records_cpu_and_oom_counter_deltas(tmp_path):
    environment = SimpleNamespace(os=TaskOS.LINUX, exec_control=AsyncMock(side_effect=[
        SimpleNamespace(stdout="usage_usec 1000\noom_kill 4\n"),
        SimpleNamespace(stdout="usage_usec 1501000\noom_kill 6\n"),
    ]))
    initial = await resource_counters(environment)
    output = tmp_path / "resource-usage.json"
    await write_resource_usage(environment, output, initial, time.monotonic())
    evidence = json.loads(output.read_text())
    assert evidence["cpu_seconds"] == 1.5
    assert evidence["oom_kills"] == 2
