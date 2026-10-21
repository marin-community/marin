# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Mini-SWE's remote deadline and subsequent verifier access on Daytona."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.errors import AgentTimeoutError


async def main():
    report = {}
    with tempfile.TemporaryDirectory(prefix="mini-deadline-probe-") as temporary:
        root = Path(temporary)
        environment_dir = root / "environment"
        environment_dir.mkdir()
        (environment_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        paths = TrialPaths(trial_dir=root / "trial")
        paths.mkdir()
        environment = DaytonaEnvironment(
            environment_dir=environment_dir,
            environment_name="mini-deadline-probe",
            session_id=f"mini-deadline-{uuid4().hex[:12]}",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(cpus=1, memory_mb=1024),
            ttl_minutes=10,
        )
        try:
            await environment.start(force_build=False)
            report["sandbox_id"] = environment._sandbox.id
            # Substitute only the external executable. Exercise the real Harbor
            # agent command, process supervisor, and Daytona transport unchanged.
            executable = root / "mini-swe-agent"
            executable.write_text("#!/bin/bash\nsleep 45\n")
            await environment.upload_file(executable, "/usr/local/bin/mini-swe-agent")
            await environment.exec("chmod +x /usr/local/bin/mini-swe-agent; mkdir -p /logs/agent /logs/verifier")
            agent = MiniSweAgent(logs_dir=root / "agent", model_name="hosted_vllm/probe")
            with environment.with_agent_timeout(12):
                try:
                    await asyncio.wait_for(agent.run("Probe", environment, AgentContext()), timeout=15)
                except AgentTimeoutError:
                    report["classified_agent_timeout"] = True
            assert report.get("classified_agent_timeout"), "Remote command did not end as AgentTimeoutError"
            result = await asyncio.wait_for(
                environment.exec("echo 1 > /logs/verifier/reward.txt; cat /logs/verifier/reward.txt"),
                timeout=8,
            )
            report["verifier_reward"] = result.stdout.strip()
            assert result.return_code == 0 and report["verifier_reward"] == "1"
            processes = await environment.exec("ps -eo args | grep '[s]leep 45' || true")
            report["remaining_agent_processes"] = processes.stdout.strip()
            assert not report["remaining_agent_processes"]
        finally:
            await environment.stop(delete=True)
            report["sandbox_deleted"] = True
            output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "mini-deadline-probe.json"
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))


if __name__ == "__main__":
    asyncio.run(main())
