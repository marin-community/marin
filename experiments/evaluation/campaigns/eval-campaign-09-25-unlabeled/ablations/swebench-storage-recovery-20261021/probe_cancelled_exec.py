# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether a cancelled Daytona command blocks later verifier commands."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


async def main():
    report = {}
    with tempfile.TemporaryDirectory(prefix="swe-cancel-probe-") as temporary:
        root = Path(temporary)
        environment_dir = root / "environment"
        environment_dir.mkdir()
        (environment_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        paths = TrialPaths(trial_dir=root / "trial")
        paths.mkdir()
        environment = DaytonaEnvironment(
            environment_dir=environment_dir,
            environment_name="swe-cancel-probe",
            session_id=f"swe-cancel-probe-{uuid4().hex[:12]}",
            trial_paths=paths,
            task_env_config=EnvironmentConfig(cpus=1, memory_mb=1024),
            ttl_minutes=10,
        )
        try:
            await environment.start(force_build=False)
            report["sandbox_id"] = environment._sandbox.id
            command = asyncio.create_task(environment.exec("sleep 45"))
            await asyncio.sleep(3)
            command.cancel()
            try:
                await command
            except asyncio.CancelledError:
                report["poll_cancelled"] = True
            try:
                result = await asyncio.wait_for(environment.exec("echo verifier-ready"), timeout=8)
                report["same_session"] = {"return_code": result.return_code, "stdout": result.stdout}
            except TimeoutError:
                report["same_session"] = "timed out after 8 seconds"
            environment._exec_session_id = None
            result = await asyncio.wait_for(
                environment.exec("ps -eo args | grep '[s]leep 45'; echo fresh-session-ready"), timeout=15
            )
            report["fresh_session"] = {"return_code": result.return_code, "stdout": result.stdout}
        finally:
            await environment.stop(delete=True)
            report["sandbox_deleted"] = True
            output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "cancelled-exec-probe.json"
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report))


if __name__ == "__main__":
    asyncio.run(main())
