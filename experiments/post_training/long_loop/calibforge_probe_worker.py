# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run released CalibForge setup and verification in fresh Linux sandboxes."""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from harbor.host_capabilities import HostCapabilities
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from shellbox.backends.daytona.connection import DaytonaConnection
from skyrl_train.dataset.harbor_import import restore_harbor_trial_task
from taskcompendium.models import TaskSpec


async def probe(inputs: Path, output: Path) -> None:
    rows = json.loads(inputs.read_text())
    report = {"passed": False, "results": []}
    connection = DaytonaConnection(os.environ["DAYTONA_RL_API_KEY"], "https://app.daytona.io/api", "us")
    try:
        for i, row in enumerate(rows):
            task = TaskSpec.model_validate(row)
            root = output.parent / f"task-{i}"
            restore_harbor_trial_task(task, root)
            config = TrialConfig.model_validate(
                {
                    "task": {"path": str(root)},
                    "agent": {"name": "nop"},
                    "environment": {"type": "daytona", "delete": True},
                    "trials_dir": str(output.parent / "trials"),
                    "trial_name": f"calibforge-shared-{i}",
                    "trial_attempt_timeout_sec": 1200,
                    "trial_cleanup_grace_sec": 120,
                }
            )
            trial = await Trial.create(config, host_capabilities=HostCapabilities(daytona=connection))
            if trial.task.config.environment.setup is None:
                raise ValueError("Release setup hook was not preserved")
            started = time.monotonic()
            result = await trial.run()
            entry = {
                "task_id": task.id,
                "elapsed_seconds": time.monotonic() - started,
                "result": result.model_dump(mode="json"),
            }
            report["results"].append(entry)
            if result.exception_info is not None or result.verifier_result is None:
                raise RuntimeError(f"CalibForge runtime did not verify: {result.exception_info}")
            if result.verifier_result.rewards is None or "reward" not in result.verifier_result.rewards:
                raise RuntimeError("CalibForge verifier produced no score")
        report["passed"] = True
    finally:
        output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(probe(args.inputs, args.output))
