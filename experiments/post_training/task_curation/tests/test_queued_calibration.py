# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consume real queued-trial artifacts through the curation calibration runner."""

import asyncio
import json
import shlex
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from daytona import DaytonaConnectionError
from harbor.environments import factory as environment_factory
from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.trial.config import TrialConfig
from harbor.trial.queue import TrialQueue
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.oracle_filter import calibrated_task, harbor_inspection
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from vendor.harbor.tests.unit.environments.transport import ProcessMachine


@pytest.mark.parametrize("outcome", ["pass", "wrong", "provider_failure"])
def test_calibration_consumes_queued_attempt_verdicts_and_infrastructure_failures(tmp_path, monkeypatch, outcome):
    native = tmp_path / "task"
    for name in ("environment", "solution", "tests"):
        (native / name).mkdir(parents=True)
    (native / "instruction.md").write_text("Write done to answer.txt.\n")
    (native / "task.toml").write_text(
        '[environment]\ndocker_image="fixture:1"\nallow_internet=false\nworkdir="/workspace"\n'
        "cpus=1\nmemory_mb=2048\nstorage_mb=2048\n[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=60\n"
    )
    solution = f"printf {'wrong' if outcome == 'wrong' else 'done'} > answer.txt\n"
    (native / "solution/solve.sh").write_text(solution)
    verifier_files = tmp_path / "guest/logs/verifier"
    (native / "tests/test.sh").write_text(
        'if test "$(cat answer.txt)" = done; then reward=1; else reward=0; fi\n'
        f'printf "%s" "$reward" > {shlex.quote(str(verifier_files / "reward.txt"))}\n'
        f'printf \'{{"status":"scored","reward":%s}}\' "$reward" > {shlex.quote(str(verifier_files / "verdict.json"))}\n'
    )
    row = prebuilt_task("task")
    task = TaskSpec.model_validate(json.loads(row["lowered_task_spec"])["task"])
    task = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={"oracle": (inline_resource("solution/solve.sh", solution.encode()),)}
            )
        }
    )
    inspection = harbor_inspection("tasks", [str(native)])[str(native)]
    mapping = {
        "task_path": str(native),
        "task_name": inspection["task_name"],
        "task_checksum": inspection["task_checksum"],
        "task_sha256": canonical_sha256(task.model_dump(mode="json")),
    }
    counters = tmp_path / "cgroup"
    counters.mkdir()
    for name, value in {
        "cpu.stat": "usage_usec 1500000\n",
        "memory.events": "oom_kill 0\n",
        "memory.peak": "67108864\n",
        "memory.current": "67108864\n",
    }.items():
        (counters / name).write_text(value)

    class Transport(ProcessMachine):
        def argument(self, value):
            return super().argument(value.replace("/sys/fs/cgroup", str(counters)))

    class Factory:
        async def create(self, spec):
            if outcome == "provider_failure":
                raise DaytonaConnectionError("Offline provider connection failed")
            return Transport(tmp_path / "guest", spec)

    class Environment(DaytonaEnvironment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.factory = Factory()

    monkeypatch.setattr(environment_factory, "_load_environment_class", lambda _: Environment)
    run_process = subprocess.run

    def execute(args, **kwargs):
        if "harbor" not in args or "start" not in args:
            return run_process(args, **kwargs)
        config = TrialConfig.model_validate_json(Path(args[-1]).read_text())
        asyncio.run(TrialQueue(n_concurrent=1).submit(config))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    calibrated = calibrated_task(task, mapping, tmp_path / "calibration", "daytona", {}, None)
    attempts = calibrated["calibration"]["attempts"]
    assert len(attempts) == 1
    receipt = attempts[0]
    assert receipt["result"]["task_checksum"] == mapping["task_checksum"]
    if outcome == "provider_failure":
        assert receipt["verdict"] is None
        assert receipt["oracle_exit_code"] is None
        assert receipt["result"]["exception_info"]["exception_type"] == "DaytonaConnectionError"
    else:
        assert receipt["verdict"]["reward"] == int(outcome == "pass")
        assert receipt["verdict"]["status"] == "scored"
        assert receipt["oracle_exit_code"] == 0
