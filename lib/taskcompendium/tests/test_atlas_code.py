# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect stale stdio runtimes without rewriting CodeNet source fixtures."""

import base64
import json

import pytest
from shellbox.machine import DockerImage, ExitReason, MachineSpec, Result
from verifyit.spec import StdioSpec, spec_to_table

from taskcompendium.datasets import atlas_code
from taskcompendium.datasets.atlas_code import exit_status_result
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import CheckStatus, RawRow
from taskcompendium.runtime.resources import resource_bytes

from . import test_executable_ingestion
from .test_executable_ingestion import GradingMachine, GradingMachines

executable_row = test_executable_ingestion.executable_row
executable_task = test_executable_ingestion.executable_task


@pytest.mark.parametrize(
    "verdict, expected_status",
    [
        ({"status": "scored", "reward": 0.0, "detail": {"reason": "runtime_error", "returncode": 7}}, CheckStatus.PASS),
        ({"status": "scored", "reward": 1.0, "detail": {}}, CheckStatus.UNSUPPORTED),
        ({"status": "scored", "reward": 0.0, "detail": {"reason": "command_failed"}}, CheckStatus.UNSUPPORTED),
        ({"status": "infra_error", "reward": None, "detail": {"error": "worker unavailable"}}, CheckStatus.INFRA_ERROR),
    ],
)
async def test_exit_status_control_checks_installed_grader_and_preserves_source(
    executable_task, monkeypatch, verdict, expected_status
):
    original_run = GradingMachine.run

    async def installed_grader(self, command):
        if command.argv[0] != "python3":
            return await original_run(self, command)
        self.files["/logs/verifier/verdict.json"] = json.dumps(verdict).encode()
        return Result(0, b"", b"", False, False, ExitReason.EXITED)

    # Replace only the external machine's response to the private grader command.
    monkeypatch.setattr(GradingMachine, "run", installed_grader)
    parameters = json.loads(executable_task.verifier.parameters_json)
    parameters["min_cases"] = 2
    task = executable_task.model_copy(
        update={"verifier": executable_task.verifier.model_copy(update={"parameters_json": json.dumps(parameters)})}
    )
    original = task.model_dump_json()
    machines = GradingMachines()
    result = await exit_status_result(
        task,
        factory=machines,
        machine_spec=MachineSpec(DockerImage(task.verifier.environment_requirements.docker_image)),
        timeout=10,
    )
    assert result.status == expected_status
    assert task.model_dump_json() == original
    uploaded = machines.machines[0].files
    assert {path for path in uploaded if path.startswith("/tests/cases/")} == {
        "/tests/cases/input_0.txt",
        "/tests/cases/output_0.txt",
        "/tests/cases/input_1.txt",
        "/tests/cases/output_1.txt",
    }
    assert not any(path.startswith("/solution/") for path in uploaded)


def test_code_snapshot_roundtrip_preserves_private_cases_and_oracle():
    payloads = {
        "tests/cases/input_0.txt": b"3 4\r\n",
        "tests/cases/output_0.txt": b"7\n",
        "tests/cases/input_1.txt": b"-5 3\n",
        "tests/cases/output_1.txt": b"-2\n",
    }
    oracle = b"printf 'print(sum(map(int,input().split())))' > /app/solution.py\n"
    row = RawRow(
        id="sum",
        source=Source(dataset="open-thoughts/TaskTrove", revision="fixture-v1", row="1", importer_revision="1"),
        data={
            "converted": {
                "instruction": "Read two integers and print their sum. Write /app/solution.py.",
                "grader_spec": spec_to_table(StdioSpec(command="python3 /app/solution.py")),
                "data_files": {path: base64.b64encode(content).decode() for path, content in payloads.items()},
                "control_files": {"solution/solve.sh": base64.b64encode(oracle).decode()},
            }
        },
    )
    normalized = atlas_code.normalize(row, "test@sha256:" + "a" * 64, output_paths=("/app/solution.py",))
    assert isinstance(normalized, TaskSpec)
    task = TaskSpec.model_validate_json(normalized.model_dump_json())
    assert not task.resources.worker and not task.resources.all
    assert {resource.path: resource_bytes(resource) for resource in task.resources.verifier} == {
        path.removeprefix("tests/"): data for path, data in payloads.items()
    }
    assert {resource.path: resource_bytes(resource) for resource in task.resources.oracle} == {
        "solution/solve.sh": oracle
    }
