# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify frozen persistent Python tool contracts on an Iris LOCAL worker."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.harness.shell import ToolContext
from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec
from rolloutengine.validation import validate_spec
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, Command, NetworkPolicy
from skyrl_gym.rollout_specs import TOOL_PROVIDERS, rollout_spec
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.environment import prepare_machine_spec


class PythonQualificationInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    original_task_sha256: str
    task_sha256: str
    task_spec: str


class PythonProbe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    original_task_sha256: str
    task_sha256: str
    tool_schema_sha256: str
    backend: str
    image: str
    namespace_isolated: bool
    network_routes_absent: bool
    state_persisted: bool


async def qualify(rows: list[PythonQualificationInput]) -> tuple[PythonProbe, ...]:
    factory = LocalMachineFactory()
    config_path = Path("vendor/skyrl/skyrl-train/skyrl_train/config/task_session_config/default.yaml")
    config = yaml.safe_load(config_path.read_text())["task_spec"]
    runtime = MachineRuntimeSpec.model_validate({**config["required_actor"], "backend": Backend.LOCAL})
    assert runtime.backend == Backend.LOCAL and runtime.network is NetworkPolicy.DENY
    host_network = os.readlink("/proc/self/ns/net")
    results = []
    for row in rows:
        encoded = row.task_spec
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if digest != row.task_sha256:
            raise ValueError("Frozen Python qualification task differs from its content hash")
        task = TaskSpec.model_validate_json(encoded)
        if task.environment_requirements.required_backend is not Backend.LOCAL:
            raise ValueError("Python qualification requires a typed LOCAL task contract")
        image = task.environment_requirements.docker_image
        if image is None:
            raise ValueError("Python qualification requires its immutable source image")
        if task.resources.all or task.resources.worker:
            raise ValueError("Python qualification tasks must declare no separate worker input files")
        session = {
            "task_session": "shellbox",
            "max_turns": None,
            "max_total_tokens": None,
            "model_turn_timeout": None,
            "tool_turn_timeout": None,
            "total_turn_timeout": None,
            "attempt_timeout": None,
            "verifier_timeout": config["minimum_verifier_timeout"],
            "cleanup_timeout": 60,
        }
        lowered = rollout_spec(task, session, task_machine=runtime)
        validate_spec(lowered, {Backend.LOCAL: factory})
        assert lowered.task_machine is not None
        selected = prepare_machine_spec(
            task.environment_requirements,
            factory,
            _machine_spec(task.environment_requirements, lowered.task_machine),
            os.environ,
        )
        machine = await factory.create(selected)
        try:
            (requirement,) = task.environment_requirements.tool_providers.values()
            provider = TOOL_PROVIDERS[requirement.action_interface]
            (tool,) = provider.tools(requirement)
            tools = await provider.open(ToolContext(machine, requirement))
            try:
                first = await tools.call(tool.name, {"code": "atlas_counter = 40; print(atlas_counter)"})
                second = await tools.call(tool.name, {"code": "atlas_counter += 2; print(atlas_counter)"})
                if (first.content, second.content) != ("40", "42"):
                    raise ValueError("Frozen source Python tool did not preserve its task-local namespace")
                network = await machine.run(
                    Command(
                        argv=(
                            "python",
                            "-c",
                            "import json,os; print(json.dumps({'namespace':os.readlink('/proc/self/ns/net'),"
                            "'routes':open('/proc/net/route').read().splitlines()[1:]}))",
                        ),
                        timeout=30,
                    )
                )
                if network.exit_code != 0:
                    raise ValueError("Cannot verify the LOCAL tool network namespace")
                isolation = json.loads(network.stdout)
                if isolation["namespace"] == host_network or isolation["routes"]:
                    raise ValueError("LOCAL Python tool does not have a private network-denied namespace")
                results.append(
                    PythonProbe(
                        source_id=row.source_id,
                        original_task_sha256=row.original_task_sha256,
                        task_sha256=digest,
                        tool_schema_sha256=hashlib.sha256(tool.model_dump_json().encode()).hexdigest(),
                        backend=Backend.LOCAL.value,
                        image=image,
                        namespace_isolated=True,
                        network_routes_absent=True,
                        state_persisted=True,
                    )
                )
            finally:
                await tools.close()
        finally:
            await machine.close()
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-commit", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    payload = StoragePath(args.inputs).read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.inputs_sha256:
        raise ValueError("Frozen Python qualification input differs from its sealed SHA")
    rows = [PythonQualificationInput.model_validate(row) for row in pq.read_table(pa.BufferReader(payload)).to_pylist()]
    if len(rows) != args.expected_tasks:
        raise ValueError("Frozen Python qualification input count differs")
    results = asyncio.run(qualify(rows))
    receipt = {
        "inputs": args.inputs,
        "inputs_sha256": args.inputs_sha256,
        "passed": True,
        "runtime_commit": args.runtime_commit,
        "scope": "Real frozen repair-candidate tool contracts; LOCAL bubblewrap, image, network and persistence",
        "results": [result.model_dump(mode="json") for result in results],
    }
    StoragePath(args.output).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
