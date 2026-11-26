# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower retained rows inside the pinned native CPU environment, without allocating agents."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from cloud.iris.launch_config import load_launch_config
from omegaconf import OmegaConf
from rolloutengine.models import MachineRuntimeSpec
from skyrl_train.dataset.tasks import canonical_task_spec
from taskcompendium.models import TaskSpec


def bind_runtime(row: dict, environment: dict, route: str) -> dict:
    """Lower the unchanged task through the native adapter, retaining published sizing."""
    task = TaskSpec.model_validate_json(row["task_spec"])
    settings = {
        **environment,
        "harness_by_dataset": dict(environment["harness_by_dataset"]),
        "task_machine_by_dataset": dict(environment["task_machine_by_dataset"]),
    }
    runtime = json.loads(row["runtime"]) if isinstance(row.get("runtime"), str) else row.get("runtime", {})
    if route == "agentic":
        settings["harness_by_dataset"][task.source.dataset] = "harbor_trial"
    if runtime.get("task_machine") is not None:
        # The native recipe's image-machine template selects Daytona and its network
        # policy; the published cohort supplies the actor's exact numeric resources.
        settings["task_machine_by_dataset"][task.source.dataset] = {
            **environment["image_verifier"],
            **runtime["task_machine"],
        }
    lowered = canonical_task_spec(task, settings)
    updates = {}
    for field in ("task_machine", "verifier_machine"):
        shape = runtime.get(field)
        if shape is None:
            continue
        machine = getattr(lowered, field)
        if machine is not None:
            updates[field] = MachineRuntimeSpec.model_validate({**machine.model_dump(), **shape})
    return {**row, "task_rollout_spec": lowered.model_copy(update=updates).model_dump_json()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("environment", type=Path)
    parser.add_argument("route", choices=("agentic", "nonagentic"))
    arguments = parser.parse_args()
    environment = OmegaConf.to_container(
        load_launch_config(arguments.environment).skyrl.environment.task_sessions.task_spec, resolve=True
    )
    rows = pq.read_table(arguments.source).to_pylist()
    pq.write_table(
        pa.Table.from_pylist([bind_runtime(row, environment, arguments.route) for row in rows]),
        arguments.output,
        compression="zstd",
    )
