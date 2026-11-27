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
from skyrl_train.dataset.tasks import canonical_task_spec
from taskcompendium.models import NoGrader, TaskSpec
from taskcompendium.runtime.integrity import require_integrity_contract

POOL_BATCH_SIZE = 64


def bind_runtime(row: dict, environment: dict) -> dict:
    """Lower the unchanged task through the native adapter, retaining published sizing."""
    task = TaskSpec.model_validate_json(row["task_spec"])
    require_integrity_contract(task)
    settings = {
        **environment,
        "harness_by_dataset": dict(environment["harness_by_dataset"]),
        "task_machine_by_dataset": dict(environment["task_machine_by_dataset"]),
        "verifier_machine_by_dataset": dict(environment.get("verifier_machine_by_dataset", {})),
    }
    dataset = task.source.dataset
    runtime = json.loads(row["runtime"]) if isinstance(row.get("runtime"), str) else row.get("runtime", {})
    # Only the original Harbor package importer delegates grading via NoGrader.
    # A scheduler's agentic route does not turn native TaskSpecs into packages.
    if "harbor" in task.tags and isinstance(task.grader, NoGrader):
        settings["harness_by_dataset"][task.source.dataset] = "harbor_trial"
    if runtime.get("task_machine") is not None:
        # The native recipe's image-machine template selects Daytona and its network
        # policy; the published cohort supplies the actor's exact numeric resources.
        # A dataset placed on its own machine template (per-task images on LOCAL) keeps that backend.
        settings["task_machine_by_dataset"][dataset] = {
            **environment["task_machine_by_dataset"].get(dataset, environment["image_verifier"]),
            **{key: value for key, value in runtime["task_machine"].items() if value is not None},
        }
    if runtime.get("verifier_machine") is not None:
        shape = {key: value for key, value in runtime["verifier_machine"].items() if value is not None}
        for field in ("image_verifier", "package_verifier"):
            settings[field] = {**environment[field], **shape}
        if dataset in settings["verifier_machine_by_dataset"]:
            settings["verifier_machine_by_dataset"][dataset] = {
                **settings["verifier_machine_by_dataset"][dataset],
                **shape,
            }
    # Native lowering applies the TaskSpec's declared allocation minima and
    # deadlines. Overlaying shapes afterward would erase those requirements.
    lowered = canonical_task_spec(task, settings)
    return {**row, "task_rollout_spec": lowered.model_dump_json()}


def lower_parquet(source: Path, output: Path, environment: dict) -> None:
    """Lower bounded batches without retaining a second copy of the whole cohort."""
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    index = schema.get_field_index("task_rollout_spec")
    field = pa.field("task_rollout_spec", pa.string())
    schema = schema.set(index, field) if index >= 0 else schema.append(field)
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for batch in parquet.iter_batches(batch_size=POOL_BATCH_SIZE):
            rows = [bind_runtime(row, environment) for row in batch.to_pylist()]
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("environment", type=Path)
    arguments = parser.parse_args()
    environment = OmegaConf.to_container(
        load_launch_config(arguments.environment).skyrl.environment.task_sessions.task_spec, resolve=True
    )
    lower_parquet(arguments.source, arguments.output, environment)
