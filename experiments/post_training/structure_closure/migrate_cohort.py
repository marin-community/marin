# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert this campaign's archived token tasks to the current rollout schema."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import RolloutLimits, TaskRolloutSpec
from taskcompendium.models import NoGrader, TaskResource, TaskSpec
from taskcompendium.runtime.resources import resource_bytes
from verifyit.json_objects import unique_object

ROLLOUT_COLUMN = "task_rollout_spec"
BATCH_SIZE = 1024


def migrate_closure_record(serialized: str) -> TaskRolloutSpec:
    document = json.loads(serialized, object_pairs_hook=unique_object)
    native = document["task"]
    session = dict(document["session"])
    if native["source"]["dataset"] != "structure-closure" or session.pop("task_session") != "structure_closure":
        raise ValueError("Only archived structure-closure tasks belong in this migration")
    if native["grader"] != {"kind": "session"} or document["runtime"] != {
        "task_machine": None,
        "verifier_machine": None,
    }:
        raise ValueError("Expected a machine-free closure session")
    configs = [resource for resource in native["resources"]["verifier"] if resource["path"] == "config.json"]
    if len(configs) != 1:
        raise ValueError("Expected one private closure config")
    contract = json.loads(resource_bytes(TaskResource.model_validate(configs[0])), object_pairs_hook=unique_object)
    task = TaskSpec.model_validate(
        {
            **{key: value for key, value in native.items() if key != "schema_version"},
            "grader": NoGrader(reason="SkyRL grades each task turn", contract=contract),
        }
    )
    # These settings described shell execution; this cohort never allocated a machine.
    del session["shell_tool"]
    del session["command_timeout"]
    return TaskRolloutSpec(
        task=task,
        harness="structure_closure",
        task_machine=None,
        verifier_machine=None,
        limits=RolloutLimits.model_validate({"max_total_tokens": None, **session}),
    )


def migrate_cohort(*, source: str, output: str, hashes: dict[str, str]) -> None:
    destination = StoragePath(output)
    destination.mkdirs()
    with TemporaryDirectory() as temporary:
        for split, expected in hashes.items():
            filename = f"{split}.parquet"
            local = Path(temporary) / filename
            (StoragePath(source) / filename).download_to(str(local))
            with local.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Archived {split} hash differs from the pinned cohort")
            converted = Path(temporary) / f"converted-{filename}"
            schema = pa.schema([pa.field(ROLLOUT_COLUMN, pa.string(), nullable=False)])
            with pq.ParquetWriter(converted, schema) as writer:
                for batch in pq.ParquetFile(local).iter_batches(batch_size=BATCH_SIZE, columns=["lowered_task_spec"]):
                    records = [migrate_closure_record(value).model_dump_json() for value in batch.column(0).to_pylist()]
                    writer.write_table(pa.Table.from_pydict({ROLLOUT_COLUMN: records}, schema=schema))
            (destination / filename).upload_from(str(converted))
