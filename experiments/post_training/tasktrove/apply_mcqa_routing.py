# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the MCQA routing artifact to verified TaskTrove rows."""

import json

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.tasktrove.convert import CONVERTED_SCHEMA
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.dataset import APPROX_SHARD_BYTES, WORKER_RESOURCES, WORKING_SHARDS
from experiments.post_training.tasktrove.mcqa_routing import (
    MCQA_SOURCE,
    ROUTE_MAPPING_FIELDS,
    ROUTE_MAPPINGS_FILENAME,
    Route,
)
from experiments.post_training.tasktrove.verify import FILTERED_GLOB

ROUTED_GLOB = "graded/*.parquet"
ROUTED_PATTERN = "graded/part-{shard:05d}.parquet"
ROUTING_COLUMNS = ROUTE_MAPPING_FIELDS[1:]
ROUTED_SCHEMA = pa.schema(
    [
        *CONVERTED_SCHEMA,
        pa.field("route", pa.string()),
        pa.field("route_source", pa.string()),
        pa.field("policy_version", pa.string()),
        pa.field("reason_codes", pa.list_(pa.string())),
    ]
)
ROUTED_SFT_STATUS = "routed:sft"
ROUTED_GARBAGE_STATUS = "routed:garbage"
ROUTED_MISSING_STATUS = "routed:missing"
ROUTING_COORDINATOR_RESOURCES = ResourceConfig(cpu=1, ram="8g", preemptible=False)


def _routing_fields(mapping: dict | None) -> dict:
    if mapping is None:
        return {name: [] if name == "reason_codes" else "" for name in ROUTING_COLUMNS}
    return {name: mapping[name] for name in ROUTING_COLUMNS}


def apply_route(row: dict, mapping: dict | None) -> dict:
    """Apply one fail-closed MCQA route and add empty route metadata elsewhere."""
    if row["source"] != MCQA_SOURCE or row["status"] != ConvertStatus.CONVERTED:
        return {**row, **_routing_fields(None)}
    routed = {**row, **_routing_fields(mapping)}
    if mapping is None:
        return {
            **routed,
            "status": ROUTED_MISSING_STATUS,
            "error": "no entry in the MCQA route-mapping artifact",
            "task_binary": None,
            "solution_binary": None,
        }

    route = Route(mapping["route"])
    detail = "; ".join(mapping["reason_codes"])
    if route is Route.RL:
        return routed
    if route is Route.SFT:
        return {**routed, "status": ROUTED_SFT_STATUS, "error": detail}
    return {
        **routed,
        "status": ROUTED_GARBAGE_STATUS,
        "error": detail,
        "task_binary": None,
        "solution_binary": None,
    }


def load_route_mappings(route_mappings_path: str) -> dict[str, dict]:
    """Load and validate the compact task-to-route JSONL artifact."""
    mappings: dict[str, dict] = {}
    for line in StoragePath(route_mappings_path).read_text().splitlines():
        if not line.strip():
            continue
        mapping = json.loads(line)
        task_id = mapping["task_id"]
        if task_id in mappings:
            raise ValueError(f"duplicate routing task ID {task_id!r}")
        mappings[task_id] = mapping
    return mappings


def route_tasks(filtered_path: str, routing_artifact_path: str, output_path: str) -> None:
    """Apply the routing artifact to verified tasks and write routed Parquet shards."""
    mappings = load_route_mappings(str(StoragePath(routing_artifact_path) / ROUTE_MAPPINGS_FILENAME))
    rows = Dataset.from_files(str(StoragePath(filtered_path) / FILTERED_GLOB)).load_parquet(
        approx_shard_bytes=APPROX_SHARD_BYTES
    )
    routed = rows.map(lambda row: apply_route(row, mappings.get(row["path"])))
    routed = routed.write_parquet(str(StoragePath(output_path) / ROUTED_PATTERN), schema=ROUTED_SCHEMA)
    ZephyrContext(
        name="tasktrove-route",
        max_workers=WORKING_SHARDS,
        resources=WORKER_RESOURCES,
        coordinator_resources=ROUTING_COORDINATOR_RESOURCES,
    ).execute(routed)
