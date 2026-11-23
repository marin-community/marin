# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live controller reads for the current moment.

Every cluster's controller database answers read-only SELECTs over the Iris
RPC raw-query endpoint, so the current occupancy is available even for
clusters whose controller does not publish mirror rows to Finelog (cw-rno2a).
History comes from the Finelog mirror instead; this module is only "now".
"""

import json
import logging
import os
from pathlib import Path

from iris.cli.connect import connect_controller, rpc_client
from iris.rpc import query_pb2

logger = logging.getLogger(__name__)

LIVE_TIMEOUT_MS = 15_000

# Stands in for the controllers with fixture rows (JSON: cluster → row lists),
# the way OCCUPANCY_FINELOG_URL stands in for the hub. Tests and journeys set
# it; production never does.
LIVE_ROWS_FILE_ENV = "OCCUPANCY_LIVE_ROWS_FILE"

TASK_STATE_RUNNING = 3

# Numeric bands from the controller's job_config, mapped like the reference tool.
PRIORITY_BANDS = {0: "inherit", 1: "production", 2: "interactive", 3: "batch", 4: "system"}

RUNNING_TASKS_SQL = f"""SELECT j.root_job_id, j.user_id, j.job_id, c.priority_band,
c.res_device_json, a.node_name, a.started_at_ms, a.topology_json
FROM tasks t
JOIN jobs j ON j.job_id = t.job_id
JOIN job_config c ON c.job_id = t.job_id
LEFT JOIN task_attempts a ON a.task_id = t.task_id AND a.attempt_id = t.current_attempt_id
WHERE t.state = {TASK_STATE_RUNNING}"""


def gpu_count_and_type(device_json: str) -> tuple[int, str]:
    """GPUs and variant held by one task, from ``job_config.res_device_json``."""
    if not device_json:
        return 0, ""
    gpu = json.loads(device_json).get("gpu")
    if not gpu:
        return 0, ""
    return int(gpu.get("count", 0)), gpu.get("variant", "") or ""


def _raw_query(cluster: str, sql: str) -> list[dict[str, object]]:
    """One read-only SELECT against a cluster's live controller database."""
    with connect_controller(cluster_name=cluster) as endpoint:
        with rpc_client(endpoint.url, endpoint.credentials, timeout_ms=LIVE_TIMEOUT_MS) as client:
            response = client.execute_raw_query(query_pb2.RawQueryRequest(sql=sql))
    columns = [column.name for column in response.columns]
    return [dict(zip(columns, json.loads(row), strict=True)) for row in response.rows]


def running_tasks(cluster: str) -> list[dict[str, object]]:
    """One row per running task placement, right now.

    The controller identifies jobs by their path, so ``root_job_id`` doubles as
    the root job's display name. CPU tasks carry no ``gpu`` device and are
    kept: they hold no GPUs, and the map drops them later.
    """
    fixture = os.environ.get(LIVE_ROWS_FILE_ENV)
    if fixture:
        rows = json.loads(Path(fixture).read_text())
        return list(rows.get(cluster, []))
    return _raw_query(cluster, RUNNING_TASKS_SQL)


def holder_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalize raw running-task rows into the app's holder shape."""
    holders: list[dict[str, object]] = []
    for row in rows:
        gpus, variant = gpu_count_and_type(row.get("res_device_json") or "")
        if gpus <= 0:
            continue  # a CPU task holds no part of the GPU map
        band = row.get("priority_band")
        holders.append(
            {
                "user_id": row.get("user_id"),
                "job_id": row.get("job_id"),
                "root_job_id": row.get("root_job_id"),
                "root_name": row.get("root_job_id"),
                "priority_band": PRIORITY_BANDS.get(band if isinstance(band, int) else -1),
                "accelerator_type": variant,
                "gpus": gpus,
                "started_ms": row.get("started_at_ms"),
                "finished_ms": None,
                "node_name": row.get("node_name") or "",
                "rack": _rack(row.get("topology_json")),
            }
        )
    return holders


def _rack(topology_json: object) -> str:
    if not isinstance(topology_json, str) or not topology_json:
        return ""
    try:
        return json.loads(topology_json).get("ds.coreweave.com/nvlink.domain") or ""
    except (TypeError, ValueError):
        return ""


def node_views(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Per-node views of the current moment: holders, GPUs, rack grouping.

    One holder per (node, job) with its tasks' GPUs summed; a node's capacity
    is what it holds right now, since the live read sees no idle inventory.
    """
    by_node: dict[str, dict[str, object]] = {}
    for holder in holder_rows(rows):
        node_name = holder.pop("node_name")
        rack = holder.pop("rack")
        node = by_node.get(node_name)
        if node is None:
            node = {"node_name": node_name, "rack": rack, "gpu_type": "", "capacity_gpus": 0, "holders": {}}
            by_node[node_name] = node
        elif not node["rack"]:
            node["rack"] = rack
        if not node["gpu_type"]:
            node["gpu_type"] = holder["accelerator_type"]
        node["capacity_gpus"] += holder["gpus"]
        key = holder["job_id"]
        existing = node["holders"].get(key)
        if existing is None:
            node["holders"][key] = holder
        else:
            existing["gpus"] += holder["gpus"]
            started = holder["started_ms"]
            if started is not None and (existing["started_ms"] is None or started < existing["started_ms"]):
                existing["started_ms"] = started

    views = []
    for node in by_node.values():
        holders = sorted(
            node["holders"].values(),
            key=lambda holder: (-holder["gpus"], holder["root_name"] or "", holder["job_id"] or ""),
        )
        views.append(
            {
                "node_name": node["node_name"],
                "rack": node["rack"],
                "gpu_type": node["gpu_type"],
                "capacity_gpus": node["capacity_gpus"],
                "holders": holders,
            }
        )
    views.sort(key=lambda node: (node["rack"] == "", node["rack"], node["node_name"]))
    return views


def legend(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """GPUs held per user across the cluster, right now."""
    per_user: dict[str, int] = {}
    for holder in holder_rows(rows):
        user = holder["user_id"] or ""
        per_user[user] = per_user.get(user, 0) + holder["gpus"]
    return [
        {"user_id": user, "gpus": gpus}
        for user, gpus in sorted(per_user.items(), key=lambda item: (-item[1], item[0]))
    ]
