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
from dataclasses import replace
from pathlib import Path

from iris.cli.connect import connect_controller, rpc_client
from iris.rpc import query_pb2

from .views import RACK_JSON_KEY, Holder, NodeView, holder_sort_key, node_sort_key

logger = logging.getLogger(__name__)

LIVE_TIMEOUT_MS = 15_000

TASK_STATE_RUNNING = 3

# The controller stores bands numerically: 0 inherit, 1 production,
# 2 interactive, 3 batch, 4 system.
PRIORITY_BANDS = {0: "inherit", 1: "production", 2: "interactive", 3: "batch", 4: "system"}

RUNNING_TASKS_SQL = f"""SELECT j.root_job_id, j.user_id, j.job_id, c.priority_band,
c.res_device_json, a.node_name, a.started_at_ms, a.topology_json
FROM tasks t
JOIN jobs j ON j.job_id = t.job_id
JOIN job_config c ON c.job_id = t.job_id
LEFT JOIN task_attempts a ON a.task_id = t.task_id AND a.attempt_id = t.current_attempt_id
WHERE t.state = {TASK_STATE_RUNNING}"""

# Stands in for the controllers with fixture rows (JSON: cluster → row lists),
# the way OCCUPANCY_FINELOG_URL stands in for the hub. Tests and journeys set
# it; production never does.
LIVE_ROWS_FILE_ENV = "OCCUPANCY_LIVE_ROWS_FILE"


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
    kept: they hold no GPUs, and the view drops them later.
    """
    return _raw_query(cluster, RUNNING_TASKS_SQL)


def fixture_running_tasks(path: str):
    """A ``running_tasks`` stand-in that serves rows from a fixture JSON file."""

    def read(cluster: str) -> list[dict[str, object]]:
        rows = json.loads(Path(path).read_text())
        return list(rows.get(cluster, []))

    return read


def _text(value: object) -> str | None:
    """The controller returns strings over the wire; keep them typed."""
    return value if isinstance(value, str) else None


def holder(row: dict[str, object]) -> Holder:
    """One normalized holder record from a raw running-task row."""
    gpus, variant = gpu_count_and_type(row.get("res_device_json") or "")
    band = row.get("priority_band")
    return Holder(
        user_id=_text(row.get("user_id")),
        job_id=_text(row.get("job_id")),
        root_job_id=_text(row.get("root_job_id")),
        root_name=_text(row.get("root_job_id")),
        priority_band=PRIORITY_BANDS.get(band if isinstance(band, int) else -1),
        accelerator_type=variant,
        gpus=gpus,
        started_ms=row.get("started_at_ms") if isinstance(row.get("started_at_ms"), int) else None,
        finished_ms=None,
    )


def _rack(topology_json: object) -> str:
    """The node's rack (NVLink domain), empty when the node reports none."""
    if not isinstance(topology_json, str) or not topology_json:
        return ""
    try:
        return json.loads(topology_json).get(RACK_JSON_KEY) or ""
    except (TypeError, ValueError):
        logger.debug("unreadable topology_json: %r", topology_json)
        return ""


def node_views(rows: list[dict[str, object]]) -> list[NodeView]:
    """Per-node views of the current moment: holders, GPUs, rack grouping.

    One holder per (node, job) with its tasks' GPUs summed; a node's capacity
    is what it holds right now, since the live read sees no idle inventory.
    CPU tasks (no ``gpu`` device) hold no part of the GPU map and drop out.
    """
    meta: dict[str, tuple[str, str | None]] = {}
    holders: dict[str, dict[object, Holder]] = {}
    for row in rows:
        record = holder(row)
        if record.gpus <= 0:
            continue
        node_name = _text(row.get("node_name")) or ""
        rack = _rack(row.get("topology_json"))
        known_meta = meta.get(node_name)
        meta[node_name] = (
            rack if known_meta is None or not known_meta[0] else known_meta[0],
            record.accelerator_type if known_meta is None or not known_meta[1] else known_meta[1],
        )
        known = holders.setdefault(node_name, {}).get(record.job_id)
        if known is None:
            holders[node_name][record.job_id] = record
        else:
            merged_gpus = known.gpus + record.gpus
            started = (
                known.started_ms
                if known.started_ms is not None and (record.started_ms is None or known.started_ms < record.started_ms)
                else record.started_ms
            )
            holders[node_name][record.job_id] = replace(known, gpus=merged_gpus, started_ms=started)

    views = [
        NodeView(
            node_name=node_name,
            rack=rack,
            gpu_type=gpu_type or "",
            capacity_gpus=sum(one.gpus for one in per_job.values()),
            holders=sorted(per_job.values(), key=holder_sort_key),
        )
        for node_name, (rack, gpu_type) in meta.items()
        for per_job in [holders[node_name]]
    ]
    views.sort(key=node_sort_key)
    return views


def legend(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """GPUs held per user across the cluster, right now."""
    per_user: dict[str, int] = {}
    for row in rows:
        record = holder(row)
        if record.gpus <= 0:
            continue
        user = record.user_id or ""
        per_user[user] = per_user.get(user, 0) + record.gpus
    return [
        {"user_id": user, "gpus": gpus} for user, gpus in sorted(per_user.items(), key=lambda item: (-item[1], item[0]))
    ]
