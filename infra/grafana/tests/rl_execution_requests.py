# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as Record

from config import ClusterTarget
from conftest import bridge_config, finelog_sql_for_duckdb
from rl_execution_fixtures import (
    BASE_EPOCH_MS,
    CLUSTER,
    DRIVER,
    JOB_ID,
    RUN_ID,
    SERVICE,
    TABLE,
    WINDOW_MS,
    WINDOW_START_MS,
    WORKER,
)
from rl_execution_observability import (
    _METRIC_NAMES,
    RL_EXECUTION_MIN_BUCKET_MS,
)
from rl_observability import _DCGM_SERIES
from server import create_app

EXPECTED_PANELS = json.loads((Path(__file__).parent / "fixtures/rl_execution_panels.json").read_text())


def _target_params(target, *, run=RUN_ID):
    variables = {
        "${cluster:csv}": CLUSTER,
        "${run}": run,
        "${job}": JOB_ID,
        "${execution:csv}": f"{DRIVER},{WORKER}",
        "${__from}": str(WINDOW_START_MS),
        "${__to}": str(WINDOW_START_MS + WINDOW_MS),
        "${__interval_ms}": str(RL_EXECUTION_MIN_BUCKET_MS),
    }
    params = {}
    for param in target["url_options"]["params"]:
        value = param["value"]
        for macro, replacement in variables.items():
            value = value.replace(macro, replacement)
        assert "${" not in value, value
        params[param["key"]] = value
    return params


def _bridge(database, *, max_rows=1000):
    queries = []

    def query_source(sql, *, max_rows):
        queries.append(sql)
        return database.execute(finelog_sql_for_duckdb(sql)).fetch_arrow_table()

    source = Record(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query_source)
    return create_app(replace(bridge_config(), max_rows=max_rows), {"marin": source}, {}, None, None, None), queries


REQUEST = {
    "clusters": CLUSTER,
    "run": RUN_ID,
    "job": JOB_ID,
    "executions": f"{DRIVER},{WORKER}",
    "from": WINDOW_START_MS,
    "to": WINDOW_START_MS + WINDOW_MS,
    "bucket_ms": RL_EXECUTION_MIN_BUCKET_MS,
}


def _long_run_store(database):
    ranks, steps = 256, 500
    window_ms = 7 * 24 * 3600 * 1000
    spacing = (window_ms - 2 * 60000) // steps
    database.execute(
        f"""CREATE TEMP TABLE long_template AS SELECT * FROM {TABLE}
        WHERE run_id='{RUN_ID}' AND job_id='{JOB_ID}' AND execution_uid IN ('{DRIVER}', '{WORKER}')
          AND name <> 'training_metric_value'
          AND COALESCE(json_get(attributes_json, 'backend'), '') <> 'megatron'"""
    )
    database.execute(f"DELETE FROM {TABLE}")
    database.execute(
        f"""INSERT INTO {TABLE}
        SELECT cluster, service, run_id, job_id, execution_uid, timestamp_ms + s.step * {spacing}, s.step,
               name, value,
               CAST(json_merge_patch(attributes_json, json_object('step', CAST(s.step AS VARCHAR))) AS VARCHAR),
               resource_attributes_json, body_json
        FROM long_template, (SELECT range AS step FROM range({steps})) s"""
    )
    database.execute(
        f"""INSERT INTO {TABLE}
        SELECT '{CLUSTER}', '{SERVICE}', '{RUN_ID}', '{JOB_ID}', '{WORKER}',
               {BASE_EPOCH_MS} + s.step * {spacing}, s.step, 'phase_duration_seconds', CAST(r.rank + 1 AS DOUBLE),
               CAST(json_object('step', CAST(s.step AS VARCHAR), 'rank', CAST(r.rank AS VARCHAR),
                    'phase', p.phase, 'outcome', 'success', 'backend', 'megatron',
                    'clock_domain', 'cpu_dispatch_wall', 'role', 'worker') AS VARCHAR),
               '{{"role":"worker","host":"learner"}}', '{{}}'
        FROM (SELECT range AS step FROM range({steps})) s, (SELECT range AS rank FROM range({ranks})) r,
             (VALUES ('ppo_train'), ('prepare'), ('forward'), ('backward'), ('optimizer'),
                     ('finalize'), ('ppo_train_residual')) p(phase)"""
    )
    metrics = ",".join("(" + repr(name) + ")" for name in _METRIC_NAMES)
    database.execute(
        f"""INSERT INTO {TABLE}
        SELECT '{CLUSTER}', '{SERVICE}', '{RUN_ID}', '{JOB_ID}', '{DRIVER}',
               {BASE_EPOCH_MS} + s.step * {spacing}, s.step, 'training_metric_value', 1.0,
               CAST(json_object('step', CAST(s.step AS VARCHAR), 'role', 'trainer', 'metric', m.metric,
                                'payload_kind', 'train') AS VARCHAR), '{{"role":"trainer"}}', '{{}}'
        FROM (SELECT range AS step FROM range({steps})) s, (VALUES {metrics}) m(metric)"""
    )
    return window_ms


def _long_run_nodes(database, nodes):
    """Seed one node before growing the same fixture to a larger node count."""
    if nodes == 1:
        database.execute(f"ALTER TABLE {TABLE} ADD COLUMN node_name VARCHAR")
        database.execute(f"ALTER TABLE {TABLE} ADD COLUMN process_index VARCHAR")
        database.execute(f"UPDATE {TABLE} SET node_name='long-node-0'")
        database.execute(f'CREATE TABLE "telemetry_v1.vllm" AS SELECT * FROM {TABLE} LIMIT 0')
        database.execute(f'CREATE TABLE "telemetry_v1.node_agent" AS SELECT * FROM {TABLE} LIMIT 0')
        database.execute(
            f"CREATE TEMP TABLE long_node_template AS SELECT * FROM {TABLE} "
            f"WHERE name='policy_step' AND execution_uid='{DRIVER}'"
        )
        names = ",".join("(" + repr(name) + ")" for name in (*_DCGM_SERIES, "gpu_power_watts", "gpu_pcie_replay_errors"))
        database.execute(
            f"""INSERT INTO "telemetry_v1.node_agent"
            SELECT cluster, 'iris-node-agent', NULL, NULL, NULL, timestamp_ms, seq,
                   n.name, CAST(seq AS DOUBLE),
                   CAST(json_object('gpu_uuid', 'GPU-0-' || CAST(g.gpu AS VARCHAR)) AS VARCHAR),
                   '{{}}', '{{}}', 'long-node-0', NULL
            FROM long_node_template, (VALUES {names}) n(name), (SELECT range AS gpu FROM range(8)) g"""
        )
        database.execute(
            """INSERT INTO "telemetry_v1.vllm"
            SELECT cluster, 'vllm', run_id, job_id, execution_uid, timestamp_ms, seq,
                   'generation_tokens_total', CAST(seq AS DOUBLE), '{}', '{}', '{}', node_name, process_index
            FROM long_node_template"""
        )
    else:
        database.execute(
            f"""INSERT INTO {TABLE}
            SELECT cluster, service, run_id, job_id, execution_uid, timestamp_ms, seq,
                   name, value, attributes_json, resource_attributes_json, body_json,
                   'long-node-' || CAST(n.node AS VARCHAR), process_index
            FROM long_node_template, (SELECT range AS node FROM range(1, {nodes})) n"""
        )
        database.execute(
            f"""INSERT INTO "telemetry_v1.node_agent"
            SELECT cluster, service, run_id, job_id, execution_uid, timestamp_ms, seq,
                   name, value,
                   CAST(json_merge_patch(attributes_json, json_object('gpu_uuid',
                        'GPU-' || CAST(n.node AS VARCHAR) || '-' ||
                        RIGHT(json_get(attributes_json, 'gpu_uuid'), 1))) AS VARCHAR),
                   resource_attributes_json, body_json,
                   'long-node-' || CAST(n.node AS VARCHAR), process_index
            FROM "telemetry_v1.node_agent", (SELECT range AS node FROM range(1, {nodes})) n
            WHERE node_name='long-node-0'"""
        )
