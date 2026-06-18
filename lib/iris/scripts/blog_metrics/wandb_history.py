# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bulk-fetch W&B run metadata to a local parquet cache (long-history source).

Iris structured logs only reach back to ~2026-05-06; W&B run history reaches
back to the project's start (~2024-04) and is the only source for pre-Iris
compute. This pulls one row per run with the few summary fields needed to
reconstruct compute over time, and caches them so ``extract``/``charts`` iterate
locally without re-hitting the API.

The high-level ``Api.runs()`` iterator is unusably slow here (~5 runs/s) because
it downloads and deserializes each run's *full* summary — and the runs we want
are the big training runs with thousands of logged metrics. Instead we issue a
raw GraphQL query that projects ``summaryMetrics(keys: ...)`` down to the seven
fields we need and filters to device-bearing runs server-side, which is ~100x
faster (hundreds of runs/s).

Requires the ``wandb`` package (run the pipeline's ``fetch`` step with
``uv run --with wandb``) and ``WANDB_API_KEY`` in the environment.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time

import config
import duckdb

logger = logging.getLogger(__name__)

_RUNTIME_KEYS = ("_runtime", "_wandb.runtime")
_TOKENS_KEY = "throughput/total_tokens"
_MFU_KEY = "throughput/mfu"
# Projected summary keys — only these are downloaded per run.
_SUMMARY_KEYS = ["num_devices", "num_hosts", "parameter_count", _TOKENS_KEY, _MFU_KEY, "backend", *_RUNTIME_KEYS]
_PAGE_SIZE = 500

_PARQUET_COLUMNS = (
    "run_path",
    "project",
    "name",
    "state",
    "created_at",
    "runtime_s",
    "num_devices",
    "num_hosts",
    "parameter_count",
    "total_tokens",
    "mfu",
    "backend",
)

# Cursor-paginated runs query. ``summaryMetrics(keys:)`` returns a JSON string
# with only the requested keys; ``filters`` drops runs lacking device info
# server-side so their (often huge) summaries are never touched.
_RUNS_QUERY = """
query BlogMetricsRuns($entity: String!, $project: String!, $cursor: String,
                     $first: Int!, $keys: [String!], $filters: JSONString) {
  project(name: $project, entityName: $entity) {
    runs(first: $first, after: $cursor, filters: $filters) {
      edges {
        cursor
        node { name displayName state createdAt summaryMetrics(keys: $keys) }
      }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

_DEVICE_FILTER = json.dumps({"summary_metrics.num_devices": {"$gt": 0}})


def _parse_created_at(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def _first_float(summary: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in summary and summary[key] is not None:
            return float(summary[key])
    return None


def _row_from_node(node: dict, entity: str, project: str) -> tuple | None:
    """Build the cache row from one GraphQL run node, or None if it lacks devices."""
    summary = json.loads(node.get("summaryMetrics") or "{}")
    num_devices = summary.get("num_devices")
    if not num_devices:
        return None
    return (
        f"{entity}/{project}/{node['name']}",
        project,
        node.get("displayName") or node["name"],
        node.get("state"),
        _parse_created_at(node.get("createdAt")),
        _first_float(summary, _RUNTIME_KEYS),
        int(num_devices),
        int(summary["num_hosts"]) if summary.get("num_hosts") is not None else None,
        int(summary["parameter_count"]) if summary.get("parameter_count") is not None else None,
        float(summary[_TOKENS_KEY]) if summary.get(_TOKENS_KEY) is not None else None,
        float(summary[_MFU_KEY]) if summary.get(_MFU_KEY) is not None else None,
        str(summary["backend"]) if summary.get("backend") is not None else None,
    )


def _fetch_project_rows(api, entity: str, project: str) -> list[tuple]:
    from wandb_gql import gql  # noqa: PLC0415  # ships with wandb; resolved alongside it

    query = gql(_RUNS_QUERY)
    logger.info("pulling %s/%s via projected GraphQL…", entity, project)
    rows: list[tuple] = []
    cursor: str | None = None
    pages = 0
    start = time.monotonic()
    while True:
        result = api.client.execute(
            query,
            variable_values={
                "entity": entity,
                "project": project,
                "cursor": cursor,
                "first": _PAGE_SIZE,
                "keys": _SUMMARY_KEYS,
                "filters": _DEVICE_FILTER,
            },
        )
        conn = result["project"]["runs"]
        for edge in conn["edges"]:
            row = _row_from_node(edge["node"], entity, project)
            if row is not None:
                rows.append(row)
        pages += 1
        if pages % 20 == 0:
            logger.info("  %d pages, %d runs (%.0fs)", pages, len(rows), time.monotonic() - start)
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
    logger.info("  %s/%s done: %d device-bearing runs (%.0fs)", entity, project, len(rows), time.monotonic() - start)
    return rows


def fetch_runs(paths: config.Paths, *, entity: str | None = None, projects: list[str] | None = None) -> None:
    """Pull run metadata for ``projects`` and cache it to ``wandb_runs.parquet``."""
    import wandb  # noqa: PLC0415  # optional dep, only needed for the W&B pull

    entity = entity or config.WANDB_ENTITY
    projects = projects or config.WANDB_PROJECTS
    api = wandb.Api(timeout=60)

    rows: list[tuple] = []
    for project in projects:
        rows.extend(_fetch_project_rows(api, entity, project))

    os.makedirs(paths.raw_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE runs ("
        " run_path VARCHAR, project VARCHAR, name VARCHAR, state VARCHAR,"
        " created_at TIMESTAMP, runtime_s DOUBLE, num_devices INTEGER, num_hosts INTEGER,"
        " parameter_count BIGINT, total_tokens DOUBLE, mfu DOUBLE, backend VARCHAR)"
    )
    con.executemany(f"INSERT INTO runs ({', '.join(_PARQUET_COLUMNS)}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute(f"COPY runs TO '{paths.wandb_runs_parquet}' (FORMAT parquet)")
    con.close()
    logger.info("wrote %d run rows -> %s", len(rows), paths.wandb_runs_parquet)
