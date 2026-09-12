# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The table behind the RL Post-training view's "Producers reporting this run" panel.

One run reports through several namespaces, and the panel shows which of them actually did. Which
namespaces exist is a property of the deployment, and DataFusion rejects a statement naming an
absent table at plan time, so the panel cannot build this query itself. Each namespace is asked
separately and a missing one is dropped; listing them up front would need a client method the
deployed finelog wheel does not have.
"""

import logging
from collections.abc import Callable, Iterable, Mapping

from finelog.errors import StatsError
from vllm_observability import VLLM_MAX_WINDOW_MS, sql_string

logger = logging.getLogger(__name__)

# `max_rows` bounds the answer, not the work: without a cap a hand-built URL asks finelog to scan
# the whole retained table. Same ceiling as the vLLM overview route.
MAX_WINDOW_MS = VLLM_MAX_WINDOW_MS

# Every namespace an RL run can report through. Each is included only when the deployment has it.
RL_PRODUCER_NAMESPACES = (
    "telemetry_v1.marinskyrl",
    "telemetry_v1.vllm",
    "telemetry_v1.harbor",
)


def producers_query(namespace: str, run: str, clusters: tuple[str, ...], start_ms: int, end_ms: int) -> str:
    """Return the SQL that rolls one namespace up into producer rows."""
    cluster_values = ", ".join(sql_string(cluster) for cluster in clusters)
    return f"""SELECT service AS producer,
       COALESCE(json_get(resource_attributes_json, 'role'), '') AS role,
       COALESCE(execution_uid, '') AS attempt,
       COALESCE(json_get(attributes_json, 'metric_source'), '') AS metric_source,
       COUNT(DISTINCT name) AS signals,
       COUNT(*) AS records,
       MAX(timestamp_ms) AS last_record_ms
FROM "{namespace}"
WHERE run_id IN ({sql_string(run)})
  AND COALESCE(NULLIF("cluster", ''), 'marin') IN ({cluster_values})
  AND timestamp_ms >= {start_ms}
  AND timestamp_ms < {end_ms}
GROUP BY 1, 2, 3, 4"""


def check_window(start_ms: int, end_ms: int) -> None:
    """Raise when the requested window is wider than this route will scan."""
    if end_ms - start_ms > MAX_WINDOW_MS:
        raise ValueError(f"window of {end_ms - start_ms} ms exceeds the {MAX_WINDOW_MS} ms maximum")


def collect_producers(
    query: Callable[[str], Iterable[Mapping[str, object]]],
    run: str,
    clusters: tuple[str, ...],
    start_ms: int,
    end_ms: int,
) -> list[dict[str, object]]:
    """Roll up every RL namespace this deployment can answer for, skipping those it cannot.

    A namespace the deployment has never received a row for has no table, and DataFusion rejects
    the statement at plan time. That is indistinguishable, from here, from "no rows", and it is the
    answer the panel wants in both cases -- so the failure is dropped rather than propagated.
    Anything else the query raises is a real fault and is left to the caller.
    """
    rows: list[dict[str, object]] = []
    for namespace in RL_PRODUCER_NAMESPACES:
        try:
            rows.extend(dict(row) for row in query(producers_query(namespace, run, clusters, start_ms, end_ms)))
        except StatsError as error:
            if "not found" not in str(error).lower():
                raise
            logger.info("rl producers: %s is absent from this deployment", namespace)
    return rows
