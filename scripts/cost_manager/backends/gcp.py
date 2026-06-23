# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP costs from the BigQuery billing export.

The Cloud Billing API exposes no actual spend, so detailed GCP cost lives only
in the **BigQuery billing export** table that the billing account must be
configured to write. This backend runs a daily-cost query via the ``bq`` CLI
(already on the GitHub runner via the Cloud SDK) and emits one
:class:`CostEvent` per (day, service, region).

Net cost adds the (negative) credits to gross ``cost``; both are cast to
``NUMERIC`` before summing to avoid float rounding.

The billing export must be enabled and readable by the runner's service
account; point ``billing_export_table`` at that dataset.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess
from collections.abc import Mapping
from typing import Any

from scripts.cost_manager.cost_event import CostEvent, CostFetchError, DateWindow, cost_event

logger = logging.getLogger(__name__)

PROVIDER = "gcp"
BQ_TIMEOUT = 180.0
MAX_ROWS = 100_000

# One row per (UTC day, service, region) with credit-adjusted net cost. The
# table name is interpolated (it is operator-controlled config, not user input)
# because BigQuery does not parameterize table identifiers.
_QUERY_TEMPLATE = """
SELECT
  FORMAT_DATE('%Y-%m-%d', DATE(usage_start_time, 'UTC')) AS day,
  service.description AS service,
  COALESCE(NULLIF(location.region, ''), NULLIF(location.location, ''), 'global') AS region,
  currency AS currency,
  SUM(
    CAST(cost AS NUMERIC)
    + IFNULL((SELECT SUM(CAST(c.amount AS NUMERIC)) FROM UNNEST(credits) c), 0)
  ) AS net_cost
FROM `{table}`
WHERE usage_start_time >= TIMESTAMP('{start}') AND usage_start_time < TIMESTAMP('{end}')
GROUP BY day, service, region, currency
HAVING ABS(net_cost) > 1e-9
ORDER BY day, net_cost DESC
"""


def fetch(config: Mapping[str, Any], window: DateWindow) -> list[CostEvent]:
    table = config.get("billing_export_table")
    if not table:
        raise CostFetchError("gcp: config.billing_export_table is required (the BigQuery billing export table)")
    location = config.get("location", "US")
    project = config.get("project")

    sql = _QUERY_TEMPLATE.format(
        table=table,
        start=window.start_dt.strftime("%Y-%m-%d %H:%M:%S+00"),
        end=window.end_exclusive_dt.strftime("%Y-%m-%d %H:%M:%S+00"),
    )
    rows = _run_bq(sql, location=location, project=project)
    events = [
        cost_event(
            provider=PROVIDER,
            day=dt.date.fromisoformat(row["day"]),
            category=row.get("service") or "unknown",
            detail=row.get("region") or "global",
            cost=float(row["net_cost"]),
            currency=str(row.get("currency", "USD")).upper(),
        )
        for row in rows
    ]
    logger.info("gcp: fetched %d cost rows for %s..%s", len(events), window.start, window.end)
    return events


def _run_bq(sql: str, *, location: str, project: str | None) -> list[dict[str, Any]]:
    cmd = ["bq", "query", "--use_legacy_sql=false", "--format=json", f"--max_rows={MAX_ROWS}", f"--location={location}"]
    if project:
        cmd.append(f"--project_id={project}")
    cmd.append(sql)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=BQ_TIMEOUT, check=True)
    except subprocess.CalledProcessError as exc:
        raise CostFetchError(f"gcp: bq query failed: {(exc.stderr or exc.stdout or str(exc)).strip()[:400]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CostFetchError(f"gcp: bq query timed out after {BQ_TIMEOUT:.0f}s") from exc
    stdout = result.stdout.strip()
    if not stdout:
        return []
    return json.loads(stdout)
