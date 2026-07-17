# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The metric API Grafana queries, backed by finelog.

A panel sends SQL and a time window; the bridge substitutes the window macros,
runs the SQL against finelog's Query RPC, shapes the Arrow result into JSON rows,
and caches the result per (cluster, SQL, window bucket) for the configured TTL.

Routes, one datasource per cluster addressed by path:

    GET /{cluster}/query?sql=&from=&to=
    GET /health

sql carries the {{from}} / {{to}} window macros. The cache key snaps the window
to a TTL-wide bucket, so a relative range keeps one key as its edges drift.
Handlers are sync defs; Starlette runs them in a threadpool.
"""

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime

import pyarrow as pa
import uvicorn
from cache import TtlCache
from config import BRIDGE_PORT, CLUSTERS, BridgeConfig, ClusterTarget
from finelog.errors import QueryResultTooLargeError
from finelog_source import FinelogSource, MetricSource
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Window macros a panel writes into its SQL, substituted with tz-naive UTC
# TIMESTAMP literals before the query runs.
FROM_MACRO = "{{from}}"
TO_MACRO = "{{to}}"

# infra/probes writes its label set as a JSON object string. The bridge expands
# it into columns under this prefix so a panel can select one as a series.
LABELS_COLUMN = "labels"
LABEL_PREFIX = "label_"


def _sql_timestamp(at: datetime) -> str:
    """Format at as the tz-naive UTC literal finelog compares timestamps against."""
    return at.strftime("%Y-%m-%d %H:%M:%S")


def substitute_time_macros(sql: str, start: datetime | None, end: datetime | None) -> str:
    """Replace {{from}} / {{to}} with TIMESTAMP literals.

    Raises ValueError if the SQL uses a macro without the matching bound.
    """
    for macro, at in ((FROM_MACRO, start), (TO_MACRO, end)):
        if macro in sql:
            if at is None:
                raise ValueError(f"SQL uses {macro} but no matching time bound was supplied")
            sql = sql.replace(macro, f"TIMESTAMP '{_sql_timestamp(at)}'")
    return sql


def _json_safe(value: object) -> object:
    """Coerce one Arrow cell into a JSON-serializable value.

    Timestamps become epoch milliseconds (naive cells read as UTC); bytes become
    text; everything else passes through.
    """
    if isinstance(value, datetime):
        at = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return round(at.timestamp() * 1000)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return value


def _flatten_labels(row: dict[str, object]) -> dict[str, object]:
    """Expand a JSON labels cell into label_<key> fields, dropping the raw cell.

    A cell that does not parse to a JSON object stays in place and is logged.
    """
    raw = row.get(LABELS_COLUMN)
    if raw is None:
        return row
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("labels is not a JSON object")
    except (ValueError, TypeError):
        logger.warning("row has unparseable labels: %.200r", raw)
        return row
    flattened = {key: value for key, value in row.items() if key != LABELS_COLUMN}
    for key, value in parsed.items():
        flattened[f"{LABEL_PREFIX}{key}"] = value
    return flattened


def rows_to_json(table: pa.Table) -> list[dict[str, object]]:
    """Turn a finelog Arrow result into JSON rows, flattening any labels column."""
    has_labels = LABELS_COLUMN in table.column_names
    rows: list[dict[str, object]] = []
    for row in table.to_pylist():
        if has_labels:
            row = _flatten_labels(row)
        rows.append({key: _json_safe(value) for key, value in row.items()})
    return rows


def _parse_time(raw: str, field: str) -> datetime:
    """Parse epoch millis (Grafana's ${__from}/${__to}) or an ISO instant."""
    try:
        return datetime.fromtimestamp(int(raw) / 1000, tz=UTC)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as err:
        raise ValueError(f"{field} must be epoch millis or an ISO instant, got {raw!r}") from err


class _BadRequest(Exception):
    """A malformed request, surfaced as HTTP 400."""


def _require(params, name: str) -> str:
    value = params.get(name)
    if not value:
        raise _BadRequest(f"missing required parameter {name!r}")
    return value


def _optional_time(params, name: str) -> datetime | None:
    raw = params.get(name)
    if not raw:
        return None
    try:
        return _parse_time(raw, name)
    except ValueError as err:
        raise _BadRequest(str(err)) from err


def _bucket(at: datetime | None, ttl: float) -> int | None:
    """Snap at to a TTL-wide bucket so a drifting window keeps one cache key."""
    return None if at is None else int(at.timestamp() // max(ttl, 1))


def _target_for(name: str, sources: Mapping[str, MetricSource]) -> ClusterTarget:
    """Return the target for name, or raise _BadRequest naming the served clusters."""
    if name not in sources:
        raise _BadRequest(f"unknown cluster {name!r}; configured: {sorted(sources)}")
    return sources[name].target


def _query(request: Request, config: BridgeConfig, sources: Mapping[str, MetricSource], cache: TtlCache):
    target = _target_for(request.path_params["cluster"], sources)
    params = request.query_params

    sql = _require(params, "sql")
    start = _optional_time(params, "from")
    end = _optional_time(params, "to")

    # Key on the SQL as written, before substitution, with each window edge snapped
    # to a TTL bucket, so a relative range stays one key as its edges drift.
    key = (target.name, sql, _bucket(start, config.cache_ttl), _bucket(end, config.cache_ttl))

    try:
        effective_sql = substitute_time_macros(sql, start, end)
    except ValueError as err:
        raise _BadRequest(str(err)) from err

    def run():
        logger.info("query %s: %s", target.name, effective_sql)
        table = sources[target.name].query(effective_sql, max_rows=config.max_rows)
        return rows_to_json(table)

    return cache.get_or_compute(key, run)


def create_app(config: BridgeConfig, sources: Mapping[str, MetricSource]) -> Starlette:
    """Build the ASGI app serving the clusters in sources."""
    cache: TtlCache = TtlCache(config.cache_ttl)

    def query(request: Request) -> JSONResponse:
        try:
            return JSONResponse(_query(request, config, sources, cache))
        except _BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except QueryResultTooLargeError as err:
            return JSONResponse({"error": f"{err}; narrow the time range or aggregate"}, status_code=400)

    def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "clusters": sorted(sources)})

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/{cluster}/query", query),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = BridgeConfig.from_environment()
    sources = {c.name: FinelogSource(c, timeout_ms=config.query_timeout_ms) for c in CLUSTERS}
    logger.info("grafana bridge serving %s on :%d", sorted(sources), BRIDGE_PORT)
    # Loopback only: Grafana fetches from the same container.
    uvicorn.run(create_app(config, sources), host="127.0.0.1", port=BRIDGE_PORT, access_log=False)


if __name__ == "__main__":
    main()
