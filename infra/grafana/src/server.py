# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The metric API Grafana queries, backed by finelog.

A panel sends SQL and the window; the bridge runs it against finelog and returns
JSON rows. finelog gates the SQL to SELECT and enforces a server-side deadline
(#7312), so the bridge does not police the query. It exists to convert Arrow to
JSON, cache results (Grafana OSS has no query caching), and call only the Query
RPC — pointing Grafana at finelog's host would expose WriteRows and DropTable,
which finelog admits from the same VPC.

Routes (one datasource per cluster, addressed by path):

    GET /{cluster}/query?sql=&from=&to=
    GET /health

``sql`` uses the ``{{from}}`` / ``{{to}}`` window macros (see results.py) so the
cache key stays stable as a relative range drifts. Handlers are sync defs:
Starlette runs those in a threadpool, so a blocking finelog query never stalls the
event loop.
"""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime

import uvicorn
from cache import TtlCache
from config import BRIDGE_PORT, CLUSTERS, BridgeConfig, ClusterTarget
from finelog.errors import QueryResultTooLargeError
from finelog_source import FinelogSource, MetricSource
from results import rows_to_json, substitute_time_macros
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)


# Panels pass Grafana's ${__from}/${__to}, which interpolate to epoch millis; a
# human poking the API by hand sends an ISO instant. Accept both.
def _parse_time(raw: str, field: str) -> datetime:
    try:
        return datetime.fromtimestamp(int(raw) / 1000, tz=UTC)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as err:
        raise ValueError(f"{field} must be epoch millis or an ISO instant, got {raw!r}") from err


class _BadRequest(Exception):
    """A malformed panel query. Surfaced as 400 — the caller is wrong, not the server."""


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
    """Snap ``at`` to a TTL-wide bucket so a window that drifts by seconds keeps one key."""
    return None if at is None else int(at.timestamp() // max(ttl, 1))


def _target_for(name: str, sources: Mapping[str, MetricSource]) -> ClusterTarget:
    """Return the target for ``name``, or raise ``_BadRequest`` naming the served clusters."""
    if name not in sources:
        raise _BadRequest(f"unknown cluster {name!r}; configured: {sorted(sources)}")
    return sources[name].target


def _query(request: Request, config: BridgeConfig, sources: Mapping[str, MetricSource], cache: TtlCache):
    target = _target_for(request.path_params["cluster"], sources)
    params = request.query_params

    sql = _require(params, "sql")
    start = _optional_time(params, "from")
    end = _optional_time(params, "to")

    # Cache on the SQL as written — before macro substitution — with each window
    # edge snapped to a TTL-wide bucket. A relative range ("now-6h to now") moves
    # both edges every refresh, so keying on the substituted timestamps would miss
    # the cache every time, precisely the case the cache exists for.
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
    """Build the ASGI app serving the clusters in ``sources``."""
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
    # Loopback only: Grafana's backend datasources fetch server-side from the same
    # container, so nothing outside it ever needs to reach this port.
    uvicorn.run(create_app(config, sources), host="127.0.0.1", port=BRIDGE_PORT, access_log=False)


if __name__ == "__main__":
    main()
