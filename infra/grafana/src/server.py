# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The metric API Grafana queries, backed by finelog.

Grafana never sends SQL here. The bridge exposes a fixed vocabulary — pick a
metric, a window, an optional label to group by, and optional label equality
filters — and writes the SQL itself. That is not a convenience: finelog runs
whatever DataFusion accepts and enforces its 64 MiB response cap only after
collecting the result, so a SQL-passthrough behind IAP would be an unbounded
query console onto the fleet's only telemetry store. A fixed vocabulary means
there is no caller-supplied SQL to police.

Routes (one datasource per cluster, addressed by path):

    GET /{cluster}/series?metric=&from=&to=[&group_by=][&label.k=v]
    GET /health

Handlers are sync defs: Starlette runs those in a threadpool, so a blocking
finelog query never stalls the event loop.
"""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import uvicorn
from cache import TtlCache
from config import BRIDGE_PORT, CLUSTERS, BridgeConfig, ClusterTarget
from finelog.errors import QueryResultTooLargeError
from finelog_source import FinelogSource, MetricSource
from series import build_sql, to_json_rows, to_series, validate_grouping
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

# Label equality filters arrive as `label.<key>=<value>` so they cannot collide
# with a reserved parameter name.
_LABEL_PREFIX = "label."


# Panels pass Grafana's ${__from}/${__to}, which interpolate to epoch millis; a
# human poking the API by hand sends an ISO instant. Accept both rather than
# making the dashboard the only usable client.
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


def _bucket(at: datetime, ttl: float) -> int:
    """Snap ``at`` to a TTL-wide bucket, so a window that drifts by seconds keeps one key."""
    return int(at.timestamp() // max(ttl, 1))


def _target_for(name: str, sources: Mapping[str, MetricSource]) -> ClusterTarget:
    """Return the target for ``name``, or raise ``_BadRequest`` naming the served clusters."""
    if name not in sources:
        raise _BadRequest(f"unknown cluster {name!r}; configured: {sorted(sources)}")
    return sources[name].target


def _series_query(request: Request, config: BridgeConfig, sources: Mapping[str, MetricSource], cache: TtlCache):
    params = request.query_params
    target = _target_for(request.path_params["cluster"], sources)

    metric = _require(params, "metric")
    try:
        start = _parse_time(_require(params, "from"), "from")
        end = _parse_time(_require(params, "to"), "to")
    except ValueError as err:
        raise _BadRequest(str(err)) from err
    group_by = params.get("group_by") or None
    match = {k[len(_LABEL_PREFIX) :]: v for k, v in params.items() if k.startswith(_LABEL_PREFIX)}

    if end <= start:
        raise _BadRequest(f"'to' ({end}) must be after 'from' ({start})")
    if end - start > timedelta(hours=config.max_window_hours):
        raise _BadRequest(
            f"window {end - start} exceeds the {config.max_window_hours}h limit; narrow the dashboard range"
        )

    # Cache on the query shape, with *both* window edges snapped to a TTL-wide
    # bucket. A dashboard on a relative range ("now-6h to now") moves both edges
    # every refresh, so keying on exact timestamps would miss the cache every
    # time — precisely the case the cache exists for. Snapping trades up to one
    # TTL of staleness at the window edges for a cache that actually hits.
    key = (
        target.name,
        metric,
        group_by,
        tuple(sorted(match.items())),
        _bucket(start, config.cache_ttl),
        _bucket(end, config.cache_ttl),
    )

    # Validate every caller-supplied identifier up front, where a rejection is
    # unambiguously the panel's fault. Past this point a ValueError means the rows
    # themselves are wrong (schema drift), which is a 500 — reporting our own data
    # bug as the caller's mistake would send whoever debugs it the wrong way.
    try:
        sql = build_sql(target.namespace, metric, start, end, limit=config.max_rows)
        validate_grouping(group_by, match)
    except ValueError as err:
        raise _BadRequest(str(err)) from err

    def run():
        logger.info("query %s: %s", target.name, sql)
        table = sources[target.name].query(sql, max_rows=config.max_rows)
        return to_json_rows(to_series(table, group_by=group_by, match=match))

    return cache.get_or_compute(key, run)


def create_app(config: BridgeConfig, sources: Mapping[str, MetricSource]) -> Starlette:
    """Build the ASGI app serving the clusters in ``sources``."""
    cache: TtlCache = TtlCache(config.cache_ttl)

    def series(request: Request) -> JSONResponse:
        try:
            return JSONResponse(_series_query(request, config, sources, cache))
        except _BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except QueryResultTooLargeError as err:
            return JSONResponse({"error": f"{err}; narrow the time range or add a label filter"}, status_code=400)

    def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "clusters": sorted(sources)})

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/{cluster}/series", series),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = BridgeConfig.from_environment()
    sources = {c.name: FinelogSource(c, timeout_ms=config.query_timeout_ms) for c in CLUSTERS}
    logger.info("grafana bridge serving %s on :%d", sorted(sources), BRIDGE_PORT)
    # Loopback only: Grafana's backend datasources fetch server-side from the
    # same container, so nothing outside it ever needs to reach this port.
    uvicorn.run(create_app(config, sources), host="127.0.0.1", port=BRIDGE_PORT, access_log=False)


if __name__ == "__main__":
    main()
