# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The data API Grafana queries: finelog SQL plus the live Iris and GitHub sources.

A finelog panel sends SQL and a time window; the bridge substitutes the window
macros, runs the SQL against finelog's Query RPC, shapes the Arrow result into JSON
rows, and caches per (cluster, SQL, window bucket). The Iris and GitHub routes are
fixed — the bridge owns their query and shape and returns flat JSON rows — so the
dashboard never sends admin RPC SQL, and every route feeds Infinity's backend parser.

Routes, grouped by source (cluster is a path segment where it applies):

    GET /finelog/{cluster}/query?sql=&from=&to=  finelog SQL (window macros, cached per bucket)
    GET /finelog/marin/fleet_health              hub query health + k8s mirror readiness
    GET /finelog/marin/alerts/fleet_health       alert rows: server labels + value(0|1)
    GET /iris/{cluster}/jobs                     root-job counts by state (in-flight + 24h terminal)
    GET /iris/{cluster}/workers                  healthy worker counts + resource totals per region
    GET /iris/{cluster}/health                   controller reachability + latency
    GET /iris/{cluster}/query?sql=               ad-hoc SELECT via ExecuteRawQuery (admin/null-auth)
    GET /github/ferries                          recent ferry runs per tier, with success rate
    GET /github/builds                           recent main commits with CI rollup state
    GET /github/nightlies                        7-day nightly-lane matrix (one row per lane/day)
    GET /wandb/{chart}                           sampled public hero-report series by chart key
    GET /k8s/control_plane                       watched components + webhook endpoints, all clusters
    GET /k8s/crashloops                          containers in backoff waiting states
    GET /k8s/pending                             Pending / SchedulingGated pods with age
    GET /k8s/termination_candidates             pods overdue past their deletion deadline
    GET /k8s/kueue                               unadmitted Kueue workloads per queue
    GET /k8s/events                              recent Warning events
    GET /k8s/health                              per-cluster API server reachability + latency
    GET /k8s/overview                            explicit workload issue counts (zeros included)
    GET /k8s/gpu_racks                           GPU nodes grouped by physical rack: trays total/ready
    GET /k8s/alerts/unreachable                  alert rows: cluster, error_class, value(0|1)
    GET /k8s/alerts/crashloops?scope=            alert rows: cluster, scope, value(count)
    GET /k8s/alerts/webhook_ready                alert rows: cluster, webhook, value(ready count)
    GET /k8s/alerts/degraded                     alert rows: cluster, component, value(desired-ready)
    GET /k8s/alerts/stuck_gpu_pods                alert rows: cluster, node, value(count)
    GET /k8s/alerts/gpu_rack_trays                alert rows: cluster, rack_name, value(trays_ready)
    GET /health                                  bridge liveness

A dead controller or GitHub returns 5xx (not empty rows), and the failure is not
cached. The k8s routes aggregate every CW cluster into one response, so a dead
cluster becomes labeled error rows while the rest render; the alert routes always
return at least one row per cluster (explicit zeros when healthy) so Grafana
rules never hit NoData. Handlers are sync defs; Starlette runs them in a
threadpool.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime

import pyarrow as pa
import uvicorn
from cache import TtlCache
from config import BRIDGE_PORT, CLUSTERS, FINELOG_SLOW_THRESHOLD_MS, K8S_CLUSTERS, BridgeConfig, ClusterTarget
from errors import UpstreamError
from finelog.errors import QueryResultTooLargeError
from finelog_health import FinelogHealth
from finelog_source import FinelogSource, MetricSource
from github_source import GithubSource
from iris_source import IrisSource
from k8s_source import K8sFleet, K8sSource
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from wandb_source import WandbSource

logger = logging.getLogger(__name__)

# Window macros a panel writes into its SQL, substituted with tz-naive UTC
# TIMESTAMP literals before the query runs.
FROM_MACRO = "{{from}}"
TO_MACRO = "{{to}}"

# infra/probes writes its label set as a JSON object string. The bridge expands
# it into columns under this prefix so a panel can select one as a series.
LABELS_COLUMN = "labels"
LABEL_PREFIX = "label_"
_K8S_TERMINATION_CANDIDATES_CACHE_KEY = "termination_candidates"
_FINELOG_HUB_CLUSTER = "marin"


def workload_overview(pending_rows: list[dict], crashloop_rows: list[dict]) -> list[dict]:
    """Summarize workload issue rows into one stat-safe row with explicit zeros."""
    return [
        {
            "pending_pods": sum("pod" in row for row in pending_rows),
            "crashlooping_containers": sum("container" in row for row in crashloop_rows),
        }
    ]


def finelog_alert_rows(health_rows: list[FinelogHealth]) -> list[dict]:
    """Project fleet health into Grafana's one-numeric-column alert contract."""
    alerts = []
    for row in health_rows:
        if not row.responsive:
            state = "unresponsive"
        elif row.latency_ms is not None and row.latency_ms >= FINELOG_SLOW_THRESHOLD_MS:
            state = "slow"
        else:
            state = "healthy"
        alerts.append(
            {
                "cluster": row.cluster,
                "server": row.server,
                "role": row.role,
                "state": state,
                "error_class": row.error_class,
                "value": 0 if state == "healthy" else 1,
            }
        )
    return alerts


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


def _labels_as_dict(raw: object) -> dict | None:
    """Coerce a labels cell to a ``{key: value}`` dict, or None if it isn't one.

    Handles both label encodings finelog serves: a JSON-string column (the probes
    EAV convention) and a native ``Map<Utf8,Utf8>`` column, which arrives from
    ``Table.to_pylist()`` as a ``list[(key, value)]`` (or a dict).
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        try:
            return dict(raw)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _flatten_labels(row: dict[str, object]) -> dict[str, object]:
    """Expand a labels cell into label_<key> fields, dropping the raw cell.

    A cell that is neither a JSON object nor a native map stays in place and is
    logged.
    """
    raw = row.get(LABELS_COLUMN)
    if raw is None:
        return row
    parsed = _labels_as_dict(raw)
    if parsed is None:
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


def _iris_for(name: str, sources: Mapping[str, IrisSource]) -> IrisSource:
    if name not in sources:
        raise _BadRequest(f"unknown cluster {name!r}; configured: {sorted(sources)}")
    return sources[name]


def create_app(
    config: BridgeConfig,
    finelog_sources: Mapping[str, MetricSource],
    iris_sources: Mapping[str, IrisSource],
    github_source: GithubSource,
    k8s_fleet: K8sFleet,
    wandb_source: WandbSource,
) -> Starlette:
    """Build the ASGI app serving finelog, Iris, GitHub, W&B, and k8s sources."""
    finelog_cache: TtlCache = TtlCache(config.cache_ttl)
    finelog_health_cache: TtlCache = TtlCache(config.k8s_cache_ttl)
    iris_cache: TtlCache = TtlCache(config.iris_cache_ttl)
    github_cache: TtlCache = TtlCache(config.github_cache_ttl)
    k8s_cache: TtlCache = TtlCache(config.k8s_cache_ttl)
    wandb_cache: TtlCache = TtlCache(config.github_cache_ttl)

    def query(request: Request) -> JSONResponse:
        try:
            return JSONResponse(_query(request, config, finelog_sources, finelog_cache))
        except _BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except QueryResultTooLargeError as err:
            return JSONResponse({"error": f"{err}; narrow the time range or aggregate"}, status_code=400)

    def fleet_health_rows() -> list[FinelogHealth]:
        _target_for(_FINELOG_HUB_CLUSTER, finelog_sources)
        return finelog_health_cache.get_or_compute(
            "fleet_health",
            lambda: [finelog_sources[_FINELOG_HUB_CLUSTER].health(), *k8s_fleet.finelog_health()],
        )

    def finelog_fleet_health(_: Request) -> JSONResponse:
        try:
            return JSONResponse([asdict(row) for row in fleet_health_rows()])
        except _BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=400)

    def finelog_alerts_fleet_health(_: Request) -> JSONResponse:
        try:
            return JSONResponse(finelog_alert_rows(fleet_health_rows()))
        except _BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=400)

    def iris_endpoint(request: Request, endpoint: str, run) -> JSONResponse:
        try:
            source = _iris_for(request.path_params["cluster"], iris_sources)
            return JSONResponse(iris_cache.get_or_compute((source.target.name, endpoint), lambda: run(source)))
        except _BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except UpstreamError as err:
            return JSONResponse({"error": str(err), "source": err.source}, status_code=err.status_code)

    def iris_jobs(request: Request) -> JSONResponse:
        return iris_endpoint(request, "jobs", lambda s: s.jobs())

    def iris_workers(request: Request) -> JSONResponse:
        return iris_endpoint(request, "workers", lambda s: s.workers())

    def iris_health(request: Request) -> JSONResponse:
        return iris_endpoint(request, "health", lambda s: s.health())

    def iris_query(request: Request) -> JSONResponse:
        # Ad-hoc SELECT: not cached (arbitrary SQL) and not used by any committed panel.
        try:
            source = _iris_for(request.path_params["cluster"], iris_sources)
            sql = _require(request.query_params, "sql")
            return JSONResponse(source.raw_query(sql))
        except _BadRequest as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except UpstreamError as err:
            return JSONResponse({"error": str(err), "source": err.source}, status_code=err.status_code)

    def github_endpoint(key: str, run) -> JSONResponse:
        try:
            return JSONResponse(github_cache.get_or_compute(key, run))
        except UpstreamError as err:
            return JSONResponse({"error": str(err), "source": err.source}, status_code=err.status_code)

    def github_ferries(_: Request) -> JSONResponse:
        return github_endpoint("ferries", github_source.ferries)

    def github_builds(_: Request) -> JSONResponse:
        return github_endpoint("builds", github_source.builds)

    def github_nightlies(_: Request) -> JSONResponse:
        return github_endpoint("nightlies", github_source.nightlies)

    def wandb_chart(request: Request) -> JSONResponse:
        chart = request.path_params["chart"]
        try:
            return JSONResponse(wandb_cache.get_or_compute(chart, lambda: wandb_source.points(chart)))
        except ValueError as err:
            return JSONResponse({"error": str(err)}, status_code=400)
        except UpstreamError as err:
            return JSONResponse({"error": str(err), "source": err.source}, status_code=err.status_code)

    def k8s_endpoint(key: str, run) -> JSONResponse:
        # Per-cluster failures are labeled rows inside the response; only a bridge
        # bug raises here, and Starlette turns that into a 500.
        return JSONResponse(k8s_cache.get_or_compute(key, run))

    def k8s_control_plane(_: Request) -> JSONResponse:
        return k8s_endpoint("control_plane", k8s_fleet.control_plane)

    def k8s_crashloops(_: Request) -> JSONResponse:
        return k8s_endpoint("crashloops", k8s_fleet.crashloops)

    def k8s_pending(_: Request) -> JSONResponse:
        return k8s_endpoint("pending", k8s_fleet.pending)

    def k8s_termination_candidates(_: Request) -> JSONResponse:
        rows = k8s_cache.get_or_compute(_K8S_TERMINATION_CANDIDATES_CACHE_KEY, k8s_fleet.termination_candidates)
        return JSONResponse([asdict(row) for row in rows])

    def k8s_kueue(_: Request) -> JSONResponse:
        return k8s_endpoint("kueue", k8s_fleet.kueue)

    def k8s_events(_: Request) -> JSONResponse:
        return k8s_endpoint("events", k8s_fleet.warning_events)

    def k8s_health(_: Request) -> JSONResponse:
        return k8s_endpoint("health", k8s_fleet.health)

    def k8s_overview(_: Request) -> JSONResponse:
        def compute() -> list[dict]:
            pending = k8s_cache.get_or_compute("pending", k8s_fleet.pending)
            crashloops = k8s_cache.get_or_compute("crashloops", k8s_fleet.crashloops)
            return workload_overview(pending, crashloops)

        return k8s_endpoint("overview", compute)

    def k8s_gpu_racks(_: Request) -> JSONResponse:
        return k8s_endpoint("gpu_racks", k8s_fleet.gpu_racks)

    def k8s_alerts_unreachable(_: Request) -> JSONResponse:
        return k8s_endpoint("alerts_unreachable", k8s_fleet.alert_unreachable)

    def k8s_alerts_crashloops(request: Request) -> JSONResponse:
        # The paging rule asks for scope=control-plane; workload backoffs stay
        # observe-only. Filtering after the cache keeps one scan per TTL.
        response = k8s_cache.get_or_compute("alerts_crashloops", k8s_fleet.alert_crashloops)
        scope = request.query_params.get("scope")
        if scope:
            response = [row for row in response if row["scope"] == scope]
        return JSONResponse(response)

    def k8s_alerts_webhook_ready(_: Request) -> JSONResponse:
        return k8s_endpoint("alerts_webhook_ready", k8s_fleet.alert_webhook_ready)

    def k8s_alerts_degraded(_: Request) -> JSONResponse:
        return k8s_endpoint("alerts_degraded", k8s_fleet.alert_degraded)

    def k8s_alerts_gpu_rack_trays(_: Request) -> JSONResponse:
        return k8s_endpoint("alerts_gpu_rack_trays", k8s_fleet.alert_gpu_rack_trays)

    def k8s_alerts_stuck_gpu_pods(_: Request) -> JSONResponse:
        # The dashboard and alert projection share one fleet LIST per cache TTL.
        rows = k8s_cache.get_or_compute(_K8S_TERMINATION_CANDIDATES_CACHE_KEY, k8s_fleet.termination_candidates)
        return JSONResponse(k8s_fleet.alert_stuck_gpu_pods(rows))

    def health(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "clusters": sorted(finelog_sources)})

    return Starlette(
        routes=[
            Route("/health", health),
            Route("/github/ferries", github_ferries),
            Route("/github/builds", github_builds),
            Route("/github/nightlies", github_nightlies),
            Route("/wandb/{chart}", wandb_chart),
            Route("/finelog/{cluster}/query", query),
            Route(f"/finelog/{_FINELOG_HUB_CLUSTER}/fleet_health", finelog_fleet_health),
            Route(f"/finelog/{_FINELOG_HUB_CLUSTER}/alerts/fleet_health", finelog_alerts_fleet_health),
            Route("/iris/{cluster}/jobs", iris_jobs),
            Route("/iris/{cluster}/workers", iris_workers),
            Route("/iris/{cluster}/health", iris_health),
            Route("/iris/{cluster}/query", iris_query),
            Route("/k8s/control_plane", k8s_control_plane),
            Route("/k8s/crashloops", k8s_crashloops),
            Route("/k8s/pending", k8s_pending),
            Route("/k8s/termination_candidates", k8s_termination_candidates),
            Route("/k8s/kueue", k8s_kueue),
            Route("/k8s/events", k8s_events),
            Route("/k8s/health", k8s_health),
            Route("/k8s/overview", k8s_overview),
            Route("/k8s/gpu_racks", k8s_gpu_racks),
            Route("/k8s/alerts/unreachable", k8s_alerts_unreachable),
            Route("/k8s/alerts/crashloops", k8s_alerts_crashloops),
            Route("/k8s/alerts/webhook_ready", k8s_alerts_webhook_ready),
            Route("/k8s/alerts/degraded", k8s_alerts_degraded),
            Route("/k8s/alerts/gpu_rack_trays", k8s_alerts_gpu_rack_trays),
            Route("/k8s/alerts/stuck_gpu_pods", k8s_alerts_stuck_gpu_pods),
        ]
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = BridgeConfig.from_environment()
    finelog_sources = {c.name: FinelogSource(c, timeout_ms=config.query_timeout_ms) for c in CLUSTERS}
    iris_sources = {c.name: IrisSource(c, timeout=config.http_timeout) for c in CLUSTERS}
    github_source = GithubSource(token=config.github_token, timeout=config.http_timeout)
    k8s_fleet = K8sFleet([K8sSource(c, token=config.cw_read_token, timeout=config.http_timeout) for c in K8S_CLUSTERS])
    wandb_source = WandbSource(timeout=config.http_timeout)
    logger.info("grafana bridge serving %s on :%d", sorted(finelog_sources), BRIDGE_PORT)
    # Loopback only: Grafana fetches from the same container.
    uvicorn.run(
        create_app(config, finelog_sources, iris_sources, github_source, k8s_fleet, wandb_source),
        host="127.0.0.1",
        port=BRIDGE_PORT,
        access_log=False,
    )


if __name__ == "__main__":
    main()
