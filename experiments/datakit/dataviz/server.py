# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit dataviz dashboard: explore a clustered-store's pipeline stages via ducky.

Point it at one datakit store (``DATAVIZ_STORE``); it resolves the store's
upstream stage datasets (:mod:`experiments.datakit.dataviz.lineage`) and serves a
single-page dashboard with a tab per stage — normalized data, decontamination,
deduplication, quality classifier, and the final cluster x quality store. All
data is fetched by issuing SQL to the ducky service
(:mod:`experiments.datakit.dataviz.ducky`); the dashboard never reads parquet
directly.

Queries run **asynchronously** (``POST /api/query`` -> ``query_id``, poll
``GET /api/result/{id}``) so a slow aggregate never trips the controller proxy's
~30 s request cap — mirroring ducky itself.

Runs two ways:

* **In-cluster** (deployed by :mod:`experiments.datakit.dataviz.deploy`): binds the
  named Iris port and registers the ``dataviz`` endpoint so the controller proxy
  routes ``/proxy/dataviz/`` to it.
* **Local** (``python -m experiments.datakit.dataviz.server --store gs://...``):
  plain uvicorn on ``--port`` for development.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import enum
import logging
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

import uvicorn
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.dashboard_common import on_shutdown
from marin.execution.artifact import read_artifact
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from experiments.datakit.dataviz.ducky import DEFAULT_BASE_URL, DuckyClient, DuckyError, iap_token_provider
from experiments.datakit.dataviz.lineage import StoreLineage, load_lineage, resolve_lineage, save_lineage
from experiments.datakit.dataviz.queries import Dataviz
from experiments.datakit.store.datakit_store import ClusteredStoreData

logger = logging.getLogger(__name__)

_INDEX_HTML = Path(__file__).with_name("index.html")
_MAX_WORKERS = 8

# Iris named port + endpoint the deployed service binds/registers; the controller
# proxy routes ``/proxy/dataviz/`` to the namespaced endpoint. Must match deploy.py.
PORT_NAME = "dataviz"
ENDPOINT_NAME = "/dataviz"


def _quality_range(quality_bucket: int, thresholds: list[float]) -> str:
    lo = "0.0" if quality_bucket == 0 else f"{thresholds[quality_bucket - 1]:g}"
    hi = "1.0" if quality_bucket >= len(thresholds) else f"{thresholds[quality_bucket]:g}"
    upper = "]" if quality_bucket >= len(thresholds) else ")"
    return f"[{lo}, {hi}{upper}"


class QueryStatus(enum.StrEnum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclasses.dataclass
class _QueryState:
    status: QueryStatus
    result: object = None
    error: str | None = None


def _source_docs(source_summary: list[dict] | None) -> dict[str, int]:
    return {r["source"]: r["docs_est"] for r in source_summary or [] if r.get("docs_est")}


def _build_views(dv: Dataviz) -> dict[str, Callable[[dict], object]]:
    """Map dashboard view name -> handler(params) -> JSON-serializable result."""
    return {
        "normalized_stats": lambda p: dv.normalized_stats(p["source"]),
        "normalized_hist": lambda p: dv.normalized_length_hist(p["source"]).dicts(),
        "normalized_samples": (
            lambda p: dv.normalized_samples(p["source"], int(p.get("n", 20)), p.get("search", "")).dicts()
        ),
        "decontam_stats": lambda p: dv.decontam_stats(p["source"]),
        "decontam_samples": lambda p: dv.decontam_samples(p["source"], int(p.get("n", 20))).dicts(),
        "quality_hist": lambda p: dv.quality_hist(p["source"]).dicts(),
        "quality_samples": (
            lambda p: dv.quality_samples(p["source"], float(p["lo"]), float(p["hi"]), int(p.get("n", 20))).dicts()
        ),
        "store_samples": lambda p: dv.store_cluster_samples(int(p["cluster"]), int(p.get("n", 12))),
    }


class QueryManager:
    """Runs dashboard views on a thread pool; the HTTP layer polls for results."""

    def __init__(self, views: dict[str, Callable[[dict], object]]):
        self._views = views
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)
        self._states: dict[str, _QueryState] = {}
        self._lock = threading.Lock()

    def submit(self, view: str, params: dict) -> str:
        if view not in self._views:
            raise KeyError(view)
        query_id = uuid.uuid4().hex
        with self._lock:
            self._states[query_id] = _QueryState(QueryStatus.RUNNING)
        self._pool.submit(self._run, query_id, view, params)
        return query_id

    def _run(self, query_id: str, view: str, params: dict) -> None:
        try:
            result = self._views[view](params)
            state = _QueryState(QueryStatus.DONE, result=result)
        except (DuckyError, KeyError, ValueError) as e:
            logger.warning("view %s failed: %s", view, e)
            state = _QueryState(QueryStatus.ERROR, error=str(e))
        except Exception as e:
            logger.exception("view %s crashed", view)
            state = _QueryState(QueryStatus.ERROR, error=f"internal error: {e}")
        with self._lock:
            self._states[query_id] = state

    def get(self, query_id: str) -> _QueryState | None:
        with self._lock:
            return self._states.get(query_id)


def build_app(
    lineage: StoreLineage,
    payload: ClusteredStoreData,
    ducky: DuckyClient,
    source_summary: list[dict] | None = None,
) -> Starlette:
    manager = QueryManager(_build_views(Dataviz(lineage, ducky, _source_docs(source_summary))))

    def overview() -> dict:
        buckets = [
            {
                "cluster_id": b.cluster_id,
                "quality_bucket": b.quality_bucket,
                "quality_range": _quality_range(b.quality_bucket, payload.quality_thresholds),
                "total_elements": b.total_elements,
                "total_tokens": b.total_tokens,
            }
            for b in payload.buckets
        ]
        return {
            "store_path": lineage.store_path,
            "data_prefix": lineage.data_prefix,
            "cluster_view": lineage.cluster_view,
            "quality_thresholds": lineage.quality_thresholds,
            "n_quality_buckets": len(lineage.quality_thresholds) + 1,
            "tokenizer": lineage.tokenizer,
            "verified": lineage.verified,
            "sources": lineage.source_names,
            "resolved": {
                "normalize": sorted(lineage.normalize),
                "decontam": sorted(lineage.decontam),
                "cluster_assign": sorted(lineage.cluster_assign),
                "quality": sorted(lineage.quality),
            },
            "dedup": lineage.dedup,
            "counters": payload.counters,
            "buckets": buckets,
            "source_summary": source_summary or [],
        }

    async def index(_request: Request) -> FileResponse:
        return FileResponse(_INDEX_HTML)

    async def api_overview(_request: Request) -> JSONResponse:
        return JSONResponse(overview())

    async def api_query(request: Request) -> JSONResponse:
        body = await request.json()
        view = body.get("view")
        try:
            query_id = manager.submit(view, body.get("params", {}))
        except KeyError:
            return JSONResponse({"error": f"unknown view {view!r}"}, status_code=400)
        return JSONResponse({"query_id": query_id}, status_code=202)

    async def api_result(request: Request) -> JSONResponse:
        state = manager.get(request.path_params["query_id"])
        if state is None:
            return JSONResponse({"error": "unknown query_id"}, status_code=404)
        if state.status is QueryStatus.RUNNING:
            return JSONResponse({"status": "running"})
        if state.status is QueryStatus.ERROR:
            return JSONResponse({"status": "error", "error": state.error})
        return JSONResponse({"status": "done", "result": state.result})

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "healthy"})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/overview", api_overview),
            Route("/api/query", api_query, methods=["POST"]),
            Route("/api/result/{query_id:str}", api_result),
            Route("/health", health),
        ]
    )


def _build_ducky(explicit_url: str | None) -> DuckyClient:
    """Pick the ducky endpoint + auth for the current environment.

    * explicit ``--ducky-url`` / ``DATAVIZ_DUCKY_URL`` — used as-is; IAP token only
      when it targets the public ``iris.oa.dev`` ingress.
    * in-cluster (``IRIS_CONTROLLER_URL`` set) — the controller's internal proxy
      (``<controller>/proxy/ducky``), no token: the internal port trusts the
      in-cluster path (IAP is enforced only on the external ingress).
    * local dev — the public IAP proxy with a service-account token.
    """
    if explicit_url:
        needs_iap = "iris.oa.dev" in explicit_url
        url = explicit_url
    elif os.environ.get("IRIS_CONTROLLER_URL"):
        url, needs_iap = f"{os.environ['IRIS_CONTROLLER_URL'].rstrip('/')}/proxy/ducky", False
    else:
        url, needs_iap = DEFAULT_BASE_URL, True
    logger.info("ducky endpoint %s (iap=%s)", url, needs_iap)
    return DuckyClient(url, token_provider=iap_token_provider() if needs_iap else None)


def _load(store_path: str, ducky: DuckyClient, cache_path: str | None) -> tuple[StoreLineage, ClusteredStoreData]:
    payload = read_artifact(store_path, ClusteredStoreData)
    if cache_path and os.path.exists(cache_path):
        logger.info("loading cached lineage from %s", cache_path)
        lineage = load_lineage(cache_path)
    else:
        logger.info("resolving lineage for %s (this issues ducky globs; ~1-2 min)", store_path)
        lineage = resolve_lineage(
            store_path,
            ducky,
            domain_centroids=os.environ.get("DATAVIZ_DOMAIN_CENTROIDS"),
            quality_model=os.environ.get("DATAVIZ_QUALITY_MODEL"),
        )
        if cache_path:
            save_lineage(lineage, cache_path)
            logger.info("cached lineage to %s", cache_path)
    return lineage, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=os.environ.get("DATAVIZ_STORE"), help="Datakit store artifact path (gs://).")
    parser.add_argument("--ducky-url", default=os.environ.get("DATAVIZ_DUCKY_URL"))
    parser.add_argument("--lineage-cache", default=os.environ.get("DATAVIZ_LINEAGE_CACHE"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not args.store:
        raise SystemExit("--store (or DATAVIZ_STORE) is required")

    ducky = _build_ducky(args.ducky_url)
    lineage, payload = _load(args.store, ducky, args.lineage_cache)
    source_summary = None
    summary_path = os.environ.get("DATAVIZ_SOURCE_SUMMARY")
    if summary_path:
        import json  # noqa: PLC0415 — only needed on this optional path

        from rigging.filesystem import open_url  # noqa: PLC0415

        with open_url(summary_path, "r") as f:
            source_summary = json.load(f)
        logger.info("loaded source summary (%d rows) from %s", len(source_summary), summary_path)
    app = build_app(lineage, payload, ducky, source_summary)
    logger.info(
        "dataviz for %s: %d sources, %d buckets",
        lineage.store_path,
        len(lineage.source_names),
        len(payload.buckets),
    )

    # In-cluster: bind the named Iris port and register the endpoint so the
    # controller proxy routes /proxy/dataviz/ here. Local dev: plain uvicorn.
    job_info = get_job_info()
    if job_info is None:
        logger.info("serving locally on http://%s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    ctx = iris_ctx()
    port = ctx.get_port(PORT_NAME)
    address = f"http://{job_info.advertise_host}:{port}"
    endpoint_id = ctx.registry.register(ENDPOINT_NAME, address, {"job_id": ctx.job_id.to_wire()})
    logger.info("dataviz registered as %s at %s", ENDPOINT_NAME, address)

    async def _on_shutdown() -> None:
        ctx.registry.unregister(endpoint_id)
        logger.info("dataviz endpoint unregistered")

    app.router.lifespan_context = on_shutdown(_on_shutdown)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
