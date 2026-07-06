# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit web explorer dashboard: explore a clustered-store's pipeline stages via ducky.

Point it at one datakit store (``WEB_EXPLORER_STORE``); it resolves the store's
upstream stage datasets (:mod:`experiments.datakit.web_explorer.lineage`) and serves a
Vue SPA (built into ``dashboard/dist``) with a tab per stage — normalized data,
decontamination, deduplication, quality classifier, and the final cluster x quality
store. Most data is fetched by issuing SQL to the ducky service
(:class:`ducky.client.DuckyClient`); the store's tokenized bucket caches are read
directly and detokenized for the "from store cache" view
(:mod:`experiments.datakit.web_explorer.store_cache`).

Queries run **asynchronously** (``POST /api/query`` -> ``query_id``, poll
``GET /api/result/{id}``) so a slow aggregate never trips the controller proxy's
~30 s request cap — mirroring ducky itself.

Runs two ways:

* **In-cluster** (deployed by :mod:`experiments.datakit.web_explorer.deploy`): binds the
  named Iris port and registers the ``web_explorer`` endpoint so the controller proxy
  routes ``/proxy/datakit_explorer/`` to it.
* **Local** (``python -m experiments.datakit.web_explorer.server --store gs://...``):
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
from ducky.client import DuckyClient, DuckyError, iap_token_provider
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.dashboard_common import on_shutdown
from marin.utils import fsspec_exists
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from experiments.datakit.store.datakit_store import ClusteredStoreData
from experiments.datakit.web_explorer.config import ENDPOINT_NAME, PORT_NAME, WebExplorerConfig
from experiments.datakit.web_explorer.lineage import (
    StoreLineage,
    load_lineage,
    read_store_payload,
    resolve_lineage,
    save_lineage,
)
from experiments.datakit.web_explorer.queries import DEFAULT_SEED, WebExplorer
from experiments.datakit.web_explorer.store_cache import StoreCacheSampler

logger = logging.getLogger(__name__)

_MAX_WORKERS = 8

# Local-dev fallback: reach ducky through the public IAP-gated proxy (needs an
# IAP token). In-cluster we use the controller's internal proxy (no token).
_LOCAL_IAP_DUCKY_URL = "https://iris.oa.dev/proxy/ducky"


# The dashboard is a bundled Vue SPA built into dashboard/dist by `npm run build`
# (gitignored; shipped in the Iris bundle via deploy's extra include). Resolve its
# dist dir: env override → the in-repo build output next to this module.
def _dashboard_dist() -> Path:
    override = os.environ.get("WEB_EXPLORER_DASHBOARD_DIST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "dashboard" / "dist"


_NOT_BUILT_HTML = (
    "<!doctype html><meta charset=utf-8><title>datakit explorer</title>"
    "<body style='font-family:system-ui;margin:3rem'><h1>datakit explorer</h1>"
    "<p>Dashboard not built — run "
    "<code>npm --prefix experiments/datakit/web_explorer/dashboard install "
    "&amp;&amp; npm --prefix experiments/datakit/web_explorer/dashboard run build</code>.</p>"
)


def _index_html(dist: Path, forwarded_prefix: str) -> HTMLResponse:
    """Serve dist/index.html, rewriting ``<base href="/">`` to the proxy sub-path.

    The controller proxy sets ``X-Forwarded-Prefix`` (e.g. ``/proxy/datakit_explorer``)
    in path-style mode; rewriting the base makes the SPA's relative asset and API URLs
    resolve under it. Empty prefix (subdomain/direct) leaves the base at ``/``.
    """
    index_path = dist / "index.html"
    if not index_path.is_file():
        return HTMLResponse(_NOT_BUILT_HTML, status_code=503)
    html = index_path.read_text(encoding="utf-8")
    prefix = forwarded_prefix.rstrip("/")
    if prefix:
        html = html.replace('<base href="/"', f'<base href="{prefix}/"', 1)
    return HTMLResponse(html)


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


def _dedup_attr_map(lineage: StoreLineage) -> dict[str, str]:
    """Map source -> per-source fuzzy-dup attr dir, from the dedup artifact.

    The dedup artifact keys its ``sources`` by each source's normalized *main*
    dir; we join that to the resolved normalize paths to get source -> attr_dir.
    """
    if not lineage.dedup:
        return {}
    import json  # noqa: PLC0415 — startup-only

    from rigging.filesystem import open_url  # noqa: PLC0415

    doc = None
    for name in (".artifact.json", "artifact.json"):
        try:
            with open_url(f"{lineage.dedup}/{name}", "r") as f:
                doc = json.load(f)
            break
        except FileNotFoundError:
            continue
    if not doc:
        return {}
    by_main = {main: entry["attr_dir"] for main, entry in doc.get("sources", {}).items()}
    return {
        src: by_main[f"{ndir}/outputs/main"]
        for src, ndir in lineage.normalize.items()
        if f"{ndir}/outputs/main" in by_main
    }


def _build_views(dv: WebExplorer, cache_sampler: StoreCacheSampler) -> dict[str, Callable[[dict], object]]:
    """Map dashboard view name -> handler(params) -> JSON-serializable result."""

    def _seed(p: dict) -> int:
        return int(p.get("seed", DEFAULT_SEED))

    return {
        "store_cache_samples": lambda p: cache_sampler.samples(
            int(p["cluster"]), int(p["quality_bucket"]), int(p.get("n", 12)), _seed(p), int(p.get("runs", 4))
        ),
        "normalized_stats": lambda p: dv.normalized_stats(p["source"]),
        "normalized_hist": lambda p: dv.normalized_length_hist(p["source"]).dicts(),
        "normalized_samples": (
            lambda p: dv.normalized_samples(p["source"], int(p.get("n", 20)), p.get("search", ""), _seed(p)).dicts()
        ),
        "decontam_stats": lambda p: dv.decontam_stats(p["source"]),
        "decontam_samples": lambda p: dv.decontam_samples(p["source"], int(p.get("n", 20)), _seed(p)).dicts(),
        "quality_hist": lambda p: dv.quality_hist(p["source"]).dicts(),
        "quality_samples": (
            lambda p: dv.quality_samples(
                p["source"], float(p["lo"]), float(p["hi"]), int(p.get("n", 20)), _seed(p)
            ).dicts()
        ),
        "store_samples": lambda p: dv.store_cluster_samples(int(p["cluster"]), int(p.get("n", 12)), seed=_seed(p)),
        "store_bucket_samples": lambda p: dv.store_bucket_samples(
            int(p["cluster"]), int(p["quality_bucket"]), int(p.get("n", 12)), seed=_seed(p)
        ),
        "dedup_examples": lambda p: dv.dedup_examples(p["source"], int(p.get("n_clusters", 6)), seed=_seed(p)),
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
    dist = _dashboard_dist()
    dv = WebExplorer(lineage, ducky, _source_docs(source_summary), _dedup_attr_map(lineage))
    cache_sampler = StoreCacheSampler(
        lineage.store_path, lineage.tokenizer, {(b.cluster_id, b.quality_bucket) for b in payload.buckets}
    )
    manager = QueryManager(_build_views(dv, cache_sampler))

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

    async def index(request: Request) -> HTMLResponse:
        return _index_html(dist, request.headers.get("x-forwarded-prefix", ""))

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

    async def api_ducky_status(_request: Request) -> JSONResponse:
        # Reachability of the (preemptible) query backend, for the viewer's banner.
        available = await run_in_threadpool(ducky.healthy)
        return JSONResponse({"available": available})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/overview", api_overview),
            Route("/api/query", api_query, methods=["POST"]),
            Route("/api/result/{query_id:str}", api_result),
            Route("/api/ducky-status", api_ducky_status),
            Route("/health", health),
            # check_dir=False: the app still boots (index shows a "not built" page)
            # when the SPA hasn't been built yet, instead of raising at startup.
            Mount("/static", StaticFiles(directory=dist / "static", check_dir=False), name="static"),
        ]
    )


def _build_ducky(explicit_url: str | None, timeout: float) -> DuckyClient:
    """Pick the ducky endpoint + auth for the current environment.

    * explicit ``--ducky-url`` / ``WEB_EXPLORER_DUCKY_URL`` — used as-is; IAP token only
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
        url, needs_iap = _LOCAL_IAP_DUCKY_URL, True
    logger.info("ducky endpoint %s (iap=%s, timeout=%.0fs)", url, needs_iap, timeout)
    return DuckyClient(url, token_provider=iap_token_provider() if needs_iap else None, timeout=timeout)


def _load(
    store_path: str, ducky: DuckyClient, cache_path: str | None, config: WebExplorerConfig
) -> tuple[StoreLineage, ClusteredStoreData]:
    payload = read_store_payload(store_path)
    # fsspec_exists handles gs:// (os.path.exists would silently miss a remote
    # cache and force a ducky-dependent re-resolve at startup).
    if cache_path and fsspec_exists(cache_path):
        logger.info("loading cached lineage from %s", cache_path)
        lineage = load_lineage(cache_path)
    else:
        logger.info("resolving lineage for %s (this issues ducky globs; ~1-2 min)", store_path)
        lineage = resolve_lineage(
            store_path,
            ducky,
            domain_centroids=config.domain_centroids,
            quality_model=config.quality_model,
        )
        if cache_path:
            save_lineage(lineage, cache_path)
            logger.info("cached lineage to %s", cache_path)
    return lineage, payload


def main() -> None:
    config = WebExplorerConfig.from_environment()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=config.store, help="Datakit store artifact path (gs://).")
    parser.add_argument("--ducky-url", default=config.ducky_url)
    parser.add_argument("--lineage-cache", default=config.lineage_cache)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not args.store:
        raise SystemExit("--store (or WEB_EXPLORER_STORE) is required")

    ducky = _build_ducky(args.ducky_url, config.query_timeout)
    lineage, payload = _load(args.store, ducky, args.lineage_cache, config)
    source_summary = None
    if config.source_summary:
        import json  # noqa: PLC0415 — only needed on this optional path

        from rigging.filesystem import open_url  # noqa: PLC0415

        with open_url(config.source_summary, "r") as f:
            source_summary = json.load(f)
        logger.info("loaded source summary (%d rows) from %s", len(source_summary), config.source_summary)
    app = build_app(lineage, payload, ducky, source_summary)
    logger.info(
        "web_explorer for %s: %d sources, %d buckets",
        lineage.store_path,
        len(lineage.source_names),
        len(payload.buckets),
    )

    # In-cluster: bind the named Iris port and register the endpoint so the
    # controller proxy routes /proxy/datakit_explorer/ here. Local dev: plain uvicorn.
    job_info = get_job_info()
    if job_info is None:
        logger.info("serving locally on http://%s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return

    ctx = iris_ctx()
    port = ctx.get_port(PORT_NAME)
    address = f"http://{job_info.advertise_host}:{port}"
    endpoint_id = ctx.registry.register(ENDPOINT_NAME, address, {"job_id": ctx.job_id.to_wire()})
    logger.info("web_explorer registered as %s at %s", ENDPOINT_NAME, address)

    async def _on_shutdown() -> None:
        ctx.registry.unregister(endpoint_id)
        logger.info("web_explorer endpoint unregistered")

    app.router.lifespan_context = on_shutdown(_on_shutdown)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
