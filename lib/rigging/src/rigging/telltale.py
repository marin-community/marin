# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-local instrumentation: Prometheus metrics and a status page.

A telltale is the ribbon on a sail that shows which way the wind is blowing. This
module is the equivalent for a Marin process: a few HTTP routes that say what the
process is doing right now, cheap enough to leave on everywhere.

``routes()`` returns the pages for mounting into an app that already serves; this
module starts no server of its own.

Instrument through ``counter``/``gauge``/``histogram``, which get-or-create
against the default ``prometheus_client`` registry. Constructing those objects
directly raises ``Duplicated timeseries`` when a process instruments the same
name twice, which a worker running a user callable repeatedly will do.

The routes are ``@public`` under ``rigging.server_auth.RouteAuthMiddleware``.
"""

import html
import logging
import re
import threading
import time
import typing
from collections.abc import Iterator, Sequence
from typing import TypeVar

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest
from prometheus_client.metrics import MetricWrapperBase
from prometheus_client.registry import Collector
from prometheus_client.samples import Sample
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from rigging.server_auth import public

logger = logging.getLogger(__name__)

_M = TypeVar("_M", bound=MetricWrapperBase)

_lock = threading.Lock()
_metrics: dict[str, MetricWrapperBase] = {}
_collectors: set[Collector] = set()
_status: str = ""
_start_time = time.time()


def _get_or_create(
    factory: type[_M],
    name: str,
    documentation: str,
    labelnames: Sequence[str],
) -> _M:
    """Return the existing metric named ``name``, or register a new one.

    Raises:
        ValueError: If ``name`` is already registered with a different metric type
            or label set. Two call sites disagreeing about a metric's shape is a
            bug that silently corrupts the series; surface it at registration.
    """
    with _lock:
        existing = _metrics.get(name)
        if existing is None:
            metric = factory(name, documentation, tuple(labelnames))
            _metrics[name] = metric
            return metric

        if not isinstance(existing, factory):
            raise ValueError(
                f"Metric {name!r} is already registered as {type(existing).__name__}, "
                f"cannot re-register as {factory.__name__}"
            )
        if tuple(existing._labelnames) != tuple(labelnames):
            raise ValueError(
                f"Metric {name!r} is already registered with labels {list(existing._labelnames)}, "
                f"cannot re-register with {list(labelnames)}"
            )
        return existing


def counter(name: str, documentation: str, labelnames: Sequence[str] = ()) -> Counter:
    """Get-or-create a cumulative counter."""
    return _get_or_create(Counter, name, documentation, labelnames)


def gauge(name: str, documentation: str, labelnames: Sequence[str] = ()) -> Gauge:
    """Get-or-create a gauge."""
    return _get_or_create(Gauge, name, documentation, labelnames)


def histogram(name: str, documentation: str, labelnames: Sequence[str] = ()) -> Histogram:
    """Get-or-create a histogram."""
    return _get_or_create(Histogram, name, documentation, labelnames)


def register_collector(collector: Collector) -> None:
    """Register a custom collector, once per process.

    For metrics a caller computes elsewhere and hands over whole — precomputed
    histogram buckets, say — which `counter`/`gauge`/`histogram` cannot express.
    """
    with _lock:
        if collector in _collectors:
            return
        REGISTRY.register(collector)
        _collectors.add(collector)


def metric_name(name: str, prefix: str = "") -> str:
    """Convert an arbitrary counter/metric key into a legal Prometheus name.

    Every character outside ``[a-zA-Z0-9_]`` becomes an underscore, so path-shaped
    keys like ``zephyr/records_in`` or ``train/loss`` become legal series names.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return f"{prefix}_{sanitized}" if prefix else sanitized


def publish_gauge(key: str, value: float, documentation: str, prefix: str = "") -> None:
    """Set a gauge named after an arbitrary metric key.

    For mirroring counters a caller already keeps, whose names it does not
    control. A key that sanitizes onto a name another metric type already holds
    is logged and dropped: exposition must never break the job it reports on.
    """
    name = metric_name(key, prefix=prefix)
    try:
        gauge(name, documentation).set(value)
    except ValueError:
        logger.warning("could not publish %r as gauge %r", key, name, exc_info=True)


def set_status(text: str) -> None:
    """Set the free-form status block shown on the index page.

    Rendered as preformatted plain text. Process-local and in-memory: a debugging
    view of a live process, neither persisted nor reported anywhere.
    """
    global _status
    with _lock:
        _status = text


def get_status() -> str:
    with _lock:
        return _status


class FamilySample(typing.NamedTuple):
    """One registry sample, tagged with the family it came from."""

    family: str
    kind: str
    """The family's metric type: counter, gauge, histogram, summary, info."""
    sample: Sample


def samples() -> Iterator[FamilySample]:
    """Yield every sample in the registry, tagged with its family and type.

    Samples rather than families, because ``collect()`` has already flattened a
    histogram into ``_bucket``/``_sum``/``_count`` samples with the bucket bound
    in an ``le`` label, and a counter into a value plus a ``_created`` sample
    holding the series' start time. A consumer that walks samples therefore needs
    no per-type special-casing, and gets the reset boundary for free.
    """
    for family in REGISTRY.collect():
        for sample in family.samples:
            yield FamilySample(family.name, family.type, sample)


@public
def _metrics_route(_request: Request) -> Response:
    """Serve the registry in Prometheus text exposition format."""
    # Must stay a sync def: Starlette runs sync handlers in a threadpool, keeping
    # collection off the event loop. An async def would serialize the whole
    # registry on the loop and stall every other request on this port per scrape.
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@public
def _health_route(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy"})


def _render_index(status: str, uptime: float) -> str:
    """Render the index page.

    Server-rendered with relative links and no scripts: the page has to work
    behind the Iris controller's ``/proxy/<name>/`` prefix, where an absolute URL
    or a bundled asset would 404.
    """
    rows = []
    for _family_name, family_type, sample in samples():
        labels = ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items()))
        rows.append(
            f"<tr><td>{html.escape(sample.name)}</td><td>{html.escape(family_type)}</td>"
            f"<td>{html.escape(labels)}</td><td>{sample.value!r}</td></tr>"
        )
    table = "\n".join(rows) or '<tr><td colspan="4">no metrics registered</td></tr>'
    return f"""<!doctype html>
<title>telltale</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
 table {{ border-collapse: collapse; }}
 td, th {{ border: 1px solid #ccc; padding: 0.2rem 0.6rem; text-align: left; }}
 pre {{ background: #f4f4f4; padding: 0.6rem; white-space: pre-wrap; }}
</style>
<h1>telltale</h1>
<p>uptime {uptime:.0f}s &middot; <a href="metrics">metrics</a> &middot; <a href="health">health</a></p>
<h2>status</h2>
<pre>{html.escape(status) or "(none set)"}</pre>
<h2>metrics ({len(rows)} samples)</h2>
<table><tr><th>sample</th><th>type</th><th>labels</th><th>value</th></tr>
{table}
</table>
"""


@public
def _index_route(_request: Request) -> HTMLResponse:
    return HTMLResponse(_render_index(get_status(), time.time() - _start_time))


def routes() -> list[Route]:
    """The telltale routes, for mounting into an app that already serves."""
    return [
        Route("/", _index_route),
        Route("/metrics", _metrics_route),
        Route("/health", _health_route),
    ]
