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

import atexit
import html
import logging
import random
import re
import threading
import time
import typing
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Protocol, TypeVar

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, Gauge, Histogram, generate_latest
from prometheus_client.metrics import MetricWrapperBase
from prometheus_client.registry import Collector
from prometheus_client.samples import Sample
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from rigging.server_auth import public
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)

_M = TypeVar("_M", bound=MetricWrapperBase)

_lock = threading.Lock()
_metrics: dict[str, MetricWrapperBase] = {}
_collectors: set[Collector] = set()
_status: str = ""
_start_time = time.time()
_global_labels: dict[str, str] = {}


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


def set_global_labels(**labels: str) -> None:
    """Merge process-wide labels describing who produced these metrics.

    The metrics themselves carry no run/process identity — a levanter gauge is
    just ``levanter_train_loss``. A producer names itself once at startup
    (``set_global_labels(run=run_id, source="levanter")``) rather than threading
    identity through each metric, and a reader of the registry can stamp every
    sample with these labels.

    Merges into the existing set; a later call overrides an existing key. Values
    are coerced to ``str``. Purely additive to the exposition path — the
    Prometheus ``/metrics`` output is unaffected.
    """
    with _lock:
        _global_labels.update((k, str(v)) for k, v in labels.items())


def get_global_labels() -> dict[str, str]:
    """A copy of the process-wide labels set via ``set_global_labels``."""
    with _lock:
        return dict(_global_labels)


# --- Forwarding the registry to a durable sink ------------------------------

#: Producer prefixes that name their own source; anything else is "process"
#: (the default Prometheus process/platform collectors).
_KNOWN_SOURCES = frozenset({"levanter", "zephyr", "iris"})

#: Seconds between registry snapshots. A durable sink typically seals at most one
#: segment per second, so this stays clear of that while keeping dashboards live.
DEFAULT_FORWARD_INTERVAL = 15.0


@dataclass(frozen=True)
class MetricIdentity:
    """The job coordinates of the process producing the metrics.

    Stamped authoritatively onto every forwarded row, so a metric's own labels can
    never masquerade as the job it came from. ``job_id`` is the job root;
    ``task_index`` is the task's index within it — the full task path is just those
    two, so it is not stored. The producer's ``source``/``run`` are separate: they
    ride on the process-global labels instead.
    """

    job_id: str | None = None
    task_index: int | None = None
    attempt: int | None = None
    worker: str | None = None
    region: str | None = None
    process_index: int | None = None


@dataclass
class TelltaleMetric:
    """One telltale sample as a durable row.

    Always-present identity — the metric ``source``, the producer's ``run``, and
    the Iris job coordinates — is flattened into typed top-level columns so a sink
    can store and filter on it directly; only leftover Prometheus labels (a
    histogram's ``le``, ad-hoc labels) stay in ``labels``. A sink keys its storage
    on ``name`` so one metric's rows cluster together, and orders a series by
    ``ts``.
    """

    key_column: ClassVar[str] = "name"

    name: str
    value: float
    kind: str
    ts: datetime
    source: str
    run: str | None = None
    job_id: str | None = None
    task_index: int | None = None
    attempt: int | None = None
    worker: str | None = None
    region: str | None = None
    process_index: int | None = None
    labels: dict[str, str] = field(default_factory=dict)


class MetricSink(Protocol):
    """A durable destination the forwarder appends scraped rows to.

    ``write`` appends a batch; ``close`` flushes anything buffered and releases
    resources. Implementations own their transport (e.g. finelog), keeping this
    module free of any storage dependency.
    """

    def write(self, rows: Sequence[TelltaleMetric]) -> None: ...

    def close(self) -> None: ...


def _source_for(name: str, source_label: str | None) -> str:
    """Resolve a row's ``source``: an explicit label wins, else the name prefix."""
    if source_label:
        return source_label
    head = name.split("_", 1)[0]
    return head if head in _KNOWN_SOURCES else "process"


def scrape_metrics(identity: MetricIdentity, ts: datetime) -> list[TelltaleMetric]:
    """Snapshot the registry into rows, stamping ``identity`` onto each. Pure; no I/O.

    ``source`` and ``run`` are lifted from the process-global labels into columns;
    the job ``identity`` is set on the row directly, so a metric's own labels can
    never spoof it. Everything else stays in ``labels``. Prometheus ``_created``
    series (a counter's start time, not a metric) are dropped.
    """
    global_labels = get_global_labels()
    rows: list[TelltaleMetric] = []
    for family in samples():
        sample = family.sample
        if sample.name.endswith("_created"):
            continue
        labels = {**sample.labels, **global_labels}
        source = _source_for(sample.name, labels.pop("source", None))
        run = labels.pop("run", None)
        rows.append(
            TelltaleMetric(
                name=sample.name,
                value=float(sample.value),
                kind=family.kind,
                ts=ts,
                source=source,
                run=run,
                job_id=identity.job_id,
                task_index=identity.task_index,
                attempt=identity.attempt,
                worker=identity.worker,
                region=identity.region,
                process_index=identity.process_index,
                labels=labels,
            )
        )
    return rows


class _Forwarder:
    """Appends the registry to a sink on a daemon thread; flushes on stop."""

    def __init__(self, sink: MetricSink, identity: MetricIdentity, interval: float) -> None:
        self._sink = sink
        self._identity = identity
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="telltale-forward", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        # A random first delay desynchronizes many processes so they don't all
        # write on the same tick.
        if self._stop.wait(random.uniform(0.0, self._interval)):
            return
        while not self._stop.is_set():
            self._scrape_once()
            self._stop.wait(self._interval)

    def _scrape_once(self) -> None:
        rows = scrape_metrics(self._identity, Timestamp.now().as_naive_utc())
        if not rows:
            return
        try:
            self._sink.write(rows)
        except Exception:
            logger.warning("telltale forward: write failed", exc_info=True)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._scrape_once()
        finally:
            try:
                self._sink.close()
            except Exception:
                logger.debug("telltale forward: sink close failed", exc_info=True)


_forward_lock = threading.Lock()
_forwarder: _Forwarder | None = None


def start_forwarding(
    sink: MetricSink,
    *,
    identity: MetricIdentity | None = None,
    interval: float = DEFAULT_FORWARD_INTERVAL,
) -> bool:
    """Begin forwarding the registry to ``sink`` on a background thread.

    Idempotent: a call while a forwarder is already running is a no-op returning
    ``False``. ``identity`` is stamped onto every row. The forwarder flushes a
    final batch at process exit.
    """
    global _forwarder
    with _forward_lock:
        if _forwarder is not None:
            return False
        forwarder = _Forwarder(sink, identity or MetricIdentity(), interval)
        forwarder.start()
        atexit.register(forwarder.stop)
        _forwarder = forwarder
        return True


def stop_forwarding() -> None:
    """Stop the active forwarder after a final flush. Idempotent."""
    global _forwarder
    with _forward_lock:
        if _forwarder is None:
            return
        atexit.unregister(_forwarder.stop)
        _forwarder.stop()
        _forwarder = None


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
