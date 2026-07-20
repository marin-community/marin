# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward a native vLLM server's operational metrics to this process's telltale.

vLLM serves its throughput, queue-depth, and latency metrics (TTFT/TPOT histograms,
prompt/generation token counters, cache utilization) in Prometheus text format at
``GET /metrics`` on the OpenAI server port. Because ``vllm serve`` runs as a
subprocess (see :mod:`marin.inference.vllm_server`), those samples live in the
subprocess's own registry and never reach the parent — so they never reach finelog,
and a serve's throughput is unobservable after the job ends.

This polls that endpoint and re-exposes the ``vllm:`` families through a
:class:`~prometheus_client.registry.Collector` in the parent's ``rigging.telltale``
registry, then starts the process's telltale forwarder. The existing
telltale->finelog forwarder (``iris.runtime.telltale``) then persists them under the
job's identity, queryable after the serve ends — the same path Levanter training
metrics already take.

Only ``vllm:``-prefixed families are mirrored. The rest of vLLM's ``/metrics`` is the
stdlib ``process_*``/``python_*`` collectors, which the parent's default registry
already holds; re-emitting them would collide (``Duplicated timeseries``) and would
double-count the subprocess against the parent.
"""

import logging
import threading
from collections.abc import Callable, Iterable

import requests
from iris.runtime import telltale as runtime_telltale
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.registry import Collector
from rigging import telltale

logger = logging.getLogger(__name__)

#: The telltale ``source`` label stamped on every forwarded vLLM row, so a reader can
#: pick vLLM serving metrics out of the shared finelog table (``WHERE source = 'vllm'``).
VLLM_METRICS_SOURCE = "vllm"

#: vLLM prefixes all of its own metrics with this. Everything else at ``/metrics`` is
#: the stdlib process/platform collectors the parent registry already exposes.
_VLLM_METRIC_PREFIX = "vllm:"

#: Seconds a single ``/metrics`` scrape may take before it is treated as a miss. A
#: busy engine can be slow to answer, but the poll must never wedge the parent.
_SCRAPE_TIMEOUT = 5.0

#: How often to refresh the mirrored snapshot. Matched to the telltale forwarder's own
#: cadence so it always reads a recent scrape rather than a stale one.
DEFAULT_POLL_INTERVAL = telltale.DEFAULT_FORWARD_INTERVAL

#: A ``GET`` of a ``/metrics`` URL, returning the body or ``None`` on any failure. The
#: one seam tests drive the poller through without a network.
Fetch = Callable[[str], str | None]


def _scrape(url: str) -> str | None:
    """GET a Prometheus ``/metrics`` endpoint; return its body or ``None`` on any failure."""
    try:
        response = requests.get(url, timeout=_SCRAPE_TIMEOUT)
    except requests.RequestException as exc:
        logger.debug("vLLM metrics scrape failed for %s: %s", url, exc)
        return None
    if response.status_code != 200:
        logger.debug("vLLM metrics scrape returned %s for %s", response.status_code, url)
        return None
    return response.text


def parse_vllm_families(body: str) -> list[Metric]:
    """Parse a ``/metrics`` body and keep only vLLM's own (``vllm:``-prefixed) families."""
    return [family for family in text_string_to_metric_families(body) if family.name.startswith(_VLLM_METRIC_PREFIX)]


class _VllmMetricsProxy(Collector):
    """Re-exposes the vLLM subprocess's latest ``vllm:`` families in the parent registry.

    Holds the most recent parsed snapshot and serves it at scrape time. Collection stays
    a fast in-memory copy — the network scrape runs on the poller thread — so a slow or
    dead engine never stalls the telltale registry's own scrape.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: list[Metric] = []

    def set_latest(self, families: Iterable[Metric]) -> None:
        with self._lock:
            self._latest = list(families)

    def collect(self) -> Iterable[Metric]:
        with self._lock:
            return list(self._latest)


class VllmMetricsForwarder:
    """Polls a vLLM ``/metrics`` endpoint and mirrors it into a telltale proxy collector.

    One daemon thread refreshes the ``proxy`` snapshot every ``interval``. A failed scrape
    keeps the previous snapshot rather than clearing it, so a transient miss does not blank
    the series and the forwarder's final flush still carries the last-seen totals.
    """

    def __init__(
        self,
        metrics_url: str,
        proxy: _VllmMetricsProxy,
        *,
        fetch: Fetch = _scrape,
        interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._metrics_url = metrics_url
        self._proxy = proxy
        self._fetch = fetch
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vllm-metrics", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def poll_once(self) -> None:
        body = self._fetch(self._metrics_url)
        if body is None:
            return
        try:
            self._proxy.set_latest(parse_vllm_families(body))
        except Exception:
            # A malformed scrape must not kill the poller; keep the last good snapshot.
            logger.warning("vLLM metrics: could not parse %s", self._metrics_url, exc_info=True)

    def _run(self) -> None:
        # Scrape immediately so the proxy has data before the first telltale forward tick.
        self.poll_once()
        while not self._stop.wait(self._interval):
            self.poll_once()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


#: One proxy per process: the default Prometheus registry rejects two collectors that
#: emit the same family, so successive serves share this and its ``set_latest`` overwrites.
_PROXY = _VllmMetricsProxy()


def start_vllm_metrics_forwarding(metrics_url: str, *, interval: float = DEFAULT_POLL_INTERVAL) -> VllmMetricsForwarder:
    """Begin forwarding a native vLLM server's ``/metrics`` to telltale and finelog.

    Ensures this process's telltale forwarder is running (``iris.runtime.telltale.start``
    is idempotent and a no-op outside an Iris job), registers the shared proxy collector,
    stamps ``source=vllm`` on the process's forwarded rows, and starts the poller.

    Best-effort: any wiring failure is logged and a (harmless) forwarder is still returned,
    so a metrics problem never breaks the serve it reports on.
    """
    try:
        runtime_telltale.start()
    except Exception:
        logger.warning("vLLM metrics: could not start the telltale forwarder", exc_info=True)
    telltale.register_collector(_PROXY)
    telltale.set_global_labels(source=VLLM_METRICS_SOURCE)

    forwarder = VllmMetricsForwarder(metrics_url, _PROXY, interval=interval)
    forwarder.start()
    logger.info("Forwarding vLLM metrics from %s to telltale (source=%s)", metrics_url, VLLM_METRICS_SOURCE)
    return forwarder
