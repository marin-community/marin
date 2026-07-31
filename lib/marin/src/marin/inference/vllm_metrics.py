# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Poll a native vLLM server and export its cumulative snapshots directly."""

import logging
import threading
from collections.abc import Callable

import requests
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families
from rigging import telemetry

logger = logging.getLogger(__name__)

VLLM_METRICS_SERVICE = "vllm"
VLLM_METRIC_SOURCE = "vllm"
_VLLM_METRIC_PREFIX = "vllm:"
_SCRAPE_TIMEOUT = 5.0
DEFAULT_POLL_INTERVAL = 15.0

Fetch = Callable[[str], str | None]


def _scrape(url: str) -> str | None:
    """Return response text, or `None` after a transport or HTTP failure."""
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
    return [family for family in text_string_to_metric_families(body) if family.name.startswith(_VLLM_METRIC_PREFIX)]


def publish_vllm_families(families: list[Metric]) -> None:
    """Export source snapshots as gauges without reinterpreting cumulative values."""
    for family in families:
        for sample in family.samples:
            cumulative = family.type in {"counter", "histogram"} or (
                family.type == "summary" and sample.name.endswith(("_count", "_sum"))
            )
            temporality = telemetry.CUMULATIVE_SNAPSHOT if cumulative else telemetry.CURRENT_SNAPSHOT
            common = {
                **telemetry.snapshot_attributes(family.type, temporality),
                "metric_source": VLLM_METRIC_SOURCE,
            }
            name = sample.name.removeprefix(_VLLM_METRIC_PREFIX)
            telemetry.gauge(name).set(float(sample.value), attributes={**sample.labels, **common})


class VllmMetricsForwarder:
    """Poll one source endpoint on a bounded daemon thread."""

    def __init__(
        self,
        metrics_url: str,
        *,
        fetch: Fetch = _scrape,
        interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._metrics_url = metrics_url
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
            publish_vllm_families(parse_vllm_families(body))
        except Exception:
            logger.warning("vLLM metrics: could not parse or publish %s", self._metrics_url, exc_info=True)

    def _run(self) -> None:
        self.poll_once()
        while not self._stop.wait(self._interval):
            self.poll_once()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)


def start_vllm_metrics_forwarding(metrics_url: str, *, interval: float = DEFAULT_POLL_INTERVAL) -> VllmMetricsForwarder:
    """Begin polling a vLLM server after the owning application configures telemetry."""
    forwarder = VllmMetricsForwarder(metrics_url, interval=interval)
    forwarder.start()
    logger.info("Forwarding vLLM metrics from %s to telemetry_v1", metrics_url)
    return forwarder
