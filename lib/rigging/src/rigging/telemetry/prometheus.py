# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward a filtered Prometheus endpoint into direct process telemetry."""

import logging
import threading
from collections.abc import Callable

import requests
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families

from rigging import telemetry

logger = logging.getLogger(__name__)

_SCRAPE_TIMEOUT = 5.0
_DEFAULT_POLL_INTERVAL = 15.0
_MAX_SCRAPE_BYTES = 16 << 20

_Fetch = Callable[[str], str | None]


def _scrape(url: str) -> str | None:
    try:
        with requests.get(url, timeout=_SCRAPE_TIMEOUT, stream=True) as response:
            if response.status_code != 200:
                logger.debug("Prometheus scrape returned %s for %s", response.status_code, url)
                return None
            try:
                content_length = int(response.headers.get("content-length", "0"))
            except ValueError:
                logger.warning("Prometheus scrape returned an invalid content length for %s", url)
                return None
            if content_length > _MAX_SCRAPE_BYTES:
                logger.warning("Prometheus scrape exceeded %d bytes for %s", _MAX_SCRAPE_BYTES, url)
                return None
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 << 10):
                size += len(chunk)
                if size > _MAX_SCRAPE_BYTES:
                    logger.warning("Prometheus scrape exceeded %d bytes for %s", _MAX_SCRAPE_BYTES, url)
                    return None
                chunks.append(chunk)
            return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
    except requests.RequestException as error:
        logger.debug("Prometheus scrape failed for %s: %s", url, error)
        return None


class PrometheusForwarder:
    """Poll one endpoint and publish only metric families with the configured prefix."""

    def __init__(
        self,
        metrics_url: str,
        *,
        metric_prefix: str,
        metric_source: str,
        fetch: _Fetch = _scrape,
    ) -> None:
        self._metrics_url = metrics_url
        self._metric_prefix = metric_prefix
        self._metric_source = metric_source
        self._fetch = fetch
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"{metric_source}-metrics", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def poll_once(self) -> None:
        body = self._fetch(self._metrics_url)
        available = 0.0
        if body is not None:
            try:
                for family in text_string_to_metric_families(body):
                    if family.name.startswith(self._metric_prefix):
                        self._publish(family)
            except Exception:
                logger.warning("could not parse or publish Prometheus metrics from %s", self._metrics_url, exc_info=True)
            else:
                available = 1.0
        telemetry.gauge("prometheus_source_available").set(
            available,
            attributes={
                "metric_source": self._metric_source,
                **telemetry.snapshot_attributes("prometheus", telemetry.CURRENT_SNAPSHOT),
            },
        )
        telemetry.record_runtime_health()

    def _publish(self, family: Metric) -> None:
        for sample in family.samples:
            cumulative = family.type in {"counter", "histogram"} or (
                family.type == "summary" and sample.name.endswith(("_count", "_sum"))
            )
            temporality = telemetry.CUMULATIVE_SNAPSHOT if cumulative else telemetry.CURRENT_SNAPSHOT
            attributes = {
                **sample.labels,
                **telemetry.snapshot_attributes(family.type, temporality),
                "metric_source": self._metric_source,
            }
            name = sample.name.removeprefix(self._metric_prefix)
            telemetry.gauge(name).set(float(sample.value), attributes=attributes)

    def _run(self) -> None:
        self.poll_once()
        while not self._stop.wait(_DEFAULT_POLL_INTERVAL):
            self.poll_once()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
