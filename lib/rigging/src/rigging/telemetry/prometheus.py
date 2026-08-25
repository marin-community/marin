# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect, process, and publish bounded Prometheus metric snapshots."""

import logging
import math
import threading
from collections.abc import Callable, Sequence
from enum import StrEnum

import requests
from prometheus_client.core import Metric as PrometheusMetric
from prometheus_client.parser import text_string_to_metric_families

from rigging import telemetry
from rigging.telemetry import metrics

logger = logging.getLogger(__name__)

DEFAULT_SCRAPE_TIMEOUT = 5.0
DEFAULT_POLL_INTERVAL = 15.0
DEFAULT_MAX_SCRAPE_BYTES = 16 << 20

type PrometheusProcessor = Callable[[tuple[PrometheusMetric, ...]], Sequence[metrics.MetricSnapshot]]


class _PrometheusStage(StrEnum):
    SCRAPE = "scrape"
    PROCESS = "process"
    PUBLISH = "publish"


class PrometheusScrapeError(RuntimeError):
    """A Prometheus endpoint could not produce a bounded parsed scrape."""


class PrometheusScraper:
    """Read and parse one bounded Prometheus text endpoint."""

    def __init__(
        self,
        metrics_url: str,
        *,
        timeout: float = DEFAULT_SCRAPE_TIMEOUT,
        max_bytes: int = DEFAULT_MAX_SCRAPE_BYTES,
    ) -> None:
        if not metrics_url.startswith(("http://", "https://")):
            raise ValueError("metrics_url must use http:// or https://")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._metrics_url = metrics_url
        self._timeout = timeout
        self._max_bytes = max_bytes

    def scrape(self) -> tuple[PrometheusMetric, ...]:
        """Return all parsed metric families.

        Raises ``PrometheusScrapeError`` for the size and status policies this
        scraper enforces. Transport and parse failures propagate with their own
        types so callers see the underlying error rather than a wrapped one.
        """
        with requests.get(self._metrics_url, timeout=self._timeout, stream=True) as response:
            if response.status_code != 200:
                raise PrometheusScrapeError(
                    f"Prometheus scrape returned HTTP {response.status_code} for {self._metrics_url}"
                )
            declared_length = response.headers.get("content-length")
            if declared_length is not None and int(declared_length) > self._max_bytes:
                raise PrometheusScrapeError(
                    f"Prometheus scrape exceeded {self._max_bytes} bytes for {self._metrics_url}"
                )
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 << 10):
                size += len(chunk)
                if size > self._max_bytes:
                    raise PrometheusScrapeError(
                        f"Prometheus scrape exceeded {self._max_bytes} bytes for {self._metrics_url}"
                    )
                chunks.append(chunk)
            body = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")

        return tuple(text_string_to_metric_families(body))


def prefixed_metric_snapshots(
    families: tuple[PrometheusMetric, ...],
    *,
    metric_prefix: str,
) -> tuple[metrics.MetricSnapshot, ...]:
    """Preserve Prometheus series whose family names match ``metric_prefix``."""
    if not metric_prefix:
        raise ValueError("metric_prefix must not be empty")
    snapshots: list[metrics.MetricSnapshot] = []
    for family in families:
        if not family.name.startswith(metric_prefix):
            continue
        for sample in family.samples:
            cumulative = family.type in {"counter", "histogram"} or (
                family.type == "summary" and sample.name.endswith(("_count", "_sum"))
            )
            snapshots.append(
                metrics.MetricSnapshot(
                    name=sample.name.removeprefix(metric_prefix),
                    value=float(sample.value),
                    unit=family.unit,
                    attributes=sample.labels,
                    source_kind=family.type,
                    source_temporality=(telemetry.CUMULATIVE_SNAPSHOT if cumulative else telemetry.CURRENT_SNAPSHOT),
                )
            )
    return tuple(snapshots)


class PrometheusCollector:
    """Periodically compose a scraper, processor, and metric snapshot publisher."""

    def __init__(
        self,
        *,
        metric_source: str,
        scraper: PrometheusScraper,
        processor: PrometheusProcessor,
        publisher: metrics.MetricSnapshotPublisher,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        if not metric_source:
            raise ValueError("metric_source must not be empty")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive and finite")
        self._metric_source = metric_source
        self._scraper = scraper
        self._processor = processor
        self._publisher = publisher
        self._poll_interval = poll_interval
        self._stage_failures = {stage: 0 for stage in _PrometheusStage}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"{metric_source}-metrics", daemon=True)

    def start(self) -> None:
        """Start periodic collection in one daemon thread."""
        self._thread.start()

    def poll_once(self) -> None:
        """Run one scrape, process, and publication cycle, isolating the failed stage.

        This runs on the daemon poll loop, so a stage failure is logged with its
        traceback and recorded against that stage rather than propagated: letting
        it escape would kill the collector thread and silently end forwarding.
        The stage cursor attributes the failure so the loss surfaces as a health
        metric instead of vanishing.
        """
        stage = _PrometheusStage.SCRAPE
        try:
            families = self._scraper.scrape()
            stage = _PrometheusStage.PROCESS
            snapshots = self._processor(families)
            stage = _PrometheusStage.PUBLISH
            result = self._publisher.publish(snapshots)
        except Exception:
            self._stage_failed(stage)
            self._record_health(source_available=stage is not _PrometheusStage.SCRAPE)
            return
        self._record_health(source_available=True, result=result)

    def _stage_failed(self, stage: _PrometheusStage) -> None:
        self._stage_failures[stage] += 1
        logger.warning(
            "Prometheus %s stage failed for %s; forwarding continues",
            stage,
            self._metric_source,
            exc_info=True,
        )

    def _record_health(
        self,
        *,
        source_available: bool,
        result: metrics.MetricPublishResult | None = None,
    ) -> None:
        current = {
            "metric_source": self._metric_source,
            **telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT),
        }
        cumulative = {
            "metric_source": self._metric_source,
            **telemetry.snapshot_attributes("counter", telemetry.CUMULATIVE_SNAPSHOT),
        }
        telemetry.gauge("prometheus_source_available").set(float(source_available), attributes=current)
        for stage, failures in self._stage_failures.items():
            telemetry.gauge("prometheus_stage_failures", unit="{failure}").set(
                failures,
                attributes={**cumulative, "stage": stage.value},
            )
        if result is not None and result.configured:
            telemetry.gauge("prometheus_enqueued_samples", unit="{sample}").set(
                result.enqueued_records,
                attributes=current,
            )
            telemetry.gauge("prometheus_dropped_samples", unit="{sample}").set(
                result.sample_limit_dropped_records,
                attributes={**current, "drop_reason": "sample_limit"},
            )
            telemetry.gauge("prometheus_dropped_samples", unit="{sample}").set(
                result.telemetry_lost_records,
                attributes={**current, "drop_reason": "telemetry_loss"},
            )
        telemetry.record_runtime_health()

    def _run(self) -> None:
        self.poll_once()
        while not self._stop.wait(self._poll_interval):
            self.poll_once()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop scheduling scrapes and wait at most ``timeout`` seconds."""
        self._stop.set()
        self._thread.join(timeout=max(0.0, timeout))
