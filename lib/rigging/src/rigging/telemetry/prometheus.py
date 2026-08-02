# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Forward a filtered Prometheus endpoint into direct process telemetry."""

import logging
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import requests
from prometheus_client.core import Metric
from prometheus_client.parser import text_string_to_metric_families

from rigging import telemetry
from rigging.telemetry import serialization

logger = logging.getLogger(__name__)

_SCRAPE_TIMEOUT = 5.0
_DEFAULT_POLL_INTERVAL = 15.0
_MAX_SCRAPE_BYTES = 16 << 20

_Fetch = Callable[[str], str | None]


@dataclass(frozen=True)
class ForwardedPrometheusSample:
    """One normalized Prometheus sample ready for direct telemetry."""

    name: str
    value: float
    unit: str
    attributes: Mapping[str, str]
    source_kind: str
    source_temporality: str


type PrometheusTransform = Callable[[tuple[Metric, ...]], Sequence[ForwardedPrometheusSample]]


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
    """Poll one endpoint and publish prefix-filtered or caller-normalized metrics.

    Configure exactly one selection mode. ``metric_prefix`` preserves matching
    Prometheus series as snapshots. ``transform`` receives the complete parsed
    scrape and returns normalized samples; that mode requires an explicit
    ``max_forwarded_samples`` publication cap.
    """

    def __init__(
        self,
        metrics_url: str,
        *,
        metric_prefix: str | None = None,
        metric_source: str,
        transform: PrometheusTransform | None = None,
        max_forwarded_samples: int | None = None,
        fetch: _Fetch = _scrape,
    ) -> None:
        if (metric_prefix is None) == (transform is None):
            raise ValueError("configure exactly one of metric_prefix or transform")
        if transform is not None and (max_forwarded_samples is None or max_forwarded_samples <= 0):
            raise ValueError("transform mode requires a positive max_forwarded_samples")
        if transform is None and max_forwarded_samples is not None:
            raise ValueError("max_forwarded_samples applies only to transform mode")
        self._metrics_url = metrics_url
        self._metric_prefix = metric_prefix
        self._metric_source = metric_source
        self._transform = transform
        self._max_forwarded_samples = max_forwarded_samples
        self._transform_failures = 0
        self._fetch = fetch
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"{metric_source}-metrics", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def poll_once(self) -> None:
        body = self._fetch(self._metrics_url)
        available = 0.0
        if body is not None:
            published = True
            try:
                families = text_string_to_metric_families(body)
                if self._transform is None:
                    assert self._metric_prefix is not None
                    for family in families:
                        if family.name.startswith(self._metric_prefix):
                            self._publish(family)
                else:
                    published = self._publish_transformed(tuple(families))
            except Exception:
                logger.warning("could not parse or publish Prometheus metrics from %s", self._metrics_url, exc_info=True)
            else:
                available = float(published)
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

    def _publish_transformed(self, families: tuple[Metric, ...]) -> bool:
        assert self._transform is not None
        assert self._max_forwarded_samples is not None
        try:
            samples = self._transform(families)
            forwarded = min(len(samples), self._max_forwarded_samples)
            dropped = len(samples) - forwarded
            selected = tuple(self._validated_sample(sample) for sample in samples[:forwarded])
        except Exception:
            self._transform_failures += 1
            self._record_transform_health(forwarded=0, dropped=0)
            logger.warning("could not transform Prometheus metrics from %s", self._metrics_url, exc_info=True)
            return False
        for sample in selected:
            telemetry.gauge(sample.name, unit=sample.unit).set(sample.value, attributes=sample.attributes)
        self._record_transform_health(forwarded=forwarded, dropped=dropped)
        return True

    def _validated_sample(self, sample: ForwardedPrometheusSample) -> ForwardedPrometheusSample:
        if not isinstance(sample, ForwardedPrometheusSample):
            raise TypeError("transform outputs must be ForwardedPrometheusSample values")
        serialization.validate_string(sample.name, "transformed sample name")
        serialization.validate_string(sample.unit, "transformed sample unit")
        serialization.validate_string(sample.source_kind, "transformed sample source kind")
        if sample.source_temporality not in {telemetry.CURRENT_SNAPSHOT, telemetry.CUMULATIVE_SNAPSHOT}:
            raise ValueError("transformed sample temporality must be current_snapshot or cumulative_snapshot")
        value = float(sample.value)
        if not math.isfinite(value):
            raise ValueError("transformed sample values must be finite")
        attributes = {
            **sample.attributes,
            **telemetry.snapshot_attributes(sample.source_kind, sample.source_temporality),
            "metric_source": self._metric_source,
        }
        serialization.validate_attributes(attributes)
        return ForwardedPrometheusSample(
            name=sample.name,
            value=value,
            unit=sample.unit,
            attributes=attributes,
            source_kind=sample.source_kind,
            source_temporality=sample.source_temporality,
        )

    def _record_transform_health(self, *, forwarded: int, dropped: int) -> None:
        current = {
            "metric_source": self._metric_source,
            **telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT),
        }
        cumulative = {
            "metric_source": self._metric_source,
            **telemetry.snapshot_attributes("counter", telemetry.CUMULATIVE_SNAPSHOT),
        }
        telemetry.gauge("prometheus_forwarded_samples", unit="{sample}").set(forwarded, attributes=current)
        telemetry.gauge("prometheus_dropped_samples", unit="{sample}").set(
            dropped,
            attributes={**current, "drop_reason": "sample_limit"},
        )
        telemetry.gauge("prometheus_transform_failures", unit="{failure}").set(
            self._transform_failures,
            attributes=cumulative,
        )

    def _run(self) -> None:
        self.poll_once()
        while not self._stop.wait(_DEFAULT_POLL_INTERVAL):
            self.poll_once()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
