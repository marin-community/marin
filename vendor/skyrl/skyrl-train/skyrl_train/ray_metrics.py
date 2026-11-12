import contextlib
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from loguru import logger
from prometheus_client.core import Metric
from rigging import telemetry as rigging_telemetry
from rigging.telemetry.metrics import MetricSnapshot, MetricSnapshotPublisher
from rigging.telemetry.prometheus import PrometheusCollector, PrometheusScraper

from skyrl_train.telemetry import process_telemetry


MAX_RAY_METRIC_SNAPSHOTS = 512
COLLECTOR_STOP_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True)
class _RayMetricSpec:
    unit: str
    labels: Mapping[str, str]
    source_kind: str = "gauge"
    source_temporality: str = rigging_telemetry.CURRENT_SNAPSHOT


_STATE = {"State": "state"}
_TYPE = {"Type": "transfer_kind"}
_RAY_METRICS = {
    "ray_tasks": _RayMetricSpec(
        unit="{task}",
        labels={"JobId": "ray_job_id", "State": "state", "Source": "source", "IsRetry": "is_retry"},
    ),
    "ray_scheduler_tasks": _RayMetricSpec(unit="{task}", labels=_STATE),
    "ray_scheduler_unscheduleable_tasks": _RayMetricSpec(unit="{task}", labels={"Reason": "reason"}),
    "ray_scheduler_failed_worker_startup_total": _RayMetricSpec(
        unit="{failure}",
        labels={"Reason": "reason"},
        source_kind="counter",
        source_temporality=rigging_telemetry.CUMULATIVE_SNAPSHOT,
    ),
    "ray_resources": _RayMetricSpec(unit="{resource}", labels={"Name": "resource", "State": "state"}),
    "ray_object_store_num_local_objects": _RayMetricSpec(unit="{object}", labels={}),
    "ray_object_store_available_memory": _RayMetricSpec(unit="By", labels={}),
    "ray_object_store_used_memory": _RayMetricSpec(unit="By", labels={}),
    "ray_object_store_fallback_memory": _RayMetricSpec(unit="By", labels={}),
    "ray_object_store_memory": _RayMetricSpec(
        unit="By",
        labels={"Location": "location", "ObjectState": "object_state"},
    ),
    "ray_pull_manager_requested_bundles": _RayMetricSpec(unit="{bundle}", labels={}),
    "ray_pull_manager_active_bundles": _RayMetricSpec(unit="{bundle}", labels={}),
    "ray_pull_manager_retries_total": _RayMetricSpec(
        unit="{retry}",
        labels={},
        source_kind="counter",
        source_temporality=rigging_telemetry.CUMULATIVE_SNAPSHOT,
    ),
    "ray_pull_manager_requests": _RayMetricSpec(unit="{request}", labels=_TYPE),
    "ray_pull_manager_num_object_pins": _RayMetricSpec(unit="{object}", labels=_TYPE),
    "ray_pull_manager_usage_bytes": _RayMetricSpec(unit="By", labels=_TYPE),
    "ray_spill_manager_objects": _RayMetricSpec(unit="{object}", labels=_STATE),
    "ray_spill_manager_objects_bytes": _RayMetricSpec(unit="By", labels=_STATE),
    "ray_spill_manager_request_total": _RayMetricSpec(
        unit="{request}",
        labels=_TYPE,
        source_kind="counter",
        source_temporality=rigging_telemetry.CUMULATIVE_SNAPSHOT,
    ),
    "ray_gcs_placement_group_count": _RayMetricSpec(unit="{placement_group}", labels=_STATE),
}
_LOGICAL_RESOURCES = frozenset({"CPU", "GPU"})
_LOWERCASE_ATTRIBUTES = frozenset(
    {"is_retry", "location", "object_state", "reason", "resource", "source", "state", "transfer_kind"}
)


def transform_ray_metrics(families: tuple[Metric, ...]) -> tuple[MetricSnapshot, ...]:
    """Normalize the bounded Ray logical-state allowlist for Finelog."""
    values: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
    for family in families:
        spec = _RAY_METRICS.get(family.name)
        if spec is None:
            continue
        for sample in family.samples:
            if sample.name != family.name:
                continue
            value = float(sample.value)
            if not math.isfinite(value):
                continue
            if family.name == "ray_resources" and sample.labels.get("Name") not in _LOGICAL_RESOURCES:
                continue
            attributes = tuple(
                sorted(
                    (
                        target,
                        sample.labels[source].lower() if target in _LOWERCASE_ATTRIBUTES else sample.labels[source],
                    )
                    for source, target in spec.labels.items()
                    if sample.labels.get(source)
                )
            )
            values[(family.name, attributes)] += value

    return tuple(
        MetricSnapshot(
            name=name,
            value=value,
            unit=_RAY_METRICS[name].unit,
            attributes=dict(attributes),
            source_kind=_RAY_METRICS[name].source_kind,
            source_temporality=_RAY_METRICS[name].source_temporality,
        )
        for (name, attributes), value in sorted(values.items())
    )


@contextlib.contextmanager
def ray_metrics_telemetry(node_ip: str, metrics_port: int, role: str) -> Iterator[None]:
    with process_telemetry(role) as owner:
        collector = owner.collector_or_inert(
            PrometheusCollector(
                metric_source="ray",
                scraper=PrometheusScraper(f"http://{node_ip}:{metrics_port}/metrics"),
                processor=transform_ray_metrics,
                publisher=MetricSnapshotPublisher(
                    max_records=MAX_RAY_METRIC_SNAPSHOTS,
                    attributes={"metric_source": "ray"},
                ),
            )
        )
        try:
            collector.start()
        except Exception:
            logger.warning("Could not start local Ray metric forwarding; training will continue", exc_info=True)
        try:
            yield
        finally:
            try:
                collector.stop(timeout=COLLECTOR_STOP_TIMEOUT_SECONDS)
            except Exception:
                logger.warning("Could not stop local Ray metric forwarding", exc_info=True)
