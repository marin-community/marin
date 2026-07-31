# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loader for the checked-in telemetry descriptor catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

_CATALOG_RESOURCE = "telemetry_catalog.v1.json"


@dataclass(frozen=True, slots=True)
class CatalogMetric:
    scope: str
    name: str
    description: str
    unit: str
    instrument_kind: str
    temporality: str
    delivery_class: str
    namespace: str
    attributes: tuple[tuple[str, tuple[str, ...]], ...]
    buckets: tuple[float, ...]
    owner: str
    cadence_seconds: int
    cardinality_limit: int
    maturity: str


@dataclass(frozen=True, slots=True)
class CatalogEvent:
    event_name: str
    namespace: str
    delivery_class: str
    owner: str
    attributes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TelemetryCatalog:
    catalog_version: str
    metrics: tuple[CatalogMetric, ...]
    events: tuple[CatalogEvent, ...]
    artifacts: tuple[str, ...]

    def metric(self, scope: str, name: str) -> CatalogMetric | None:
        return next(
            (metric for metric in self.metrics if (metric.scope, metric.name) == (scope, name)),
            None,
        )


def _required(record: dict[str, Any], name: str, expected: type) -> Any:
    value = record[name]
    if not isinstance(value, expected):
        raise ValueError(f"telemetry catalog field {name!r} must be {expected.__name__}")
    return value


def _metric(record: dict[str, Any]) -> CatalogMetric:
    attributes = _required(record, "attributes", dict)
    return CatalogMetric(
        scope=_required(record, "scope", str),
        name=_required(record, "name", str),
        description=_required(record, "description", str),
        unit=_required(record, "unit", str),
        instrument_kind=_required(record, "instrument_kind", str),
        temporality=_required(record, "temporality", str),
        delivery_class=_required(record, "delivery_class", str),
        namespace=_required(record, "namespace", str),
        attributes=tuple(
            (name, tuple(_required({"values": values}, "values", list))) for name, values in attributes.items()
        ),
        buckets=tuple(float(bound) for bound in _required(record, "buckets", list)),
        owner=_required(record, "owner", str),
        cadence_seconds=_required(record, "cadence_seconds", int),
        cardinality_limit=_required(record, "cardinality_limit", int),
        maturity=_required(record, "maturity", str),
    )


def _event(record: dict[str, Any]) -> CatalogEvent:
    return CatalogEvent(
        event_name=_required(record, "event_name", str),
        namespace=_required(record, "namespace", str),
        delivery_class=_required(record, "delivery_class", str),
        owner=_required(record, "owner", str),
        attributes=tuple(_required(record, "attributes", list)),
    )


@cache
def load_catalog() -> TelemetryCatalog:
    document = json.loads(files("rigging").joinpath(_CATALOG_RESOURCE).read_text())
    if not isinstance(document, dict):
        raise ValueError("telemetry catalog must be a JSON object")
    catalog = TelemetryCatalog(
        catalog_version=_required(document, "catalog_version", str),
        metrics=tuple(_metric(record) for record in _required(document, "metrics", list)),
        events=tuple(_event(record) for record in _required(document, "events", list)),
        artifacts=tuple(_required(document, "artifacts", list)),
    )
    metric_keys = [(metric.scope, metric.name) for metric in catalog.metrics]
    if len(metric_keys) != len(set(metric_keys)):
        raise ValueError("telemetry catalog metric keys must be unique")
    event_names = [event.event_name for event in catalog.events]
    if len(event_names) != len(set(event_names)):
        raise ValueError("telemetry catalog event names must be unique")
    return catalog
