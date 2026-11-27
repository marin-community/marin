# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project Finelog records onto the fields used by Hublot."""

import json
import math
import re
from typing import Any

from .queries import CAPACITY_EVENT, CONFIG_EVENT, GPU_METRICS, GRADING_METRICS, TRAINING_METRIC_EVENT
from .query_contract import (
    ATTRIBUTE_FIELDS,
    BODY_FIELDS,
    CUMULATIVE_NAMES,
    DRIVER_LINE_PATTERNS,
    DRIVER_TIMESTAMP,
    SNAPSHOT_NAMES,
)

_CONFIG_FIELDS = (
    "mini_batch_size",
    "max_staleness_steps",
    "generation_workers",
    "policy_nodes",
    "policy_gpus_per_node",
    "policy_tp",
    "policy_pp",
    "policy_cp",
    "policy_ep",
    "strategy",
)
_SOURCE_FIELDS = ("cluster", "service", "run_id", "job_id", "execution_uid", "region", "node_name", "process_index")
_ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")
_LOG_TIME = re.compile(DRIVER_TIMESTAMP)
_DRIVER_FACT = re.compile("|".join(DRIVER_LINE_PATTERNS))


def _object(raw: str | None) -> dict[str, Any] | None:
    try:
        value = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _scalars(values: dict[str, Any], fields: tuple[str, ...]) -> dict[str, str | int | float | bool]:
    return {
        key: value
        for key in fields
        if isinstance((value := values.get(key)), (str, int, float))
        and (not isinstance(value, float) or math.isfinite(value))
    }


def _stable_count(values: set[float]) -> int | None:
    if len(values) != 1:
        return None
    value = next(iter(values))
    return int(value) if 0 < value < 2**53 and int(value) == value else None


def run_facts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect run settings from the scoped overview read."""
    configurations: set[str | None] = set()
    counts: dict[str, set[float]] = {key: set() for key in (*GPU_METRICS.values(), "bufferCapacity")}
    separate_grading = False
    for row in rows:
        name = row["name"]
        value = row["value"]
        if name == CONFIG_EVENT:
            configurations.add(row["body_json"])
        elif name == CAPACITY_EVENT and value is not None:
            counts["bufferCapacity"].add(value)
        elif name == TRAINING_METRIC_EVENT:
            metric = (_object(row["attributes_json"]) or {}).get("metric")
            if not isinstance(metric, str):
                continue
            if metric in GPU_METRICS and value is not None:
                counts[GPU_METRICS[metric]].add(value)
            if metric in GRADING_METRICS:
                separate_grading = True
    configuration = _scalars(_object(configurations.pop()) or {}, _CONFIG_FIELDS) if len(configurations) == 1 else None
    settings = {key: _stable_count(values) for key, values in counts.items()}
    return {
        "configuration": configuration,
        "runSettings": {**settings, "separateGrading": separate_grading},
    }


def telemetry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Preserve the full source and engine-label distinctions without returning their raw metadata.
    origins: dict[tuple, int] = {}
    series_ids: dict[tuple, int] = {}
    output = []
    for row in rows:
        attributes = _object(row["attributes_json"]) or {}
        resource_text = row["resource_attributes_json"]
        resource_data = _object(resource_text)
        # Keep unreadable resource labels distinct so their counters are not merged.
        resource = json.dumps(resource_data, sort_keys=True) if resource_data is not None else resource_text or "{}"
        origin = (*(row[key] for key in _SOURCE_FIELDS), resource)
        source = origins.setdefault(origin, len(origins))
        name = row["name"]
        measurement = {
            "name": name,
            "source": source,
            "seq": row["seq"],
            "timestamp_ms": row["timestamp_ms"],
            "value": row["value"] if row["value"] is not None and math.isfinite(row["value"]) else None,
            "attributes_json": _scalars(attributes, ATTRIBUTE_FIELDS),
            "body_json": _scalars(_object(row["body_json"]) or {}, BODY_FIELDS),
        }
        if name in CUMULATIVE_NAMES or name in SNAPSHOT_NAMES:
            if name in SNAPSHOT_NAMES:
                attributes.pop("step", None)
            labels = json.dumps(attributes, sort_keys=True)
            measurement["engineSeries"] = series_ids.setdefault((source, labels), len(series_ids))
        output.append(measurement)
    return output


def driver_facts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts = []
    for row in rows:
        line = _ANSI_COLOR.sub("", row["line"])
        match = _DRIVER_FACT.search(line)
        if match is None:
            continue
        stamp = _LOG_TIME.search(line)
        prefix = stamp.group() + " " if stamp else ""
        facts.append({"key": row["key"], "epoch_ms": row["epoch_ms"], "seq": row["seq"], "line": prefix + match.group()})
    return facts
