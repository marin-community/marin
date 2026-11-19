# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check every precomputed curriculum count through the API, without a browser walk."""

import math
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from .server.curriculum import TOTAL_METRICS

OUTCOMES = tuple(metric for metric in TOTAL_METRICS if metric != "denominator")


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return row["subject_id"], row["node_id"]


def _normalized(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.casefold().replace("&", "and"))


@dataclass(frozen=True)
class CountProjection:
    totals: dict[str, int]
    areas: dict[tuple[str, str], int]


def count_projection(data: dict[str, Any], level: int) -> CountProjection:
    """Project the same exported vector coordinates selected by the viewer."""
    confidence = data.get("confidence")
    if not confidence:
        return CountProjection(data["totals"], {_key(row): row["count"] for row in data["area_counts"]})
    totals, areas = {}, {}
    for row in confidence["statistics"]:
        if not row["subject_id"] and not row["node_id"]:
            totals[row["metric"]] = row["instances"][level]
        elif row["metric"] == "placement":
            areas[_key(row)] = row["instances"][level]
    return CountProjection({metric: totals.get(metric, 0) for metric in TOTAL_METRICS}, areas)


def representative_domain(data: dict[str, Any]) -> str:
    """Prefer a measured small domain so a smoke check need not download the whole sample union."""
    areas = count_projection(data, 0).areas
    choices = [
        (areas.get(_key(row), 0), row["subject_id"])
        for row in data["hierarchy"]
        if row["kind"] == "subject" and areas.get(_key(row), 0) > 0
    ]
    return min(choices)[1] if choices else ""


def check_curriculum_document(document: dict[str, Any]) -> dict[str, Any]:
    """Reconcile API projections against independent source vectors and the archived taxonomy."""
    release = document["release_id"]
    taxonomy = {_key(row): row for row in document["taxonomy"]}
    if len(taxonomy) != len(document["taxonomy"]):
        raise ValueError(f"{release}: duplicate taxonomy nodes")
    children: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for key, row in taxonomy.items():
        if row["parent_id"] is not None:
            parent = row["subject_id"], row["parent_id"]
            if parent not in taxonomy:
                raise ValueError(f"{release}: missing taxonomy parent {parent}")
            children[parent].append(key)
    subjects = {row["subject_id"]: row for row in taxonomy.values() if row["kind"] == "subject"}
    sources = {row["source_id"]: row for row in document["sources"]}
    if len(sources) != len(document["sources"]):
        raise ValueError(f"{release}: duplicate source statistics")
    cases = []
    for use in ("all", "train", "eval"):
        data = document["views"][use]
        if "sources" in data and {row["source_id"] for row in data["sources"]} != sources.keys():
            raise ValueError(f"{release}/{use}: displayed source inventory differs")
        nodes = {_key(row): row for row in data["hierarchy"]}
        if len(nodes) != len(data["hierarchy"]) or nodes.keys() != taxonomy.keys():
            raise ValueError(f"{release}/{use}: taxonomy inventory differs, including empty nodes")
        for key, node in nodes.items():
            if node["kind"] != taxonomy[key]["kind"] or node["parent_id"] != taxonomy[key]["parent_id"]:
                raise ValueError(f"{release}/{use}: taxonomy structure differs at {key}")
            if node["kind"] != "subject" and _normalized(node["name"]) == _normalized(
                nodes[(key[0], subjects[key[0]]["node_id"])]["name"]
            ):
                raise ValueError(f"{release}/{use}: capability label equals its subject: {key}")
        confidence = data.get("confidence")
        width = len(confidence["levels"]) if confidence else 1
        if confidence:
            aggregate_keys = set()
            for row in confidence["statistics"]:
                key = row["metric"], row["subject_id"], row["node_id"]
                values = row["instances"]
                if key in aggregate_keys or len(values) != width or any(type(v) is not int or v < 0 for v in values):
                    raise ValueError(f"{release}/{use}: invalid aggregate confidence vector {key}")
                aggregate_keys.add(key)
        sums: dict[tuple[str, str, str], list[int]] = defaultdict(lambda width=width: [0] * width)
        for source in sources.values():
            alignment = source["alignment"]
            selected = (
                alignment is not None
                and alignment["basis"] == "matched"
                and (use == "all" or alignment["intended_use"] == use)
            )
            records = source["confidence_statistics"] if confidence else source["statistics"]
            observed = {}
            for metric, subject, node, raw in records:
                key = metric, subject, node
                values = raw if confidence else [raw]
                if key in observed or len(values) != width or any(type(v) is not int or v < 0 for v in values):
                    raise ValueError(f"{release}: invalid source vector {source['source_id']} {key}")
                if (metric in TOTAL_METRICS and (subject or node)) or (
                    metric not in TOTAL_METRICS
                    and (metric not in {"placement", "capability"} or (subject, node) not in taxonomy)
                ):
                    raise ValueError(f"{release}: invalid source vector coordinate {key}")
                observed[key] = values
                if selected:
                    sums[key] = [a + b for a, b in zip(sums[key], values, strict=True)]
            if alignment is not None and alignment["basis"] == "matched":
                if observed.get(("denominator", "", "")) != [alignment["task_count"]] * width:
                    raise ValueError(f"{release}: source population differs: {source['source_id']}")
                for level in range(width):
                    if (
                        sum(observed.get((metric, "", ""), [0] * width)[level] for metric in OUTCOMES)
                        != alignment["task_count"]
                    ):
                        raise ValueError(f"{release}: source outcomes do not partition {source['source_id']}")
            elif any(any(values) for values in observed.values()):
                raise ValueError(f"{release}: unmeasured source claims statistics: {source['source_id']}")
        for level in range(width):
            projected = count_projection(data, level)
            totals, areas = projected.totals, projected.areas
            if sum(totals.get(metric, 0) for metric in OUTCOMES) != totals["denominator"]:
                raise ValueError(f"{release}/{use}/{level}: outcomes do not partition the denominator")
            for metric in TOTAL_METRICS:
                if totals.get(metric, 0) != sums[(metric, "", "")][level]:
                    raise ValueError(f"{release}/{use}/{level}: aggregate {metric} differs from sources")
            if not areas.keys() <= taxonomy.keys():
                raise ValueError(f"{release}/{use}/{level}: unknown projected area")
            for key in taxonomy:
                count = areas.get(key, 0)
                if count != sums[("placement", *key)][level]:
                    raise ValueError(f"{release}/{use}/{level}: source breakdown differs for {key}")
                child_count = sum(areas.get(child, 0) for child in children[key])
                if child_count > count:
                    raise ValueError(f"{release}/{use}/{level}: children exceed parent {key}")
            cases.append(
                {
                    "use": use,
                    "level": level,
                    "domains": len(subjects),
                    "nodes": len(taxonomy),
                    "denominator": totals["denominator"],
                }
            )
        if confidence:
            original = next(
                (
                    i
                    for i, level in enumerate(confidence["levels"])
                    if math.isclose(level["subject_min_similarity"], 1 - data["thresholds"]["subject_distance"])
                    and math.isclose(level["capability_min_similarity"], 1 - data["thresholds"]["capability_distance"])
                ),
                None,
            )
            if original is None:
                raise ValueError(f"{release}: recorded cutoffs have no exported confidence level")
            projected = count_projection(data, original)
            recorded_areas = {_key(row): row["count"] for row in data["area_counts"]}
            if projected.totals != data["totals"] or any(
                projected.areas.get(key, 0) != recorded_areas.get(key, 0) for key in taxonomy
            ):
                raise ValueError(f"{release}/{use}: displayed release counts differ from API confidence counts")
    return {"release_id": release, "cases": cases, "source_vectors": len(sources)}


def curriculum_invariants(get: Callable[[str], Any], releases: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    results = [
        check_curriculum_document(get("api/curriculum/invariants?" + urlencode({"release": release["release_id"]})))
        for release in releases
    ]
    return {"releases": results, "seconds": time.monotonic() - started}
