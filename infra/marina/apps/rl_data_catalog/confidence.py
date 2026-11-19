# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the immutable, precomputed confidence vectors before a curriculum load."""

from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumCoverage

from .server.source_identity import canonical_source_id

OUTCOMES = ("assigned", "subject_only", "out_of_scope", "insufficient_context", "skipped")


def confidence_rows(document: dict[str, Any], release: AlignmentRelease, coverage: CurriculumCoverage) -> Iterator[dict]:
    """Reconcile every source, node and aggregate at every exported confidence level."""
    if document.get("example_only") or document["schema_version"] != 1:
        raise ValueError("Confidence input must be a measured version-one export")
    if (document["release"]["release_id"], document["coverage"]["coverage_id"]) != (
        release.release_id,
        coverage.coverage_id,
    ):
        raise ValueError("Confidence input belongs to another release")
    if document["levels"] != [level.model_dump(mode="json") for level in coverage.confidence_levels]:
        raise ValueError("Confidence levels differ from the coverage manifest")
    if (document["samples"]["paths"], document["samples"]["task_count"]) != (
        coverage.sample_paths,
        coverage.sample_task_count,
    ):
        raise ValueError("Confidence sample union differs from coverage")
    width = len(document["levels"])
    if not width:
        raise ValueError("Confidence export has no levels")
    nodes = {(node.subject_id, node.node_id): node for node in release.nodes}
    children_by_parent: dict[tuple, list[str]] = defaultdict(list)
    for node in release.nodes:
        children_by_parent[(node.subject_id, node.parent_id)].append(node.node_id)
    expected = {canonical_source_id(source.source_id): source for source in release.sources}
    sums: dict[str, dict[tuple, list[int]]] = {
        scope: defaultdict(lambda: [0] * width) for scope in ("all", "train", "eval")
    }
    seen_sources = set()

    def checked(scope: str, statistics: list[dict]) -> dict[tuple, list[int]]:
        result = {}
        for row in statistics:
            key = (row["metric"], row["subject_id"], row["node_id"])
            values = row["instances"]
            if key in result or len(values) != width or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"Invalid or repeated confidence vector: {scope} {key}")
            if row["metric"] in ("placement", "capability"):
                if key[1:] not in nodes:
                    raise ValueError(f"Unknown confidence node: {key}")
            elif row["metric"] not in ("denominator", *OUTCOMES) or key[1:] != ("", ""):
                raise ValueError(f"Unknown confidence metric: {key}")
            result[key] = values
        zero = [0] * width
        denominator = result.get(("denominator", "", ""), zero)
        if any(
            sum(result.get((metric, "", ""), zero)[i] for metric in OUTCOMES) != denominator[i] for i in range(width)
        ):
            raise ValueError(f"Confidence outcomes do not partition the population: {scope}")
        for subject, node_id in nodes:
            children = children_by_parent[(subject, node_id)]
            if children and any(
                sum(result.get(("placement", subject, child), zero)[i] for child in children)
                > result.get(("placement", subject, node_id), zero)[i]
                for i in range(width)
            ):
                raise ValueError(f"Confidence children exceed their parent: {scope} {node_id}")
        return result

    for source in document["sources"]:
        source_id = canonical_source_id(source["source_id"])
        if source_id in seen_sources or source_id not in expected:
            raise ValueError(f"Unknown or repeated confidence source: {source_id}")
        seen_sources.add(source_id)
        alignment = expected[source_id]
        if source["intended_use"] != alignment.intended_use:
            raise ValueError(f"Confidence Use differs from the release: {source_id}")
        statistics = checked(source_id, source["statistics"])
        if statistics.get(("denominator", "", "")) != [alignment.task_count] * width:
            raise ValueError(f"Confidence denominator differs from the release: {source_id}")
        for key, values in statistics.items():
            for scope in ("all", source["intended_use"]):
                sums[scope][key] = [a + b for a, b in zip(sums[scope][key], values, strict=True)]
            yield dict(scope_id=source_id, metric=key[0], subject_id=key[1], node_id=key[2], instances=values)
    if seen_sources != expected.keys():
        raise ValueError("Confidence source inventory differs from the release")
    for scope, statistics in document["aggregates"].items():
        if scope not in sums:
            raise ValueError(f"Unknown confidence aggregate: {scope}")
        values = checked(scope, statistics)
        for key in values.keys() | sums[scope].keys():
            if values.get(key, [0] * width) != sums[scope].get(key, [0] * width):
                raise ValueError(f"Confidence aggregate differs from source vectors: {scope} {key}")
        for key, vector in values.items():
            yield dict(scope_id="@" + scope, metric=key[0], subject_id=key[1], node_id=key[2], instances=vector)
    if document["aggregates"].keys() != sums.keys():
        raise ValueError("Confidence aggregates must cover all, train and eval")
