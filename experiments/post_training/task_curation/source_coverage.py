# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Each source's use, converted example count and leading curriculum subject.

The values are measurements, so they live in one generated table,
``source_coverage.json``, rather than in hand-edited declarations.
``regenerate_coverage`` writes the table from curriculum releases.
``all_sources()`` attaches each entry to its declaration and rejects a catalog in
which any source has none. Unmeasured fields require an explicit reason.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path

from finestore.rl.curation import CurriculumArea, SourceCoverage, SourceUse

COVERAGE_TABLE = Path(__file__).with_name("source_coverage.json")
PLACEMENT_METRIC = "placement"
SUBJECT_NODE_PREFIX = "subject:"


class CoverageGap(StrEnum):
    """Why a coverage value has no measurement."""

    NO_CONVERTER = "no conversion pipeline, so no converted tasks exist to count or place"
    NOT_EMBEDDED = "converted tasks have not been embedded in a curriculum release"
    NO_PLACEMENT = "no converted task matched a curriculum subject within the subject cutoff"
    NO_PINNED_COUNT = "the pinned payload has no established row count"


def load_coverage(path: Path = COVERAGE_TABLE) -> dict[str, SourceCoverage]:
    return {source_id: SourceCoverage.model_validate(row) for source_id, row in json.loads(path.read_text()).items()}


def require_coverage(source_id: str, coverage: SourceCoverage | None) -> None:
    """Require ingestion fields, allowing documented unknown counts and areas."""
    if coverage is None:
        raise ValueError(f"Source {source_id} has no coverage entry; update {COVERAGE_TABLE.name}")
    gaps = {gap.value for gap in CoverageGap}
    if coverage.unavailable_reason is not None and coverage.unavailable_reason not in gaps:
        raise ValueError(f"Source {source_id} gives an unrecognized coverage gap: {coverage.unavailable_reason}")
    if (coverage.example_count is None or coverage.top_area is None) and not coverage.unavailable_reason:
        raise ValueError(f"Source {source_id} leaves coverage unmeasured without a reason")


def top_subjects(statistics: Iterable[Mapping], names: Mapping[str, str]) -> dict[str, CurriculumArea]:
    """The subject with the most placed tasks per source, ties broken by subject ID as Atlas does."""
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for row in statistics:
        if row["metric"] == PLACEMENT_METRIC and row["node_id"] == SUBJECT_NODE_PREFIX + row["subject_id"]:
            if row["instances"] > 0:
                counts[row["source_id"]][row["subject_id"]] = row["instances"]
    return {
        source_id: CurriculumArea(subject_id=subject, name=names[subject], instances=subjects[subject])
        for source_id, subjects in counts.items()
        for subject in [min(subjects, key=lambda item: (-subjects[item], item))]
    }


def release_coverage(release: Mapping, release_uri: str, statistics: Iterable[Mapping]) -> dict[str, SourceCoverage]:
    """Coverage for each source a curriculum release matched, from its full-population statistics.

    ``release`` is the parsed ``release.json``. Releases published before declared
    rows existed carry no ``basis``; every source in them was matched.
    """
    names = {node["subject_id"]: node["name"] for node in release["nodes"] if node["kind"] == "subject"}
    tops = top_subjects(statistics, names)
    scope = "a seeded sample" if release.get("sample_size") is not None else "every"
    result = {}
    for row in release["sources"]:
        if row.get("basis", "matched") != "matched":
            continue
        count = row["converted_rows"] if row.get("converted_rows") is not None else row["task_count"]
        top = tops.get(row["source_id"])
        result[row["source_id"]] = SourceCoverage(
            intended_use=SourceUse(row["intended_use"]),
            example_count=count,
            top_area=top,
            basis=(
                f"curriculum release {release_uri}: {row['population']} population of {count} tasks, "
                f"{scope} task embedded; leading subject by placed-task count"
            ),
            unavailable_reason=None if top is not None else CoverageGap.NO_PLACEMENT.value,
        )
    return result


def declared_coverage(declaration) -> SourceCoverage:
    """Coverage of a source no release matched, from its pinned declaration only."""
    gap = CoverageGap.NO_CONVERTER if declaration.pipeline is None else CoverageGap.NOT_EMBEDDED
    return SourceCoverage(
        intended_use=SourceUse.EVAL if "eval" in declaration.tags else SourceUse.TRAIN,
        example_count=declaration.task_count,
        top_area=None,
        basis="declaration: pinned row count before conversion; use from the eval tag",
        unavailable_reason=(gap if declaration.task_count is not None else CoverageGap.NO_PINNED_COUNT).value,
    )
