# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare bounded post-curation evidence and summarize sampled model opinions."""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_parquet

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.execution_telemetry import SourceTelemetry, execute_phase
from taskcompendium.pipeline.review import review_payload
from taskcompendium.pipeline.sampling import merge_sample_rows, seeded_order, seeded_sample

PACKET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("cohort", pa.string()),
        ("packet_json", pa.string()),
        ("omissions", pa.list_(pa.string())),
    ]
)
PACKET_SHARDS = 4


class Cohort(StrEnum):
    ADMITTED = "admitted"
    EXCLUDED = "excluded"


class Verdict(StrEnum):
    GOOD = "good"
    SOME_ISSUES = "some_issues"
    BAD = "bad"
    UNKNOWN = "unknown"


class TaskJudgment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    verdict: Verdict
    explanation: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)


@dataclass(frozen=True)
class AssessmentSampling:
    admitted: int = 100
    excluded: int = 20
    seed: int = 0
    max_packet_characters: int = 64_000


@dataclass(frozen=True)
class RowLocation:
    path: str
    index: int


def read_json(path: str) -> dict[str, Any]:
    return json.loads(StoragePath(path).read_text())


def write_json(path: str, value: dict[str, Any]) -> None:
    with StoragePath(path).open("wt", auto_mkdir=True) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _indexed_rows(path: str, *, excluded: bool) -> Iterator[dict[str, Any]]:
    columns = ["task_id", "admission", "filter_reasons"] if excluded else ["task_id"]
    for index, row in enumerate(load_parquet(InputFileSpec(path, columns=columns))):
        reason = None
        if excluded:
            if row["admission"] == "admitted":
                continue
            reason = (row["filter_reasons"] or [row["admission"]])[0]
        yield {"task_id": row["task_id"], "path": path, "index": index, "reason": reason}


def _task_id(row: dict[str, Any]) -> str:
    return row["task_id"]


def _unique_row(task_id: str, rows: Iterator[dict[str, Any]]) -> dict[str, Any]:
    first = next(rows)
    if next(rows, None) is not None:
        raise ValueError(f"Duplicate assessment task ID: {task_id}")
    return first


def _order(row: dict[str, Any], *, seed: int) -> tuple[str, str]:
    return seeded_order(row["task_id"], seed)


def _sample_rows(rows: Iterator[dict[str, Any]], *, size: int, seed: int) -> tuple[int, list[dict[str, Any]]]:
    return seeded_sample(rows, size=size, key=partial(_order, seed=seed))


def _merge_rows(samples: Iterator[tuple[int, list[dict[str, Any]]]], *, size: int, seed: int):
    return merge_sample_rows(samples, size=size, key=partial(_order, seed=seed))


def _reason_samples(rows: Iterator[dict[str, Any]], *, size: int, seed: int) -> dict[str, tuple[int, list[dict]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    priorities: dict[str, tuple[str, str]] = {}
    # Each group retains at most N candidates, independent of the population size.
    for row in rows:
        reason = row["reason"]
        counts[reason] += 1
        candidates = groups[reason]
        priorities[row["task_id"]] = seeded_order(row["task_id"], seed)
        candidates.append(row)
        if len(candidates) > size:
            worst = max(candidates, key=lambda candidate: priorities[candidate["task_id"]])
            candidates.remove(worst)
            del priorities[worst["task_id"]]
    return {reason: (counts[reason], candidates) for reason, candidates in groups.items()}


def _merge_reasons(samples: Iterator[dict[str, tuple[int, list[dict]]]], *, size: int, seed: int):
    groups: dict[str, list[tuple[int, list[dict]]]] = defaultdict(list)
    for sample in samples:
        for reason, rows in sample.items():
            groups[reason].append(rows)
    return {
        reason: merge_sample_rows(rows, size=size, key=partial(_order, seed=seed)) for reason, rows in groups.items()
    }


def _diagnostic_sample(groups: dict[str, tuple[int, list[dict]]], size: int, seed: int) -> list[dict]:
    """Allocate one item per reason per round, with seeded ordering within groups."""
    pending = {reason: sorted(rows, key=partial(_order, seed=seed)) for reason, (_, rows) in groups.items()}
    selected = []
    reasons = sorted(pending, key=lambda reason: seeded_order(reason, seed))
    while len(selected) < size and any(pending.values()):
        for reason in reasons:
            if pending[reason] and len(selected) < size:
                selected.append(pending[reason].pop(0))
    return selected


def _selected_locations(path: str, *, task_ids: frozenset[str]) -> Iterator[tuple[str, RowLocation]]:
    for index, row in enumerate(load_parquet(InputFileSpec(path, columns=["task_id"]))):
        if row["task_id"] in task_ids:
            yield row["task_id"], RowLocation(path, index)


def _read_row(location: RowLocation) -> dict[str, Any]:
    return next(load_parquet(InputFileSpec(location.path, row_start=location.index, row_end=location.index + 1)))


def _packet(
    selected: dict[str, Any], *, locations: dict[str, dict[str, RowLocation]], sampling: AssessmentSampling
) -> dict[str, Any]:
    task_id = selected["task_id"]
    evidence = {}
    omissions = []
    for view, indexed in locations.items():
        location = indexed.get(task_id)
        if location is None:
            omissions.append(f"{view}: no row evidence")
            continue
        row = _read_row(location)
        if view == "normalize":
            task_json = row.pop("task_json")
            if task_json is None:
                omissions.append("task: no converted TaskSpec")
            elif len(task_json) > sampling.max_packet_characters * 8:
                omissions.append("task: encoded TaskSpec exceeds evidence budget")
            else:
                task = TaskSpec.model_validate_json(task_json)
                if task.id != task_id:
                    raise ValueError(f"TaskSpec ID disagrees with row: {task_id}")
                evidence["task"] = review_payload(task)
        evidence[view] = row
    # Previous reviews are opinions. Preserve them as a separate cited evidence section.
    if len(json.dumps(evidence, ensure_ascii=False)) > sampling.max_packet_characters:
        evidence.pop("task", None)
        omissions.append("task: projected evidence exceeds packet budget")
    if len(json.dumps(evidence, ensure_ascii=False)) > sampling.max_packet_characters:
        evidence = {view: {"task_id": task_id, "omitted": "section exceeds packet budget"} for view in evidence}
        omissions.append("audit: evidence exceeds packet budget")
    return {
        "task_id": task_id,
        "cohort": selected["cohort"],
        "packet_json": json.dumps(evidence, ensure_ascii=False, allow_nan=False),
        "omissions": omissions,
    }


def prepare_assessment(
    context: ZephyrContext, *, curated_root: str, sampling: AssessmentSampling, output_root: str
) -> None:
    """Select admitted and excluded rows, then materialize bounded evidence on workers."""
    if sampling.admitted < 1 or sampling.excluded < 0 or sampling.max_packet_characters < 1:
        raise ValueError("Assessment sampling and packet budgets must be positive")
    manifest = read_json(str(StoragePath(curated_root) / "manifest.json"))
    views = manifest["datasets"]
    telemetry = SourceTelemetry(manifest["source"], output_root)
    with telemetry.record():
        with telemetry.phase("select") as phase:
            admitted = (
                Dataset.from_files(str(StoragePath(views["final"]) / "*.parquet"))
                .flat_map(partial(_indexed_rows, excluded=False))
                .group_by(_task_id, reducer=_unique_row)
                .reduce(
                    partial(_sample_rows, size=sampling.admitted, seed=sampling.seed),
                    partial(_merge_rows, size=sampling.admitted, seed=sampling.seed),
                )
            )
            admitted_count, admitted_rows = execute_phase(
                context, admitted, telemetry=phase, operation="admitted"
            ).results[0]
            excluded = (
                Dataset.from_files(str(StoragePath(views["verify"]) / "*.parquet"))
                .flat_map(partial(_indexed_rows, excluded=True))
                .group_by(_task_id, reducer=_unique_row)
                .reduce(
                    partial(_reason_samples, size=sampling.excluded, seed=sampling.seed),
                    partial(_merge_reasons, size=sampling.excluded, seed=sampling.seed),
                )
            )
            groups = execute_phase(context, excluded, telemetry=phase, operation="excluded").results[0]
            selected = [
                *({**row, "cohort": Cohort.ADMITTED} for row in admitted_rows),
                *(
                    {**row, "cohort": Cohort.EXCLUDED}
                    for row in _diagnostic_sample(groups, sampling.excluded, sampling.seed)
                ),
            ]
            if len({_task_id(row) for row in selected}) != len(selected):
                raise ValueError("Assessment cohorts overlap")
        with telemetry.phase("locate") as phase:
            task_ids = frozenset(_task_id(row) for row in selected)
            locations = {}
            for view in ("normalize", "review", "verify"):
                pairs = execute_phase(
                    context,
                    Dataset.from_files(str(StoragePath(views[view]) / "*.parquet")).flat_map(
                        partial(_selected_locations, task_ids=task_ids)
                    ),
                    telemetry=phase,
                    operation=view,
                ).results
                if len({task_id for task_id, _ in pairs}) != len(pairs):
                    raise ValueError(f"Duplicate selected ID in {view}")
                locations[view] = dict(pairs)
        with telemetry.phase("packets") as phase:
            paths = execute_phase(
                context,
                Dataset.from_list(selected)
                .reshard(min(PACKET_SHARDS, max(1, len(selected))))
                .map(partial(_packet, locations=locations, sampling=sampling))
                .write_parquet(str(StoragePath(output_root) / "packets/part-{shard:05d}.parquet"), schema=PACKET_SCHEMA),
                telemetry=phase,
            ).results
        write_json(str(StoragePath(output_root) / "curation.json"), manifest)
        write_json(
            str(StoragePath(output_root) / "manifest.json"),
            {
                "schema_version": 1,
                "source": manifest["source"],
                "curated_root": curated_root,
                "scope": manifest["mode"],
                "sampling": asdict(sampling),
                "admitted_population": admitted_count,
                "excluded_population": sum(count for count, _ in groups.values()),
                "excluded_reasons": {reason: count for reason, (count, _) in groups.items()},
                "sample": [{"task_id": row["task_id"], "cohort": row["cohort"]} for row in selected],
                "packet_paths": paths,
            },
        )


def assessment_summary(inputs_root: str, judgments_root: str) -> dict[str, Any]:
    """Keep release opinions, diagnostic opinions and executed controls separate."""
    inputs = read_json(str(StoragePath(inputs_root) / "manifest.json"))
    judgments = read_json(str(StoragePath(judgments_root) / "manifest.json"))
    packets = {row["task_id"]: row for path in inputs["packet_paths"] for row in load_parquet(path)}
    grades = [row for path in judgments["grade_paths"] for row in load_parquet(path)]
    validate_coverage(inputs["sample"], grades)
    cohorts = {}
    for cohort in Cohort:
        rows = [row for row in grades if row["cohort"] == cohort]
        counts = Counter(row["verdict"] for row in rows)
        missing = sum(bool(packets[row["task_id"]]["omissions"]) for row in rows)
        complete = all(row["status"] == "reviewed" for row in rows)
        grade = Verdict.UNKNOWN
        if rows and complete and not missing:
            if counts[Verdict.BAD]:
                grade = Verdict.BAD
            elif counts[Verdict.SOME_ISSUES]:
                grade = Verdict.SOME_ISSUES
            elif counts[Verdict.UNKNOWN] == 0:
                grade = Verdict.GOOD
        cohorts[cohort] = {
            "population": inputs[f"{cohort}_population"],
            "sample_count": len(rows),
            "reviewed": sum(row["status"] == "reviewed" for row in rows),
            "verdict_counts": dict(counts),
            "missing_evidence": missing,
            "grade": grade,
            "examples": [row for row in rows if row["verdict"] != Verdict.GOOD][:5],
        }
    return {
        "schema_version": 1,
        "source": inputs["source"],
        "scope": inputs["scope"],
        "basis": "sampled_model_opinions",
        "assessment": cohorts,
        "curation": read_json(str(StoragePath(inputs_root) / "curation.json")),
        "model": judgments["model"],
        "rubric": judgments["rubric"],
        "evidence": {"inputs": inputs_root, "judgments": judgments_root},
    }


def validate_coverage(selected: Iterable[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    expected = {(row["task_id"], row["cohort"]) for row in selected}
    actual = {(row["task_id"], row["cohort"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Grade records do not match the selected assessment sample")
