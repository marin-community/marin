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
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_parquet, open_file
from zephyr.writers import write_parquet_file

from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    AnswerType,
    ScriptGrader,
    TaskSpec,
    VerifyitGrader,
    verifyit_answer_file,
    verifyit_spec,
)
from taskcompendium.pipeline.execution_telemetry import SourceTelemetry, execute_phase
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.review import private_evidence_summary, review_payload
from taskcompendium.pipeline.sampling import merge_sample_rows, seeded_order, seeded_sample
from taskcompendium.submission import submission_compatibility, submission_instruction

PACKET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("cohort", pa.string()),
        ("packet_json", pa.string()),
        ("omissions", pa.list_(pa.string())),
    ]
)
EVIDENCE_READ_BLOCK_BYTES = 1_048_576
AUDIT_PREVIEW_CHARACTERS = 2048


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
    max_task_json_characters: int = 4_000_000


@dataclass(frozen=True)
class RowLocation:
    path: str
    row_group: int
    index: int


@dataclass(frozen=True)
class SubmissionContract:
    answer_type: AnswerType
    instruction: str | None
    answer_path: str | None
    conversation_path: str | None
    output_paths: tuple[str, ...]
    compatibility_reasons: tuple[str, ...]
    handling: str


def submission_contract(task: TaskSpec) -> SubmissionContract:
    """Describe the existing runtime's answer extraction and grader-file staging."""
    answer_path = None
    conversation_path = None
    handling = "No runnable grader is available; this is a declared source contract."
    if isinstance(task.grader, ScriptGrader):
        answer_path = task.grader.answer_path
        conversation_path = task.grader.conversation_path
        handling = (
            "The runtime extracts the answer from the conversation using answer_format and writes it to answer_path "
            "before invoking the script. The solver need not write that file."
            if answer_path is not None
            else "The runtime captures declared workspace outputs and available state."
        )
    elif isinstance(task.grader, VerifyitGrader):
        if task.grader.environment is None:
            handling = (
                "In-process Verifyit receives the extracted submission directly; "
                "answer_file does not require solver file delivery."
            )
        else:
            answer_path = verifyit_answer_file(verifyit_spec(task.grader))
            handling = (
                "The runtime extracts the conversation answer using answer_format and stages it at answer_path "
                "for Verifyit. The solver need not write that file."
                if task.answer_type in CONVERSATION_ANSWERS
                else "The runtime captures declared output files or state from the solver."
            )
    return SubmissionContract(
        task.answer_type,
        submission_instruction(task.answer_format) if task.answer_type in CONVERSATION_ANSWERS else None,
        answer_path,
        conversation_path,
        task.output_paths,
        submission_compatibility(task).reasons if task.answer_type in CONVERSATION_ANSWERS else (),
        handling,
    )


def _audit_preview(value: str) -> str | dict[str, Any]:
    if len(value) <= AUDIT_PREVIEW_CHARACTERS:
        return value
    return private_evidence_summary(value, "Full value retained in source audit", AUDIT_PREVIEW_CHARACTERS)


def read_json(path: str) -> dict[str, Any]:
    return json.loads(StoragePath(path).read_text())


def write_json(path: str, value: dict[str, Any]) -> None:
    with StoragePath(path).open("wt", auto_mkdir=True) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _indexed_rows(path: str, *, excluded: bool) -> Iterator[dict[str, Any]]:
    columns = ["task_id", "admission", "filter_reasons"] if excluded else ["task_id"]
    for row in load_parquet(InputFileSpec(path, columns=columns)):
        reason = None
        if excluded:
            if row["admission"] == "admitted":
                continue
            reason = (row["filter_reasons"] or [row["admission"]])[0]
        yield {"task_id": row["task_id"], "reason": reason}


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
    values = pa.array(sorted(task_ids), type=pa.string())
    with open_file(path, block_size=EVIDENCE_READ_BLOCK_BYTES) as stream:
        parquet = pq.ParquetFile(stream)
        for group in range(parquet.num_row_groups):
            if parquet.metadata.row_group(group).num_rows == 0:
                continue
            table = parquet.read_row_group(group, columns=["task_id"])
            matches = pc.call_function("is_in", [table["task_id"]], options=pc.SetLookupOptions(value_set=values))
            indices = pc.call_function("indices_nonzero", [matches]).to_pylist()
            for index in indices:
                yield table["task_id"][index].as_py(), RowLocation(path, group, index)


def _project_evidence(view: str, row: dict[str, Any], sampling: AssessmentSampling) -> dict[str, Any]:
    evidence = {}
    omissions = []
    for change in row.get("normalization_changes") or []:
        for key in ("original", "replacement"):
            if change[key] is not None:
                if len(change[key]) > AUDIT_PREVIEW_CHARACTERS:
                    omissions.append(f"{view}: normalization {change['field']} {key} preview truncated")
                change[key] = _audit_preview(change[key])
    if row.get("raw_json") is not None:
        if len(row["raw_json"]) > AUDIT_PREVIEW_CHARACTERS:
            omissions.append(f"{view}: original source row preview truncated")
        row["raw_json"] = _audit_preview(row["raw_json"])
    if view == "normalize":
        task_json = row.pop("task_json")
        if task_json is None:
            omissions.append("task: no converted TaskSpec")
        elif len(task_json) > sampling.max_task_json_characters:
            omissions.append("task: encoded TaskSpec exceeds evidence budget")
        else:
            task = TaskSpec.model_validate_json(task_json)
            if task.id != row["task_id"]:
                raise ValueError(f"TaskSpec ID disagrees with row: {row['task_id']}")
            evidence["task"] = review_payload(task)
            evidence["submission"] = asdict(submission_contract(task))
    evidence[view] = row
    if len(json.dumps(evidence, ensure_ascii=False)) > sampling.max_packet_characters:
        evidence.pop("task", None)
        omissions.append("task: projected evidence exceeds packet budget")
    if len(json.dumps(evidence, ensure_ascii=False)) > sampling.max_packet_characters:
        evidence = {view: {"task_id": row["task_id"], "omitted": "section exceeds packet budget"}}
        omissions.append(f"{view}: evidence exceeds packet budget")
    return {"task_id": row["task_id"], "evidence": evidence, "omissions": omissions}


def _selected_evidence(
    file: tuple[str, str, dict[int, dict[int, str]]], *, sampling: AssessmentSampling
) -> Iterator[dict[str, Any]]:
    """Decode each selected row group once, projecting only selected records."""
    view, path, groups = file
    with open_file(path, block_size=EVIDENCE_READ_BLOCK_BYTES) as stream:
        parquet = pq.ParquetFile(stream)
        for group, selected in groups.items():
            columns = [
                name
                for name in parquet.schema_arrow.names
                if (name != "task_json" or view == "normalize")
                and (name != "normalization_changes" or view == "normalize")
                and (name != "raw_json" or view == "review")
            ]
            table = parquet.read_row_group(group, columns=columns)
            rows = table.take(pa.array(list(selected), type=pa.int64())).to_pylist()
            for task_id, row in zip(selected.values(), rows, strict=True):
                if row["task_id"] != task_id:
                    raise ValueError(f"Selected row changed after ID scan: {task_id}")
                yield _project_evidence(view, row, sampling)


def _packet(
    selected: dict[str, Any], *, records: dict[str, list[dict[str, Any]]], sampling: AssessmentSampling
) -> dict[str, Any]:
    task_id = selected["task_id"]
    evidence = {}
    omissions = []
    for record in records.get(task_id, []):
        evidence.update(record["evidence"])
        omissions.extend(record["omissions"])
    for view in ("normalize", "review", "verify"):
        if view not in evidence:
            omissions.append(f"{view}: no row evidence")
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
        "packet_json": json.dumps(evidence, ensure_ascii=False, allow_nan=False, sort_keys=True),
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
        with telemetry.phase("evidence") as phase:
            files: dict[tuple[str, str], dict[int, dict[int, str]]] = {}
            for view, indexed in locations.items():
                for task_id, location in indexed.items():
                    files.setdefault((view, location.path), {}).setdefault(location.row_group, {})[
                        location.index
                    ] = task_id
            projected = execute_phase(
                context,
                Dataset.from_list([(view, path, indices) for (view, path), indices in files.items()]).flat_map(
                    partial(_selected_evidence, sampling=sampling)
                ),
                telemetry=phase,
            ).results
            records: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in projected:
                records[row["task_id"]].append(row)
        with telemetry.phase("packets"):
            packet_path = str(StoragePath(output_root) / "packets/part-00000.parquet")
            write_parquet_file(
                (_packet(row, records=records, sampling=sampling) for row in selected), packet_path, schema=PACKET_SCHEMA
            )
            paths = [packet_path]
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
                "population_scope": (
                    "Admitted final rows and processed non-admitted verify rows; unprocessed rows excluded"
                ),
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
        missing = sum("task" not in json.loads(packets[row["task_id"]]["packet_json"]) for row in rows)
        omissions = sum(bool(packets[row["task_id"]]["omissions"]) for row in rows)
        complete = all(row["status"] == ReviewStatus.REVIEWED for row in rows)
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
            "reviewed": sum(row["status"] == ReviewStatus.REVIEWED for row in rows),
            "verdict_counts": dict(counts),
            "missing_evidence": missing,
            "packets_with_omissions": omissions,
            "grade": grade,
            "examples": [row for row in rows if row["verdict"] != Verdict.GOOD][:5],
        }
    return {
        "schema_version": 1,
        "source": inputs["source"],
        "scope": inputs["scope"],
        "basis": "sampled_model_opinions",
        "grade_rule": (
            "Empty sample, missing TaskSpec or unavailable judgments give unknown; "
            "otherwise bad takes precedence over some_issues, then unknown, then good"
        ),
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
