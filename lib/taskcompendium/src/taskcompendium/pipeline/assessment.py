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
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles
from taskcompendium.pipeline.models import Admission, ReviewStatus
from taskcompendium.pipeline.review import private_evidence_summary, review_payload
from taskcompendium.pipeline.sampling import merge_sample_rows, seeded_order, seeded_sample
from taskcompendium.pipeline.source_evidence import READ_BLOCK_BYTES, OriginalRow, SourceEvidenceFile, original_rows
from taskcompendium.pipeline.sources import prepared_component_path, prepared_manifest_path, staged_inputs
from taskcompendium.submission import submission_compatibility, submission_instruction

PACKET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("cohort", pa.string()),
        ("packet_json", pa.string()),
        ("omissions", pa.list_(pa.string())),
    ]
)
AUDIT_PREVIEW_CHARACTERS = 2048
AUDIT_FIELD_OWNERS = {"task_json": "normalize", "normalization_changes": "normalize"}


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
class SelectedFile:
    view: str
    path: str
    groups: dict[int, dict[int, str]]


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
            else (
                "The runtime captures workspace outputs and available state, "
                "and stages the conversation at conversation_path."
            )
        )
        if answer_path is not None:
            handling += " The runtime also stages the conversation at conversation_path."
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
                else (
                    "The runtime captures declared output files from the solver workspace, plus answer_path when "
                    "configured. Supplied state can be graded without an answer file."
                )
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


def _audit_preview(value: str) -> tuple[str | dict[str, Any], bool]:
    if len(value) <= AUDIT_PREVIEW_CHARACTERS:
        return value, False
    return private_evidence_summary(value, "Full value retained in source audit", AUDIT_PREVIEW_CHARACTERS), True


def read_json(path: str) -> dict[str, Any]:
    return json.loads(StoragePath(path).read_text())


def write_json(path: str, value: dict[str, Any]) -> None:
    with StoragePath(path).open("wt", auto_mkdir=True) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _indexed_rows(path: str, *, cohort: Cohort) -> Iterator[dict[str, Any]]:
    columns = ["task_id", "admission", "filter_reasons"] if cohort == Cohort.EXCLUDED else ["task_id"]
    for row in load_parquet(InputFileSpec(path, columns=columns)):
        reason = None
        if cohort == Cohort.EXCLUDED:
            if row["admission"] == Admission.ADMITTED:
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
    with open_file(path, block_size=READ_BLOCK_BYTES) as stream:
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
                change[key], truncated = _audit_preview(change[key])
                if truncated:
                    omissions.append(f"{view}: normalization {change['field']} {key} preview truncated")
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
    return {
        "task_id": row["task_id"],
        "source_locator": row.get("source_locator"),
        "raw_input_sha256": row.get("raw_input_sha256"),
        "evidence": evidence,
        "omissions": omissions,
    }


def _selected_evidence(file: SelectedFile, *, sampling: AssessmentSampling) -> Iterator[dict[str, Any]]:
    """Yield bounded evidence for selected records from a Parquet file."""
    with open_file(file.path, block_size=READ_BLOCK_BYTES) as stream:
        parquet = pq.ParquetFile(stream)
        columns = [
            name
            for name in parquet.schema_arrow.names
            if name != "raw_json" and AUDIT_FIELD_OWNERS.get(name, file.view) == file.view
        ]
        # Selected rows from the same group share one decode.
        for group, selected in file.groups.items():
            table = parquet.read_row_group(group, columns=columns)
            rows = table.take(pa.array(list(selected), type=pa.int64())).to_pylist()
            for task_id, row in zip(selected.values(), rows, strict=True):
                if row["task_id"] != task_id:
                    raise ValueError(f"Selected row changed after ID scan: {task_id}")
                yield _project_evidence(file.view, row, sampling)


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


def _original_files(records: dict[str, list[dict[str, Any]]], download: dict[str, Any]) -> list[SourceEvidenceFile]:
    grouped: dict[str, dict[int, OriginalRow]] = {}
    for task_id, evidence in records.items():
        identity = next((row for row in evidence if row.get("source_locator") is not None), None)
        if identity is None:
            raise ValueError(f"Selected task has no original-row locator: {task_id}")
        file, index_text = identity["source_locator"].rsplit(":", 1)
        if file.startswith("/") or ".." in file.split("/"):
            raise ValueError(f"Original source file must be relative: {file}")
        if file not in download["staged_files"]:
            raise ValueError(f"Original source file is outside the recorded inputs: {file}")
        index = int(index_text)
        rows = grouped.setdefault(file, {})
        if index in rows:
            raise ValueError(f"Selected tasks share an original-row locator: {file}:{index}")
        rows[index] = OriginalRow(task_id, identity["source_locator"], identity["raw_input_sha256"])
    overrides = download["source_file_overrides"]
    component = download["files"].get("prepared_component")
    layout = download["files"].get("prepared_source_revision")
    if component is not None and layout not in ("2", "3"):
        raise ValueError(f"Unknown recorded prepared-source layout: {layout}")
    return [
        SourceEvidenceFile(
            file,
            overrides[file]["path"] if file in overrides else str(StoragePath(download["source_input"]) / file),
            rows,
            prepared_path=(
                str(
                    prepared_manifest_path(StoragePath(download["source_input"]) / file).parent
                    if layout == "2"
                    else prepared_component_path(StoragePath(download["source_input"]) / file, component)
                )
                if component is not None and file not in overrides
                else None
            ),
        )
        for file, rows in sorted(grouped.items())
    ]


def prepare_assessment(
    context: ZephyrContext,
    *,
    curated_root: str,
    sampling: AssessmentSampling,
    raw_source: SourceFiles | None,
    raw_source_identity: dict[str, Any] | None,
    output_root: str,
) -> None:
    """Select admitted and excluded rows, then materialize bounded evidence on workers."""
    if sampling.admitted < 1 or sampling.excluded < 0 or sampling.max_packet_characters < 1:
        raise ValueError("Assessment sampling and packet budgets must be positive")
    manifest = read_json(str(StoragePath(curated_root) / "manifest.json"))
    download = None
    if raw_source is not None:
        download = read_json(str(StoragePath(manifest["datasets"]["download"]) / "manifest.json"))
        # Preparation layout is acquisition metadata, not the source reader's semantics.
        expected_reader = json.loads(json.dumps(raw_source_identity))
        recorded_reader = dict(download["files"])
        expected_reader.pop("prepared_source_revision", None)
        recorded_reader.pop("prepared_source_revision", None)
        if expected_reader != recorded_reader:
            raise ValueError("Original source reader differs from the curation invocation")
    views = manifest["datasets"]
    telemetry = SourceTelemetry(manifest["source"], output_root)
    with telemetry.record():
        with telemetry.phase("select") as phase:
            admitted = (
                Dataset.from_files(str(StoragePath(views["final"]) / "*.parquet"))
                .flat_map(partial(_indexed_rows, cohort=Cohort.ADMITTED))
                .group_by(_task_id, reducer=_unique_row)
                .reduce(
                    partial(seeded_sample, size=sampling.admitted, key=partial(_order, seed=sampling.seed)),
                    partial(merge_sample_rows, size=sampling.admitted, key=partial(_order, seed=sampling.seed)),
                )
            )
            admitted_count, admitted_rows = execute_phase(
                context, admitted, telemetry=phase, operation="admitted"
            ).results[0]
            excluded = (
                Dataset.from_files(str(StoragePath(views["verify"]) / "*.parquet"))
                .flat_map(partial(_indexed_rows, cohort=Cohort.EXCLUDED))
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
                Dataset.from_list(
                    [SelectedFile(view, path, indices) for (view, path), indices in files.items()]
                ).flat_map(partial(_selected_evidence, sampling=sampling)),
                telemetry=phase,
            ).results
            records: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in projected:
                records[row["task_id"]].append(row)
        source_rows = 0
        if raw_source is not None and selected:
            with telemetry.phase("source") as phase:
                assert download is not None
                recovered = execute_phase(
                    context,
                    Dataset.from_list(_original_files(records, download)).flat_map(
                        partial(
                            original_rows,
                            spec=raw_source,
                            context=ConversionContext(staged_inputs(download["inputs"]), None),
                            preview_characters=AUDIT_PREVIEW_CHARACTERS,
                        )
                    ),
                    telemetry=phase,
                ).results
                if {row["task_id"] for row in recovered} != task_ids or len(recovered) != len(selected):
                    raise ValueError("Original source evidence does not cover the selected sample")
                for row in recovered:
                    records[row["task_id"]].append(row)
                source_rows = sum(row["evidence"]["source"]["available"] for row in recovered)
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
                "source_evidence": {
                    "reader_declared": raw_source is not None,
                    "rows": source_rows,
                    "unavailable_rows": len(selected) - source_rows if raw_source is not None else 0,
                },
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
    """Build a source report from prepared evidence and validated judgments."""
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
        "source_evidence": inputs["source_evidence"],
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
