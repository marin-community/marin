# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare bounded samples from TaskSpecs and recorded curation stages."""

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finestore.rl.curation import Cohort, ModelReference, Verdict
from finestore.schema import arrow_schema
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
from taskcompendium.pipeline.models import Admission
from taskcompendium.pipeline.review import private_evidence_summary, review_payload
from taskcompendium.pipeline.sampling import merge_sample_rows, seeded_order, seeded_sample
from taskcompendium.submission import submission_compatibility, submission_instruction

PACKET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("cohort", pa.string()),
        ("packet_json", pa.string()),
        ("omissions", pa.list_(pa.string())),
        ("exclusion_reason", pa.string()),
    ]
)
AUDIT_PREVIEW_CHARACTERS = 2048
AUDIT_FIELD_OWNERS = {"task_json": "normalize", "normalization_changes": "normalize"}
PARQUET_READ_BLOCK_BYTES = 4 * 1024 * 1024
EVIDENCE_VIEWS = ("normalize", "review", "verify")
INPUTS_METADATA = "inputs.parquet"
JUDGMENTS_METADATA = "judgments.parquet"


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


class SelectedTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    task_id: str
    cohort: Cohort


class SamplePopulation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    cohort: Cohort
    population: int
    sample_count: int


class AssessmentInputs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: str
    scope: str
    sampling: AssessmentSampling
    populations: list[SamplePopulation]
    sample: list[SelectedTask]
    packet_paths: list[str]


class AssessmentJudgments(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    grade_paths: list[str]
    model: ModelReference
    rubric: str


def assessment_inputs(root: str) -> AssessmentInputs:
    rows = list(load_parquet(str(StoragePath(root) / INPUTS_METADATA)))
    (row,) = rows
    return AssessmentInputs.model_validate(row)


def assessment_judgments(root: str) -> AssessmentJudgments:
    rows = list(load_parquet(str(StoragePath(root) / JUDGMENTS_METADATA)))
    (row,) = rows
    return AssessmentJudgments.model_validate(row)


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


def _evidence_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _evidence_preview(value: Any, *, text_characters: int, entries: int) -> Any:
    if isinstance(value, str):
        if len(value) <= text_characters:
            return value
        prefix = text_characters // 2
        summary = private_evidence_summary(value, "Full value retained in curation audit", prefix)
        summary["suffix"] = value[-(text_characters - prefix) :]
        return summary
    if isinstance(value, list):
        if len(value) <= entries:
            return [_evidence_preview(item, text_characters=text_characters, entries=entries) for item in value]
        indices = [index * (len(value) - 1) // (entries - 1) for index in range(entries)]
        return {
            "total_count": len(value),
            "sampled_indices": indices,
            "items": [
                _evidence_preview(value[index], text_characters=text_characters, entries=entries) for index in indices
            ],
        }
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            key: _evidence_preview(item, text_characters=text_characters, entries=entries)
            for key, item in items[:entries]
        }
        if len(items) > entries:
            return {"total_fields": len(items), "fields": result, "truncated": True}
        return result
    return value


def _bounded_evidence(value: dict[str, Any], budget: int) -> dict[str, Any]:
    if len(_evidence_json(value)) <= budget:
        return value
    # Shrink from the original on each pass so nested previews never become evidence.
    for text_characters, entries in ((4096, 64), (2048, 32), (1024, 16), (512, 8), (256, 4), (128, 2)):
        preview = {
            key: _evidence_preview(item, text_characters=text_characters, entries=entries) for key, item in value.items()
        }
        if len(_evidence_json(preview)) <= budget:
            return preview
    raise ValueError(f"Evidence structure cannot fit the {budget}-character section budget")


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
    with open_file(path, block_size=PARQUET_READ_BLOCK_BYTES) as stream:
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
    return {
        "task_id": row["task_id"],
        "evidence": evidence,
        "omissions": omissions,
    }


def _selected_evidence(file: SelectedFile, *, sampling: AssessmentSampling) -> Iterator[dict[str, Any]]:
    """Yield bounded evidence for selected records from a Parquet file."""
    with open_file(file.path, block_size=PARQUET_READ_BLOCK_BYTES) as stream:
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
    for view in EVIDENCE_VIEWS:
        if view not in evidence:
            omissions.append(f"{view}: no row evidence")
    if len(_evidence_json(evidence)) > sampling.max_packet_characters:
        # Reserve most of the packet for the task; audits must not crowd it out.
        available = sampling.max_packet_characters - 256
        budgets = {"task": available * 3 // 4}
        other_sections = [section for section in evidence if section != "task"]
        audit_budget = (available - budgets["task"]) // max(len(other_sections), 1)
        budgets.update({section: audit_budget for section in other_sections})
        for section, value in evidence.items():
            bounded = _bounded_evidence(value, budgets[section])
            if bounded != value:
                omissions.append(f"{section}: bounded preview; full evidence retained in curation audit")
            evidence[section] = bounded
    packet_json = _evidence_json(evidence)
    if len(packet_json) > sampling.max_packet_characters:
        raise ValueError(f"Assessment packet exceeds its budget: {task_id}")
    return {
        "task_id": task_id,
        "cohort": selected["cohort"],
        "packet_json": packet_json,
        "omissions": omissions,
        "exclusion_reason": selected["reason"],
    }


def prepare_assessment(
    context: ZephyrContext,
    *,
    curated_root: str,
    sampling: AssessmentSampling,
    output_root: str,
) -> None:
    """Select admitted and excluded rows, then materialize bounded evidence on workers."""
    if sampling.admitted < 1 or sampling.excluded < 0 or sampling.max_packet_characters < 1:
        raise ValueError("Assessment sampling and packet budgets must be positive")
    manifest = json.loads((StoragePath(curated_root) / "manifest.json").read_text())
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
            for view in EVIDENCE_VIEWS:
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
        with telemetry.phase("packets"):
            packet_path = str(StoragePath(output_root) / "packets/part-00000.parquet")
            write_parquet_file(
                (_packet(row, records=records, sampling=sampling) for row in selected), packet_path, schema=PACKET_SCHEMA
            )
            paths = [packet_path]
        populations = {Cohort.ADMITTED: admitted_count, Cohort.EXCLUDED: sum(count for count, _ in groups.values())}
        metadata = AssessmentInputs(
            source=manifest["source"],
            scope=manifest["mode"],
            sampling=sampling,
            populations=[
                SamplePopulation(
                    cohort=cohort,
                    population=populations[cohort],
                    sample_count=sum(row["cohort"] == cohort for row in selected),
                )
                for cohort in Cohort
            ],
            sample=[SelectedTask(task_id=row["task_id"], cohort=row["cohort"]) for row in selected],
            packet_paths=paths,
        )
        write_parquet_file(
            [metadata.model_dump(mode="json")],
            str(StoragePath(output_root) / INPUTS_METADATA),
            schema=arrow_schema(AssessmentInputs),
        )


def validate_coverage(selected: Iterable[SelectedTask], rows: list[dict[str, Any]]) -> None:
    expected = {(row.task_id, row.cohort) for row in selected}
    actual = {(row["task_id"], row["cohort"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Grade records do not match the selected assessment sample")
