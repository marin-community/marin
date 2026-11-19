# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import NamedTuple

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finestore.rl.rl_progress import Comparison, ComparisonRole, Invariant
from finestore.rl.run_records import AttemptRecord, EvaluationAnswerRow, EvaluationTaskRow, read_evaluations
from finestore.rl.strata import MembershipState, SelectionStrata
from marin.rl.skyrl import read_skyrl_attempt
from marin.rl.skyrl_records import (
    RecordedLaunch,
    RecordedSource,
    attempt_tracking,
    read_skyrl_metrics,
    read_stored_skyrl_attempt,
    recorded_launch,
)
from pydantic import BaseModel
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grading_result import Outcome

from experiments.post_training import rl_presets as presets
from experiments.post_training import rl_recipe as recipe
from experiments.post_training.task_curation.selection import TASK_SPEC_COLUMN

SLICE_COLUMNS = ("family", "aai_category", "source", "topic", "difficulty", "agentic")
AAII_WEIGHTS = {"scientific_reasoning": 0.2, "coding": 0.2, "general": 0.3, "agents": 0.3}

EVAL_SCORE_SCHEMA = pa.schema(
    [
        ("evaluation", pa.string()),
        ("step", pa.int32()),
        ("item", pa.string()),
        ("attempt", pa.int32()),
        *((column, pa.bool_() if column == SLICE_COLUMNS[-1] else pa.string()) for column in SLICE_COLUMNS),
        ("score", pa.float64()),
        ("scored", pa.bool_()),
        ("response_token_count", pa.int32()),
    ]
)


class InvalidComparison(ValueError):
    """The recorded runs cannot support this calibrated comparison."""


def read_recorded_launch(receipt: AttemptRecord) -> RecordedLaunch:
    try:
        return recorded_launch(receipt)
    except ValueError as error:
        raise InvalidComparison(str(error)) from None


@dataclass(frozen=True)
class EvalLiteVersionScore:
    score: float | None
    scored: int
    missing: tuple[str, ...]
    digest_mismatches: tuple[str, ...]


def _verified_score(answer: EvaluationAnswerRow) -> bool:
    return answer.verification_status == "verified" and answer.verifier_score is not None


def _scored_answer(answer: EvaluationAnswerRow) -> bool:
    return _verified_score(answer) and not answer.error_type and not answer.error_treatment


def score_eval_lite_version(
    rows: Sequence[EvaluationTaskRow], panel: SelectionStrata, *, expected_answers: int | None = None
) -> EvalLiteVersionScore:
    """Score one attempt, evaluation and step against an EvalLite version's active tasks.

    Every task needs a matching digest and verified answers. Each task contributes
    the mean of its normalized answer scores, with equal weight across tasks.
    """
    if len({(row.attempt_id, row.evaluation, row.step) for row in rows}) > 1:
        raise ValueError("select one evaluation, attempt and step before scoring an EvalLite version")
    measured = {row.task_id: row for row in rows}
    if len(measured) != len(rows):
        raise ValueError("duplicate per-task evaluation rows")

    scores, missing, mismatches = [], [], []
    for task_id, expected in sorted(panel.tasks.items()):
        if expected.state != MembershipState.ACTIVE:
            continue
        row = measured.get(task_id)
        if row is None or (expected_answers is not None and len(row.answers) != expected_answers):
            missing.append(task_id)
        elif row.task_digest is None or row.task_digest != expected.task_digest:
            mismatches.append(task_id)
        else:
            verified = [
                answer.verifier_score
                for answer in row.answers
                if _scored_answer(answer) and answer.verifier_score is not None and 0 <= answer.verifier_score <= 1
            ]
            if verified and len(verified) == len(row.answers):
                scores.append(statistics.mean(verified))
            else:
                missing.append(task_id)

    score = statistics.mean(scores) if scores and not (missing or mismatches) else None
    return EvalLiteVersionScore(score, len(scores), tuple(missing), tuple(mismatches))


def _selected_evaluation_record(
    run_output: str, launch: RecordedLaunch | None, receipt: AttemptRecord | None
) -> tuple[AttemptRecord, RecordedLaunch]:
    record = receipt or read_stored_skyrl_attempt(run_output)
    if record is None:
        raise InvalidComparison(f"evaluation attempt receipt is unavailable: {run_output}")
    if record.evaluation_storage_version != 1:
        raise InvalidComparison("per-task evaluation records require storage version 1")
    selected = read_recorded_launch(record)
    if (record.run_id, record.attempt_id) != (selected.run_id, selected.attempt_id):
        raise InvalidComparison("evaluation receipt identity differs from its configuration")
    if launch is not None and (launch.run_id, launch.attempt_id, launch.export_root) != (
        selected.run_id,
        selected.attempt_id,
        selected.export_root,
    ):
        raise InvalidComparison("evaluation launch differs from the selected receipt")
    return record, launch or selected


def recorded_sources(launch: RecordedLaunch, name: str) -> list[RecordedSource]:
    return launch.evaluation_data.get(name, launch.validation_data)


def native_prefix(launch: RecordedLaunch, name: str) -> str:
    return f"eval/{name}/" if name in launch.evaluation_data else "eval/"


class RecordedTaskSet(NamedTuple):
    task_ids: tuple[str, ...]
    task_set_sha256: str


def _task_specs(path: StoragePath) -> Iterator[tuple[str, str]]:
    fs, key = filesystem_for(str(path))
    with fs.open(key, "rb") as stream:
        for raw in pq.read_table(stream, columns=[TASK_SPEC_COLUMN])[TASK_SPEC_COLUMN].to_pylist():
            yield json.loads(raw)["id"], raw


def read_recorded_task_set(launch: RecordedLaunch, name: str) -> RecordedTaskSet:
    """Read evaluation task IDs and a hash of the sorted raw TaskSpec strings."""
    pairs = [
        pair
        for source in recorded_sources(launch, name)
        for pair in _task_specs(StoragePath(source.uri) / source.relative_path)
    ]
    payload = json.dumps(sorted(pairs), separators=(",", ":"), ensure_ascii=False).encode()
    return RecordedTaskSet(tuple(task_id for task_id, _ in pairs), hashlib.sha256(payload).hexdigest())


def evaluation_protocol(launch: RecordedLaunch, name: str) -> dict:
    """The evaluation request recorded by native configuration composition."""
    sampling = deepcopy(launch.eval_sampling_params)
    samples = launch.eval_n_samples_per_prompt
    length = launch.eval_max_generate_length
    if name in launch.evaluation_data:
        profile = next(
            (
                profile
                for callback in launch.callbacks or ()
                if callback.type == "evaluation"
                for key, profile in (callback.additional_evaluations or {}).items()
                if key == name
            ),
            None,
        )
        if profile is None:
            raise InvalidComparison(f"expected one recorded evaluation profile for {name}")
        sampling = deepcopy(profile.sampling_params)
        samples = profile.n_samples_per_prompt
        length = profile.max_generate_length
    return {
        "sampling_params": sampling,
        "n_samples_per_prompt": samples,
        "context_budget": {
            "max_new_tokens_per_turn": length,
            "max_turns": launch.max_turns,
        },
        "max_prompt_length": launch.max_prompt_length,
        "chat_template": launch.chat_template,
        "task_env": {key: value for key, value in launch.task_env.items() if not key.startswith("VLLM_")},
    }


def evaluation_scores(
    evaluation: presets.Evaluation,
    run_output: str,
    *,
    launch: RecordedLaunch | None = None,
    receipt: AttemptRecord | None = None,
) -> pa.Table:
    """Read the verified answer evidence for one recorded evaluation."""
    record, launch = _selected_evaluation_record(run_output, launch, receipt)
    tasks = read_evaluations(
        launch.export_root, attempt_id=record.attempt_id, evaluation=native_prefix(launch, evaluation.name)
    )
    if not tasks:
        raise InvalidComparison(f"per-task evaluation records are missing for attempt {record.attempt_id}")
    rows = []
    attempts = Counter()
    for task, answer in sorted(
        ((row, value) for row in tasks for value in row.answers), key=lambda pair: (pair[0].step, pair[1].ordinal)
    ):
        verified = _verified_score(answer)
        rows.append(
            {
                "evaluation": evaluation.name,
                "step": task.step,
                "item": task.task_id,
                "attempt": attempts[task.step, task.task_id],
                **task.strata.model_dump(),
                "score": answer.verifier_score if verified else 0.0,
                "scored": _scored_answer(answer),
                "response_token_count": answer.response_token_count,
            }
        )
        attempts[task.step, task.task_id] += 1
    return pa.Table.from_pylist(rows, schema=EVAL_SCORE_SCHEMA)


def score_slices(scores: pa.Table) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in scores.to_pylist():
        if not row["scored"]:
            continue
        evaluation = row["evaluation"]
        keys = [f"{evaluation}/mean"]
        for column in SLICE_COLUMNS:
            value = row[column]
            if value is not None:
                value = str(value).lower() if column == SLICE_COLUMNS[-1] else value
                key = f"{evaluation}/{column}/{value}"
                keys.append(key + "/mean")
        for key in keys:
            groups[key].append(row)
    for evaluation in {key.split("/", 1)[0] for key in groups}:
        agentic = [key for key in groups if key.startswith(f"{evaluation}/{SLICE_COLUMNS[-1]}/")]
        if len(agentic) == 1:
            # A task set without both kinds of task has no agentic split; its one slice repeats the overall mean.
            del groups[agentic[0]]
    return dict(groups)


def evaluation_statistics(scores: pa.Table) -> dict[str, float]:
    sessions = defaultdict(list)
    for row in scores.to_pylist():
        sessions[row["evaluation"]].append(row)
    result = {}
    for key, rows in score_slices(scores).items():
        tasks = defaultdict(list)
        for row in rows:
            tasks[row["item"]].append(row["score"])
        result[key] = (
            math.fsum(row["score"] for row in rows) / len(rows)
            if key.count("/") == 1
            else math.fsum(math.fsum(values) / len(values) for values in tasks.values()) / len(tasks)
        )
    for evaluation, rows in sessions.items():
        if len({row["step"] for row in rows}) != 1:
            raise ValueError(f"select one weight step before aggregating {evaluation}")
        scored = [row for row in rows if row["scored"]]
        result[f"{evaluation}/attempted"] = len(rows)
        result[f"{evaluation}/scored"] = len(scored)
        result[f"{evaluation}/scored_fraction"] = len(scored) / len(rows)
        represented = {
            category: weight
            for category, weight in AAII_WEIGHTS.items()
            if any(row["aai_category"] == category for row in rows)
        }
        weight_sum = math.fsum(represented.values())
        result[f"{evaluation}/aai_category_weight_coverage"] = weight_sum
        if weight_sum:
            result.update(
                {
                    f"{evaluation}/aai_category/{category}/weight": weight / weight_sum
                    for category, weight in represented.items()
                }
            )
        if not scored:
            continue
        tasks = defaultdict(list)
        for row in scored:
            tasks[row["item"]].append(row)
        counts = [(sum(row["score"] == 1 for row in group), len(group)) for group in tasks.values()]
        for name, accepts in (
            ("mixed", lambda right, n: 0 < right < n),
            ("all_right", lambda right, n: right == n),
            ("all_wrong", lambda right, n: right == 0),
        ):
            if name == "all_right" or any(n > 1 for n in Counter(row["item"] for row in rows).values()):
                result[f"{evaluation}/{name}_fraction"] = sum(accepts(right, n) for right, n in counts) / len(counts)
        families = [
            value
            for key, value in result.items()
            if key.startswith(f"{evaluation}/{SLICE_COLUMNS[0]}/") and key.count("/") == 3
        ]
        if families:
            result[f"{evaluation}/family_mean"] = math.fsum(families) / len(families)
        if weight_sum and all(f"{evaluation}/aai_category/{category}/mean" in result for category in represented):
            result[f"{evaluation}/aai_weighted_mean"] = (
                math.fsum(
                    weight * result[f"{evaluation}/aai_category/{category}/mean"]
                    for category, weight in represented.items()
                )
                / weight_sum
            )
    return result


GATE_INPUTS_FILENAME = "gate_inputs.json"
GRADER_ERROR_LIMIT = 0.01


class GateInputs(BaseModel, frozen=True):
    preset_name: str
    settings: tuple[str, ...] = ()
    tasks: str | None = None
    init: presets.StoredInitializer | None = None
    requirements: tuple[str, ...] = ()


def read_gate_inputs(run_output: str) -> GateInputs:
    """Read the typed RL inputs saved beside a native launch receipt."""
    return GateInputs.model_validate_json((StoragePath(run_output) / GATE_INPUTS_FILENAME).read_text())


@dataclass(frozen=True)
class Requirement:
    key: str
    higher_is_better: bool
    bound: float | None


def parse_requirement(text: str) -> Requirement:
    match = re.fullmatch(r"([^<>=\s]+)(<=|>=)(control|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if match is None:
        raise click.BadParameter(f"expected KEY<=NUMBER, KEY>=NUMBER, KEY<=control or KEY>=control: {text}")
    key, operator, value = match.groups()
    bound = None if value == "control" else float(value)
    if bound is not None and not math.isfinite(bound):
        raise click.BadParameter(f"requirement bound must be finite: {text}")
    return Requirement(key, operator == ">=", bound)


def _mismatch_counts(rows: Sequence[Mapping[str, float]], prefix: str) -> list[float] | None:
    counts = [row.get(prefix + "/selected_tokens", math.nan) for row in rows]
    if all(math.isfinite(count) and count >= 0 for count in counts) and math.fsum(counts) > 0:
        return counts
    return None


def _window_values(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    rows = [metrics.get(step, {}) for step in preset.measured]
    phases = sorted(
        key
        for key in set().union(*(row.keys() for row in rows))
        if key.startswith("timing/step_wall/") and key.rsplit("/", 1)[-1] not in ("evaluation", "checkpoint_work")
    )
    values = {}
    declared = {metric.key: metric.window for metric in (*preset.guards, *preset.objectives)}
    extra = (
        set().union(*(row.keys() for row in metrics.values()))
        - declared.keys()
        - {"step", "attempt_id", "metric_attempt_ids"}
    )
    for key in (*declared, *sorted(extra)):
        window = declared.get(
            key, presets.WindowStatistic.FIRST if key.startswith("startup/") else presets.WindowStatistic.MEAN
        )
        if window == presets.WindowStatistic.FIRST:
            first = next((row[key] for _, row in sorted(metrics.items()) if key in row), None)
            if first is not None:
                values[key] = float(first)
            continue
        if window == presets.WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN:
            prefix = "/".join(key.split("/")[:3])
            counts = _mismatch_counts(rows, prefix)
            if counts is None:
                continue
            total = math.fsum(counts)
            nonempty = [(row, count) for row, count in zip(rows, counts, strict=True) if count > 0]
            if all(
                math.isfinite(row.get(key, math.nan)) and row.get(prefix + "/finite_fraction") == 1
                for row, _ in nonempty
            ):
                values[key] = math.fsum(count / total * row[key] for row, count in nonempty)
            continue
        if window == presets.WindowStatistic.PHASE_SUM_MEDIAN:
            series = [math.fsum(row[key] for key in phases) for row in rows if phases and set(phases) <= set(row)]
        else:
            series = [row[key] for row in rows if key in row]
        if series and len(series) == len(rows):
            reducer = statistics.median if window == presets.WindowStatistic.PHASE_SUM_MEDIAN else statistics.mean
            values[key] = float(reducer(series))
    return values


def _grader_errors(
    rows: Sequence[Mapping[str, float]], prefix: str, name: str, *, answers: int | None = None
) -> Invariant:
    recorded = all(prefix + outcome in row for row in rows for outcome in (*Outcome, "rollout_failure"))

    def count(*outcomes: str) -> float:
        return math.fsum(row.get(prefix + outcome, 0) for row in rows for outcome in outcomes)

    total = count(*Outcome)
    bad = count(Outcome.INFRA_ERROR, Outcome.UNAVAILABLE, Outcome.INVALID_TASK)
    failures = count("rollout_failure")
    upper = min(total, bad + failures) / total if total else 1.0
    if answers is not None and (not recorded or total != answers or failures > total):
        upper = 1.0
    return Invariant(
        name=name,
        value=upper,
        limit=GRADER_ERROR_LIMIT,
        comparator="<=",
        holds=upper <= GRADER_ERROR_LIMIT and recorded,
    )


def _session_invariants(
    name: str,
    table: pa.Table,
    evaluation: presets.Evaluation,
    preset: presets.Preset,
    expected: int,
    task_ids: Sequence[str],
) -> list[Invariant]:
    rows = table.to_pylist()
    counts = Counter(row["item"] for row in rows)
    holds = len(rows) == expected and set(counts.values()) == {evaluation.answers} and set(counts) == set(task_ids)
    result = [Invariant(name=f"{name}/batch_finished", value=len(rows), limit=expected, comparator="==", holds=holds)]
    if evaluation == preset.guarded or (preset.checkpoint_every and evaluation.held_out):
        coverage = len({row["item"] for row in rows if row["scored"]}) / (expected // evaluation.answers)
        result.append(
            Invariant(
                name=f"{name}/scored_fraction",
                value=coverage,
                limit=presets.MINIMUM_SCORED_FRACTION,
                comparator=">=",
                holds=coverage >= presets.MINIMUM_SCORED_FRACTION,
            )
        )
    return result


@dataclass(frozen=True)
class RunData:
    config: dict
    recorded: RecordedLaunch
    path: str
    eval_lite: str | None
    provenance: dict
    values: dict[str, float]
    invariants: tuple[Invariant, ...]
    evaluations: tuple[Comparison, ...]
    slices: dict[str, int]
    evaluation_identity: dict[str, tuple[str | None, dict]]
    gpu_hours: float | None
    engine: dict
    attempt_id: str


class EvaluationStatus(BaseModel):
    evaluations: tuple[Comparison, ...] = ()
    evaluation_invariants: tuple[Invariant, ...] = ()
    evaluation_error: str | None = None


def evaluation_status(
    path: str, preset: presets.Preset, receipt: AttemptRecord, launch: RecordedLaunch
) -> EvaluationStatus:
    """Read successful-attempt evaluation summaries and invariants for status and publication."""
    if not preset.evaluations or receipt.native.result.state != "succeeded":
        return EvaluationStatus()
    try:
        effective = preset.with_updates(launch.max_steps if launch.max_steps is not None else preset.updates)
        try:
            inputs = read_gate_inputs(path)
        except FileNotFoundError:
            inputs = None
        data = load_run(path, effective, inputs, receipt=receipt)
        return EvaluationStatus(
            evaluations=data.evaluations,
            evaluation_invariants=tuple(
                fact
                for fact in data.invariants
                if any(fact.name.startswith(evaluation.name + "/") for evaluation in preset.evaluations)
            ),
        )
    except (InvalidComparison, FileNotFoundError) as error:
        return EvaluationStatus(
            evaluation_error=str(error),
            evaluation_invariants=(
                Invariant(name="evaluation_records_available", value=False, limit=True, comparator="==", holds=False),
            ),
        )


def training_invariants(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> tuple[Invariant, ...]:
    """Check planned update completion and finite training loss and gradient evidence."""
    return preset.recipe.layout.workload.invariants(preset.updates, metrics)


def load_run(
    path: str,
    preset: presets.Preset,
    inputs: GateInputs | None = None,
    *,
    receipt: AttemptRecord | None = None,
) -> RunData:
    inputs = inputs or GateInputs(preset_name=preset.name)
    terminal = receipt or read_skyrl_attempt(path)
    if terminal is None:
        raise FileNotFoundError(f"no launch receipt at {path}")
    if terminal.native.result.state != "succeeded":
        raise InvalidComparison(f"native job {terminal.native.result.iris_job_id} is {terminal.native.result.state}")
    if terminal.native.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {path}")
    metrics = read_skyrl_metrics(path, attempt_id=terminal.attempt_id)
    workload = preset.recipe.layout.workload
    planned = workload.planned_steps(preset.updates)
    training = [row for step, row in sorted(metrics.items()) if step in planned]
    invariants = list(training_invariants(preset, metrics))
    launch = read_recorded_launch(terminal)
    tracking = attempt_tracking(terminal, launch)
    try:
        recorded_task_sets = {
            evaluation.name: read_recorded_task_set(launch, evaluation.name) for evaluation in preset.evaluations
        }
        identities = {
            evaluation.name: (
                (recorded_task_sets[evaluation.name].task_set_sha256 if evaluation.held_out else None),
                evaluation_protocol(launch, evaluation.name),
            )
            for evaluation in preset.evaluations
        }
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidComparison(f"recorded evaluation identity is malformed: {error}") from error
    default, selection = recipe.task_selections(
        preset, recipe.selection_artifact(inputs.tasks) if inputs.tasks else None
    )
    task_set = preset.held_out_tasks if preset.held_out is not None else None
    scores, initial_scores = [EVAL_SCORE_SCHEMA.empty_table()], []
    evaluations = []
    response_token_counts: set[int | None] = set()
    primary_answers = 0
    for index, evaluation in enumerate(preset.evaluations):
        quality = evaluation.held_out and (evaluation == preset.guarded or bool(preset.checkpoint_every))
        expected = recipe.evaluation_answer_count(
            evaluation, recipe.evaluation_source(evaluation, selection, default, preset.held_out_tasks)
        )
        all_scores = evaluation_scores(evaluation, path, launch=launch, receipt=terminal)
        if index == 0:
            primary_answers = all_scores.num_rows
            response_token_counts.update(all_scores["response_token_count"].to_pylist())
        table = all_scores
        initial = table.filter(pc.equal(table["step"], 0))
        table = table.filter(pc.equal(table["step"], workload.scored_step(preset.updates)))
        scores.append(table)
        initial_scores.append(initial)
        ids = frozenset(table["item"].to_pylist())
        sessions = [(evaluation.name, table, workload.metric_step(preset.updates))]
        if evaluation.at_start:
            if initial.num_rows and frozenset(initial["item"].to_pylist()) != ids:
                raise InvalidComparison(f"initial and final evaluation task IDs differ: {path}")
            sessions.append((evaluation.name + "/initial", initial, 0))
        if preset.checkpoint_every:
            scheduled = set(range(evaluation.every, preset.updates, evaluation.every)) if evaluation.every else set()
            sessions.extend(
                (f"{evaluation.name}/step-{step}", all_scores.filter(pc.equal(all_scores["step"], step)), step)
                for step in sorted(
                    (set(all_scores["step"].to_pylist()) | scheduled) - {0, workload.scored_step(preset.updates)}
                )
            )
        prefix = native_prefix(launch, evaluation.name) + "all/outcome/"
        for session_name, session, metric_step in sessions:
            statistics = evaluation_statistics(session)
            slices = score_slices(session)
            for key, value in statistics.items():
                evaluations.append(
                    Comparison(
                        metric=f"{evaluation.name}@{metric_step}/{key.removeprefix(evaluation.name + '/')}",
                        role=ComparisonRole.SLICE,
                        run=value,
                        n_run=1,
                        n_control=0,
                        tasks=len({row["item"] for row in slices[key]}) if key in slices else None,
                    )
                )
            invariants.extend(
                _session_invariants(
                    session_name, session, evaluation, preset, expected, recorded_task_sets[evaluation.name].task_ids
                )
            )
            if quality:
                invariants.append(
                    _grader_errors(
                        [metrics.get(metric_step, {})],
                        prefix,
                        session_name + "/grader_error_fraction",
                        answers=session.num_rows,
                    )
                )
    prefix = workload.outcome_prefix
    invariants.append(_grader_errors(training, prefix, "grader_error_fraction"))
    fixed = preset.recipe.fixed_reply_tokens
    if fixed is not None:
        if response_token_counts == {None} and isinstance(workload, presets.Generation):
            lengths = metrics.get(0, {})
            if (
                lengths.get("eval/all/response_tokens_max") == fixed
                and lengths.get("eval/all/response_tokens") == fixed * primary_answers
            ):
                response_token_counts = {fixed}
        holds = workload.fixed_reply_holds(fixed, training, response_token_counts)
        invariants.append(Invariant(name="fixed_reply_tokens", value=holds, limit=True, comparator="==", holds=holds))
    final = pa.concat_tables(scores)
    values = {
        **_window_values(preset, metrics),
        **evaluation_statistics(final),
    }
    for key, value in evaluation_statistics(pa.concat_tables([scores[0], *initial_scores])).items():
        values[key + "/initial"] = value
        if key in values:
            values[key + "/gain"] = values[key] - value
    result = terminal.native.result
    gpu_hours = None
    if result.started_at and result.finished_at:
        seconds = (recipe.utc_millis(result.finished_at) - recipe.utc_millis(result.started_at)) / 1000
        if seconds > 0:
            gpu_hours = seconds * launch.num_nodes * launch.gpus_per_node / 3600
    slices = {key: len({row["item"] for row in rows}) for key, rows in score_slices(final).items()}
    return RunData(
        config=terminal.native.config,
        recorded=launch,
        path=path,
        eval_lite=task_set,
        provenance=terminal.native.result.provenance,
        values=values,
        invariants=tuple(invariants),
        evaluations=tuple(evaluations),
        slices=slices,
        evaluation_identity=identities,
        gpu_hours=gpu_hours,
        engine=tracking.get("engine", {}),
        attempt_id=terminal.attempt_id,
    )


def with_objectives(candidate: RunData, preset: presets.Preset) -> RunData:
    values = dict(candidate.values)
    for metric in preset.objectives:
        if metric.window == presets.WindowStatistic.GAIN_PER_GPU_HOUR:
            gain = values.get(re.sub(r"_per_[^/]+_hour$", "", metric.key))
            if gain is not None and candidate.gpu_hours:
                values[metric.key] = gain / candidate.gpu_hours
    return replace(candidate, values=values)


def with_requirements(candidate: RunData, requirements: Sequence[str]) -> RunData:
    invariants = list(candidate.invariants)
    values = candidate.values
    for text in requirements:
        requirement = parse_requirement(text)
        value = values.get(requirement.key)
        if value is None or not math.isfinite(value):
            raise InvalidComparison(f"missing or non-finite requirement metric: {requirement.key}")
        if requirement.bound is not None:
            holds = value >= requirement.bound if requirement.higher_is_better else value <= requirement.bound
            invariants.append(
                Invariant(
                    name=f"require/{requirement.key}",
                    value=value,
                    limit=requirement.bound,
                    comparator=">=" if requirement.higher_is_better else "<=",
                    holds=holds,
                )
            )
    return replace(candidate, invariants=tuple(invariants))
