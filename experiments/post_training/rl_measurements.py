# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read metrics, evaluation scores and invariants from stored RL runs."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finestore.rl.rl_progress import Comparison, ComparisonRole, Invariant
from finestore.rl.run_records import AttemptRecord, read_evaluations
from finestore.rl.strata import TaskStrata
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
from taskcompendium.identity import canonical_sha256

from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_scoring import ANSWER_SCHEMA, SLICE_COLUMNS, answer_table, score_answers, score_slices
from experiments.post_training.task_curation.selection import TASK_SPEC_COLUMN


class InvalidComparison(ValueError):
    """The recorded runs cannot support this calibrated comparison."""


def read_recorded_launch(attempt: AttemptRecord) -> RecordedLaunch:
    try:
        return recorded_launch(attempt)
    except ValueError as error:
        raise InvalidComparison(str(error)) from None


def _selected_evaluation_record(
    run_output: str, launch: RecordedLaunch | None, attempt: AttemptRecord | None
) -> tuple[AttemptRecord, RecordedLaunch]:
    record = attempt or read_stored_skyrl_attempt(run_output)
    if record is None:
        raise InvalidComparison(f"evaluation attempt record is unavailable: {run_output}")
    if record.evaluation_storage_version != 1:
        raise InvalidComparison("per-task evaluation records require storage version 1")
    selected = read_recorded_launch(record)
    if (record.run_id, record.attempt_id) != (selected.run_id, selected.attempt_id):
        raise InvalidComparison("evaluation attempt record identity differs from its configuration")
    if launch is not None and (launch.run_id, launch.attempt_id, launch.export_root) != (
        selected.run_id,
        selected.attempt_id,
        selected.export_root,
    ):
        raise InvalidComparison("evaluation launch differs from the selected attempt record")
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


def _source_task_specs(sources: Sequence[RecordedSource]) -> list[tuple[str, str]]:
    return [pair for source in sources for pair in _task_specs(StoragePath(source.uri) / source.relative_path)]


def read_task_set(sources: Sequence[RecordedSource]) -> RecordedTaskSet:
    """Read task IDs and a hash of the sorted raw TaskSpec strings."""
    return _task_set(_source_task_specs(sources))


def _task_set(pairs: Sequence[tuple[str, str]]) -> RecordedTaskSet:
    payload = json.dumps(sorted(pairs), separators=(",", ":"), ensure_ascii=False).encode()
    return RecordedTaskSet(tuple(task_id for task_id, _ in pairs), hashlib.sha256(payload).hexdigest())


def read_recorded_task_set(launch: RecordedLaunch, name: str) -> RecordedTaskSet:
    """Read the task set bound to a recorded evaluation."""
    return read_task_set(recorded_sources(launch, name))


def held_out_identity(preset: presets.Preset, launch: RecordedLaunch) -> str | None:
    """Return the recorded selection's NAME@VERSION, or None for a preset without held-out evaluation."""
    evaluation = preset.held_out
    if evaluation is None:
        return None
    identities = {source.identity for source in recorded_sources(launch, evaluation.name)}
    if len(identities) != 1:
        raise InvalidComparison(f"expected one recorded selection identity for {evaluation.name}")
    identity = identities.pop().removeprefix(recipe.SELECTION_PREFIX + "/")
    return identity.split(":", 1)[0]


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


def read_answers(
    evaluation: presets.Evaluation,
    run_output: str,
    *,
    launch: RecordedLaunch | None = None,
    attempt: AttemptRecord | None = None,
) -> pa.Table:
    """Read the verified answer evidence for one recorded evaluation."""
    record, launch = _selected_evaluation_record(run_output, launch, attempt)
    tasks = read_evaluations(
        launch.export_root, attempt_id=record.attempt_id, evaluation=native_prefix(launch, evaluation.name)
    )
    if not tasks:
        raise InvalidComparison(f"per-task evaluation records are missing for attempt {record.attempt_id}")
    return answer_table(evaluation.name, tasks)


RUN_INPUTS_FILENAME = "run_inputs.json"
GRADER_ERROR_LIMIT = 0.01


class RunOptions(BaseModel, frozen=True):
    checkpoints: presets.CheckpointPolicy
    checkpoint_every: int | None
    export: presets.CheckpointPolicy
    priority: str

    @classmethod
    def from_preset(cls, preset: presets.Preset) -> RunOptions:
        return cls(
            checkpoints=preset.checkpoints,
            checkpoint_every=preset.checkpoint_every,
            export=preset.export,
            priority=preset.recipe.layout.priority,
        )


class RunInputs(BaseModel, frozen=True):
    preset_name: str
    settings: tuple[str, ...] = ()
    tasks: str | None = None
    init: presets.StoredInitializer | None = None
    resume: presets.StoredResume | None = None
    requirements: tuple[str, ...] = ()
    options: RunOptions | None = None
    sft_preset: str | None = None


def run_inputs(
    preset_name: str,
    settings: Sequence[str] = (),
    tasks: str | None = None,
    init: presets.StoredInitializer | None = None,
    requirements: Sequence[str] = (),
    *,
    resume: presets.StoredResume | None = None,
    options: RunOptions | None = None,
    sft_preset: str | None = None,
) -> RunInputs:
    values = recipe.override_document(tuple(settings))
    return RunInputs(
        preset_name=preset_name,
        settings=tuple(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in sorted(values.items())),
        tasks=tasks.removeprefix(recipe.SELECTION_PREFIX + "/") if tasks else None,
        init=init,
        resume=resume,
        requirements=tuple(dict.fromkeys(requirements)),
        options=options,
        sft_preset=sft_preset,
    )


def read_run_inputs(run_output: str) -> RunInputs:
    """Read the typed RL inputs saved beside a native launch receipt."""
    return RunInputs.model_validate_json((StoragePath(run_output) / RUN_INPUTS_FILENAME).read_text())


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
    samples: int,
    task_ids: Sequence[str],
) -> list[Invariant]:
    rows = table.to_pylist()
    counts = Counter(row["task_id"] for row in rows)
    holds = len(rows) == expected and set(counts.values()) == {samples} and set(counts) == set(task_ids)
    result = [Invariant(name=f"{name}/answers_complete", value=len(rows), limit=expected, comparator="==", holds=holds)]
    if _checks_task_coverage(preset, evaluation):
        coverage = len({row["task_id"] for row in rows if row["scored"]}) / (expected // samples)
        result.append(
            Invariant(
                name=f"{name}/task_coverage",
                value=coverage,
                limit=presets.MINIMUM_TASK_COVERAGE,
                comparator=">=",
                holds=coverage >= presets.MINIMUM_TASK_COVERAGE,
            )
        )
    return result


def _checks_task_coverage(preset: presets.Preset, evaluation: presets.Evaluation) -> bool:
    return evaluation == preset.guarded or (
        evaluation.held_out and isinstance(preset.recipe.layout.workload, presets.Training)
    )


@dataclass(frozen=True)
class HeldOutAnswers:
    final: pa.Table
    initial: pa.Table


@dataclass(frozen=True)
class RunMeasurements:
    config: dict
    recorded: RecordedLaunch
    path: str
    held_out_tasks: str | None
    provenance: dict
    values: dict[str, float]
    invariants: tuple[Invariant, ...]
    evaluations: tuple[Comparison, ...]
    slices: dict[str, int]
    evaluation_identity: dict[str, tuple[str | None, dict]]
    held_out_answers: dict[str, HeldOutAnswers]
    started_at: str | None
    gpu_hours: float | None
    engine: dict
    attempt_id: str


def _evaluation_values(final: pa.Table, initial: pa.Table) -> tuple[dict[str, float], dict[str, int]]:
    values = score_answers(final)
    for key, value in score_answers(initial).items():
        values[key + "/initial"] = value
        if key in values:
            values[key + "/gain"] = values[key] - value
    slices = {key: len({row["task_id"] for row in rows}) for key, rows in score_slices(final).items()}
    return values, slices


def restrict_evaluation(
    run: RunMeasurements, name: str, digests: set[str], strata: Mapping[str, TaskStrata]
) -> RunMeasurements:
    """Recompute evaluation metrics on the selected digests with shared strata labels."""
    answers = run.held_out_answers[name]

    selected_digests = pa.array(sorted(digests), type=pa.string())

    def project(table: pa.Table) -> pa.Table:
        selected = table.filter(pc.is_in(table["task_digest"], value_set=selected_digests))
        labels = [strata[digest] for digest in selected["task_digest"].to_pylist()]
        for column in SLICE_COLUMNS:
            position = ANSWER_SCHEMA.get_field_index(column)
            selected = selected.set_column(
                position,
                ANSWER_SCHEMA.field(position),
                pa.array([getattr(label, column) for label in labels], type=ANSWER_SCHEMA.field(position).type),
            )
        return selected

    values, slices = _evaluation_values(project(answers.final), project(answers.initial))
    prefix = name + "/"
    return replace(
        run,
        values={**{key: value for key, value in run.values.items() if not key.startswith(prefix)}, **values},
        slices={**{key: value for key, value in run.slices.items() if not key.startswith(prefix)}, **slices},
    )


def _bind_task_digests(table: pa.Table, task_specs: Sequence[tuple[str, str]]) -> pa.Table:
    expected = {name: canonical_sha256(json.loads(raw)) for name, raw in task_specs}
    digests = []
    for name, recorded in zip(table["task_id"].to_pylist(), table["task_digest"].to_pylist(), strict=True):
        digest = expected.get(name)
        if recorded is not None and recorded != digest:
            raise InvalidComparison(f"recorded held-out TaskSpec content differs for {name}")
        digests.append(digest)
    position = ANSWER_SCHEMA.get_field_index("task_digest")
    return table.set_column(position, ANSWER_SCHEMA.field(position), pa.array(digests, type=pa.string()))


class EvaluationStatus(BaseModel):
    evaluations: tuple[Comparison, ...] = ()
    evaluation_invariants: tuple[Invariant, ...] = ()
    evaluation_error: str | None = None


def evaluation_status(
    path: str, preset: presets.Preset, attempt: AttemptRecord, launch: RecordedLaunch
) -> EvaluationStatus:
    """Read successful-attempt evaluation summaries and invariants for status and publication."""
    if not preset.evaluations or attempt.native.result.state != "succeeded":
        return EvaluationStatus()
    try:
        effective = preset.with_updates(launch.max_steps if launch.max_steps is not None else preset.updates)
        try:
            inputs = read_run_inputs(path)
        except FileNotFoundError:
            inputs = None
        data = read_measurements(path, effective, inputs, attempt=attempt)
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


def read_measurements(
    path: str,
    preset: presets.Preset,
    inputs: RunInputs | None = None,
    *,
    attempt: AttemptRecord | None = None,
) -> RunMeasurements:
    inputs = inputs or RunInputs(preset_name=preset.name)
    terminal = attempt or read_skyrl_attempt(path)
    if terminal is None:
        raise FileNotFoundError(f"no terminal attempt record at {path}")
    if terminal.native.result.state != "succeeded":
        raise InvalidComparison(f"native job {terminal.native.result.iris_job_id} is {terminal.native.result.state}")
    if terminal.native.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {path}")
    launch = read_recorded_launch(terminal)
    metrics = read_skyrl_metrics(path, attempt_id=terminal.attempt_id)
    workload = preset.recipe.layout.workload
    planned = workload.planned_steps(preset.updates)
    training = [row for step, row in sorted(metrics.items()) if step in planned]
    invariants = list(training_invariants(preset, metrics))
    tracking = attempt_tracking(terminal, launch)
    try:
        recorded_tasks = {
            evaluation.name: _source_task_specs(recorded_sources(launch, evaluation.name))
            for evaluation in preset.evaluations
        }
        protocols = {evaluation.name: evaluation_protocol(launch, evaluation.name) for evaluation in preset.evaluations}
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidComparison(f"recorded evaluation identity is malformed: {error}") from error
    default, selection = recipe.task_selections(
        preset, recipe.selection_artifact(inputs.tasks) if inputs.tasks else None
    )
    task_set = held_out_identity(preset, launch)
    scores, initial_scores = [ANSWER_SCHEMA.empty_table()], []
    held_out_answers = {}
    evaluations = []
    response_token_counts: set[int | None] = set()
    primary_answers = 0
    for index, evaluation in enumerate(preset.evaluations):
        quality = evaluation.held_out and _checks_task_coverage(preset, evaluation)
        task_specs = recorded_tasks[evaluation.name]
        samples = evaluation.answers
        if evaluation.held_out:
            samples = protocols[evaluation.name]["n_samples_per_prompt"]
            expected = len(task_specs) * samples
        else:
            expected = recipe.evaluation_answer_count(
                evaluation, recipe.evaluation_source(evaluation, selection, default, preset.held_out_tasks)
            )
        if samples < 1 or expected < samples:
            raise InvalidComparison(f"recorded evaluation task set is empty for {evaluation.name}")
        all_scores = read_answers(evaluation, path, launch=launch, attempt=terminal)
        if evaluation.held_out:
            all_scores = _bind_task_digests(all_scores, task_specs)
        if index == 0:
            primary_answers = all_scores.num_rows
            response_token_counts.update(all_scores["response_token_count"].to_pylist())
        table = all_scores
        initial = table.filter(pc.equal(table["step"], 0))
        table = table.filter(pc.equal(table["step"], workload.scored_step(preset.updates)))
        scores.append(table)
        initial_scores.append(initial)
        if evaluation.held_out:
            held_out_answers[evaluation.name] = HeldOutAnswers(table, initial)
        ids = frozenset(table["task_id"].to_pylist())
        sessions = [(evaluation.name, table, workload.metric_step(preset.updates))]
        if evaluation.at_start:
            if initial.num_rows and frozenset(initial["task_id"].to_pylist()) != ids:
                raise InvalidComparison(f"initial and final evaluation task IDs differ: {path}")
            sessions.append((evaluation.name + "/initial", initial, 0))
        if preset.checkpoint_every or evaluation.every:
            scheduled = set(range(evaluation.every, preset.updates, evaluation.every)) if evaluation.every else set()
            sessions.extend(
                (f"{evaluation.name}/step-{step}", all_scores.filter(pc.equal(all_scores["step"], step)), step)
                for step in sorted(
                    (set(all_scores["step"].to_pylist()) | scheduled) - {0, workload.scored_step(preset.updates)}
                )
            )
        prefix = native_prefix(launch, evaluation.name) + "all/outcome/"
        for session_name, session, metric_step in sessions:
            statistics = score_answers(session)
            slices = score_slices(session)
            for key, value in statistics.items():
                evaluations.append(
                    Comparison(
                        metric=f"{evaluation.name}@{metric_step}/{key.removeprefix(evaluation.name + '/')}",
                        role=ComparisonRole.SLICE,
                        run=value,
                        n_run=1,
                        n_control=0,
                        tasks=len({row["task_id"] for row in slices[key]}) if key in slices else None,
                    )
                )
            invariants.extend(
                _session_invariants(
                    session_name,
                    session,
                    evaluation,
                    preset,
                    expected,
                    samples,
                    tuple(task_id for task_id, _ in task_specs),
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
    values, slices = _evaluation_values(pa.concat_tables(scores), pa.concat_tables([scores[0], *initial_scores]))
    result = terminal.native.result
    gpu_hours = None
    if result.started_at and result.finished_at:
        seconds = (recipe.utc_millis(result.finished_at) - recipe.utc_millis(result.started_at)) / 1000
        if seconds > 0:
            gpu_hours = seconds * launch.num_nodes * launch.gpus_per_node / 3600
    return RunMeasurements(
        config=terminal.native.config,
        recorded=launch,
        path=path,
        held_out_tasks=task_set,
        provenance=terminal.native.result.provenance,
        values={**_window_values(preset, metrics), **values},
        invariants=tuple(invariants),
        evaluations=tuple(evaluations),
        slices=slices,
        evaluation_identity={
            evaluation.name: (
                _task_set(recorded_tasks[evaluation.name]).task_set_sha256 if evaluation.held_out else None,
                protocols[evaluation.name],
            )
            for evaluation in preset.evaluations
        },
        held_out_answers=held_out_answers,
        started_at=result.started_at,
        gpu_hours=gpu_hours,
        engine=tracking.get("engine", {}),
        attempt_id=terminal.attempt_id,
    )
