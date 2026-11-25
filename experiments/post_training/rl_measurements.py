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
from dataclasses import dataclass
from typing import NamedTuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from finestore.rl.rl_progress import Comparison, ComparisonRole, ControlReplicate, EvaluationScore, Invariant
from finestore.rl.run_records import AttemptRecord, LineageEntry, attempt_lineage, read_evaluations
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

from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_preset_plan import InvalidComparison as InvalidComparison
from experiments.post_training.rl_scoring import (
    ANSWER_SCHEMA,
    answer_table,
    score_answers,
    score_membership,
    score_slices,
)
from experiments.post_training.task_curation.selection import TASK_SPEC_COLUMN


def read_recorded_launch(attempt: AttemptRecord) -> RecordedLaunch:
    try:
        return recorded_launch(attempt)
    except ValueError as error:
        raise InvalidComparison(str(error)) from None


class NativeWindow(NamedTuple):
    seconds: float
    gpus: int


def _native_window(attempt: AttemptRecord, launch: RecordedLaunch) -> NativeWindow | None:
    result = attempt.native.result
    if result.started_at is None or result.finished_at is None:
        return None
    seconds = (recipe.utc_millis(result.finished_at) - recipe.utc_millis(result.started_at)) / 1000
    if seconds < 0:
        raise ValueError("Native completion precedes its recorded start")
    return NativeWindow(seconds, launch.num_nodes * launch.gpus_per_node)


def native_gpu_hours(attempt: AttemptRecord, launch: RecordedLaunch) -> float | None:
    """Multiply the stored native job window by its GPU allocation."""
    window = _native_window(attempt, launch)
    return window.seconds * window.gpus / 3600 if window is not None and window.seconds > 0 else None


def lineage_windows(history: Sequence[LineageEntry]) -> list[NativeWindow] | None:
    """Read the native windows and allocations for a complete checkpoint lineage."""
    if not history or history[-1].record.resumed_step not in (None, 0):
        return None
    windows = []
    for entry in history:
        record = entry.record
        if record.config is None:
            return None
        try:
            launch = read_recorded_launch(record)
        except InvalidComparison:
            return None
        window = _native_window(record, launch)
        if window is None:
            return None
        windows.append(window)
    return windows


def lineage_gpu_hours(attempt: AttemptRecord, launch: RecordedLaunch) -> float | None:
    """Sum native allocations over the complete recorded checkpoint lineage."""
    if not attempt.native.result.state.is_terminal:
        return None
    windows = lineage_windows(attempt_lineage(launch.export_root, attempt.attempt_id))
    return sum(window.seconds * window.gpus for window in windows) / 3600 if windows else None


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


def read_answers(
    evaluation: presets.Evaluation | StoredEvaluation,
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


class StoredEvaluation(BaseModel, frozen=True):
    name: str
    primary: bool
    held_out: bool
    requires_coverage: bool
    at_start: bool = False
    every: int | None = None


class RunInputs(BaseModel, frozen=True):
    preset_name: str
    control: ControlReplicate | None = None
    settings: tuple[str, ...] = ()
    tasks: str | None = None
    init: presets.StoredInitializer | None = None
    resume: presets.StoredResume | None = None
    requirements: tuple[str, ...] = ()
    options: RunOptions | None = None
    sft_preset: str | None = None
    evaluation_profiles: tuple[StoredEvaluation, ...] | None = None


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
    preset: presets.Preset | None = None,
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
        evaluation_profiles=_declared_evaluations(preset) if preset is not None else None,
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
    task_ids: Sequence[str],
    answers_per_task: int,
    required_coverage: bool,
) -> list[Invariant]:
    rows = table.to_pylist()
    counts = Counter(row["task_id"] for row in rows)
    expected = len(task_ids) * answers_per_task
    holds = len(rows) == expected and set(counts.values()) == {answers_per_task} and set(counts) == set(task_ids)
    result = [Invariant(name=f"{name}/answers_complete", value=len(rows), limit=expected, comparator="==", holds=holds)]
    if required_coverage:
        coverage = len({row["task_id"] for row in rows if row["scored"]}) / len(task_ids) if task_ids else 0.0
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


def _declared_evaluations(preset: presets.Preset) -> tuple[StoredEvaluation, ...]:
    return tuple(
        StoredEvaluation(
            name=evaluation.name,
            primary=index == 0,
            held_out=evaluation.held_out,
            requires_coverage=evaluation == preset.guarded
            or (evaluation.held_out and isinstance(preset.recipe.layout.workload, presets.Training)),
        )
        for index, evaluation in enumerate(preset.evaluations)
    )


def _recorded_evaluations(
    preset: presets.Preset, launch: RecordedLaunch, inputs: RunInputs
) -> tuple[StoredEvaluation, ...]:
    declarations = inputs.evaluation_profiles
    if declarations is None:
        declarations = _declared_evaluations(preset)
        primary_files = {source.relative_path for source in launch.validation_data}
        if preset.evaluations and f"{preset.evaluations[0].file}.parquet" not in primary_files:
            declarations = tuple(row for row in declarations if not row.primary)
    by_name = {profile.name: profile for profile in declarations if not profile.primary}
    profiles = []
    if launch.validation_data:
        primary = next((profile for profile in declarations if profile.primary), None)
        profiles.append(
            primary or StoredEvaluation(name="primary", primary=True, held_out=False, requires_coverage=False)
        )
    for name in launch.evaluation_data:
        profiles.append(
            by_name.get(name) or StoredEvaluation(name=name, primary=False, held_out=False, requires_coverage=False)
        )
    callback = next((row for row in launch.callbacks or () if row.type == "evaluation"), None)
    at_start = launch.eval_before_train if callback is None else callback.eval_before_train
    every = launch.eval_interval if callback is None else callback.eval_steps
    if isinstance(preset.recipe.layout.workload, presets.Generation):
        at_start, every = False, None
    return tuple(
        row.model_copy(update={"at_start": at_start is True, "every": every if every and every > 0 else None})
        for row in profiles
    )


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
    evaluation_task_set: str | None
    gpu_hours: float | None
    engine: dict
    attempt_id: str


class EvaluationStatus(BaseModel):
    evaluations: tuple[Comparison, ...] = ()
    evaluation_invariants: tuple[Invariant, ...] = ()
    evaluation_error: str | None = None
    evaluation_task_set: str | None = None


def evaluation_status(
    path: str,
    preset: presets.Preset,
    attempt: AttemptRecord,
    launch: RecordedLaunch,
    *,
    inputs: RunInputs | None = None,
) -> EvaluationStatus:
    """Read successful-attempt evaluation summaries and invariants for status and publication."""
    if not (launch.validation_data or launch.evaluation_data) or attempt.native.result.state != "succeeded":
        return EvaluationStatus()
    try:
        effective = preset.with_updates(launch.max_steps if launch.max_steps is not None else preset.updates)
        if inputs is None:
            try:
                inputs = read_run_inputs(path)
            except FileNotFoundError:
                inputs = RunInputs(preset_name=preset.name)
        data = read_measurements(path, effective, inputs, attempt=attempt)
        names = tuple(profile.name + "/" for profile in _recorded_evaluations(effective, launch, inputs))
        return EvaluationStatus(
            evaluations=data.evaluations,
            evaluation_task_set=data.evaluation_task_set,
            evaluation_invariants=tuple(fact for fact in data.invariants if fact.name.startswith(names)),
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
    metrics = read_skyrl_metrics(path, attempt_id=terminal.attempt_id)
    workload = preset.recipe.layout.workload
    planned = workload.planned_steps(preset.updates)
    training = [row for step, row in sorted(metrics.items()) if step in planned]
    invariants = list(training_invariants(preset, metrics))
    launch = read_recorded_launch(terminal)
    profiles = _recorded_evaluations(preset, launch, inputs)
    tracking = attempt_tracking(terminal, launch)
    try:
        recorded_task_sets = {
            evaluation.name: read_recorded_task_set(launch, evaluation.name) for evaluation in profiles
        }
        identities = {
            evaluation.name: (
                (recorded_task_sets[evaluation.name].task_set_sha256 if evaluation.held_out else None),
                evaluation_protocol(launch, evaluation.name),
            )
            for evaluation in profiles
        }
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidComparison(f"recorded evaluation identity is malformed: {error}") from error
    task_set = preset.held_out_tasks if preset.held_out is not None else None
    scores, initial_scores = [ANSWER_SCHEMA.empty_table()], []
    evaluations = []
    response_token_counts: set[int | None] = set()
    primary_answers = 0
    for index, evaluation in enumerate(profiles):
        required_coverage = evaluation.requires_coverage
        recorded_tasks = recorded_task_sets[evaluation.name].task_ids
        recorded_answers = identities[evaluation.name][1]["n_samples_per_prompt"]
        all_scores = read_answers(evaluation, path, launch=launch, attempt=terminal)
        if index == 0:
            primary_answers = all_scores.num_rows
            response_token_counts.update(all_scores["response_token_count"].to_pylist())
        table = all_scores
        initial = table.filter(pc.equal(table["step"], 0))
        table = table.filter(pc.equal(table["step"], workload.scored_step(preset.updates)))
        scores.append(table)
        initial_scores.append(initial)
        ids = frozenset(table["task_id"].to_pylist())
        intended_step = workload.metric_step(preset.updates)
        sessions = [(evaluation.name, table, intended_step)]
        if evaluation.held_out:
            task_digest, protocol = identities[evaluation.name]
            identities[evaluation.name] = (
                task_digest,
                {
                    **protocol,
                    "family_scoring": score_membership(table, "family_mean"),
                    "aai_scoring": score_membership(table, "aai_weighted_mean"),
                },
            )
        if evaluation.at_start:
            if initial.num_rows and frozenset(initial["task_id"].to_pylist()) != ids:
                raise InvalidComparison(f"initial and final evaluation task IDs differ: {path}")
            sessions.append((evaluation.name + "/initial", initial, 0))
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
                statistic = key.removeprefix(evaluation.name + "/")
                coordinate = None
                family_slice = statistic.startswith("family/") and statistic.endswith("/mean")
                if statistic in ("mean", "family_mean") or family_slice:
                    final = metric_step == intended_step
                    comparison_digest = None
                    if final and ids and ids == frozenset(recorded_tasks):
                        comparison_digest = hashlib.sha256(
                            json.dumps(
                                (
                                    recorded_task_sets[evaluation.name].task_set_sha256,
                                    evaluation_protocol(launch, evaluation.name),
                                    score_membership(session, statistic, scored_only=True),
                                ),
                                sort_keys=True,
                            ).encode()
                        ).hexdigest()
                    coordinate = EvaluationScore(
                        profile=evaluation.name,
                        step=metric_step,
                        statistic=statistic,
                        final=final,
                        comparability=comparison_digest,
                    )
                evaluations.append(
                    Comparison(
                        metric=f"{evaluation.name}@{metric_step}/{statistic}",
                        evaluation_score=coordinate,
                        role=ComparisonRole.SLICE,
                        run=value,
                        n_run=1,
                        n_control=0,
                        tasks=len({row["task_id"] for row in slices[key]}) if key in slices else None,
                    )
                )
            invariants.extend(
                _session_invariants(session_name, session, recorded_tasks, recorded_answers, required_coverage)
            )
            if evaluation.held_out and required_coverage:
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
        **score_answers(final),
    }
    for key, value in score_answers(pa.concat_tables([scores[0], *initial_scores])).items():
        values[key + "/initial"] = value
        if key in values:
            values[key + "/gain"] = values[key] - value
    gpu_hours = native_gpu_hours(terminal, launch)
    slices = {key: len({row["task_id"] for row in rows}) for key, rows in score_slices(final).items()}
    task_sets = [(name, recorded.task_set_sha256) for name, recorded in sorted(recorded_task_sets.items())]
    return RunMeasurements(
        config=terminal.native.config,
        recorded=launch,
        path=path,
        held_out_tasks=task_set,
        provenance=terminal.native.result.provenance,
        values=values,
        invariants=tuple(invariants),
        evaluations=tuple(evaluations),
        slices=slices,
        evaluation_identity=identities,
        evaluation_task_set=hashlib.sha256(json.dumps(task_sets).encode()).hexdigest() if task_sets else None,
        gpu_hours=gpu_hours,
        engine=tracking.get("engine", {}),
        attempt_id=terminal.attempt_id,
    )
