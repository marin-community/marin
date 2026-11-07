# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a saved Snowball run, or one control replicate, with the preset's pinned controls.

Reads saved metrics and panel scores only, so it needs no GPU. Writes gate_report.json and
gate_report.md under the caller's namespace and prints the report as JSON. snowball_rl runs this
comparison itself after a gated run; use this command to re-score a finished run, for example
with a new --require or --baseline, or to check calibration by scoring a control against the other controls.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl_gate`:
  --preset learning-medium --candidate RUN_NAME --require 'test_panel/aaii_category/coding/mean>=control'
  --preset speed-medium --leave-out r0
  --preset learning-medium --candidate RUN_NAME --baseline EARLIER_RUN@2026.10.27
"""

from __future__ import annotations

import importlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import NamedTuple, TypeGuard

import click
import pyarrow as pa
import pyarrow.compute as pc
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, run
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    SkyRLReceiptLookupError,
    SkyRLRun,
    read_skyrl_attempt_tracking,
    read_skyrl_launch_config,
    read_skyrl_launch_receipt,
)
from pydantic import BaseModel
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from scipy.stats import t
from tabulate import tabulate
from taskcompendium.grading_result import Outcome

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.snowball_rl_scoring import InvalidComparison

GATE_REVISION = "2026.11.07"
GATE_REPORT_FILENAME = "gate_report.json"
GRADER_ERROR_LIMIT = 0.01
OUTCOME_PREFIX = "generate/outcome/"
REWARD_PREFIX = "reward/domain/"


@dataclass(frozen=True)
class ControlPin:
    preset: str
    main_commit: str
    revision: str
    runs: int


CONTROLS = {
    name: (ControlPin(name, "648b5687f0", "2026.11.06.1", count),)
    for name, count in (
        ("speed-short", 5),
        ("data-short", 4),
        ("speed-medium", 4),
        ("learning-medium", 4),
        ("speed-long", 4),
    )
}

SHARED_CONTROLS = {"data-medium": "learning-medium", "data-long": "learning-long"}


_DETAIL_METRICS = {
    presets.Focus.SPEED: (
        presets.Metric("async/performance/loss_tokens_per_configured_policy_gpu_second", True),
        presets.Metric("async/performance/response_tokens_per_configured_inference_gpu_second", True),
    ),
    presets.Focus.LEARNING: (
        presets.Metric("async/rejected_rate", False),
        presets.Metric("consumed/length_stop_fraction", False),
        presets.Metric("generate/std_num_tokens", False),
    ),
}


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INVALID = "invalid"


class Label(StrEnum):
    IMPROVES = "improves"
    TRADE_OFF = "trade-off"
    NO_CHANGE = "no change"
    REGRESSION = "regression"


class Change(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    SAME = "same"
    UNAVAILABLE = "unavailable"


class GateInputs(BaseModel, frozen=True):
    settings: tuple[str, ...] = ()
    tasks: str | None = None
    init: str | None = None
    requirements: tuple[str, ...] = ()
    baselines: tuple[str, ...] = ()


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


def _first_control(preset: presets.Preset, pins: Sequence[ControlPin]) -> tuple[pa.Table | None, RunData]:
    rates_path = _pass_rates_path(preset)
    rates = scoring.read_pass_rates(rates_path) if rates_path is not None else None
    return rates, _load_run(control_path(preset, pins[0], 0), preset, rates)


def validate_candidate(preset: presets.Preset, requirements: Sequence[str], *, control: bool) -> None:
    """Validate a candidate's control pin and user requirements before GPU submission."""
    gated = presets.gated(preset)
    if requirements and (control or not gated):
        raise click.BadParameter("--require needs a gated candidate preset, without --control")
    keys = [parse_requirement(text).key for text in requirements]
    pins = control_pins(preset.name) if not control and gated else ()
    if not requirements:
        return
    reference = _with_objectives(_first_control(preset, pins)[1], preset)
    missing = [key for key in keys if key not in reference.values]
    if missing:
        raise click.BadParameter(f"requirement metrics that the gate does not compute for control r0: {missing}")


def gate_inputs(
    settings: Sequence[presets.Setting] = (),
    tasks: str | None = None,
    init: str | None = None,
    requirements: Sequence[str] = (),
    baselines: Sequence[str] = (),
) -> GateInputs:
    values = {setting.key: setting.value for setting in settings}
    return GateInputs(
        settings=tuple(f"{key}={json.dumps(value, default=str)}" for key, value in sorted(values.items())),
        tasks=tasks.removeprefix(presets.SELECTION_PREFIX + "/") if tasks else None,
        init=init,
        requirements=tuple(requirements),
        baselines=tuple(sorted(set(baselines))),
    )


class Invariant(BaseModel):
    name: str
    value: bool | int | float
    limit: bool | int | float
    holds: bool
    replicate: int = 0


class ErrorBoundInvariant(Invariant):
    lower_bound: float
    upper_bound: float


class Comparison(BaseModel):
    metric: str
    higher_is_better: bool
    run: float | None = None
    control: float | None = None
    delta: float | None = None
    tolerance: float | None = None
    sd: float | None = None
    df: int | None = None
    n_run: int
    n_control: int


class GuardComparison(Comparison):
    passes: bool
    change: Change


class ReportedComparison(Comparison):
    change: Change
    tasks: int | None = None
    initializer: float | None = None
    control_min: float | None = None
    control_max: float | None = None


class ObjectiveComparison(Comparison):
    change: Change
    delta_percent: float | None = None
    noise_percent: float | None = None


class RunMetadata(GateInputs):
    tree: str | None = None
    base_commit: str | None = None
    dirty: bool | None = None
    replicates: int
    panel: str | None = None
    engines: tuple[dict, ...] = ()


class ControlMetadata(BaseModel):
    main_commit: str
    revision: str
    runs: int
    gate_revision: str = GATE_REVISION
    engines: tuple[dict, ...] = ()
    panel: str | None = None


class GateReport(Artifact):
    preset: str
    verdict: Verdict
    label: Label | None
    reason: str | None
    run: RunMetadata
    control: ControlMetadata
    invariants: tuple[ErrorBoundInvariant | Invariant, ...]
    guards: tuple[GuardComparison, ...]
    reported: tuple[ReportedComparison, ...]
    objectives: tuple[ObjectiveComparison, ...] = ()
    constraints: tuple[GuardComparison, ...] = ()
    vs_baseline: tuple[ReportedComparison, ...] = ()
    vs_baseline_reason: str | None = None
    gaps: tuple[str, ...]


@dataclass(frozen=True)
class MetricNoise:
    mean: float
    sd: float
    df: int


def calibrate(control_values: Mapping[ControlPin, Sequence[Mapping[str, float]]]) -> dict[str, MetricNoise]:
    groups = list(control_values.values())
    runs = [run for group in groups for run in group]
    keys = set.intersection(*(set(run) for run in runs)) if runs else set()
    df = sum(len(group) - 1 for group in groups)
    if df < 1:
        raise InvalidComparison("calibration needs at least two replicates within a control pin")
    result = {}
    for key in keys:
        values = [run[key] for run in runs]
        if all(math.isfinite(value) for value in values):
            squared = math.fsum(
                (run[key] - statistics.mean(item[key] for item in group)) ** 2 for group in groups for run in group
            )
            result[key] = MetricNoise(statistics.mean(values), math.sqrt(squared / df), df)
    return result


def _comparison(
    metric: presets.Metric, rows: Sequence[Mapping[str, float]], noise: MetricNoise | None, n_control: int, count: int
) -> Comparison:
    values = [row.get(metric.key, math.nan) for row in rows]
    value = statistics.mean(values) if all(math.isfinite(item) for item in values) else None
    row = Comparison(
        metric=metric.key,
        higher_is_better=metric.higher_is_better,
        run=value,
        control=noise.mean if noise else None,
        n_run=len(values),
        n_control=n_control,
    )
    if value is None or noise is None:
        return row
    tolerance = float(t.ppf(1 - 0.025 / count, noise.df)) * noise.sd * math.sqrt(1 / len(values) + 1 / n_control)
    delta = (value - noise.mean) * (1 if metric.higher_is_better else -1)
    return row.model_copy(update={"delta": delta, "tolerance": tolerance, "sd": noise.sd, "df": noise.df})


def _ambiguous(row: Invariant) -> TypeGuard[ErrorBoundInvariant]:
    return isinstance(row, ErrorBoundInvariant) and row.lower_bound <= row.limit < row.upper_bound


class VerdictResult(NamedTuple):
    verdict: Verdict
    label: Label | None
    reason: str | None


def _verdict(
    invariants: Sequence[Invariant], guards: Sequence[GuardComparison], declared: Sequence[ReportedComparison]
) -> VerdictResult:
    ambiguous = [row for row in invariants if _ambiguous(row)]
    if any(not row.holds and not _ambiguous(row) for row in invariants) or any(not row.passes for row in guards):
        verdict = Verdict.FAIL
    elif ambiguous:
        return VerdictResult(
            Verdict.INVALID,
            None,
            "; ".join(
                f"grader-error union is ambiguous: lower={row.lower_bound:.12g}, upper={row.upper_bound:.12g}, "
                f"limit={row.limit}; available counters do not determine the grader-error fraction"
                for row in ambiguous
            ),
        )
    else:
        verdict = Verdict.PASS
    better = any(row.change == Change.BETTER for row in (*guards, *declared))
    worse = verdict == Verdict.FAIL or any(row.change == Change.WORSE for row in (*guards, *declared))
    if better:
        return VerdictResult(verdict, Label.TRADE_OFF if worse else Label.IMPROVES, None)
    return VerdictResult(verdict, Label.REGRESSION if worse else Label.NO_CHANGE, None)


def read_metrics(root: StoragePath) -> dict[int, dict]:
    """Read the saved per-update metrics in the run's checkpoint lineage."""
    return {
        int(path.name.removeprefix("step-").removesuffix(".json")): json.loads(presets.read_text(path))
        for path in scoring.paths(root / "exports/metrics/step-*.json")
    }


def _mismatch_counts(rows: Sequence[Mapping[str, float]], prefix: str) -> list[float] | None:
    counts = [row.get(prefix + "/selected_tokens", math.nan) for row in rows]
    if all(math.isfinite(count) and count >= 0 for count in counts) and math.fsum(counts) > 0:
        return counts
    return None


def _window_values(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    rows = [metrics.get(step, {}) for step in preset.measured]
    phases = [
        key
        for key in rows[0]
        if key.startswith("timing/step_wall/") and key.rsplit("/", 1)[-1] not in ("evaluation", "checkpoint_work")
    ]
    values = {}
    declared = {metric.key: metric for metric in (*preset.guards, *preset.reported)}
    extra = set().union(*(row.keys() for row in metrics.values())) - declared.keys() - {"step", "attempt_id"}
    for metric in (*declared.values(), *(presets.Metric(key, True) for key in sorted(extra))):
        if metric.key.startswith("startup/"):
            first = next((row[metric.key] for _, row in sorted(metrics.items()) if metric.key in row), None)
            if first is not None:
                values[metric.key] = float(first)
            continue
        if metric.window == presets.WindowStatistic.SELECTED_TOKEN_WEIGHTED_MEAN:
            prefix = "/".join(metric.key.split("/")[:3])
            counts = _mismatch_counts(rows, prefix)
            if counts is None:
                continue
            total = math.fsum(counts)
            nonempty = [(row, count) for row, count in zip(rows, counts, strict=True) if count > 0]
            if all(
                math.isfinite(row.get(metric.key, math.nan)) and row.get(prefix + "/finite_fraction") == 1
                for row, _ in nonempty
            ):
                values[metric.key] = math.fsum(count / total * row[metric.key] for row, count in nonempty)
            continue
        if metric.key == "speed/update_seconds":
            series = [math.fsum(row[key] for key in phases) for row in rows if phases and set(phases) <= set(row)]
        else:
            series = [row[metric.key] for row in rows if metric.key in row]
        if series and len(series) == len(rows):
            reducer = statistics.median if metric.window == presets.WindowStatistic.MEDIAN else statistics.mean
            values[metric.key] = float(reducer(series))
    return values


def _grader_errors(
    rows: Sequence[Mapping[str, float]], prefix: str, name: str, *, answers: int | None = None
) -> ErrorBoundInvariant:
    recorded = all(prefix + outcome in row for row in rows for outcome in (*Outcome, "rollout_failure"))

    def count(*outcomes: str) -> float:
        return math.fsum(row.get(prefix + outcome, 0) for row in rows for outcome in outcomes)

    total = count(*Outcome)
    bad = count(Outcome.INFRA_ERROR, Outcome.UNAVAILABLE, Outcome.INVALID_TASK)
    failures = count("rollout_failure")
    lower, upper = (max(bad, failures) / total, min(total, bad + failures) / total) if total else (1.0, 1.0)
    if answers is not None and (not recorded or total != answers or failures > total):
        lower, upper = 0.0, 1.0
    return ErrorBoundInvariant(
        name=name,
        value=upper,
        lower_bound=lower,
        upper_bound=upper,
        limit=GRADER_ERROR_LIMIT,
        holds=upper <= GRADER_ERROR_LIMIT and recorded,
    )


def _session_invariants(
    name: str, table: pa.Table, evaluation: presets.Evaluation, preset: presets.Preset, expected: int
) -> list[Invariant]:
    rows = table.to_pylist()
    counts = Counter(row["item"] for row in rows)
    holds = len(rows) == expected and set(counts.values()) == {evaluation.answers}
    result = [Invariant(name=f"{name}/batch_finished", value=len(rows), limit=expected, holds=holds)]
    if any(presets.guards_evaluation(evaluation, metric) for metric in preset.guards):
        coverage = len({row["item"] for row in rows if row["scored"]}) / (expected // evaluation.answers)
        result.append(
            Invariant(
                name=f"{name}/panel_scored_fraction",
                value=coverage,
                limit=presets.MINIMUM_PANEL_COVERAGE,
                holds=coverage >= presets.MINIMUM_PANEL_COVERAGE,
            )
        )
    return result


@dataclass(frozen=True)
class RunData:
    path: str
    version: str
    panel: str | None
    provenance: dict
    values: dict[str, float]
    panel_ids: dict[str, frozenset[str]]
    invariants: tuple[Invariant, ...]
    missing_tokens: bool
    slices: dict[str, tuple[int, float | None]]
    evaluation_identity: dict[str, tuple[str | None, dict]]
    h100_hours: float | None
    engine: dict


def _recorded_panel(launch: dict, evaluation: presets.Evaluation) -> str:
    try:
        identities = {
            "@".join(scoring.recorded_identity(source)) for source in scoring.recorded_sources(launch, evaluation.name)
        }
        if len(identities) != 1:
            raise ValueError("expected one panel identity")
        identity = identities.pop()
        return identity
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise InvalidComparison(f"recorded panel version is missing or malformed: {error}") from error


def _panel_identity(preset: presets.Preset, launch: dict, inputs: GateInputs) -> str | None:
    evaluation = presets.panel_evaluation(preset)
    if evaluation is None:
        return None
    default = presets.default_selection()
    selection = presets.selection_artifact(inputs.tasks) if inputs.tasks else default
    selected = evaluation.tasks(selection, default)
    panel = f"{selected.name.removeprefix(presets.SELECTION_PREFIX + '/')}@{selected.version}"
    binding = selected
    if evaluation.sample is not None:
        binding = presets.sampled_tasks_step(
            selected,
            evaluation.file,
            evaluation.sample,
            launch["skyrl"]["trainer"]["seed"],
        )
    expected = f"{binding.name}@{binding.version}"
    sources = scoring.recorded_sources(launch, evaluation.name)
    recorded = _recorded_panel(launch, evaluation)
    if recorded != expected:
        raise InvalidComparison(
            f"recorded panel differs from preset: expected {panel} ({expected}); recorded {recorded}"
        )
    if evaluation.file != scoring.TRAIN_STEM and any(
        source["relative_path"] != f"{evaluation.file}.parquet" for source in sources
    ):
        raise InvalidComparison(f"recorded panel file differs from preset {panel}: expected {evaluation.file}.parquet")
    return panel


def training_invariants(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> tuple[Invariant, ...]:
    """Check planned update completion and finite training loss and gradient evidence."""
    generating = preset.recipe.layout.role_plan.generate_only
    planned = {0} if generating else set(range(1, preset.updates + 1))
    completed = planned.intersection(metrics)
    training = [metrics[step] for step in sorted(completed)]
    invariants = [
        Invariant(name="planned_updates", value=len(completed), limit=len(planned), holds=completed == planned)
    ]
    if not generating:
        finite = all(
            "policy/raw_grad_norm" in row
            and any(key.startswith("loss/") for key in row)
            and all(math.isfinite(row[key]) for key in row if key.startswith("loss/") or key == "policy/raw_grad_norm")
            for row in training
        )
        invariants.append(Invariant(name="finite_losses_and_gradients", value=finite, limit=True, holds=finite))

    return tuple(invariants)


def _load_run(
    path: str,
    preset: presets.Preset,
    rates: pa.Table | None,
    inputs: GateInputs = GateInputs(),
) -> RunData:
    terminal = read_skyrl_launch_receipt(path)
    if terminal is None:
        raise FileNotFoundError(f"no launch receipt at {path}")
    if terminal.result.state != "succeeded":
        raise InvalidComparison(native_failure_reason(terminal.result.state, terminal.result.iris_job_id))
    if terminal.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {path}")
    metrics = read_metrics(StoragePath(path))
    generating = preset.recipe.layout.role_plan.generate_only
    training = [
        row for step, row in sorted(metrics.items()) if (step == 0 if generating else 1 <= step <= preset.updates)
    ]
    invariants = list(training_invariants(preset, metrics))
    answers = scoring.evaluation_answers(path) if preset.evaluations else []
    launch = read_skyrl_launch_config(path)
    attempt = launch["run"]["attempt_id"]
    tracking = read_skyrl_attempt_tracking(path, attempt)
    try:
        identities = {
            evaluation.name: (
                (
                    scoring.read_recorded_panel(launch, evaluation.name).task_set_sha256
                    if evaluation.file != scoring.TRAIN_STEM
                    else None
                ),
                scoring.evaluation_protocol(launch, evaluation.name),
            )
            for evaluation in preset.evaluations
        }
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidComparison(f"recorded evaluation identity is malformed: {error}") from error
    panel = _panel_identity(preset, launch, inputs)
    scores, initial_scores, panel_ids = [scoring.EVAL_SCORE_SCHEMA.empty_table()], [], {}
    default = presets.default_selection()
    selection = presets.selection_artifact(inputs.tasks) if inputs.tasks else default
    for evaluation in preset.evaluations:
        quality = evaluation.file != scoring.TRAIN_STEM and any(
            presets.guards_evaluation(evaluation, metric) for metric in preset.guards
        )
        expected = presets.evaluation_answer_count(evaluation, evaluation.tasks(selection, default))
        table = scoring.panel_scores(evaluation, path, answers)
        initial = table.filter(pc.equal(table["step"], 0))
        table = table.filter(pc.equal(table["step"], -1 if generating else preset.updates))
        scores.append(table)
        initial_scores.append(initial)
        ids = frozenset(table["item"].to_pylist())
        if evaluation.file != scoring.TRAIN_STEM:
            panel_ids[evaluation.name] = ids
        sessions = [(evaluation.name, table, 0 if generating else preset.updates)]
        if evaluation.at_start:
            if initial.num_rows and frozenset(initial["item"].to_pylist()) != ids:
                raise InvalidComparison(f"initial and final panel task IDs differ: {path}")
            sessions.append((evaluation.name + "/initial", initial, 0))
        prefix = scoring.native_prefix(launch, evaluation.name) + "all/outcome/"
        for session_name, session, metric_step in sessions:
            invariants.extend(_session_invariants(session_name, session, evaluation, preset, expected))
            if quality:
                invariants.append(
                    _grader_errors(
                        [metrics.get(metric_step, {})],
                        prefix,
                        session_name + "/grader_error_fraction",
                        answers=session.num_rows,
                    )
                )
    invariants.append(_grader_errors(training, OUTCOME_PREFIX, "grader_error_fraction"))
    fixed = preset.recipe.fixed_reply_tokens
    if fixed is not None:
        if generating:
            holds = {len(row["response_token_ids"]) for _, _, row in answers} == {fixed}
        else:
            holds = all(
                row.get("generate/min_num_tokens") == row.get("generate/max_num_tokens") == fixed for row in training
            )
        invariants.append(Invariant(name="fixed_reply_tokens", value=holds, limit=fixed, holds=holds))
    missing = any(
        metric.key.startswith("policy/mismatch/")
        and _mismatch_counts([metrics.get(step, {}) for step in preset.measured], metric.key.rsplit("/", 1)[0]) is None
        for metric in preset.guards
    )
    final = pa.concat_tables(scores)
    values = {
        **_window_values(preset, metrics),
        **scoring.evaluation_statistics(final, rates),
    }
    for key, value in scoring.evaluation_statistics(pa.concat_tables([scores[0], *initial_scores])).items():
        values[key + "/initial"] = value
        if key in values:
            values[key + "/gain"] = values[key] - value
    result = terminal.result
    h100_hours = None
    if result.started_at and result.finished_at:
        seconds = (presets.utc_millis(result.finished_at) - presets.utc_millis(result.started_at)) / 1000
        if seconds > 0:
            allocation = launch["iris"]["allocation"]
            h100_hours = seconds * allocation["num_nodes"] * allocation["gpus_per_node"] / 3600
    initializer = {row["task_id"]: row["mean_score"] for row in rates.to_pylist()} if rates is not None else {}
    slices = {}
    for key, rows in scoring.score_slices(final, rates).items():
        ids = {row["item"] for row in rows}
        initial = statistics.mean(initializer[item] for item in ids) if ids <= initializer.keys() else None
        slices[key] = len(ids), initial
    return RunData(
        path=path,
        version=presets.artifact_record(path).version,
        panel=panel,
        provenance=terminal.result.provenance,
        values=values,
        panel_ids=panel_ids,
        invariants=tuple(invariants),
        missing_tokens=missing,
        slices=slices,
        evaluation_identity=identities,
        h100_hours=h100_hours,
        engine=tracking.get("engine", {}),
    )


def native_failure_reason(state: str, job_id: str | None) -> str:
    return f"native job {job_id} is {state}"


def result_text(verdict: str | None, label: str | None) -> str:
    return f"{verdict} ({label})" if label else verdict or "pending"


def _with_objectives(candidate: RunData, preset: presets.Preset) -> RunData:
    values = dict(candidate.values)
    for metric in preset.objectives:
        if metric.key.endswith("/gain_per_h100_hour"):
            gain = values.get(metric.key.removesuffix("_per_h100_hour"))
            if gain is not None and candidate.h100_hours:
                values[metric.key] = gain / candidate.h100_hours
    return replace(candidate, values=values)


def _with_requirements(candidate: RunData, requirements: Sequence[str]) -> RunData:
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
                Invariant(name=f"require/{requirement.key}", value=value, limit=requirement.bound, holds=holds)
            )
    return replace(candidate, invariants=tuple(invariants))


def _check_control(run: RunData, pin: ControlPin) -> None:
    if run.provenance["base_commit"] != pin.main_commit or run.provenance["dirty"]:
        raise InvalidComparison(f"control does not match its pin: {run.path}")
    if run.missing_tokens or any(not row.holds for row in run.invariants):
        raise InvalidComparison(f"control has missing mismatch tokens or failed invariants: {run.path}")


def _change(row: Comparison) -> Change:
    if row.delta is None:
        return Change.UNAVAILABLE
    return Change.BETTER if row.delta > row.tolerance else Change.WORSE if row.delta < -row.tolerance else Change.SAME


def _breakdown_group(key: str) -> str:
    if key.startswith(REWARD_PREFIX):
        return "Training reward per source"
    if key.endswith(("/initial", "/gain")):
        return key.split("/", 1)[0] + ": initial weights"
    parts = key.split("/")
    dimension = parts[1] if len(parts) > 2 else "overall"
    if dimension == "agentic":
        dimension = "overall"
    if "difficulty" in parts and dimension != "difficulty":
        dimension += " x difficulty"
    return f"{parts[0]}: {dimension}"


class ComparisonRows(NamedTuple):
    guards: list[GuardComparison]
    declared: list[ReportedComparison]
    details: list[ReportedComparison]


def _compare(
    preset: presets.Preset,
    runs: Sequence[RunData],
    controls: Mapping[ControlPin, Sequence[RunData]],
    requirements: Sequence[str] = (),
    *,
    baselines: Sequence[RunData] = (),
) -> ComparisonRows:
    """Return guard comparisons, declared report rows and breakdown detail rows."""
    noise = calibrate({pin: [run.values for run in group] for pin, group in controls.items()})
    values = [run.values for run in runs]
    n_control = sum(map(len, controls.values()))
    reference_noise: dict[str, MetricNoise | None] = dict(noise)
    if baselines:
        n_control = len(baselines)
        for key, value in noise.items():
            samples = [run.values.get(key, math.nan) for run in baselines]
            reference_noise[key] = (
                replace(value, mean=statistics.mean(samples)) if all(map(math.isfinite, samples)) else None
            )
    guards = []
    relative = [parse_requirement(text) for text in requirements if parse_requirement(text).bound is None]
    metrics = tuple(
        {
            (metric.key, metric.higher_is_better): metric
            for metric in (*preset.guards, *(presets.Metric(item.key, item.higher_is_better) for item in relative))
        }.values()
    )
    for metric in metrics:
        row = _comparison(metric, values, reference_noise.get(metric.key), n_control, len(metrics))
        if row.delta is None and not baselines:
            raise InvalidComparison(f"missing or non-finite guard metric: {metric.key}")
        if row.sd == 0 and not baselines:
            raise InvalidComparison(f"control variation is zero for guard metric: {metric.key}; use a number bound")
        guards.append(
            GuardComparison(
                **row.model_dump(), passes=row.delta is not None and row.delta >= -row.tolerance, change=_change(row)
            )
        )
    guard_keys = {metric.key for metric in metrics}
    declared = {metric.key: metric for metric in preset.reported if metric.key not in guard_keys}
    prefixes = tuple(evaluation.name + "/" for evaluation in preset.evaluations)
    keys = dict.fromkeys(
        key
        for row in (*values, noise)
        for key in row
        if (key.startswith(prefixes) or key.startswith(REWARD_PREFIX))
        and (not key.startswith(REWARD_PREFIX) or key in noise)
        and key not in declared
        and key not in guard_keys
    )
    details = (
        {}
        if preset.focus == presets.Focus.SPEED
        else {key: presets.Metric(key, not key.endswith("all_wrong_fraction")) for key in keys}
    )
    telemetry = _DETAIL_METRICS.get(presets.PRESETS[SHARED_CONTROLS.get(preset.name, preset.name)].focus, ())
    details.update({metric.key: metric for metric in telemetry if metric.key in noise and metric.key not in guard_keys})
    counts = Counter(map(_breakdown_group, (*details, *(key for key in declared if key.startswith(prefixes)))))
    reported = []
    reference = baselines or [run for group in controls.values() for run in group]
    for metric in (*declared.values(), *details.values()):
        row = _comparison(
            metric,
            values,
            reference_noise.get(metric.key),
            n_control,
            (
                counts[_breakdown_group(metric.key)]
                if metric.key not in declared and metric.key.startswith((*prefixes, REWARD_PREFIX))
                else 1
            ),
        )
        slices = [run.slices.get(metric.key, (None, None)) for run in runs]
        tasks = slices[0][0] if len({count for count, _ in slices}) == 1 else None
        initializer = (
            statistics.mean(initial for _, initial in slices)
            if all(initial is not None for _, initial in slices)
            else None
        )
        control_values = [run.values[metric.key] for run in reference if metric.key in run.values]
        reported.append(
            ReportedComparison(
                **row.model_dump(),
                change=Change.UNAVAILABLE if metric.key not in declared and row.sd == 0 else _change(row),
                tasks=tasks,
                initializer=initializer,
                control_min=min(control_values) if row.control is not None else None,
                control_max=max(control_values) if row.control is not None else None,
            )
        )
    return ComparisonRows(guards, reported[: len(declared)], reported[len(declared) :])


def objective_metrics(preset: presets.Preset) -> tuple[presets.Metric, ...]:
    return preset.objectives


def _objectives(
    preset: presets.Preset,
    runs: Sequence[RunData],
    grouped: Mapping[ControlPin, Sequence[RunData]],
    guards: Sequence[GuardComparison],
) -> list[ObjectiveComparison]:
    noise = calibrate({pin: [run.values for run in group] for pin, group in grouped.items()})
    guarded = {row.metric: row for row in guards}
    rows = [
        (
            guarded[metric.key]
            if metric.key in guarded
            else _comparison(
                metric, [run.values for run in runs], noise.get(metric.key), sum(map(len, grouped.values())), 1
            )
        )
        for metric in preset.objectives
    ]
    return [
        ObjectiveComparison(
            **row.model_dump(include=set(Comparison.model_fields)),
            change=_change(row),
            delta_percent=(
                100 * (row.run - row.control) / abs(row.control) if row.run is not None and row.control else None
            ),
            noise_percent=(
                100 * row.tolerance / abs(row.control) if row.tolerance is not None and row.control else None
            ),
        )
        for row in rows
    ]


def _render_markdown(report: GateReport) -> str:
    lines = [f"# {report.preset}", "", "## Objectives", ""]
    rows = [
        {
            "metric": row.metric,
            "control": row.control,
            "run": row.run,
            "delta %": row.delta_percent,
            "noise +/- %": row.noise_percent,
            "change": row.change,
        }
        for row in report.objectives
    ]
    lines += [tabulate(rows, "keys", "github", floatfmt=".8g", missingval="unavailable"), "", "## Constraints", ""]
    constraints = [
        {"metric": row.metric, "control": row.control, "run": row.run, "passes": row.passes}
        for row in report.constraints
    ]
    constraints += [
        {"metric": row.name, "value": row.value, "limit": row.limit, "passes": row.holds} for row in report.invariants
    ]
    lines += [
        tabulate(constraints, "keys", "github", floatfmt=".8g", missingval="unavailable"),
        "",
        f"Result: {result_text(report.verdict, report.label)}",
    ]
    if report.reason:
        lines += ["", report.reason]
    for title, value in (("Run", report.run), ("Control", report.control)):
        lines += ["", f"## {title}", ""]
        lines += [f"- {key}: {json.dumps(item, default=str)}" for key, item in value.model_dump().items()]
    prefixes = (*(evaluation.name + "/" for evaluation in presets.PRESETS[report.preset].evaluations), REWARD_PREFIX)
    breakdown = [row for row in report.reported if row.metric.startswith(prefixes)]
    summary = [row for row in report.reported if not row.metric.startswith(prefixes)]
    for title, rows in (("Guards", report.guards), ("Reported", summary), ("Invariants", report.invariants)):
        rows = [row.model_dump() for row in rows]
        lines += ["", f"## {title}", "", tabulate(rows, "keys", "github", floatfmt=".8g", missingval="unavailable")]
    if report.run.baselines:
        rows = [
            {key.replace("control", "baseline"): value for key, value in row.model_dump().items()}
            for row in report.vs_baseline
        ]
        lines += [
            "",
            "## Vs baseline",
            "",
            report.vs_baseline_reason or tabulate(rows, "keys", "github", floatfmt=".8g", missingval="unavailable"),
        ]
    groups = {}
    for row in breakdown:
        control = f"{row.control:.8g} ({row.control_min:.8g}-{row.control_max:.8g})" if row.control is not None else None
        values = (row.metric, row.tasks, row.initializer, control, row.run, row.delta, row.tolerance, row.change)
        groups.setdefault(_breakdown_group(row.metric), []).append(values)
    lines += ["", "## Breakdown"]
    headers = ("slice", "tasks", "initializer", "control mean (min-max)", "run", "Δ", "τ", "change")
    for name, rows in groups.items():
        lines += [
            "",
            f"### {name}",
            "",
            tabulate(rows, headers, "github", floatfmt=".8g", missingval="unavailable"),
        ]
    return "\n".join([*lines, "", "## Gaps", "", *[f"- {gap}" for gap in report.gaps], ""])


def _write_gate(
    preset_name: str,
    run_paths: tuple[str, ...],
    control_paths: tuple[tuple[str, ...], ...],
    pins: tuple[ControlPin, ...],
    inputs: GateInputs,
    pass_rates_path: str | None,
    output: str,
) -> GateReport:
    preset = presets.PRESETS[preset_name]
    metadata = RunMetadata(**inputs.model_dump(), replicates=len(run_paths))
    control = ControlMetadata(
        main_commit=pins[0].main_commit, revision=pins[0].revision, runs=sum(map(len, control_paths))
    )
    invariants, guards, reported, vs_baseline = [], [], [], []
    objectives, constraints = [], []
    baseline_reason = None
    verdict, label, reason = Verdict.INVALID, None, None
    try:
        rates = scoring.read_pass_rates(pass_rates_path) if pass_rates_path is not None else None
        if rates is not None:
            evaluation = presets.panel_evaluation(preset)
            default = presets.default_selection()
            manifest = scoring.task_manifest(StoragePath(evaluation.tasks(default, default).path()))
            if set(rates["task_id"].to_pylist()) != set(manifest):
                raise InvalidComparison("pass-rate panel task IDs differ from the panel manifest")
        runs = [
            _with_requirements(_with_objectives(_load_run(path, preset, rates, inputs), preset), inputs.requirements)
            for path in run_paths
        ]
        if runs:
            fields = {"tree": "tree_hash", "base_commit": "base_commit", "dirty": "dirty"}
            metadata = metadata.model_copy(
                update={
                    "panel": runs[0].panel,
                    "engines": tuple(run.engine for run in runs),
                    **{field: runs[0].provenance[key] for field, key in fields.items()},
                }
            )
        invariants = [
            row.model_copy(update={"replicate": index})
            for index, candidate in enumerate(runs)
            for row in candidate.invariants
        ]
        grouped = {
            pin: [_with_objectives(_load_run(path, preset, rates), preset) for path in paths]
            for pin, paths in zip(pins, control_paths, strict=True)
        }
        controls = [candidate for group in grouped.values() for candidate in group]
        if not runs or not controls:
            raise InvalidComparison("comparison needs recorded runs and control replicates")
        control = control.model_copy(
            update={"panel": controls[0].panel, "engines": tuple(run.engine for run in controls)}
        )
        for pin, group in grouped.items():
            for candidate in group:
                _check_control(candidate, pin)
        if any(candidate.version != controls[0].version for candidate in (*runs, *controls)):
            raise InvalidComparison(
                "preset revisions differ, including effective evaluation sampling: "
                f"candidate={sorted({candidate.version for candidate in runs})}; "
                f"controls={sorted({candidate.version for candidate in controls})}"
            )
        if any(candidate.version != presets.PRESET_REVISION for candidate in runs):
            raise InvalidComparison("preset revision differs")
        if any(candidate.panel_ids != controls[0].panel_ids for candidate in (*runs, *controls)):
            raise InvalidComparison("held-out panel task IDs differ")
        if any(candidate.evaluation_identity != controls[0].evaluation_identity for candidate in (*runs, *controls)):
            raise InvalidComparison("evaluation protocol or held-out TaskSpec content differs")
        if any(candidate.missing_tokens for candidate in (*runs, *controls)):
            raise InvalidComparison("mismatch guard has missing or zero selected-token counts")
        comparisons = _compare(preset, runs, grouped, inputs.requirements)
        guards = comparisons.guards
        reported = [*comparisons.declared, *comparisons.details]
        objectives = _objectives(preset, runs, grouped, guards)
        objective_keys = {metric.key for metric in objective_metrics(preset)}
        constraints = [row for row in guards if row.metric not in objective_keys]
        decision = _verdict(invariants, guards, comparisons.declared)
        verdict, label, reason = decision.verdict, decision.label, decision.reason
        try:
            baselines = _baseline_runs(inputs.baselines, preset, rates, controls[0])
            if baselines:
                baseline_rows = _compare(preset, runs, grouped, inputs.requirements, baselines=baselines)
                vs_baseline = (
                    [ReportedComparison(**row.model_dump(exclude={"passes"})) for row in baseline_rows.guards]
                    + baseline_rows.declared
                    + baseline_rows.details
                )
        except (InvalidComparison, SkyRLReceiptLookupError, FileNotFoundError, click.BadParameter) as error:
            baseline_reason = str(error)
    except (InvalidComparison, SkyRLReceiptLookupError, FileNotFoundError) as error:
        reason = str(error)
    report = GateReport(
        preset=preset_name,
        verdict=verdict,
        label=label,
        reason=reason,
        run=metadata,
        control=control,
        invariants=invariants,
        guards=guards,
        reported=reported,
        objectives=objectives,
        constraints=constraints,
        vs_baseline=vs_baseline,
        vs_baseline_reason=baseline_reason,
        gaps=[gap.value for gap in preset.gaps],
    )
    for name, value in (
        (
            GATE_REPORT_FILENAME,
            json.dumps(report.model_dump(mode="json", exclude={"path"}), indent=2, allow_nan=False) + "\n",
        ),
        ("gate_report.md", _render_markdown(report)),
    ):
        fs, key = filesystem_for(str(StoragePath(output) / name))
        fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
        with fs.open(key, "wt") as stream:
            stream.write(value)
    return report


def control_pins(preset: str) -> tuple[ControlPin, ...]:
    control_preset = SHARED_CONTROLS.get(preset, preset)
    known = CONTROLS.get(control_preset, ())
    pins = tuple(pin for pin in known if pin.revision == presets.PRESET_REVISION)
    if not pins:
        name, replicas = (known[0].preset, known[0].runs) if known else (control_preset, 4)
        revisions = sorted({pin.revision for pin in known})
        raise InvalidComparison(
            f"no control pin for {preset} at {presets.PRESET_REVISION}; recorded control revisions={revisions}; "
            f"collect with python -m experiments.post_training.snowball_rl --preset {name} "
            f"--version {presets.PRESET_REVISION} --control --replicates {replicas} --run"
        )
    return pins


def control_path(preset: presets.Preset, pin: ControlPin, index: int) -> str:
    return presets.run_path(
        preset.recipe.layout.cluster, presets.control_name(pin.preset, pin.main_commit, index), pin.revision
    )


class PassRatesInput(NamedTuple):
    identity: str
    path: str


def _pass_rates_input(preset: presets.Preset) -> PassRatesInput | None:
    for evaluation in preset.evaluations:
        if (
            not any(presets.guards_evaluation(evaluation, metric) for metric in preset.guards)
            or evaluation.file == scoring.TRAIN_STEM
        ):
            continue
        default = presets.default_selection()
        selection = evaluation.tasks(default, default)
        reference = presets.selection_summary(selection)["recipe"]["pass_rates"]
        return PassRatesInput(f"{selection.name}@{selection.version}", str(StoragePath(reference["uri"]).parent))
    return None


def _pass_rates_path(preset: presets.Preset) -> str | None:
    source = _pass_rates_input(preset)
    return source.path if source is not None else None


class ReceiptEvidence(NamedTuple):
    path: str | None
    native_state: str | None
    attempt_id: str | None


def receipt_evidence(path: str | None) -> ReceiptEvidence:
    """Hash payload (path, native state, attempt ID); unresolved lookups share one stable state."""
    if path is None:
        return ReceiptEvidence(None, None, None)
    try:
        receipt = read_skyrl_launch_receipt(path)
        return (
            ReceiptEvidence(path, receipt.result.state, receipt.config.get("run", {}).get("attempt_id"))
            if receipt
            else ReceiptEvidence(path, None, None)
        )
    except (FileNotFoundError, SkyRLReceiptLookupError):
        return ReceiptEvidence(path, "unresolved", None)


def _gate_artifact(
    name: str,
    preset: presets.Preset,
    candidates: tuple[str, ...],
    inputs: GateInputs,
    leave_out: int | None = None,
    *,
    run_paths: tuple[str, ...],
) -> ArtifactStep[GateReport]:
    pins = control_pins(preset.name)
    controls = tuple(
        tuple(
            (
                f"{presets.control_name(pin.preset, pin.main_commit, index)}@{pin.revision}",
                control_path(preset, pin, index),
            )
            for index in range(pin.runs)
            if pin != pins[0] or index != leave_out
        )
        for pin in pins
    )
    rates = _pass_rates_input(preset)
    name += "-req-" + fingerprint_hash(canonical_json(inputs.requirements)) if inputs.requirements else ""
    name += "-base-" + fingerprint_hash(canonical_json(inputs.baselines)) if inputs.baselines else ""
    attempts = [
        receipt_evidence(path)._replace(path=identity) for identity, path in zip(candidates, run_paths, strict=True)
    ]
    name += "-evidence-" + fingerprint_hash(canonical_json(attempts))
    step = apply(
        name,
        _write_gate,
        version=GATE_REVISION,
        artifact_type=GateReport,
        preset_name=preset.name,
        run_paths=candidates,
        control_paths=tuple(tuple(identity for identity, _ in group) for group in controls),
        pins=pins,
        inputs=inputs,
        pass_rates_path=rates.identity if rates is not None else None,
        output=OUT,
    )

    def build_config(ctx: StepContext) -> dict:
        config = step.build_config(ctx)
        if not ctx.is_fingerprint:
            config.update(ctx.runtime_arg("paths"))
        return config

    return replace(
        step,
        build_config=build_config,
        runtime_args={
            "paths": {
                "run_paths": run_paths,
                "control_paths": tuple(tuple(path for _, path in group) for group in controls),
                "pass_rates_path": rates.path if rates is not None else None,
            }
        },
    )


def gate_step(
    preset: presets.Preset, runs: tuple[ArtifactStep[SkyRLRun], ...], *, inputs: GateInputs
) -> ArtifactStep[GateReport]:
    pins = control_pins(preset.name)
    if inputs.baselines:
        try:
            _baseline_runs(inputs.baselines, preset, *_first_control(preset, pins))
        except (InvalidComparison, SkyRLReceiptLookupError, FileNotFoundError) as error:
            raise click.BadParameter(str(error), param_hint="--baseline") from error
    name = runs[0].name.rsplit("-r", 1)[0] + f"/gate-n{len(runs)}-{pins[0].main_commit}"
    paths = tuple(candidate.path(presets.artifact_prefix(preset.recipe.layout.cluster)) for candidate in runs)
    identities = tuple(f"{candidate.name}@{candidate.version}" for candidate in runs)
    return replace(_gate_artifact(name, preset, identities, inputs, run_paths=paths), deps=runs)


def _recorded_run(path: str) -> tuple[str | None, GateInputs]:
    try:
        receipt = read_skyrl_launch_receipt(path)
    except SkyRLReceiptLookupError as error:
        raise InvalidComparison(f"cannot recover recorded command options for {path}: {error}") from error
    if receipt is None:
        raise InvalidComparison(f"no launch receipt at {path}")
    options = recorded_run_options((receipt.result.provenance or {}).get("command_line", []))
    return options.preset_name, options.inputs


class RecordedRunOptions(BaseModel):
    preset_name: str | None
    inputs: GateInputs
    resume_from: str | None


def recorded_run_options(argv: Sequence[str]) -> RecordedRunOptions:
    """Parse stored launcher arguments into the gate inputs and preset name."""
    context = presets.run_command()(lambda **kwargs: ()).make_context(
        "snowball", list(argv), resilient_parsing=True, ignore_unknown_options=True, allow_extra_args=True
    )
    params = context.params
    settings = tuple(
        leaf for text in params.get("settings", ()) for leaf in presets.setting_leaves(presets.parse_setting(text))
    )
    return RecordedRunOptions(
        preset_name=params.get("preset_name"),
        inputs=gate_inputs(
            settings,
            params.get("tasks_name"),
            params.get("init_name"),
            params.get("requirements", ()),
            params.get("baselines", ()),
        ),
        resume_from=params.get("resume_name"),
    )


def _baseline_runs(
    references: tuple[str, ...],
    preset: presets.Preset,
    rates: pa.Table | None,
    control: RunData,
) -> list[RunData]:
    runs = {}
    for reference in references:
        artifact = presets.published_artifact(reference)
        path = artifact.path()
        if path in runs:
            continue
        preset_name, recorded = _recorded_run(path)
        if preset_name != preset.name:
            raise InvalidComparison(f"baseline is not a recorded {preset.name} run: {reference}")
        inputs = recorded.model_copy(update={"requirements": (), "baselines": ()})
        baseline = _with_objectives(_load_run(path, preset, rates, inputs), preset)
        if baseline.version != presets.PRESET_REVISION:
            raise InvalidComparison(f"baseline preset revision differs: {reference}")
        if baseline.panel_ids != control.panel_ids:
            raise InvalidComparison(f"baseline held-out panel task IDs differ: {reference}")
        if baseline.evaluation_identity != control.evaluation_identity:
            raise InvalidComparison(f"baseline evaluation protocol or held-out TaskSpec content differs: {reference}")
        if baseline.missing_tokens or any(not row.holds for row in baseline.invariants):
            raise InvalidComparison(f"baseline has missing tokens or failed invariants: {reference}")
        runs[path] = baseline
    return list(runs.values())


@click.command(cls=presets.SnowballCommand, help=__doc__)
@click.option(
    "--preset",
    "preset_name",
    type=click.Choice(sorted(name for name, preset in presets.PRESETS.items() if presets.gated(preset))),
    required=True,
    help="Gated preset whose pinned controls to compare with.",
)
@click.option(
    "--candidate",
    "candidate_name",
    metavar="NAME[@VERSION]",
    help="Saved run to re-score, by the name and version the status block prints for it; VERSION defaults "
    "to PRESET_REVISION. Give this or --leave-out.",
)
@click.option(
    "--leave-out",
    "leave_out",
    metavar="rN",
    help="Score control replicate N against the remaining controls. Each replicate should PASS; use it "
    "after pinning new controls. Give this or --candidate.",
)
@click.option(
    "--require",
    "requirements",
    multiple=True,
    metavar="KEY<=|>=BOUND",
    help="Add a condition, as in snowball_rl --require, to those the run recorded at launch. Repeatable.",
)
@click.option(
    "--baseline",
    "baselines",
    multiple=True,
    metavar="NAME@VERSION",
    help="Compare with earlier finished runs; the verdict still uses pinned controls. Repeatable.",
)
def main(
    preset_name: str,
    candidate_name: str | None,
    leave_out: str | None,
    requirements: tuple[str, ...],
    baselines: tuple[str, ...],
) -> None:
    if (candidate_name is None) == (leave_out is None):
        raise click.UsageError("choose --candidate NAME[@VERSION] or --leave-out r<index>")
    preset = presets.PRESETS[preset_name]
    try:
        pins = control_pins(preset_name)
        index = None
        if leave_out is not None:
            if not re.fullmatch(r"r\d+", leave_out) or int(leave_out[1:]) >= pins[0].runs:
                raise click.BadParameter(f"choose r0 through r{pins[0].runs - 1}", param_hint="--leave-out")
            index = int(leave_out[1:])
            candidate_name = presets.control_name(pins[0].preset, pins[0].main_commit, index)
        name, separator, version = candidate_name.partition("@")
        source = presets.run_path(preset.recipe.layout.cluster, name, version if separator else presets.PRESET_REVISION)
        namespace = user_owned_name(
            f"{presets.RUN_PREFIX}/rescore/{preset_name}-{fingerprint_hash(canonical_json((source, pins, index)))}"
        )
        _, recorded = _recorded_run(source)
        inputs = recorded.model_copy(
            update={
                "requirements": (*recorded.requirements, *requirements),
                "baselines": tuple(sorted({*recorded.baselines, *baselines})),
            }
        )
        for text in inputs.requirements:
            parse_requirement(text)
        step = _gate_artifact(
            f"{namespace}/gate-n1-{pins[0].main_commit}",
            preset,
            (f"{name}@{version if separator else presets.PRESET_REVISION}",),
            inputs,
            index,
            run_paths=(source,),
        )
    except InvalidComparison as error:
        click.echo(json.dumps({"preset": preset_name, "verdict": "invalid", "reason": str(error)}))
        raise click.exceptions.Exit(presets.REFUSAL_EXIT_CODE) from error
    report = run(step)[0]
    payload = {
        "report": str(StoragePath(report.path) / GATE_REPORT_FILENAME),
        **report.model_dump(mode="json", exclude={"path"}),
    }
    click.echo(json.dumps(payload))
    raise click.exceptions.Exit(presets.EXIT_CODES[report.verdict])


def completed_gate(
    preset: presets.Preset, runs: Sequence[ArtifactStep], gate: ArtifactStep | None
) -> ArtifactStep | None:
    """Bind a gate to the current receipts of its completed native runs."""
    if gate is None:
        return None
    config = gate.build_config(StepContext.for_fingerprint(gate.runtime_args, gate.deps))
    return gate_step(preset, runs, inputs=config["inputs"])


if __name__ == "__main__":
    # The importable module owns the persisted GateReport type.
    importlib.import_module("experiments.post_training.snowball_rl_gate").main()
