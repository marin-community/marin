# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a saved Snowball run, or one control replicate, with the preset's pinned controls.

Checks saved measurements against current preset defaults and pinned controls on CPU. Writes gate_report.json and
gate_report.md under the caller's namespace and prints the report as JSON. snowball_rl runs this
comparison itself after a gated run; use this command to re-score a finished run, for example
with a new --require, or to check calibration by scoring a control against the other controls.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl_gate`:
  --preset learning-medium --candidate RUN_NAME --require 'eval_lite/aaii_category/coding/mean>=control'
  --preset speed-medium --leave-out r0
"""

from __future__ import annotations

import importlib
import json
import math
import re
import statistics
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.compute as pc
from marin.execution.artifact import Artifact, read_record
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, run
from marin.experiment.namespacing import user_owned_name
from marin.rl.rl_progress import Change, Comparison, ComparisonRole, Invariant, Label, Verdict
from marin.rl.skyrl import (
    MISSING_CONFIG_VALUE,
    SkyRLReceiptLookupError,
    SkyRLRun,
    declared_config_value,
    read_skyrl_launch_receipt,
    set_config_value,
    skyrl_cpu_command,
)
from marin.rl.skyrl_records import attempt_tracking, read_skyrl_metrics, read_stored_skyrl_receipt
from pydantic import BaseModel
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import Provenance
from scipy.stats import t
from tabulate import tabulate
from taskcompendium.grading_result import Outcome

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_recipe as recipe
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.snowball_rl_scoring import InvalidComparison

GATE_REVISION = "2026.11.10"
GATE_INPUTS_FILENAME = "gate_inputs.json"
GATE_REPORT_FILENAME = "gate_report.json"
GATE_MARKDOWN_FILENAME = "gate_report.md"
GRADER_ERROR_LIMIT = 0.01
OUTCOME_PREFIX = "generate/outcome/"


@dataclass(frozen=True)
class ControlPin:
    preset: str
    commit: str
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

SHARED_CONTROLS = {"data-medium": "learning-medium"}


REPORTED_PREFIXES = ("async/", "consumed/", "generate/", "loss/", "policy/", "reward/", "startup/", "timing/", "vllm/")


EXIT_CODES = {Verdict.PASS: 0, Verdict.FAIL: 1, Verdict.INVALID: 2, None: 3}
REFUSAL_EXIT_CODE = 4


class GateInputs(BaseModel, frozen=True):
    preset_name: str
    settings: tuple[str, ...] = ()
    tasks: str | None = None
    init: str | None = None
    requirements: tuple[str, ...] = ()


def read_gate_inputs(run_output: str) -> GateInputs:
    """Read the typed Snowball inputs saved beside a native launch receipt."""
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


def validate_requirements(preset: presets.Preset, requirements: Sequence[str], *, control: bool) -> None:
    """Validate a candidate's control pin and user requirements before GPU submission."""
    gated = presets.gated(preset)
    if requirements and (control or not gated):
        raise click.BadParameter("--require needs a gated candidate preset, without --control")
    keys = [parse_requirement(text).key for text in requirements]
    pins = control_pins(preset) if not control and gated else ()
    if not requirements:
        return
    reference = _with_objectives(_load_run(control_path(preset, pins[0], 0), preset), preset)
    missing = [key for key in keys if key not in reference.values]
    if missing:
        raise click.BadParameter(f"requirement metrics that the gate does not compute for control r0: {missing}")


def validate_candidate(preset: presets.Preset) -> None:
    """Refuse stale pinned control configurations before submitting a candidate."""
    pins = control_pins(preset)
    defaults, _ = _current_configs(preset, GateInputs(preset_name=preset.name))
    current = comparison_config(defaults)
    for pin in pins:
        for index in range(pin.runs):
            path = control_path(preset, pin, index)
            receipt = read_stored_skyrl_receipt(path)
            if receipt is None:
                raise InvalidComparison(f"control receipt is unavailable: {path}")
            _check_control_config(receipt.native.config, current)


def gate_inputs(
    preset_name: str,
    settings: Sequence[presets.Setting] = (),
    tasks: str | None = None,
    init: str | None = None,
    requirements: Sequence[str] = (),
) -> GateInputs:
    values = {setting.key: setting for setting in settings}
    return GateInputs(
        preset_name=preset_name,
        settings=tuple(
            f"{'+' if setting.allows_new_key else ''}{key}={json.dumps(setting.value, default=str)}"
            for key, setting in sorted(values.items())
        ),
        tasks=tasks.removeprefix(presets.SELECTION_PREFIX + "/") if tasks else None,
        init=init,
        requirements=tuple(dict.fromkeys(requirements)),
    )


class RunMetadata(GateInputs):
    tree: str | None = None
    base_commit: str | None = None
    dirty: bool | None = None
    replicates: int
    panel: str | None = None
    engines: tuple[dict, ...] = ()
    attempt_ids: tuple[str, ...] = ()
    config_changes: tuple[str, ...] = ()


class ControlMetadata(BaseModel):
    commit: str
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
    invariants: tuple[Invariant, ...]
    comparisons: tuple[Comparison, ...]
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
    key: str,
    role: ComparisonRole,
    rows: Sequence[Mapping[str, float]],
    noise: MetricNoise | None,
    n_control: int,
    count: int = 1,
    *,
    higher_is_better: bool | None = None,
) -> Comparison:
    values = [row.get(key, math.nan) for row in rows]
    value = statistics.mean(values) if values and all(math.isfinite(item) for item in values) else None
    control = noise.mean if noise else None
    delta = value - control if value is not None and control is not None else None
    tolerance = (
        float(t.ppf(1 - 0.025 / count, noise.df)) * noise.sd * math.sqrt(1 / len(values) + 1 / n_control)
        if delta is not None and noise.sd > 0
        else None
    )
    change = Change.UNAVAILABLE
    if tolerance is not None:
        if abs(delta) <= tolerance:
            change = Change.SAME
        elif higher_is_better is None:
            change = Change.CHANGED
        else:
            change = Change.BETTER if (delta > 0) == higher_is_better else Change.WORSE
    return Comparison(
        metric=key,
        role=role,
        higher_is_better=higher_is_better,
        run=value,
        control=control,
        delta=delta,
        tolerance=tolerance,
        sd=noise.sd if noise else None,
        df=noise.df if noise else None,
        n_run=len(values),
        n_control=n_control,
        change=change,
        delta_percent=100 * delta / abs(control) if delta is not None and control else None,
        noise_percent=100 * tolerance / abs(control) if tolerance is not None and control else None,
    )


def _verdict(invariants: Sequence[Invariant], comparisons: Sequence[Comparison]) -> tuple[Verdict, Label]:
    verdict = (
        Verdict.FAIL
        if any(not row.holds for row in invariants) or any(row.passes is False for row in comparisons)
        else Verdict.PASS
    )
    relevant = [row for row in comparisons if row.role in (ComparisonRole.OBJECTIVE, ComparisonRole.CONSTRAINT)]
    better = any(row.change == Change.BETTER for row in relevant)
    worse = any(row.change == Change.WORSE for row in relevant)
    label = (
        Label.TRADE_OFF
        if better and worse
        else Label.IMPROVES if better else Label.REGRESSION if worse else Label.NO_CHANGE
    )
    return verdict, label


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
        window = declared.get(key, presets.WindowStatistic.MEAN)
        if key.startswith("startup/"):
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
        if key == presets.UPDATE_SECONDS_METRIC:
            series = [math.fsum(row[key] for key in phases) for row in rows if phases and set(phases) <= set(row)]
        else:
            series = [row[key] for row in rows if key in row]
        if series and len(series) == len(rows):
            reducer = statistics.median if window == presets.WindowStatistic.MEDIAN else statistics.mean
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
    if any(metric.key.startswith(evaluation.name + "/") for metric in preset.guards):
        coverage = len({row["item"] for row in rows if row["scored"]}) / (expected // evaluation.answers)
        result.append(
            Invariant(
                name=f"{name}/panel_scored_fraction",
                value=coverage,
                limit=presets.MINIMUM_PANEL_COVERAGE,
                comparator=">=",
                holds=coverage >= presets.MINIMUM_PANEL_COVERAGE,
            )
        )
    return result


@dataclass(frozen=True)
class RunData:
    config: dict
    path: str
    version: str
    panel: str | None
    provenance: dict
    values: dict[str, float]
    panel_ids: dict[str, frozenset[str]]
    invariants: tuple[Invariant, ...]
    missing_tokens: bool
    slices: dict[str, int]
    evaluation_identity: dict[str, tuple[str | None, dict]]
    h100_hours: float | None
    engine: dict
    attempt_id: str


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
    default = presets.default_selection(preset)
    selection = presets.selection_artifact(inputs.tasks) if inputs.tasks else default
    selected = evaluation.tasks(selection, default, preset.eval_lite)
    panel = f"{selected.name.removeprefix(presets.SELECTION_PREFIX + '/')}@{selected.version}"
    binding = selected
    if evaluation.sample is not None:
        binding = presets.sampled_tasks_step(
            selected,
            evaluation.file,
            evaluation.sample,
            launch["skyrl"]["trainer"]["seed"],
            revision=preset.revision,
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
        Invariant(
            name="planned_updates", value=len(completed), limit=len(planned), comparator="==", holds=completed == planned
        )
    ]
    if not generating:
        finite = all(
            all(
                math.isfinite(row.get(key, math.nan))
                for key in ("policy/raw_grad_norm", "policy/policy_loss", "policy/final_loss")
            )
            and any(key.startswith("loss/") for key in row)
            and all(math.isfinite(row[key]) for key in row if key.startswith("loss/") or key == "policy/raw_grad_norm")
            for row in training
        )
        invariants.append(
            Invariant(name="finite_losses_and_gradients", value=finite, limit=True, comparator="==", holds=finite)
        )

    return tuple(invariants)


def _load_run(
    path: str,
    preset: presets.Preset,
    inputs: GateInputs | None = None,
) -> RunData:
    inputs = inputs or GateInputs(preset_name=preset.name)
    terminal = read_skyrl_launch_receipt(path)
    if terminal is None:
        raise FileNotFoundError(f"no launch receipt at {path}")
    if terminal.native.result.state != "succeeded":
        raise InvalidComparison(f"native job {terminal.native.result.iris_job_id} is {terminal.native.result.state}")
    if terminal.native.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {path}")
    metrics = read_skyrl_metrics(path)
    generating = preset.recipe.layout.role_plan.generate_only
    training = [
        row for step, row in sorted(metrics.items()) if (step == 0 if generating else 1 <= step <= preset.updates)
    ]
    invariants = list(training_invariants(preset, metrics))
    answers = scoring.evaluation_answers(path) if preset.evaluations else []
    launch = terminal.native.config
    tracking = attempt_tracking(terminal)
    try:
        recorded_panels = {
            evaluation.name: scoring.read_recorded_panel(launch, evaluation.name) for evaluation in preset.evaluations
        }
        identities = {
            evaluation.name: (
                (recorded_panels[evaluation.name].task_set_sha256 if evaluation.file != scoring.TRAIN_STEM else None),
                scoring.evaluation_protocol(launch, evaluation.name),
            )
            for evaluation in preset.evaluations
        }
    except (KeyError, TypeError, ValueError) as error:
        raise InvalidComparison(f"recorded evaluation identity is malformed: {error}") from error
    panel = _panel_identity(preset, launch, inputs)
    scores, initial_scores, panel_ids = [scoring.EVAL_SCORE_SCHEMA.empty_table()], [], {}
    default = presets.default_selection(preset)
    selection = presets.selection_artifact(inputs.tasks) if inputs.tasks else default
    for evaluation in preset.evaluations:
        quality = evaluation.file != scoring.TRAIN_STEM and any(
            metric.key.startswith(evaluation.name + "/") for metric in preset.guards
        )
        expected = presets.evaluation_answer_count(evaluation, evaluation.tasks(selection, default, preset.eval_lite))
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
            invariants.extend(
                _session_invariants(
                    session_name, session, evaluation, preset, expected, recorded_panels[evaluation.name].task_ids
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
    prefix = "eval/all/outcome/" if generating else OUTCOME_PREFIX
    invariants.append(_grader_errors(training, prefix, "grader_error_fraction"))
    fixed = preset.recipe.fixed_reply_tokens
    if fixed is not None:
        if generating:
            holds = {len(row["response_token_ids"]) for _, _, row in answers} == {fixed}
        else:
            holds = all(
                row.get("generate/min_num_tokens") == row.get("generate/max_num_tokens") == fixed for row in training
            )
        invariants.append(Invariant(name="fixed_reply_tokens", value=holds, limit=True, comparator="==", holds=holds))
    missing = any(
        metric.key.startswith("policy/mismatch/")
        and _mismatch_counts([metrics.get(step, {}) for step in preset.measured], metric.key.rsplit("/", 1)[0]) is None
        for metric in preset.guards
    )
    final = pa.concat_tables(scores)
    values = {
        **_window_values(preset, metrics),
        **scoring.evaluation_statistics(final),
    }
    for key, value in scoring.evaluation_statistics(pa.concat_tables([scores[0], *initial_scores])).items():
        values[key + "/initial"] = value
        if key in values:
            values[key + "/gain"] = values[key] - value
    result = terminal.native.result
    h100_hours = None
    if result.started_at and result.finished_at:
        seconds = (presets.utc_millis(result.finished_at) - presets.utc_millis(result.started_at)) / 1000
        if seconds > 0:
            allocation = launch["iris"]["allocation"]
            h100_hours = seconds * allocation["num_nodes"] * allocation["gpus_per_node"] / 3600
    slices = {key: len({row["item"] for row in rows}) for key, rows in scoring.score_slices(final).items()}
    record = read_record(path)
    if record is None:
        raise FileNotFoundError(f"no artifact record at {path}")
    return RunData(
        config=launch,
        path=path,
        version=record.version,
        panel=panel,
        provenance=terminal.native.result.provenance,
        values=values,
        panel_ids=panel_ids,
        invariants=tuple(invariants),
        missing_tokens=missing,
        slices=slices,
        evaluation_identity=identities,
        h100_hours=h100_hours,
        engine=tracking.get("engine", {}),
        attempt_id=terminal.attempt_id,
    )


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
                Invariant(
                    name=f"require/{requirement.key}",
                    value=value,
                    limit=requirement.bound,
                    comparator=">=" if requirement.higher_is_better else "<=",
                    holds=holds,
                )
            )
    return replace(candidate, invariants=tuple(invariants))


def _check_control(run: RunData, pin: ControlPin) -> None:
    if run.provenance["base_commit"] != pin.commit or run.provenance["dirty"]:
        raise InvalidComparison(f"control does not match its pin: {run.path}")
    if run.missing_tokens or any(not row.holds for row in run.invariants):
        raise InvalidComparison(f"control has missing mismatch tokens or failed invariants: {run.path}")


def _compare(
    preset: presets.Preset,
    runs: Sequence[RunData],
    controls: Mapping[ControlPin, Sequence[RunData]],
    requirements: Sequence[str] = (),
) -> list[Comparison]:
    noise = calibrate({pin: [run.values for run in group] for pin, group in controls.items()})
    values = [run.values for run in runs]
    n_control = sum(map(len, controls.values()))
    objectives = {metric.key: metric for metric in preset.objectives}
    rows = []
    for metric in preset.guards:
        role = ComparisonRole.OBJECTIVE if metric.key in objectives else ComparisonRole.CONSTRAINT
        row = _comparison(
            metric.key,
            role,
            values,
            noise.get(metric.key),
            n_control,
            len(preset.guards),
            higher_is_better=metric.higher_is_better,
        )
        if row.delta is None:
            raise InvalidComparison(f"missing or non-finite guard metric: {metric.key}")
        if row.sd == 0:
            raise InvalidComparison(
                f"control variation is zero for guard metric: {metric.key}; calibration is unavailable"
            )
        signed = row.delta * (1 if metric.higher_is_better else -1)
        rows.append(row.model_copy(update={"passes": signed >= -row.tolerance}))
    built_in = {(metric.key, metric.higher_is_better) for metric in preset.guards}
    for text in dict.fromkeys(requirements):
        requirement = parse_requirement(text)
        if requirement.bound is not None or (requirement.key, requirement.higher_is_better) in built_in:
            continue
        row = _comparison(
            requirement.key,
            ComparisonRole.REQUIREMENT,
            values,
            noise.get(requirement.key),
            n_control,
            higher_is_better=requirement.higher_is_better,
        )
        if row.delta is None or row.tolerance is None:
            raise InvalidComparison(
                f"missing or zero-variation requirement metric: {requirement.key}; use a number bound"
            )
        signed = row.delta * (1 if requirement.higher_is_better else -1)
        rows.append(row.model_copy(update={"passes": signed >= -row.tolerance}))
    guard_keys = {metric.key for metric in preset.guards}
    for metric in preset.objectives:
        if metric.key not in guard_keys:
            rows.append(
                _comparison(
                    metric.key,
                    ComparisonRole.OBJECTIVE,
                    values,
                    noise.get(metric.key),
                    n_control,
                    len(preset.objectives),
                    higher_is_better=metric.higher_is_better,
                )
            )
    selected = {row.metric for row in rows}
    prefixes = tuple(evaluation.name + "/" for evaluation in preset.evaluations)
    for key in sorted(set().union(*values, noise) - selected):
        if not key.startswith((*prefixes, *REPORTED_PREFIXES)):
            continue
        role = ComparisonRole.SLICE if key.startswith(prefixes) else ComparisonRole.REPORTED
        row = _comparison(key, role, values, noise.get(key), n_control)
        reference = [run.values[key] for group in controls.values() for run in group] if key in noise else []
        sizes = {run.slices.get(key) for run in runs}
        row = row.model_copy(
            update={
                "control_min": min(reference) if reference else None,
                "control_max": max(reference) if reference else None,
                "tasks": sizes.pop() if len(sizes) == 1 else None,
            }
        )
        if role == ComparisonRole.SLICE:
            row = row.model_copy(update={"tolerance": None, "noise_percent": None, "change": None})
        rows.append(row)
    return rows


def _render_markdown(report: GateReport) -> str:
    lines = [f"# {report.preset}"]
    for title, rows in (
        (
            "Objectives",
            [
                row.model_dump(
                    include={"metric", "control", "run", "delta_percent", "noise_percent", "passes", "change"}
                )
                for row in report.comparisons
                if row.role == ComparisonRole.OBJECTIVE
            ],
        ),
        (
            "Constraints",
            [
                row.model_dump(include={"metric", "control", "run", "passes"})
                for row in report.comparisons
                if row.role in (ComparisonRole.CONSTRAINT, ComparisonRole.REQUIREMENT)
            ],
        ),
        ("Invariants", [row.model_dump() for row in report.invariants]),
        (
            "Reported",
            [
                row.model_dump(include={"metric", "control", "run", "delta", "change"})
                for row in report.comparisons
                if row.role == ComparisonRole.REPORTED
            ],
        ),
        (
            "Breakdown",
            [
                row.model_dump(include={"metric", "tasks", "control", "control_min", "control_max", "run"})
                for row in report.comparisons
                if row.role == ComparisonRole.SLICE
            ],
        ),
    ):
        lines += ["", f"## {title}", "", tabulate(rows, "keys", "github", floatfmt=".8g", missingval="unavailable")]
    lines += ["", f"Result: {report.verdict}" + (f" ({report.label})" if report.label else "")]
    if report.reason:
        lines += ["", report.reason]
    for title, value in (("Run", report.run), ("Control", report.control)):
        lines += [
            "",
            f"## {title}",
            "",
            *[f"- {key}: {json.dumps(item, default=str)}" for key, item in value.model_dump().items()],
        ]
    return "\n".join([*lines, "", "## Gaps", "", *[f"- {gap}" for gap in report.gaps], ""])


def _config_values(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, (dict, list)) and value:
        items = value.items() if isinstance(value, dict) else enumerate(value)
        return {
            path: leaf
            for key, child in items
            for path, leaf in _config_values(child, f"{prefix}.{key}".lstrip(".")).items()
        }
    return {prefix: value}


def comparison_config(launch: dict) -> dict[str, object]:
    """Compare complete native configuration with declared runtime aliases resolved."""
    config = deepcopy(launch)
    inputs = launch["inputs"]

    def replace_alias(path, aliases, replacement):
        current = declared_config_value(config, path)
        if any(current == alias for alias in aliases):
            set_config_value(config, path, replacement)

    def sources_at(sources, prefix):
        normalized = []
        for index, source in enumerate(sources):
            copied = dict(source)
            copied["local_path"] = f"<staging:{prefix}.{index}>"
            normalized.append(copied)
        return normalized

    def bound_data(values, sources, normalized, path):
        if not isinstance(values, list) or len(values) != len(sources):
            raise InvalidComparison(f"native data binding differs from declared inputs: {path}")
        for value, source in zip(values, sources, strict=True):
            aliases = [prefix_join(source[key], source["relative_path"]) for key in ("uri", "local_path")]
            if value != source and value not in aliases:
                raise InvalidComparison(f"native data binding differs from declared inputs: {path}")
        return normalized

    for name, path in (("train_data", "skyrl.data.train_data"), ("validation_data", "skyrl.data.val_data")):
        sources = inputs[name]
        normalized = sources_at(sources, f"inputs.{name}")
        config["inputs"][name] = normalized
        if sources:
            set_config_value(config, path, bound_data(declared_config_value(config, path), sources, normalized, path))
        else:
            filename = "train.parquet" if name == "train_data" else "validation.parquet"
            aliases = [[str(Path(home) / "data/gsm8k" / filename)] for home in (str(Path.home()), "/root")]
            replace_alias(path, aliases, [f"${{oc.env:HOME}}/data/gsm8k/{filename}"])
    for name, sources in inputs.get("evaluation_data", {}).items():
        normalized = sources_at(sources, f"inputs.evaluation_data.{name}")
        config["inputs"]["evaluation_data"][name] = normalized
        for index, callback in enumerate(config["skyrl"]["trainer"]["callbacks"]):
            if name in callback.get("additional_evaluations", {}):
                profile = callback["additional_evaluations"][name]
                profile["data_files"] = (
                    normalized
                    if profile.get("data_files") is None
                    else bound_data(
                        profile["data_files"],
                        sources,
                        normalized,
                        f"skyrl.trainer.callbacks.{index}.additional_evaluations.{name}.data_files",
                    )
                )

    model = inputs["model"]
    aliases = [model["uri"], model["local_path"]]
    config["inputs"]["model"]["local_path"] = "<model-staging>"
    for path in (
        "skyrl.trainer.policy.model.path",
        "skyrl.trainer.ref.model.path",
        "skyrl.trainer.policy.model.tokenizer_path",
        "skyrl.trainer.ref.model.tokenizer_path",
        "skyrl.generator.model_name",
        "skyrl.generator.trajectory_retention.model_path",
    ):
        value = declared_config_value(config, path)
        if value is MISSING_CONFIG_VALUE:
            continue
        if value not in aliases:
            raise InvalidComparison(f"native model binding differs from declared inputs: {path}")
        set_config_value(config, path, "<model-staging>")
    replace_alias(
        "skyrl.generator.engine_init_kwargs.served_model_name",
        [uri.rstrip("/").rsplit("/", 1)[-1] for uri in aliases],
        "<served-model-name>",
    )
    for field in ("checkpoint_root", "export_root", "attempts_root", "resolved_config_uri", "terminal_manifest_uri"):
        config["artifacts"][field] = f"<output:{field}>"
    for path, field, suffix in (
        ("skyrl.trainer.ckpt_path", "checkpoint_root", ""),
        ("skyrl.trainer.export_path", "export_root", ""),
        ("ray.rendezvous_dir", "attempts_root", "rendezvous"),
        ("ray.log_dir", "attempts_root", "ray-logs"),
        ("skyrl.generator.trajectory_retention.output_path", "attempts_root", "trajectories"),
    ):
        root = launch["artifacts"][field]
        replace_alias(path, [prefix_join(root, suffix) if suffix else root], f"<output:{field}/{suffix}>")
    for path in (
        "run.id",
        "run.attempt_id",
        "run.coordinator_iris_job_id",
        "iris.job_name",
        "iris.wandb_entity",
    ):
        set_config_value(config, path, f"<identity:{path}>")
    replace_alias("skyrl.trainer.run_name", [launch["iris"]["job_name"]], "<launcher-run-name>")
    replace_alias(
        "skyrl.generator.trajectory_retention.run_id",
        [launch["run"]["id"], launch["iris"]["job_name"]],
        "<run-identity>",
    )
    runtime = config["skyrl"].get("runtime", {})
    for field in ("run_id", "attempt_id", "coordinator_iris_job_id", "iris_job_id"):
        runtime.pop(field, None)
    if not runtime:
        config["skyrl"].pop("runtime", None)
    return _config_values(config)


def _current_configs(preset: presets.Preset, inputs: GateInputs) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory(prefix="snowball-current-config-") as directory:
        root = Path(directory)
        paths = []
        for index, candidate in enumerate((GateInputs(preset_name=preset.name), inputs)):
            try:
                spec = recipe.run_spec(
                    preset,
                    name=f"current-config-{index}",
                    tasks=presets.selection_artifact(candidate.tasks) if candidate.tasks else None,
                    init=(
                        presets.published_artifact(candidate.init, revision=preset.revision) if candidate.init else None
                    ),
                    settings=tuple(map(presets.parse_setting, candidate.settings)),
                )
                step = recipe.run_step(preset, spec)
                context = StepContext.for_run(
                    str(root / f"run-{index}"),
                    presets.artifact_prefix(preset.recipe.layout.cluster),
                    runtime_args=step.runtime_args,
                    deps=step.deps,
                )
                path = root / f"config-{index}.yaml"
                path.write_text(step.build_config(context).launch_config_yaml)
                paths.append(str(path))
            except (ValueError, click.BadParameter) as error:
                raise InvalidComparison("recorded inputs cannot build the current preset") from error

        native_root = str(presets.REPO_ROOT / "vendor/skyrl")
        program = (
            "import json,sys\nfrom pathlib import Path\nfrom omegaconf import OmegaConf\n"
            "from cloud.iris import launch_config\nimport skyrl_train,marinskyrl\n"
            f"assert all(Path(module.__file__).resolve().is_relative_to({native_root!r}) "
            "for module in (launch_config,skyrl_train,marinskyrl))\n"
            "print(json.dumps([OmegaConf.to_container(launch_config.load_launch_config(path), resolve=True) "
            "for path in sys.argv[1:]]))\n"
        )
        try:
            result = subprocess.run(
                skyrl_cpu_command("python", "-c", program, *paths), capture_output=True, text=True, timeout=120
            )
            result.check_returncode()
            defaults, expected = json.loads(result.stdout.splitlines()[-1])
        except (subprocess.SubprocessError, ValueError, IndexError) as error:
            raise InvalidComparison("current preset recipe cannot be composed") from error
    return defaults, expected


def _changed_config_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {
        key
        for key in left.keys() | right.keys()
        if key not in left or key not in right or canonical_json(left[key]) != canonical_json(right[key])
    }


def _setting_keys(inputs: GateInputs, keys: set[str]) -> set[str]:
    prefixes = []
    for text in inputs.settings:
        for setting in presets.setting_leaves(presets.parse_setting(text)):
            prefix = "skyrl." + setting.key
            if setting.key.startswith("extra_env."):
                prefix = "runtime.task_env." + setting.key.removeprefix("extra_env.")
            prefixes.append(prefix)
    allowed = {key for key in keys if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)}
    return allowed


def _check_control_config(config: dict, current: dict[str, object]) -> None:
    differences = _changed_config_keys(comparison_config(config), current)
    if differences:
        raise InvalidComparison(f"control config differs from current preset defaults: {', '.join(sorted(differences))}")


def _write_gate(
    preset_name: str,
    run_paths: tuple[str, ...],
    control_paths: tuple[tuple[str, ...], ...],
    pins: tuple[ControlPin, ...],
    inputs: GateInputs,
    output: str,
) -> GateReport:
    preset = presets.PRESETS[preset_name]
    metadata = RunMetadata(**inputs.model_dump(), replicates=len(run_paths))
    control = ControlMetadata(commit=pins[0].commit, revision=pins[0].revision, runs=sum(map(len, control_paths)))
    invariants, comparisons = [], []
    verdict, label, reason = Verdict.INVALID, None, None
    try:
        runs = [
            _with_requirements(_with_objectives(_load_run(path, preset, inputs), preset), inputs.requirements)
            for path in run_paths
        ]
        if runs:
            fields = {"tree": "tree_hash", "base_commit": "base_commit", "dirty": "dirty"}
            metadata = metadata.model_copy(
                update={
                    "panel": runs[0].panel,
                    "engines": tuple(run.engine for run in runs),
                    "attempt_ids": tuple(run.attempt_id for run in runs),
                    **{field: runs[0].provenance[key] for field, key in fields.items()},
                }
            )
        invariants = [
            row.model_copy(update={"replicate": index})
            for index, candidate in enumerate(runs)
            for row in candidate.invariants
        ]
        grouped = {
            pin: [_with_objectives(_load_run(path, preset), preset) for path in paths]
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
        if any(candidate.version != preset.revision for candidate in runs):
            raise InvalidComparison("preset revision differs")
        if any(candidate.panel_ids != controls[0].panel_ids for candidate in (*runs, *controls)):
            raise InvalidComparison("held-out panel task IDs differ")
        if any(candidate.evaluation_identity != controls[0].evaluation_identity for candidate in (*runs, *controls)):
            raise InvalidComparison("evaluation protocol or held-out TaskSpec content differs")
        if any(candidate.missing_tokens for candidate in (*runs, *controls)):
            raise InvalidComparison("mismatch guard has missing or zero selected-token counts")
        current_config, expected_config = _current_configs(preset, inputs)
        current, expected = comparison_config(current_config), comparison_config(expected_config)
        changed = _changed_config_keys(current, expected)
        explicit = _setting_keys(inputs, set(current) | set(expected))
        metadata = metadata.model_copy(update={"config_changes": tuple(sorted(changed | explicit))})
        for candidate in runs:
            differences = _changed_config_keys(comparison_config(candidate.config), expected)
            if differences:
                raise InvalidComparison(
                    f"candidate config differs from recorded inputs: {', '.join(sorted(differences))}"
                )
        for candidate in controls:
            _check_control_config(candidate.config, current)
        comparisons = _compare(preset, runs, grouped, inputs.requirements)
        verdict, label = _verdict(invariants, comparisons)
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
        comparisons=comparisons,
        gaps=[gap.value for gap in preset.gaps],
    )
    for name, value in (
        (
            GATE_REPORT_FILENAME,
            json.dumps(report.model_dump(mode="json", exclude={"path"}), indent=2, allow_nan=False) + "\n",
        ),
        (GATE_MARKDOWN_FILENAME, _render_markdown(report)),
    ):
        fs, key = filesystem_for(str(StoragePath(output) / name))
        fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
        with fs.open(key, "wt") as stream:
            stream.write(value)
    return report


def control_pins(preset: presets.Preset) -> tuple[ControlPin, ...]:
    control_preset = SHARED_CONTROLS.get(preset.name, preset.name)
    known = CONTROLS.get(control_preset, ())
    pins = tuple(pin for pin in known if pin.revision == preset.revision)
    if not pins:
        name, replicas = (known[0].preset, known[0].runs) if known else (control_preset, 4)
        revisions = sorted({pin.revision for pin in known})
        raise InvalidComparison(
            f"no control pin for {preset.name} at {preset.revision}; recorded control revisions={revisions}; "
            f"collect with python -m experiments.post_training.snowball_rl --preset {name} "
            f"--version {preset.revision} --control --replicates {replicas} --run"
        )
    return pins


def control_path(preset: presets.Preset, pin: ControlPin, index: int) -> str:
    return presets.run_path(
        preset.recipe.layout.cluster, presets.control_name(pin.preset, pin.commit, index), pin.revision
    )


def _gate_artifact(
    name: str,
    preset: presets.Preset,
    candidates: tuple[str, ...],
    inputs: GateInputs,
    leave_out: int | None = None,
    *,
    run_paths: tuple[str, ...],
) -> ArtifactStep[GateReport]:
    pins = control_pins(preset)
    defaults = recipe.run_step(preset, recipe.run_spec(preset, name="current-defaults"))
    name += "-defaults-" + fingerprint_hash(
        canonical_json((defaults.fingerprint(), defaults.runtime_args, Provenance.capture(presets.REPO_ROOT).tree_hash))
    )
    controls = tuple(
        tuple(
            (
                f"{presets.control_name(pin.preset, pin.commit, index)}@{pin.revision}",
                control_path(preset, pin, index),
            )
            for index in range(pin.runs)
            if pin != pins[0] or index != leave_out
        )
        for pin in pins
    )
    name += "-req-" + fingerprint_hash(canonical_json(inputs.requirements)) if inputs.requirements else ""
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
        output=OUT,
    )

    def build_config(ctx: StepContext) -> dict:
        config = step.build_config(ctx)
        if not ctx.is_fingerprint:
            config.update(ctx.runtime_arg("paths"))
        return config

    return _gate_with_evidence(
        replace(
            step,
            build_config=build_config,
            runtime_args={
                "name": name,
                "paths": {
                    "run_paths": run_paths,
                    "control_paths": tuple(tuple(path for _, path in group) for group in controls),
                },
            },
        )
    )


def _gate_with_evidence(step: ArtifactStep[GateReport]) -> ArtifactStep[GateReport]:
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    candidates = config["run_paths"]
    run_paths = step.runtime_args["paths"]["run_paths"]
    attempts = []
    for identity, path in zip(candidates, run_paths, strict=True):
        try:
            receipt = read_skyrl_launch_receipt(path)
        except (FileNotFoundError, SkyRLReceiptLookupError):
            state, attempt = "unresolved", None
        else:
            state = receipt.native.result.state if receipt else None
            attempt = receipt.native.config["run"]["attempt_id"] if receipt else None
        attempts.append({"run": identity, "native_state": state, "attempt_id": attempt})
    return replace(step, name=step.runtime_args["name"] + "-evidence-" + fingerprint_hash(canonical_json(attempts)))


def gate_step(
    preset: presets.Preset, runs: tuple[ArtifactStep[SkyRLRun], ...], *, inputs: GateInputs
) -> ArtifactStep[GateReport]:
    pins = control_pins(preset)
    name = runs[0].name.rsplit("-r", 1)[0] + f"/gate-n{len(runs)}-{pins[0].commit}"
    paths = tuple(candidate.path(presets.artifact_prefix(preset.recipe.layout.cluster)) for candidate in runs)
    identities = tuple(f"{candidate.name}@{candidate.version}" for candidate in runs)
    return replace(_gate_artifact(name, preset, identities, inputs, run_paths=paths), deps=runs)


class SnowballCommand(click.Command):
    def make_context(self, *args, **kwargs):
        try:
            return super().make_context(*args, **kwargs)
        except click.UsageError as error:
            error.exit_code = REFUSAL_EXIT_CODE
            raise

    def invoke(self, context):
        configure_coreweave_s3()
        try:
            return super().invoke(context)
        except click.UsageError as error:
            error.exit_code = REFUSAL_EXIT_CODE
            raise


@click.command(cls=SnowballCommand, help=__doc__)
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
    "to the preset's revision. Give this or --leave-out.",
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
def main(
    preset_name: str,
    candidate_name: str | None,
    leave_out: str | None,
    requirements: tuple[str, ...],
) -> None:
    if (candidate_name is None) == (leave_out is None):
        raise click.UsageError("choose --candidate NAME[@VERSION] or --leave-out r<index>")
    preset = presets.PRESETS[preset_name]
    try:
        pins = control_pins(preset)
        index = None
        if leave_out is not None:
            if not re.fullmatch(r"r\d+", leave_out) or int(leave_out[1:]) >= pins[0].runs:
                raise click.BadParameter(f"choose r0 through r{pins[0].runs - 1}", param_hint="--leave-out")
            index = int(leave_out[1:])
            candidate_name = presets.control_name(pins[0].preset, pins[0].commit, index)
        name, separator, version = candidate_name.partition("@")
        source = presets.run_path(preset.recipe.layout.cluster, name, version if separator else preset.revision)
        namespace = user_owned_name(
            f"{presets.RUN_PREFIX}/rescore/{preset_name}-{fingerprint_hash(canonical_json((source, pins, index)))}"
        )
        recorded = GateInputs(preset_name=pins[0].preset) if index is not None else read_gate_inputs(source)
        if index is None and recorded.preset_name != preset_name:
            raise InvalidComparison(f"recorded preset is {recorded.preset_name}, not {preset_name}")
        inputs = recorded.model_copy(
            update={"requirements": tuple(dict.fromkeys((*recorded.requirements, *requirements)))}
        )
        for text in inputs.requirements:
            parse_requirement(text)
        step = _gate_artifact(
            f"{namespace}/gate-n1-{pins[0].commit}",
            preset,
            (f"{name}@{version if separator else preset.revision}",),
            inputs,
            index,
            run_paths=(source,),
        )
    except (InvalidComparison, FileNotFoundError) as error:
        click.echo(json.dumps({"preset": preset_name, "verdict": Verdict.INVALID.value, "reason": str(error)}))
        raise click.exceptions.Exit(REFUSAL_EXIT_CODE) from error
    report = run(step)[0]
    payload = {
        "report": str(StoragePath(report.path) / GATE_REPORT_FILENAME),
        **report.model_dump(mode="json", exclude={"path"}),
    }
    click.echo(json.dumps(payload))
    raise click.exceptions.Exit(EXIT_CODES[report.verdict])


def completed_gate(gate: ArtifactStep[GateReport] | None) -> ArtifactStep[GateReport] | None:
    """Bind current native attempt evidence to the gate identity fixed at submission."""
    return _gate_with_evidence(gate) if gate is not None else None


if __name__ == "__main__":
    # The importable module owns the persisted GateReport type.
    importlib.import_module("experiments.post_training.snowball_rl_gate").main()
