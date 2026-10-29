# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Snowball runs with pinned control replicates."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeGuard

import click
import pyarrow as pa
import pyarrow.compute as pc
from marin.execution.artifact import Artifact, ArtifactRecord
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply, run
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    SkyRLLaunchReceipt,
    SkyRLReceiptLookupError,
    SkyRLRun,
    read_skyrl_launch_receipt,
)
from pydantic import BaseModel
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from scipy.stats import t
from tabulate import tabulate
from taskcompendium.grading_result import Outcome

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.snowball_rl_scoring import (
    EVAL_SCORE_SCHEMA,
    PASS_RATES_NAME,
    InvalidPassRates,
    InvalidScores,
    evaluation_answers,
    evaluation_statistics,
    panel_scores,
    paths,
    read_pass_rates,
)

GATE_REVISION = "2026.10.29"


@dataclass(frozen=True)
class ControlPin:
    preset: str
    main_commit: str
    revision: str
    runs: int


CONTROLS = {
    name: (ControlPin(name, "09026fd100", "2026.10.27", count),)
    for name, count in (("speed-short", 5), ("data-short", 4), ("speed-medium", 4), ("learning-medium", 4))
}
CONTROLS["data-medium"] = CONTROLS["learning-medium"]


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


class InvalidComparison(ValueError):
    """The recorded runs cannot support this calibrated comparison."""


@dataclass(frozen=True)
class GateInputs:
    settings: tuple[tuple[str, object], ...] = ()
    tasks: str | None = None
    init: str | None = None


def gate_inputs(
    settings: Sequence[presets.Setting] = (),
    tasks: ArtifactStep | str | None = None,
    init: ArtifactStep | str | None = None,
) -> GateInputs:
    values = {leaf.key: leaf.value for setting in settings for leaf in presets.setting_leaves(setting)}
    task_name = f"{tasks.name}@{tasks.version}" if isinstance(tasks, ArtifactStep) else tasks
    init_name = f"{init.name}@{init.version}" if isinstance(init, ArtifactStep) else init
    return GateInputs(
        tuple(sorted(values.items())),
        task_name.removeprefix(presets.SELECTION_PREFIX + "/") if task_name else None,
        init_name,
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


class ReportedComparison(Comparison):
    change: Change


class RunMetadata(BaseModel):
    tree: str | None = None
    base_commit: str | None = None
    dirty: bool | None = None
    settings: tuple[str, ...]
    tasks: str | None
    init: str | None
    replicates: int


class ControlMetadata(BaseModel):
    main_commit: str
    revision: str
    runs: int
    gate_revision: str = GATE_REVISION


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
    metric: presets.Metric, values: Sequence[float], noise: MetricNoise, n_control: int, count: int
) -> Comparison:
    value = statistics.mean(values)
    return Comparison(
        metric=metric.key,
        higher_is_better=metric.higher_is_better,
        run=value,
        control=noise.mean,
        delta=(value - noise.mean) * (1 if metric.higher_is_better else -1),
        tolerance=float(t.ppf(1 - 0.025 / count, noise.df)) * noise.sd * math.sqrt(1 / len(values) + 1 / n_control),
        sd=noise.sd,
        df=noise.df,
        n_run=len(values),
        n_control=n_control,
    )


def _label(verdict: Verdict, reported: Sequence[ReportedComparison]) -> Label:
    better = any(row.change == Change.BETTER for row in reported)
    worse = any(row.change == Change.WORSE for row in reported)
    if better:
        return Label.TRADE_OFF if verdict == Verdict.FAIL or worse else Label.IMPROVES
    return Label.REGRESSION if verdict == Verdict.FAIL or worse else Label.NO_CHANGE


def _ambiguous(row: Invariant) -> TypeGuard[ErrorBoundInvariant]:
    return isinstance(row, ErrorBoundInvariant) and row.lower_bound <= row.limit < row.upper_bound


def _verdict(invariants: Sequence[Invariant], guards: Sequence[GuardComparison]) -> tuple[Verdict, str | None]:
    ambiguous = [row for row in invariants if _ambiguous(row)]
    if any(not row.holds and not _ambiguous(row) for row in invariants) or any(not row.passes for row in guards):
        return Verdict.FAIL, None
    if ambiguous:
        return Verdict.INVALID, "; ".join(
            f"grader-error union is ambiguous: lower={row.lower_bound:.12g}, upper={row.upper_bound:.12g}, "
            f"limit={row.limit}; recorded counts do not identify overlap"
            for row in ambiguous
        )
    return Verdict.PASS, None


def _metrics(root: StoragePath) -> dict[int, dict]:
    return {
        int(PurePosixPath(str(path)).stem.removeprefix("step-")): json.loads(presets.read_text(path))
        for path in paths(root / "exports/metrics/step-*.json")
    }


def _window_values(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    values = {}
    phases = [
        key
        for key in metrics.get(preset.measured.start, {})
        if key.startswith("timing/step_wall/") and key.rsplit("/", 1)[-1] not in ("evaluation", "checkpoint_work")
    ]
    for metric in (*preset.guards, *preset.reported):
        if metric.key.startswith("startup/"):
            observations = [row[metric.key] for _, row in sorted(metrics.items()) if metric.key in row]
            if observations:
                values[metric.key] = float(observations[0])
            continue
        series = []
        for step in preset.measured:
            row = metrics.get(step, {})
            if metric.key == "speed/update_seconds":
                if not phases or not all(key in row for key in phases):
                    break
                series.append(math.fsum(row[key] for key in phases))
            elif metric.key in row:
                series.append(row[metric.key])
            else:
                break
        if series and len(series) == len(preset.measured):
            reducer = statistics.median if metric.window == presets.WindowStatistic.MEDIAN else statistics.mean
            values[metric.key] = float(reducer(series))
    return values


def _gated(evaluation: presets.Evaluation, preset: presets.Preset) -> bool:
    return any(metric.key.startswith(evaluation.name + "/") for metric in preset.guards)


def _session_invariants(
    name: str, table: pa.Table, evaluation: presets.Evaluation, preset: presets.Preset
) -> list[Invariant]:
    rows = table.to_pylist()
    counts = Counter(row["item"] for row in rows)
    expected = presets.evaluation_answer_count(evaluation, presets.default_selection())
    result = [
        Invariant(
            name=f"{name}/batch_finished",
            value=len(rows),
            limit=expected,
            holds=len(counts) * evaluation.answers == expected and set(counts.values()) == {evaluation.answers},
        )
    ]
    if _gated(evaluation, preset):
        coverage = len({row["item"] for row in rows if row["scored"]}) / (expected / evaluation.answers)
        result.append(
            Invariant(name=f"{name}/panel_scored_fraction", value=coverage, limit=0.99, holds=coverage >= 0.99)
        )
    return result


@dataclass(frozen=True)
class RunData:
    path: str
    record: ArtifactRecord
    terminal: SkyRLLaunchReceipt
    values: dict[str, float]
    scores: pa.Table
    panel_ids: dict[str, frozenset[str]]
    invariants: tuple[Invariant, ...]
    missing_tokens: bool
    initial_scores: pa.Table


def _load_run(path: str, preset: presets.Preset, inputs: GateInputs = GateInputs()) -> RunData:
    try:
        terminal = read_skyrl_launch_receipt(path)
    except SkyRLReceiptLookupError as error:
        raise InvalidComparison(str(error)) from error
    if terminal is None:
        raise FileNotFoundError(f"no launch receipt at {path}")
    if terminal.result.state != "succeeded":
        raise InvalidComparison(f"run crashed: {terminal.result.iris_job_id}")
    metrics = _metrics(StoragePath(path))
    generating = preset.recipe.layout.role_plan.generate_only
    planned = {0} if generating else set(range(1, preset.updates + 1))
    completed = planned.intersection(metrics)
    invariants = [
        Invariant(name="planned_updates", value=len(completed), limit=len(planned), holds=completed == planned)
    ]
    scores, initial_scores, panel_ids = [], [], {}
    try:
        answers = evaluation_answers(path) if preset.evaluations else []
        for evaluation in preset.evaluations:
            all_scores = panel_scores(evaluation, path, answers)
            initial = all_scores.filter(pc.equal(all_scores["step"], 0))
            table = all_scores.filter(pc.equal(all_scores["step"], -1 if generating else preset.updates))
            scores.append(table)
            initial_scores.append(initial)
            ids = frozenset(table["item"].to_pylist())
            if evaluation.file != "train":
                panel_ids[evaluation.name] = ids
            invariants.extend(_session_invariants(evaluation.name, table, evaluation, preset))
            if inputs.init is not None:
                if initial.num_rows and frozenset(initial["item"].to_pylist()) != ids:
                    raise InvalidComparison(f"initial and final panel task IDs differ: {path}")
                invariants.extend(_session_invariants(evaluation.name + "/initial", initial, evaluation, preset))
    except InvalidScores as error:
        raise InvalidComparison(str(error)) from error
    table = pa.concat_tables(scores) if scores else pa.Table.from_pylist([], schema=EVAL_SCORE_SCHEMA)
    training = [metrics[step] for step in sorted(completed)]
    if not generating:
        finite = all(
            math.isfinite(value)
            for row in training
            for key, value in row.items()
            if key.startswith("loss/") or key == "policy/raw_grad_norm"
        )
        finite = finite and all(
            "policy/raw_grad_norm" in row and any(key.startswith("loss/") for key in row) for row in training
        )
        invariants.append(Invariant(name="finite_losses_and_gradients", value=finite, limit=True, holds=finite))
    outcome_keys = (*(f"generate/outcome/{outcome.value}" for outcome in Outcome), "generate/outcome/rollout_failure")
    total = math.fsum(row.get(f"generate/outcome/{outcome.value}", 0) for row in training for outcome in Outcome)
    bad = math.fsum(
        row.get(f"generate/outcome/{outcome.value}", 0)
        for row in training
        for outcome in (Outcome.INFRA_ERROR, Outcome.UNAVAILABLE, Outcome.INVALID_TASK)
    )
    failures = math.fsum(row.get("generate/outcome/rollout_failure", 0) for row in training)
    lower, upper = (max(bad, failures) / total, min(total, bad + failures) / total) if total else (1.0, 1.0)
    invariants.append(
        ErrorBoundInvariant(
            name="grader_error_fraction",
            value=upper,
            lower_bound=lower,
            upper_bound=upper,
            limit=0.01,
            holds=upper <= 0.01 and all(key in row for row in training for key in outcome_keys),
        )
    )
    fixed = preset.recipe.fixed_reply_tokens
    if fixed is not None:
        lengths = [len(row["response_token_ids"]) for _, _, row in answers] if generating else []
        holds = (
            bool(lengths) and set(lengths) == {fixed}
            if generating
            else all(
                row.get("generate/min_num_tokens") == row.get("generate/max_num_tokens") == fixed for row in training
            )
        )
        invariants.append(Invariant(name="fixed_reply_tokens", value=holds, limit=fixed, holds=holds))
    missing = any(
        metric.key.startswith("policy/mismatch/")
        and any(
            metrics.get(step, {}).get(metric.key.rsplit("/", 1)[0] + "/selected_tokens", 0) <= 0
            for step in preset.measured
        )
        for metric in preset.guards
    )
    return RunData(
        path=path,
        record=presets.artifact_record(path),
        terminal=terminal,
        values=_window_values(preset, metrics),
        scores=table,
        panel_ids=panel_ids,
        invariants=tuple(invariants),
        missing_tokens=missing,
        initial_scores=pa.concat_tables(initial_scores) if initial_scores else table.slice(0, 0),
    )


def _provenance(run: RunData) -> dict:
    if run.terminal.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {run.path}")
    return run.terminal.result.provenance


def _check_control(run: RunData, pin: ControlPin) -> None:
    provenance = _provenance(run)
    if provenance["base_commit"] != pin.main_commit or provenance["dirty"]:
        raise InvalidComparison(f"control does not match its pin: {run.path}")
    if run.missing_tokens or any(not row.holds for row in run.invariants):
        raise InvalidComparison(f"control has missing mismatch tokens or failed invariants: {run.path}")


def _difficulty_statistics(scores: pa.Table, rates: pa.Table) -> dict[str, float]:
    pass_rates = {row["task_id"]: row["passed"] / row["samples"] for row in rates.to_pylist()}
    result = {}
    for band, accepts in (
        ("easy", lambda rate: rate > 0.7),
        ("learnable", lambda rate: 0.1 <= rate <= 0.7),
        ("hard", lambda rate: rate < 0.1),
    ):
        rows = [row for row in scores.to_pylist() if row["item"] in pass_rates and accepts(pass_rates[row["item"]])]
        if rows:
            for key, value in evaluation_statistics(pa.Table.from_pylist(rows, schema=EVAL_SCORE_SCHEMA)).items():
                if key.endswith("/mean") and key.count("/") == 1:
                    result[f"{key.removesuffix('/mean')}/difficulty/{band}/mean"] = value
    return result


def _run_values(run: RunData, rates: pa.Table | None) -> dict[str, float]:
    values = {
        **run.values,
        **evaluation_statistics(run.scores),
        **(_difficulty_statistics(run.scores, rates) if rates is not None else {}),
    }
    for key, value in evaluation_statistics(run.initial_scores).items():
        values[key + "/initial"] = value
        if key in values:
            values[key + "/gain"] = values[key] - value
    return values


def _compare(
    preset: presets.Preset,
    runs: Sequence[RunData],
    controls: Mapping[ControlPin, Sequence[RunData]],
    rates: pa.Table | None,
) -> tuple[list[GuardComparison], list[ReportedComparison]]:
    grouped = {pin: [_run_values(run, rates) for run in group] for pin, group in controls.items()}
    noise = calibrate(grouped)
    values = [_run_values(run, rates) for run in runs]
    n_control = sum(map(len, controls.values()))
    guards = []
    for metric in preset.guards:
        if metric.key not in noise or any(metric.key not in row or not math.isfinite(row[metric.key]) for row in values):
            raise InvalidComparison(f"missing or non-finite guard metric: {metric.key}")
        if noise[metric.key].sd == 0:
            raise InvalidComparison(f"control variation is zero for guard metric: {metric.key}")
        row = _comparison(
            metric, [value[metric.key] for value in values], noise[metric.key], n_control, len(preset.guards)
        )
        guards.append(GuardComparison(**row.model_dump(), passes=row.delta >= -row.tolerance))
    metrics = {metric.key: metric for metric in preset.reported}
    declared = set(metrics)
    for row in values if preset.focus != presets.Focus.SPEED else ():
        for key in row:
            if key.startswith(tuple(evaluation.name + "/" for evaluation in preset.evaluations)):
                metrics.setdefault(key, presets.Metric(key, not key.endswith("all_wrong_fraction")))
    detail_count = len(set(metrics) - declared)
    reported = []
    for metric in metrics.values():
        available = all(metric.key in row and math.isfinite(row[metric.key]) for row in values)
        if metric.key in noise and available:
            row = _comparison(
                metric,
                [value[metric.key] for value in values],
                noise[metric.key],
                n_control,
                1 if metric.key in declared else detail_count,
            )
            change = (
                Change.BETTER
                if row.delta > row.tolerance
                else Change.WORSE if row.delta < -row.tolerance else Change.SAME
            )
            reported.append(ReportedComparison(**row.model_dump(), change=change))
        else:
            reported.append(
                ReportedComparison(
                    metric=metric.key,
                    higher_is_better=metric.higher_is_better,
                    run=statistics.mean(row[metric.key] for row in values) if available else None,
                    control=noise[metric.key].mean if metric.key in noise else None,
                    n_run=len(runs),
                    n_control=n_control,
                    change=Change.UNAVAILABLE,
                )
            )
    return guards, reported


def _render_markdown(report: GateReport) -> str:
    lines = [f"# {report.preset}", "", f"Verdict: {report.verdict}", f"Label: {report.label or 'unavailable'}"]
    if report.reason:
        lines += ["", report.reason]
    for title, value in (("Run", report.run), ("Control", report.control)):
        lines += ["", f"## {title}", ""]
        lines += [f"- {key}: {json.dumps(item, default=str)}" for key, item in value.model_dump().items()]
    for title, rows in (("Guards", report.guards), ("Reported", report.reported), ("Invariants", report.invariants)):
        table = tabulate(
            [row.model_dump() for row in rows],
            headers="keys",
            tablefmt="github",
            floatfmt=".8g",
            missingval="unavailable",
        )
        lines += ["", f"## {title}", "", table]
    return "\n".join([*lines, "", "## Gaps", "", *[f"- {gap}" for gap in report.gaps], ""])


def _write_report(report: GateReport, output: str) -> GateReport:
    fs, key = filesystem_for(output)
    fs.makedirs(key, exist_ok=True)
    for name, text in (
        (
            "gate_report.json",
            json.dumps(report.model_dump(mode="json", exclude={"path"}), indent=2, allow_nan=False) + "\n",
        ),
        ("gate_report.md", _render_markdown(report)),
    ):
        destination_fs, destination = filesystem_for(str(StoragePath(output) / name))
        with destination_fs.open(destination, "wt") as stream:
            stream.write(text)
    return report


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
    metadata = RunMetadata(
        settings=tuple(f"{key}={json.dumps(value, default=str)}" for key, value in inputs.settings),
        tasks=inputs.tasks,
        init=inputs.init,
        replicates=len(run_paths),
    )
    control = ControlMetadata(
        main_commit=pins[0].main_commit, revision=pins[0].revision, runs=sum(map(len, control_paths))
    )
    invariants, guards, reported = [], [], []
    verdict, reason, label = Verdict.INVALID, None, None
    try:
        runs = [_load_run(path, preset, inputs) for path in run_paths]
        if runs:
            provenance = _provenance(runs[0])
            metadata = metadata.model_copy(
                update={
                    "tree": provenance["tree_hash"],
                    "base_commit": provenance["base_commit"],
                    "dirty": provenance["dirty"],
                }
            )
        invariants = [
            row.model_copy(update={"replicate": index})
            for index, candidate in enumerate(runs)
            for row in candidate.invariants
        ]
        grouped = {
            pin: [_load_run(path, preset) for path in paths] for pin, paths in zip(pins, control_paths, strict=True)
        }
        controls = [candidate for group in grouped.values() for candidate in group]
        if not runs or not controls:
            raise InvalidComparison("comparison needs recorded runs and control replicates")
        for pin, group in grouped.items():
            for candidate in group:
                _check_control(candidate, pin)
        if any(candidate.record.version != presets.PRESET_REVISION for candidate in runs):
            raise InvalidComparison("preset revision differs")
        if any(candidate.panel_ids != controls[0].panel_ids for candidate in (*runs, *controls)):
            raise InvalidComparison("held-out panel task IDs differ")
        if any(candidate.missing_tokens for candidate in (*runs, *controls)):
            raise InvalidComparison("mismatch guard has missing or zero selected-token counts")
        try:
            rates = read_pass_rates(pass_rates_path) if pass_rates_path is not None else None
        except InvalidPassRates as error:
            raise InvalidComparison(str(error)) from error
        guards, reported = _compare(preset, runs, grouped, rates)
        verdict, reason = _verdict(invariants, guards)
        if verdict != Verdict.INVALID:
            declared = {metric.key for metric in preset.reported}
            label = _label(verdict, [row for row in reported if row.metric in declared])
    except (InvalidComparison, FileNotFoundError) as error:
        reason = str(error)
    return _write_report(
        GateReport(
            preset=preset_name,
            verdict=verdict,
            label=label,
            reason=reason,
            run=metadata,
            control=control,
            invariants=tuple(invariants),
            guards=tuple(guards),
            reported=tuple(reported),
            gaps=tuple(gap.value for gap in preset.gaps),
        ),
        output,
    )


def _pins(preset: str) -> tuple[ControlPin, ...]:
    return tuple(pin for pin in CONTROLS[preset] if pin.revision == presets.PRESET_REVISION)


def control_paths(pins: tuple[ControlPin, ...], root: str, leave_out: int | None = None) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            str(StoragePath(root) / presets.control_name(pin.preset, pin.main_commit, index) / pin.revision)
            for index in range(pin.runs)
            if pin != pins[0] or index != leave_out
        )
        for pin in pins
    )


def _gate_artifact(
    name: str,
    preset: presets.Preset,
    candidates: tuple[ArtifactStep[SkyRLRun], ...] | tuple[str, ...],
    pins: tuple[ControlPin, ...],
    controls: tuple[tuple[str, ...], ...],
    inputs: GateInputs,
) -> ArtifactStep[GateReport]:
    rates = (
        str(
            StoragePath(presets.artifact_prefix(preset.recipe.layout.cluster))
            / PASS_RATES_NAME
            / presets.PRESET_REVISION
        )
        if any(evaluation.source == "default" and _gated(evaluation, preset) for evaluation in preset.evaluations)
        else None
    )
    return apply(
        name,
        _write_gate,
        version=GATE_REVISION,
        artifact_type=GateReport,
        preset_name=preset.name,
        run_paths=candidates,
        control_paths=controls,
        pins=pins,
        inputs=inputs,
        pass_rates_path=rates,
        output=OUT,
    )


def gate_step(
    preset: presets.Preset, runs: tuple[ArtifactStep[SkyRLRun], ...], *, inputs: GateInputs
) -> ArtifactStep[GateReport]:
    pins = _pins(preset.name)
    if not pins:
        raise click.BadParameter(
            f"no control pin for {preset.name} at {presets.PRESET_REVISION}; collect with "
            f"python -m experiments.post_training.snowball_rl --preset {CONTROLS[preset.name][0].preset} "
            f"--version {presets.PRESET_REVISION} --control --replicates {CONTROLS[preset.name][0].runs} --run"
        )
    root = presets.artifact_prefix(preset.recipe.layout.cluster)
    name = runs[0].name.rsplit("-r", 1)[0] + f"/gate-n{len(runs)}-{pins[0].main_commit}"
    return _gate_artifact(name, preset, runs, pins, control_paths(pins, root), inputs)


def recorded_inputs(path: str) -> GateInputs:
    receipt = read_skyrl_launch_receipt(path)
    if receipt is None or receipt.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {path}")
    argv = receipt.result.provenance.get("command_line", [])
    command = presets.run_command()(lambda **kwargs: ())
    with command.make_context(
        "snowball", list(argv), resilient_parsing=True, ignore_unknown_options=True, allow_extra_args=True
    ) as context:
        params = context.params
    return gate_inputs(
        tuple(presets.parse_setting(value) for value in params.get("settings", ())),
        params.get("tasks_name"),
        params.get("init_name"),
    )


@click.command(help=__doc__)
@click.option("--preset", "preset_name", type=click.Choice(sorted(CONTROLS)), required=True)
@click.option("--candidate", "candidate_name")
@click.option("--leave-out", "leave_out")
def main(preset_name: str, candidate_name: str | None, leave_out: str | None) -> None:
    if (candidate_name is None) == (leave_out is None):
        raise click.UsageError("choose --candidate NAME[@VERSION] or --leave-out r<index>")
    preset, pins = presets.PRESETS[preset_name], _pins(preset_name)
    index = None
    if leave_out is not None:
        if not re.fullmatch(r"r\d+", leave_out) or int(leave_out[1:]) >= pins[0].runs:
            raise click.BadParameter(f"choose r0 through r{pins[0].runs-1}", param_hint="--leave-out")
        index = int(leave_out[1:])
        candidate_name = presets.control_name(pins[0].preset, pins[0].main_commit, index)
    root = presets.artifact_prefix(preset.recipe.layout.cluster)
    name, separator, version = candidate_name.partition("@")
    source = str(StoragePath(root) / name / (version if separator else presets.PRESET_REVISION))
    controls = control_paths(pins, root, index)
    identity = fingerprint_hash(json.dumps((source, controls)))
    namespace = user_owned_name(f"checkpoints/snowball-rl/rescore/{preset_name}-{identity}")
    step = _gate_artifact(
        namespace + f"/gate-n1-{pins[0].main_commit}",
        preset,
        (source,),
        pins,
        controls,
        recorded_inputs(source),
    )
    result = run(step)[0]
    click.echo(
        json.dumps(
            {
                "report": str(StoragePath(result.path) / "gate_report.json"),
                **result.model_dump(mode="json", exclude={"path"}),
            }
        )
    )


if __name__ == "__main__":
    main()
