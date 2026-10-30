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
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TypeGuard

import click
import pyarrow as pa
import pyarrow.compute as pc
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, apply, run
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import SkyRLReceiptLookupError, SkyRLRun, read_skyrl_launch_receipt
from pydantic import BaseModel
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from scipy.stats import t
from tabulate import tabulate
from taskcompendium.grading_result import Outcome

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.snowball_rl_scoring import InvalidComparison

GATE_REVISION = "2026.10.30.1"


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


_DETAIL_METRICS = {
    "speed-medium": (
        presets.Metric("async/performance/loss_tokens_per_configured_policy_gpu_second", True),
        presets.Metric("async/performance/response_tokens_per_configured_inference_gpu_second", True),
    ),
    "learning-medium": (
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


@dataclass(frozen=True)
class GateInputs:
    settings: tuple[str, ...] = ()
    tasks: str | None = None
    init: str | None = None
    requirements: tuple[str, ...] = ()


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


def check_requirements(preset: presets.Preset, requirements: Sequence[str], *, control: bool) -> None:
    if not requirements:
        return
    if control or preset.name not in CONTROLS:
        raise click.BadParameter("--require needs a gated candidate preset, without --control")
    prefixes = tuple(evaluation.name + "/" for evaluation in preset.evaluations)
    raw = [item.key for item in map(parse_requirement, requirements) if not item.key.startswith(prefixes)]
    if raw:
        pin = CONTROLS[preset.name][0]
        root = StoragePath(presets.artifact_prefix(preset.recipe.layout.cluster))
        path = root / presets.control_name(pin.preset, pin.main_commit, 0) / pin.revision
        metrics = json.loads(presets.read_text(path / f"exports/metrics/step-{preset.measured[0]:06d}.json"))
        for key in raw:
            if key not in metrics:
                raise click.BadParameter(f"requirement metric absent from control r0 first measured step: {key}")


def gate_inputs(
    settings: Sequence[presets.Setting] = (),
    tasks: ArtifactStep | str | None = None,
    init: ArtifactStep | str | None = None,
    requirements: Sequence[str] = (),
) -> GateInputs:
    values = {leaf.key: leaf.value for setting in settings for leaf in presets.setting_leaves(setting)}
    tasks, init = (f"{item.name}@{item.version}" if isinstance(item, ArtifactStep) else item for item in (tasks, init))
    return GateInputs(
        tuple(f"{key}={json.dumps(value, default=str)}" for key, value in sorted(values.items())),
        tasks.removeprefix(presets.SELECTION_PREFIX + "/") if tasks else None,
        init,
        tuple(requirements),
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


class RunMetadata(BaseModel):
    tree: str | None = None
    base_commit: str | None = None
    dirty: bool | None = None
    settings: tuple[str, ...]
    tasks: str | None
    init: str | None
    replicates: int
    requirements: tuple[str, ...]


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


def _verdict(
    invariants: Sequence[Invariant], guards: Sequence[GuardComparison], declared: Sequence[ReportedComparison]
) -> tuple[Verdict, Label | None, str | None]:
    ambiguous = [row for row in invariants if _ambiguous(row)]
    if any(not row.holds and not _ambiguous(row) for row in invariants) or any(not row.passes for row in guards):
        verdict = Verdict.FAIL
    elif ambiguous:
        return (
            Verdict.INVALID,
            None,
            "; ".join(
                f"grader-error union is ambiguous: lower={row.lower_bound:.12g}, upper={row.upper_bound:.12g}, "
                f"limit={row.limit}; recorded counts do not identify overlap"
                for row in ambiguous
            ),
        )
    else:
        verdict = Verdict.PASS
    better = any(row.change == Change.BETTER for row in (*guards, *declared))
    worse = verdict == Verdict.FAIL or any(row.change == Change.WORSE for row in (*guards, *declared))
    if better:
        return verdict, Label.TRADE_OFF if worse else Label.IMPROVES, None
    return verdict, Label.REGRESSION if worse else Label.NO_CHANGE, None


def _metrics(root: StoragePath) -> dict[int, dict]:
    return {
        int(path.name.removeprefix("step-").removesuffix(".json")): json.loads(presets.read_text(path))
        for path in scoring.paths(root / "exports/metrics/step-*.json")
    }


def _window_values(preset: presets.Preset, metrics: Mapping[int, Mapping[str, float]]) -> dict[str, float]:
    rows = [metrics.get(step, {}) for step in preset.measured]
    phases = [
        key
        for key in rows[0]
        if key.startswith("timing/step_wall/") and key.rsplit("/", 1)[-1] not in ("evaluation", "checkpoint_work")
    ]
    values = {}
    declared = {metric.key: metric for metric in (*preset.guards, *preset.reported)}
    extra = set().union(*(row.keys() for row in metrics.values())) - declared.keys()
    for metric in (*declared.values(), *(declared.get(key, presets.Metric(key, True)) for key in extra)):
        if metric.key.startswith("startup/"):
            first = next((row[metric.key] for _, row in sorted(metrics.items()) if metric.key in row), None)
            if first is not None:
                values[metric.key] = float(first)
            continue
        if metric.key == "speed/update_seconds":
            series = [math.fsum(row[key] for key in phases) for row in rows if phases and set(phases) <= set(row)]
        else:
            series = [row[metric.key] for row in rows if metric.key in row]
        if series and len(series) == len(rows):
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
    holds = len(counts) * evaluation.answers == expected and set(counts.values()) == {evaluation.answers}
    result = [Invariant(name=f"{name}/batch_finished", value=len(rows), limit=expected, holds=holds)]
    if _gated(evaluation, preset):
        coverage = len({row["item"] for row in rows if row["scored"]}) / (expected / evaluation.answers)
        result.append(
            Invariant(name=f"{name}/panel_scored_fraction", value=coverage, limit=0.99, holds=coverage >= 0.99)
        )
    return result


@dataclass(frozen=True)
class RunData:
    path: str
    version: str
    provenance: dict
    values: dict[str, float]
    panel_ids: dict[str, frozenset[str]]
    invariants: tuple[Invariant, ...]
    missing_tokens: bool
    slices: dict[str, tuple[int, float | None]]


def _load_run(path: str, preset: presets.Preset, rates: pa.Table | None, inputs: GateInputs = GateInputs()) -> RunData:
    terminal = read_skyrl_launch_receipt(path)
    if terminal is None:
        raise FileNotFoundError(f"no launch receipt at {path}")
    if terminal.result.state != "succeeded":
        raise InvalidComparison(f"run crashed: {terminal.result.iris_job_id}")
    if terminal.result.provenance is None:
        raise InvalidComparison(f"run has no recorded source provenance: {path}")
    metrics = _metrics(StoragePath(path))
    generating = preset.recipe.layout.role_plan.generate_only
    planned = {0} if generating else set(range(1, preset.updates + 1))
    completed = planned.intersection(metrics)
    training = [metrics[step] for step in sorted(completed)]
    invariants = [
        Invariant(name="planned_updates", value=len(completed), limit=len(planned), holds=completed == planned)
    ]
    answers = scoring.evaluation_answers(path) if preset.evaluations else []
    scores, initial_scores, panel_ids = [scoring.EVAL_SCORE_SCHEMA.empty_table()], [], {}
    for evaluation in preset.evaluations:
        table = scoring.panel_scores(evaluation, path, answers)
        initial = table.filter(pc.equal(table["step"], 0))
        table = table.filter(pc.equal(table["step"], -1 if generating else preset.updates))
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
    if not generating:
        finite = all(
            "policy/raw_grad_norm" in row
            and any(key.startswith("loss/") for key in row)
            and all(math.isfinite(row[key]) for key in row if key.startswith("loss/") or key == "policy/raw_grad_norm")
            for row in training
        )
        invariants.append(Invariant(name="finite_losses_and_gradients", value=finite, limit=True, holds=finite))

    def count(*outcomes: str) -> float:
        return math.fsum(row.get(f"generate/outcome/{outcome}", 0) for row in training for outcome in outcomes)

    total, bad, failures = (
        count(*Outcome),
        count(Outcome.INFRA_ERROR, Outcome.UNAVAILABLE, Outcome.INVALID_TASK),
        count("rollout_failure"),
    )
    lower, upper = (max(bad, failures) / total, min(total, bad + failures) / total) if total else (1.0, 1.0)
    recorded = all(f"generate/outcome/{outcome}" in row for row in training for outcome in (*Outcome, "rollout_failure"))
    holds = upper <= 0.01 and recorded
    invariants.append(
        ErrorBoundInvariant(
            name="grader_error_fraction", value=upper, lower_bound=lower, upper_bound=upper, limit=0.01, holds=holds
        )
    )
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
        and any(
            metrics.get(step, {}).get(metric.key.rsplit("/", 1)[0] + "/selected_tokens", 0) <= 0
            for step in preset.measured
        )
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
    for text in inputs.requirements:
        requirement = parse_requirement(text)
        value = values.get(requirement.key)
        if value is None or not math.isfinite(value):
            raise InvalidComparison(f"missing or non-finite requirement metric: {requirement.key}")
        if requirement.bound is not None:
            holds = value >= requirement.bound if requirement.higher_is_better else value <= requirement.bound
            invariants.append(
                Invariant(name=f"require/{requirement.key}", value=value, limit=requirement.bound, holds=holds)
            )
    initializer = {row["task_id"]: row["mean_score"] for row in rates.to_pylist()} if rates is not None else {}
    slices = {}
    for key, rows in scoring.score_slices(final, rates).items():
        ids = {row["item"] for row in rows}
        initial = statistics.mean(initializer[item] for item in ids) if ids <= initializer.keys() else None
        slices[key] = len(ids), initial
    return RunData(
        path=path,
        version=presets.artifact_record(path).version,
        provenance=terminal.result.provenance,
        values=values,
        panel_ids=panel_ids,
        invariants=tuple(invariants),
        missing_tokens=missing,
        slices=slices,
    )


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
    if key.startswith("reward/domain/"):
        return "Training reward per source"
    parts = key.split("/")
    dimension = parts[1] if len(parts) > 2 else "overall"
    if dimension == "agentic":
        dimension = "overall"
    if "difficulty" in parts and dimension != "difficulty":
        dimension += " x difficulty"
    return f"{parts[0]}: {dimension}"


def _compare(
    preset: presets.Preset,
    runs: Sequence[RunData],
    controls: Mapping[ControlPin, Sequence[RunData]],
    requirements: Sequence[str] = (),
) -> tuple[list[GuardComparison], list[ReportedComparison], list[ReportedComparison]]:
    noise = calibrate({pin: [run.values for run in group] for pin, group in controls.items()})
    values = [run.values for run in runs]
    n_control = sum(map(len, controls.values()))
    guards = []
    relative = [parse_requirement(text) for text in requirements if parse_requirement(text).bound is None]
    metrics = tuple(
        {
            (metric.key, metric.higher_is_better): metric
            for metric in (*preset.guards, *(presets.Metric(item.key, item.higher_is_better) for item in relative))
        }.values()
    )
    for metric in metrics:
        row = _comparison(metric, values, noise.get(metric.key), n_control, len(metrics))
        if row.delta is None:
            raise InvalidComparison(f"missing or non-finite guard metric: {metric.key}")
        if row.sd == 0:
            raise InvalidComparison(f"control variation is zero for guard metric: {metric.key}; use a number bound")
        guards.append(GuardComparison(**row.model_dump(), passes=row.delta >= -row.tolerance, change=_change(row)))
    guard_keys = {metric.key for metric in metrics}
    declared = {metric.key: metric for metric in preset.reported if metric.key not in guard_keys}
    prefixes = tuple(evaluation.name + "/" for evaluation in preset.evaluations)
    keys = dict.fromkeys(
        key
        for row in (*values, noise)
        for key in row
        if (key.startswith(prefixes) or key.startswith("reward/domain/"))
        and (not key.startswith("reward/domain/") or key in noise)
        and key not in declared
        and key not in guard_keys
    )
    details = (
        {}
        if preset.focus == presets.Focus.SPEED
        else {key: presets.Metric(key, not key.endswith("all_wrong_fraction")) for key in keys}
    )
    telemetry = _DETAIL_METRICS.get(CONTROLS[preset.name][0].preset, ())
    details.update({metric.key: metric for metric in telemetry if metric.key in noise and metric.key not in guard_keys})
    counts = Counter(map(_breakdown_group, (*details, *(key for key in declared if key.startswith(prefixes)))))
    reported = []
    for metric in (*declared.values(), *details.values()):
        row = _comparison(
            metric,
            values,
            noise.get(metric.key),
            n_control,
            (
                counts[_breakdown_group(metric.key)]
                if metric.key.startswith(prefixes) or metric.key.startswith("reward/domain/")
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
        control_values = [
            run.values[metric.key] for group in controls.values() for run in group if metric.key in run.values
        ]
        reported.append(
            ReportedComparison(
                **row.model_dump(),
                change=_change(row),
                tasks=tasks,
                initializer=initializer,
                control_min=min(control_values) if control_values else None,
                control_max=max(control_values) if control_values else None,
            )
        )
    return guards, reported[: len(declared)], reported[len(declared) :]


def _render_markdown(report: GateReport) -> str:
    lines = [f"# {report.preset}", "", f"Verdict: {report.verdict}", f"Label: {report.label or 'unavailable'}"]
    if report.reason:
        lines += ["", report.reason]
    for title, value in (("Run", report.run), ("Control", report.control)):
        lines += ["", f"## {title}", ""]
        lines += [f"- {key}: {json.dumps(item, default=str)}" for key, item in value.model_dump().items()]
    prefixes = (*(evaluation.name + "/" for evaluation in presets.PRESETS[report.preset].evaluations), "reward/domain/")
    breakdown = [row for row in report.reported if row.metric.startswith(prefixes)]
    summary = [row for row in report.reported if not row.metric.startswith(prefixes)]
    for title, rows in (("Guards", report.guards), ("Reported", summary), ("Invariants", report.invariants)):
        rows = [row.model_dump() for row in rows]
        lines += ["", f"## {title}", "", tabulate(rows, "keys", "github", floatfmt=".8g", missingval="unavailable")]
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
    metadata = RunMetadata(**asdict(inputs), replicates=len(run_paths))
    control = ControlMetadata(
        main_commit=pins[0].main_commit, revision=pins[0].revision, runs=sum(map(len, control_paths))
    )
    invariants, guards, reported = [], [], []
    verdict, label, reason = Verdict.INVALID, None, None
    try:
        rates = scoring.read_pass_rates(pass_rates_path) if pass_rates_path is not None else None
        runs = [_load_run(path, preset, rates, inputs) for path in run_paths]
        if runs:
            fields = {"tree": "tree_hash", "base_commit": "base_commit", "dirty": "dirty"}
            metadata = metadata.model_copy(update={field: runs[0].provenance[key] for field, key in fields.items()})
        invariants = [
            row.model_copy(update={"replicate": index})
            for index, candidate in enumerate(runs)
            for row in candidate.invariants
        ]
        grouped = {
            pin: [_load_run(path, preset, rates) for path in paths]
            for pin, paths in zip(pins, control_paths, strict=True)
        }
        controls = [candidate for group in grouped.values() for candidate in group]
        if not runs or not controls:
            raise InvalidComparison("comparison needs recorded runs and control replicates")
        for pin, group in grouped.items():
            for candidate in group:
                _check_control(candidate, pin)
        if any(candidate.version != presets.PRESET_REVISION for candidate in runs):
            raise InvalidComparison("preset revision differs")
        if any(candidate.panel_ids != controls[0].panel_ids for candidate in (*runs, *controls)):
            raise InvalidComparison("held-out panel task IDs differ")
        if any(candidate.missing_tokens for candidate in (*runs, *controls)):
            raise InvalidComparison("mismatch guard has missing or zero selected-token counts")
        guards, declared, details = _compare(preset, runs, grouped, inputs.requirements)
        reported = [*declared, *details]
        verdict, label, reason = _verdict(invariants, guards, declared)
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
        gaps=[gap.value for gap in preset.gaps],
    )
    for name, value in (
        (
            "gate_report.json",
            json.dumps(report.model_dump(mode="json", exclude={"path"}), indent=2, allow_nan=False) + "\n",
        ),
        ("gate_report.md", _render_markdown(report)),
    ):
        fs, key = filesystem_for(str(StoragePath(output) / name))
        fs.makedirs(key.rsplit("/", 1)[0], exist_ok=True)
        with fs.open(key, "wt") as stream:
            stream.write(value)
    return report


def _pins(preset: str) -> tuple[ControlPin, ...]:
    return tuple(pin for pin in CONTROLS[preset] if pin.revision == presets.PRESET_REVISION)


def _gate_artifact(
    name: str,
    preset: presets.Preset,
    candidates: tuple[ArtifactStep[SkyRLRun], ...] | tuple[str, ...],
    inputs: GateInputs,
    leave_out: int | None = None,
) -> ArtifactStep[GateReport]:
    pins = _pins(preset.name)
    root = StoragePath(presets.artifact_prefix(preset.recipe.layout.cluster))
    controls = tuple(
        tuple(
            str(root / presets.control_name(pin.preset, pin.main_commit, index) / pin.revision)
            for index in range(pin.runs)
            if pin != pins[0] or index != leave_out
        )
        for pin in pins
    )
    rates = any(evaluation.source == "default" and _gated(evaluation, preset) for evaluation in preset.evaluations)
    name += "-req-" + fingerprint_hash(canonical_json(inputs.requirements)) if inputs.requirements else ""
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
        pass_rates_path=str(root / scoring.PASS_RATES_NAME / scoring.PASS_RATE_REVISION) if rates else None,
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
    name = runs[0].name.rsplit("-r", 1)[0] + f"/gate-n{len(runs)}-{pins[0].main_commit}"
    return _gate_artifact(name, preset, runs, inputs)


def recorded_inputs(path: str) -> GateInputs:
    try:
        receipt = read_skyrl_launch_receipt(path)
    except SkyRLReceiptLookupError:
        return GateInputs()
    argv = (receipt.result.provenance or {}).get("command_line", []) if receipt else []
    context = presets.run_command()(lambda **kwargs: ()).make_context(
        "snowball", list(argv), resilient_parsing=True, ignore_unknown_options=True, allow_extra_args=True
    )
    settings = tuple(map(presets.parse_setting, context.params.get("settings", ())))
    return gate_inputs(
        settings,
        context.params.get("tasks_name"),
        context.params.get("init_name"),
        context.params.get("requirements", ()),
    )


@click.command(help=__doc__)
@click.option("--preset", "preset_name", type=click.Choice(sorted(CONTROLS)), required=True)
@click.option("--candidate", "candidate_name")
@click.option("--leave-out", "leave_out")
@click.option("--require", "requirements", multiple=True)
def main(preset_name: str, candidate_name: str | None, leave_out: str | None, requirements: tuple[str, ...]) -> None:
    if (candidate_name is None) == (leave_out is None):
        raise click.UsageError("choose --candidate NAME[@VERSION] or --leave-out r<index>")
    preset, pins = presets.PRESETS[preset_name], _pins(preset_name)
    index = None
    if leave_out is not None:
        if not re.fullmatch(r"r\d+", leave_out) or int(leave_out[1:]) >= pins[0].runs:
            raise click.BadParameter(f"choose r0 through r{pins[0].runs-1}", param_hint="--leave-out")
        index = int(leave_out[1:])
        candidate_name = presets.control_name(pins[0].preset, pins[0].main_commit, index)
    name, separator, version = candidate_name.partition("@")
    root = StoragePath(presets.artifact_prefix(preset.recipe.layout.cluster))
    source = str(root / name / (version if separator else presets.PRESET_REVISION))
    namespace = user_owned_name(
        f"checkpoints/snowball-rl/rescore/{preset_name}-{fingerprint_hash(canonical_json((source, pins, index)))}"
    )
    recorded = recorded_inputs(source)
    inputs = GateInputs(recorded.settings, recorded.tasks, recorded.init, (*recorded.requirements, *requirements))
    for text in inputs.requirements:
        parse_requirement(text)
    step = _gate_artifact(f"{namespace}/gate-n1-{pins[0].main_commit}", preset, (source,), inputs, index)
    report = run(step)[0]
    payload = {
        "report": str(StoragePath(report.path) / "gate_report.json"),
        **report.model_dump(mode="json", exclude={"path"}),
    }
    click.echo(json.dumps(payload))


if __name__ == "__main__":
    main()
