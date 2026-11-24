# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare stored RL runs with the selected preset table's pinned controls."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.compute as pc
from finestore.rl.rl_progress import Change, Comparison, ComparisonRole, Effect, Invariant, Verdict
from finestore.rl.strata import TaskStrata
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, materialized_config, run
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import (
    ArtifactHfModel,
    ResolvedModelLocator,
    SkyRLPolicyModel,
    SkyRLReceiptLookupError,
    SkyRLRun,
    SkyRLSpec,
    read_skyrl_attempt,
    skyrl_cpu_command,
    skyrl_execution,
    skyrl_step,
)
from marin.rl.skyrl_records import RecordedLaunch, read_stored_skyrl_attempt
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import Provenance
from scipy.stats import t
from skyrlrecipe.documents import MISSING, get_path, set_path
from tabulate import tabulate

from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_command import EXIT_CODES, RLCommand
from experiments.post_training.rl_measurements import (
    InvalidComparison,
    RunInputs,
    RunMeasurements,
    held_out_identity,
    read_measurements,
    read_recorded_launch,
    read_run_inputs,
    restrict_evaluation,
)
from experiments.post_training.rl_presets import ControlPin, ModelPresets
from experiments.post_training.rl_scoring import AAII_WEIGHTS, SLICE_COLUMNS, score_answers

GATE_REVISION = "2026.11.24"
GATE_REPORT_FILENAME = "gate_report.json"
GATE_MARKDOWN_FILENAME = "gate_report.md"
MIN_SHARED_TASK_FRACTION = 0.9


REPORTED_PREFIXES = ("async/", "consumed/", "generate/", "loss/", "policy/", "reward/", "startup/", "timing/", "vllm/")


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


def with_objectives(candidate: RunMeasurements, preset: presets.Preset) -> RunMeasurements:
    values = dict(candidate.values)
    for metric in preset.objectives:
        if metric.window == presets.WindowStatistic.GAIN_PER_GPU_HOUR:
            gain = values.get(metric.key.removesuffix("_per_gpu_hour"))
            if gain is not None and candidate.gpu_hours:
                values[metric.key] = gain / candidate.gpu_hours
    return replace(candidate, values=values)


def with_requirements(candidate: RunMeasurements, requirements: Sequence[str]) -> RunMeasurements:
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


def validate_requirements(
    preset: presets.Preset, requirements: Sequence[str], *, table: ModelPresets, control: bool
) -> None:
    """Validate a candidate's control pin and user requirements before GPU submission."""
    if not requirements:
        return
    if control or not preset.gated:
        raise click.BadParameter("--require needs a gated candidate preset, without --control")
    keys = [parse_requirement(text).key for text in requirements]
    pin = control_pin(preset, table=table)
    reference = with_objectives(read_measurements(control_path(preset, pin, 0, table=table), preset), preset)
    missing = [key for key in keys if key not in reference.values]
    if missing:
        raise click.BadParameter(f"requirement metrics that the gate does not compute for control r0: {missing}")


def validate_candidate(preset: presets.Preset, inputs: RunInputs, *, table: ModelPresets, control: bool = False) -> dict:
    """Return the composed candidate after checking gated controls against current defaults."""
    if control or not preset.gated:
        (candidate,) = _compose_plans(preset, (inputs,))
        return candidate
    defaults, candidate = _compose_plans(preset, (RunInputs(preset_name=preset.name), inputs))
    pin = control_pin(preset, table=table)
    panel_defaults = {preset.held_out_tasks: defaults}
    for index in range(pin.replicates):
        path = control_path(preset, pin, index, table=table)
        receipt = read_stored_skyrl_attempt(path)
        if receipt is None:
            raise InvalidComparison(f"control receipt is unavailable: {path}")
        launch = read_recorded_launch(receipt)
        baseline = _panel_defaults(preset, launch, panel_defaults)
        _check_control_config(receipt.native.config, baseline, launch, preset=preset)
    return candidate


class RunMetadata(RunInputs):
    tree: str | None = None
    base_commit: str | None = None
    dirty: bool | None = None
    replicates: int
    held_out_tasks: str | None = None
    engines: tuple[dict, ...] = ()
    attempt_ids: tuple[str, ...] = ()
    config_changes: tuple[str, ...] = ()


class ControlMetadata(BaseModel):
    commit: str
    revision: str
    replicates: int
    gate_revision: str = GATE_REVISION
    engines: tuple[dict, ...] = ()
    held_out_tasks: str | None = None


class GateReport(Artifact):
    preset: str
    verdict: Verdict
    effect: Effect | None
    reason: str | None
    run: RunMetadata
    control: ControlMetadata
    invariants: tuple[Invariant, ...]
    comparisons: tuple[Comparison, ...]
    gaps: tuple[str, ...]

    def rows(self, *roles: ComparisonRole) -> tuple[Comparison, ...]:
        return tuple(row for row in self.comparisons if row.role in roles)


def read_gate_report(path: str | None) -> GateReport | None:
    """Read a completed gate report, or None while its artifact is unavailable."""
    if path is None:
        return None
    try:
        return read_artifact(path, GateReport)
    except FileNotFoundError:
        return None


@dataclass(frozen=True)
class MetricNoise:
    mean: float
    sd: float
    df: int


def calibrate(controls: Sequence[Mapping[str, float]]) -> dict[str, MetricNoise]:
    if len(controls) < 2:
        raise InvalidComparison("calibration needs at least two control replicates")
    keys = set.intersection(*(set(run) for run in controls))
    result = {}
    for key in keys:
        values = [run[key] for run in controls]
        if all(math.isfinite(value) for value in values):
            result[key] = MetricNoise(statistics.mean(values), statistics.stdev(values), len(controls) - 1)
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


def _verdict(invariants: Sequence[Invariant], comparisons: Sequence[Comparison]) -> tuple[Verdict, Effect]:
    verdict = (
        Verdict.FAIL
        if any(not row.holds for row in invariants) or any(row.passes is False for row in comparisons)
        else Verdict.PASS
    )
    relevant = [row for row in comparisons if row.role in (ComparisonRole.OBJECTIVE, ComparisonRole.GUARD)]
    better = any(row.change == Change.BETTER for row in relevant)
    worse = any(row.change == Change.WORSE for row in relevant)
    effect = (
        Effect.TRADE_OFF
        if better and worse
        else Effect.IMPROVES if better else Effect.REGRESSION if worse else Effect.NO_CHANGE
    )
    return verdict, effect


def _check_control(run: RunMeasurements, pin: ControlPin) -> None:
    if run.provenance["base_commit"] != pin.commit or run.provenance["dirty"]:
        raise InvalidComparison(f"control does not match its pin: {run.path}")
    if any(not row.holds for row in run.invariants):
        raise InvalidComparison(f"control has failed invariants: {run.path}")


def _task_labels(table: pa.Table) -> dict[str, TaskStrata]:
    rows = table.select(["task_digest", *SLICE_COLUMNS]).to_pylist()
    return {digest: TaskStrata.model_validate(row) for row in rows if (digest := row.pop("task_digest")) is not None}


def _check_protocols(runs: Sequence[RunMeasurements], controls: Sequence[RunMeasurements]) -> None:
    if any(run.evaluation_protocols != controls[0].evaluation_protocols for run in (*runs, *controls)):
        raise InvalidComparison("evaluation protocol differs")


def _shared_task_fraction(name: str, shared: set[str], control_labels: Sequence[Mapping[str, TaskStrata]]) -> Invariant:
    minimum = 1.0
    for labels in control_labels:
        families: dict[str, set[str]] = {}
        for digest, strata in labels.items():
            families.setdefault(strata.family, set()).add(digest)
        for label, total in [(name, set(labels)), *((f"{name}/family/{key}", ids) for key, ids in families.items())]:
            accepted = len(shared & total)
            required = max(1, math.ceil(MIN_SHARED_TASK_FRACTION * len(total)))
            if accepted < required:
                raise InvalidComparison(f"{label}: {accepted} of {len(total)} control tasks shared; minimum {required}")
            if label == name:
                minimum = min(minimum, accepted / len(total))
    return Invariant(
        name=f"{name}/shared_tasks", value=minimum, limit=MIN_SHARED_TASK_FRACTION, comparator=">=", holds=True
    )


def _unshared_score(
    name: str,
    runs: Sequence[RunMeasurements],
    extra: Sequence[set[str]],
) -> Comparison | None:
    if not any(extra):
        return None
    key = f"{name}/mean"
    means = []
    for candidate, digests in zip(runs, extra, strict=True):
        table = candidate.held_out_answers[name].final
        value_set = pa.array(sorted(digests), type=pa.string())
        score = score_answers(table.filter(pc.is_in(table["task_digest"], value_set=value_set))).get(key)
        if score is not None:
            means.append(score)
    return Comparison(
        metric=f"{name}/unshared/mean",
        role=ComparisonRole.REPORTED,
        run=statistics.mean(means) if means else None,
        n_run=len(means),
        n_control=0,
        tasks=len(set.union(*extra)),
    )


@dataclass(frozen=True)
class SharedEvaluation:
    coverage: Invariant
    final_tasks: int
    gain_tasks: int
    same_task_count: bool
    unshared: Comparison | None


def _align_evaluation(
    evaluation: presets.Evaluation,
    cohort: Sequence[RunMeasurements],
    n_runs: int,
    newest: int,
) -> tuple[list[RunMeasurements], SharedEvaluation]:
    name = evaluation.name
    labels = [_task_labels(run.held_out_answers[name].final) for run in cohort]
    sets = [set(tasks) for tasks in labels]
    shared = set.intersection(*sets)
    coverage = _shared_task_fraction(name, shared, labels[n_runs:])
    gain = shared.copy()
    if evaluation.at_start:
        for member in cohort:
            gain.intersection_update(member.held_out_answers[name].initial["task_digest"].to_pylist())
    baseline = set.union(*sets[n_runs:])
    extra = _unshared_score(name, cohort[:n_runs], [tasks - baseline for tasks in sets[:n_runs]])
    strata = labels[newest]
    aligned = [
        (
            restrict_evaluation(member, name, shared, gain, strata)
            if tasks != shared or gain != shared or member_labels != strata
            else member
        )
        for member, tasks, member_labels in zip(cohort, sets, labels, strict=True)
    ]
    return aligned, SharedEvaluation(coverage, len(shared), len(gain), len({len(tasks) for tasks in sets}) == 1, extra)


def _compare_evaluations(
    preset: presets.Preset,
    runs: Sequence[RunMeasurements],
    controls: Sequence[RunMeasurements],
    requirements: tuple[str, ...] = (),
) -> tuple[list[Invariant], list[Comparison]]:
    cohort = [*runs, *controls]
    newest = max(
        range(len(cohort)),
        key=lambda index: recipe.utc_millis(cohort[index].started_at) if cohort[index].started_at else -1,
    )
    shared = {}
    for evaluation in preset.evaluations:
        if evaluation.held_out:
            cohort, shared[evaluation.name] = _align_evaluation(evaluation, cohort, len(runs), newest)
    unavailable = {
        metric.key
        for metric in preset.objectives
        if metric.window == presets.WindowStatistic.GAIN_PER_GPU_HOUR
        and (alignment := shared.get(metric.key.split("/", 1)[0])) is not None
        and not alignment.same_task_count
    }
    cohort = [with_objectives(run, preset) for run in cohort]
    if unavailable:
        cohort = [
            replace(run, values={key: value for key, value in run.values.items() if key not in unavailable})
            for run in cohort
        ]
    selected_runs = [with_requirements(run, requirements) for run in cohort[: len(runs)]]
    selected_controls = cohort[len(runs) :]
    invariants = [alignment.coverage for alignment in shared.values()]
    invariants.extend(
        row.model_copy(update={"replicate": index}) for index, run in enumerate(selected_runs) for row in run.invariants
    )
    comparisons = _compare(preset, selected_runs, selected_controls, requirements)
    for index, row in enumerate(comparisons):
        alignment = shared.get(row.metric.split("/", 1)[0])
        if alignment is not None and row.role in (ComparisonRole.GUARD, ComparisonRole.OBJECTIVE):
            tasks = alignment.gain_tasks if "/gain" in row.metric else alignment.final_tasks
            comparisons[index] = row.model_copy(update={"tasks": tasks})
    comparisons.extend(alignment.unshared for alignment in shared.values() if alignment.unshared is not None)
    return invariants, comparisons


def _compare(
    preset: presets.Preset,
    runs: Sequence[RunMeasurements],
    controls: Sequence[RunMeasurements],
    requirements: Sequence[str] = (),
) -> list[Comparison]:
    noise = calibrate([run.values for run in controls])
    values = [run.values for run in runs]
    n_control = len(controls)
    objectives = {metric.key for metric in preset.objectives}
    built_in = {(metric.key, metric.higher_is_better) for metric in preset.guards}
    declarations = [
        (
            metric,
            ComparisonRole.OBJECTIVE if metric.key in objectives else ComparisonRole.GUARD,
            len(preset.guards),
            True,
        )
        for metric in preset.guards
    ]
    declarations.extend(
        (requirement, ComparisonRole.REQUIREMENT, 1, True)
        for text in dict.fromkeys(requirements)
        if (requirement := parse_requirement(text)).bound is None
        and (requirement.key, requirement.higher_is_better) not in built_in
    )
    declarations.extend(
        (metric, ComparisonRole.OBJECTIVE, len(preset.objectives), False)
        for metric in preset.objectives
        if metric.key not in {guard.key for guard in preset.guards}
    )
    rows = []
    for metric, role, count, required in declarations:
        row = _comparison(
            metric.key, role, values, noise.get(metric.key), n_control, count, higher_is_better=metric.higher_is_better
        )
        if required:
            kind = "requirement" if role == ComparisonRole.REQUIREMENT else "guard"
            if row.delta is None:
                evaluation = metric.key.split("/", 1)[0]
                unavailable = [
                    candidate for candidate in runs if not math.isfinite(candidate.values.get(metric.key, math.nan))
                ]
                if (
                    role != ComparisonRole.REQUIREMENT
                    and row.sd is not None
                    and row.sd > 0
                    and unavailable
                    and all(
                        metric.key not in candidate.values
                        and any(
                            fact.name == f"{evaluation}/task_coverage" and not fact.holds
                            for fact in candidate.invariants
                        )
                        for candidate in unavailable
                    )
                ):
                    rows.append(row)
                    continue
                raise InvalidComparison(f"missing or non-finite {kind} metric: {metric.key}")
            if row.sd == 0:
                raise InvalidComparison(
                    f"control variation is zero for {kind} metric: {metric.key}; "
                    "use a numeric bound for a relative requirement"
                )
            signed = row.delta * (1 if metric.higher_is_better else -1)
            row = row.model_copy(update={"passes": signed >= -row.tolerance})
        rows.append(row)
    selected = {row.metric for row in rows}
    prefixes = tuple(evaluation.name + "/" for evaluation in preset.evaluations)
    for key in sorted(set().union(*values, noise) - selected):
        if not key.startswith((*prefixes, *REPORTED_PREFIXES)):
            continue
        role = ComparisonRole.SLICE if key.startswith(prefixes) else ComparisonRole.REPORTED
        row = _comparison(key, role, values, noise.get(key), n_control)
        reference = [run.values[key] for run in controls] if key in noise else []
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
    lines = [
        f"# {report.preset}",
        "",
        "AAII v4.3.2 category weights: "
        + ", ".join(f"{category} {weight:.0%}" for category, weight in AAII_WEIGHTS.items())
        + ". Weights are normalized over represented categories; format has weight zero. "
        "A represented category without scored tasks has no weighted mean.",
    ]
    tables = (
        (
            "Objectives",
            (ComparisonRole.OBJECTIVE,),
            {"metric", "control", "run", "delta_percent", "noise_percent", "passes", "change"},
        ),
        (
            "Guards and requirements",
            (ComparisonRole.GUARD, ComparisonRole.REQUIREMENT),
            {"metric", "control", "run", "passes"},
        ),
        ("Invariants", (), None),
        ("Reported", (ComparisonRole.REPORTED,), {"metric", "control", "run", "delta", "change"}),
        ("Breakdown", (ComparisonRole.SLICE,), {"metric", "tasks", "control", "control_min", "control_max", "run"}),
    )
    for title, roles, columns in tables:
        rows = [row.model_dump(include=columns) for row in (report.rows(*roles) if roles else report.invariants)]
        lines += ["", f"## {title}", "", tabulate(rows, "keys", "github", floatfmt=".8g", missingval="unavailable")]
    lines += ["", f"Result: {report.verdict}" + (f" ({report.effect})" if report.effect else "")]
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


def _comparison_document(launch: dict, recorded: RecordedLaunch | None = None) -> dict:
    """Normalize native configuration and resolve its runtime aliases for comparison."""
    recorded = recorded or RecordedLaunch.from_document(launch, attempt=launch["run"]["attempt_id"])
    config = deepcopy(launch)
    if get_path(config, "skyrl.trainer.rollout_buffer.groups_per_family") == {}:
        del config["skyrl"]["trainer"]["rollout_buffer"]["groups_per_family"]
    inputs = launch["inputs"]

    def replace_alias(path, aliases, replacement):
        current = get_path(config, path)
        if any(current == alias for alias in aliases):
            set_path(config, path, replacement)

    def sources_at(sources, prefix):
        normalized = []
        for index, source in enumerate(sources):
            copied = dict(source)
            copied["local_path"] = f"<staging:{prefix}.{index}>"
            normalized.append(copied)
        return normalized

    def bound_data(values, sources, descriptors, normalized, path):
        if not isinstance(values, list) or len(values) != len(sources):
            raise InvalidComparison(f"native data binding differs from declared inputs: {path}")
        for value, source, descriptor in zip(values, sources, descriptors, strict=True):
            aliases = [prefix_join(root, descriptor.relative_path) for root in (descriptor.uri, descriptor.local_path)]
            if value != source and value not in aliases:
                raise InvalidComparison(f"native data binding differs from declared inputs: {path}")
        return normalized

    for name, path, descriptors in (
        ("train_data", "skyrl.data.train_data", recorded.train_data),
        ("validation_data", "skyrl.data.val_data", recorded.validation_data),
    ):
        sources = inputs[name]
        normalized = sources_at(sources, f"inputs.{name}")
        config["inputs"][name] = normalized
        if sources:
            set_path(config, path, bound_data(get_path(config, path), sources, descriptors, normalized, path))
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
                        recorded.evaluation_data[name],
                        normalized,
                        f"skyrl.trainer.callbacks.{index}.additional_evaluations.{name}.data_files",
                    )
                )

    aliases = [recorded.model_uri, recorded.model_local_path]
    config["inputs"]["model"]["local_path"] = "<model-staging>"
    for path in (
        "skyrl.trainer.policy.model.path",
        "skyrl.trainer.ref.model.path",
        "skyrl.trainer.policy.model.tokenizer_path",
        "skyrl.trainer.ref.model.tokenizer_path",
        "skyrl.generator.model_name",
        "skyrl.generator.trajectory_retention.model_path",
    ):
        value = get_path(config, path)
        if value is MISSING:
            continue
        if value not in aliases:
            raise InvalidComparison(f"native model binding differs from declared inputs: {path}")
        set_path(config, path, "<model-staging>")
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
        "run.model",
        "run.coordinator_iris_job_id",
        "iris.job_name",
        "iris.wandb_entity",
    ):
        set_path(config, path, f"<identity:{path}>")
    replace_alias("skyrl.trainer.run_name", [launch["iris"]["job_name"]], "<launcher-run-name>")
    replace_alias(
        "skyrl.generator.trajectory_retention.run_id",
        [recorded.run_id, launch["iris"]["job_name"]],
        "<run-identity>",
    )
    runtime = config["skyrl"].get("runtime", {})
    for field in ("run_id", "model", "attempt_id", "coordinator_iris_job_id", "iris_job_id"):
        runtime.pop(field, None)
    if checkpoint := inputs.get("starting_checkpoint"):
        for field, identity in (
            ("model_source_identity", inputs["model"]["identity"]),
            ("starting_checkpoint_identity", checkpoint["identity"]),
        ):
            if field in runtime and runtime.pop(field) != identity:
                raise InvalidComparison(f"native resume binding differs from declared inputs: skyrl.runtime.{field}")
    if not runtime:
        config["skyrl"].pop("runtime", None)
    for section in ("trainer", "checkpoint_export"):
        export = config["skyrl"].get(section, {})
        if "hf_hub_private" in export and not export.get("hf_hub_repo_id"):
            # Without an upload destination, compare visibility using the stored controls' False default.
            export["hf_hub_private"] = False
    return config


def comparison_config(launch: dict, recorded: RecordedLaunch | None = None) -> dict[str, object]:
    """Compare complete native configuration with declared runtime aliases resolved."""
    return _config_values(_comparison_document(launch, recorded))


def _gate_config(launch: dict, recorded: RecordedLaunch | None = None) -> dict[str, object]:
    config = _comparison_document(launch, recorded)
    config["iris"].pop("priority", None)
    return _config_values(config)


def _compose_plans(
    preset: presets.Preset, inputs: Sequence[RunInputs], *, model: ResolvedModelLocator | None = None
) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="rl-current-config-") as directory:
        root = Path(directory)
        paths = []
        for index, candidate in enumerate(inputs):
            try:
                spec = recipe.run_spec(
                    preset,
                    name=f"current-config-{index}",
                    tasks=recipe.selection_artifact(candidate.tasks) if candidate.tasks else None,
                    init=(recipe.model_from_reference(candidate.init, resolved=model) if candidate.init else None),
                    resume=candidate.resume,
                    settings=candidate.settings,
                )
                if isinstance(spec.model, ArtifactHfModel) and spec.model.relative_path is None:
                    spec = replace(spec, model=replace(spec.model, relative_path="<terminal-hf-export>"))
                step = recipe.run_step(preset, spec)
                path = root / f"config-{index}.yaml"
                path.write_text(materialized_config(step, recipe.run_prefix(preset)).launch_config_yaml)
                paths.append(str(path))
            except (ValueError, click.BadParameter) as error:
                raise InvalidComparison(f"recorded inputs cannot build the current preset: {error}") from error

        return compose_configs(paths)


def _native_json(*command: str, timeout: int = 120):
    try:
        result = subprocess.run(
            skyrl_cpu_command(*command),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result.check_returncode()
        return json.loads(result.stdout.splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        cause = f"timed out after {error.timeout:g} s" if isinstance(error, subprocess.TimeoutExpired) else str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr and error.stderr.strip():
            cause = error.stderr.strip().splitlines()[-1]
        raise InvalidComparison(f"native configuration cannot be composed: {cause}") from error


def _native_program(native_root: str, modules: tuple[str, ...], body: str) -> str:
    return (
        "import json,sys\nfrom pathlib import Path\nfrom omegaconf import OmegaConf\n"
        "from importlib import import_module\n"
        f"assert all(Path(import_module(name).__file__).resolve().is_relative_to({native_root!r}) "
        f"for name in {modules!r})\n" + body
    )


def compose_configs(paths: Sequence[str], *, validate_driver: bool = False) -> list[dict]:
    """Compose launch files and optionally check the driver configuration in SkyRL's CPU environment."""
    program = _native_program(
        str(recipe.REPO_ROOT / "vendor/skyrl"),
        ("cloud.iris.launch_config", "skyrl_train", "marinskyrl"),
        "from cloud.iris import launch_config\n"
        "from rigging.filesystem.s3_compat import configure_coreweave_s3\n"
        "configure_coreweave_s3()\n"
        "configs = [launch_config.load_launch_config(path) for path in sys.argv[1:]]\n"
        "documents = [OmegaConf.to_container(config, resolve=True) for config in configs]\n"
        f"if {validate_driver!r}:\n"
        "    for config in configs:\n"
        "        launch_config.validate_launch_driver_config(config)\n"
        "print(json.dumps(documents))\n",
    )
    return _native_json("python", "-c", program, *paths)


def validate_plan(preset: presets.Preset, spec: SkyRLSpec, step: ArtifactStep[SkyRLRun]) -> None:
    """Validate a planned launch through the native CPU composer and driver."""
    # The artifact root supplies the storage reference before the initializer's export exists.
    model = spec.model
    if isinstance(model, SkyRLPolicyModel):
        model = ArtifactHfModel(model.step, relative_path="")
    elif isinstance(model, ArtifactHfModel) and model.relative_path is None:
        model = replace(model, relative_path="")
    if model is not spec.model:
        execution = skyrl_execution(step)
        assert execution is not None
        step = skyrl_step(replace(spec, model=model), execution, export_hf=preset.export != "none")
    config = materialized_config(step, recipe.run_prefix(preset))
    with tempfile.TemporaryDirectory(prefix="rl-plan-") as directory:
        path = Path(directory) / "launch.yaml"
        path.write_text(config.launch_config_yaml)
        compose_configs([str(path)], validate_driver=True)


def _changed_config_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {
        key
        for key in left.keys() | right.keys()
        if key not in left or key not in right or canonical_json(left[key]) != canonical_json(right[key])
    }


def _setting_keys(inputs: RunInputs, keys: set[str]) -> set[str]:
    values = recipe.override_document(inputs.settings)
    allowed = set()
    for key in keys:
        if key.startswith("skyrl."):
            path = key.removeprefix("skyrl.")
        elif key.startswith("runtime.task_env."):
            path = "extra_env." + key.removeprefix("runtime.task_env.")
        else:
            continue
        parts = path.split(".")
        if any(
            (value := get_path(values, ".".join(parts[:index]))) is not MISSING
            and (index == len(parts) or isinstance(value, list))
            for index in range(1, len(parts) + 1)
        ):
            allowed.add(key)
    return allowed


@lru_cache(maxsize=16)
def _schema_defaults(native_project: str, source_identity: str | None, groups: tuple[tuple[str, str], ...]) -> dict:
    program = _native_program(
        native_project,
        ("cloud.iris.launch_config", "cloud.iris.rl_config_translation"),
        "from cloud.iris import launch_config, rl_config_translation\n"
        "config = rl_config_translation.compose_config_groups(json.loads(sys.argv[1]), config_name='ppo_base_config')\n"
        "defaults = OmegaConf.to_container(\n"
        "    OmegaConf.structured(launch_config.SkyRLLaunchConfig), resolve=False, enum_to_str=True)\n"
        "defaults['skyrl'] = OmegaConf.to_container(config, resolve=True, enum_to_str=True)\n"
        "print(json.dumps(defaults))\n",
    )
    return _native_json("python", "-c", program, json.dumps(dict(groups)))


def _check_control_config(
    config: dict, current_config: dict, recorded: RecordedLaunch | None = None, *, preset: presets.Preset
) -> None:
    observed, current = _gate_config(config, recorded), _gate_config(current_config)
    differences = _changed_config_keys(observed, current)
    missing = {key for key in differences if key not in observed}
    if missing:
        groups = tuple(sorted(preset.recipe.config.to_skyrl().get("config_groups", {}).items()))
        defaults = _schema_defaults(
            str(recipe.REPO_ROOT / "vendor/skyrl"), Provenance.capture(recipe.REPO_ROOT).tree_hash, groups
        )
        for key in missing:
            value = get_path(defaults, key)
            if value is MISSING:
                continue
            projected = deepcopy(current_config)
            set_path(projected, key, value)
            try:
                default_value = _gate_config(projected).get(key, MISSING)
            except InvalidComparison:
                continue
            if default_value == current[key]:
                differences.discard(key)
    if differences:
        raise InvalidComparison(f"control config differs from current preset defaults: {', '.join(sorted(differences))}")


def _panel_defaults(preset: presets.Preset, recorded: RecordedLaunch, cache: dict[str | None, dict]) -> dict:
    identity = held_out_identity(preset, recorded)
    if identity is None:
        return cache[preset.held_out_tasks]
    if identity not in cache:
        bound = replace(preset, held_out_tasks=identity)
        (cache[identity],) = _compose_plans(bound, (RunInputs(preset_name=preset.name),))
    return cache[identity]


def _write_gate(
    preset: presets.Preset,
    run_paths: tuple[str, ...],
    control_paths: tuple[str, ...],
    control_pin: ControlPin,
    inputs: RunInputs,
    output: str,
) -> GateReport:
    preset_name = preset.name
    metadata = RunMetadata(**inputs.model_dump(), replicates=len(run_paths))
    control = ControlMetadata(commit=control_pin.commit, revision=control_pin.revision, replicates=len(control_paths))
    invariants, comparisons = [], []
    verdict, effect, reason = Verdict.INVALID, None, None
    try:
        if control_pin.revision != preset.revision:
            raise _pin_error(preset, control_pin)
        receipts = [read_stored_skyrl_attempt(path) for path in run_paths]
        metadata = metadata.model_copy(
            update={"attempt_ids": tuple(receipt.attempt_id for receipt in receipts if receipt)}
        )
        runs = [read_measurements(path, preset, inputs) for path in run_paths]
        if runs:
            provenance_fields = {"tree": "tree_hash", "base_commit": "base_commit", "dirty": "dirty"}
            metadata = metadata.model_copy(
                update={
                    "held_out_tasks": runs[0].held_out_tasks,
                    "engines": tuple(run.engine for run in runs),
                    "attempt_ids": tuple(run.attempt_id for run in runs),
                    **{field: runs[0].provenance[key] for field, key in provenance_fields.items()},
                }
            )
        invariants = [
            row.model_copy(update={"replicate": index})
            for index, candidate in enumerate(runs)
            for row in candidate.invariants
        ]
        controls = [read_measurements(path, preset) for path in control_paths]
        if not runs or not controls:
            raise InvalidComparison("comparison needs recorded runs and control replicates")
        control = control.model_copy(
            update={"held_out_tasks": controls[0].held_out_tasks, "engines": tuple(run.engine for run in controls)}
        )
        for candidate in controls:
            _check_control(candidate, control_pin)
        _check_protocols(runs, controls)
        model = runs[0].config["inputs"]["model"]
        resolved_model = ResolvedModelLocator(
            **{field.name: model[field.name] for field in fields(ResolvedModelLocator)}
        )
        current_config, expected_config = _compose_plans(
            preset, (RunInputs(preset_name=preset.name), inputs), model=resolved_model
        )
        current, expected = _gate_config(current_config), _gate_config(expected_config)
        changed = _changed_config_keys(current, expected)
        explicit = _setting_keys(inputs, set(current) | set(expected))
        metadata = metadata.model_copy(update={"config_changes": tuple(sorted(changed | explicit))})
        for candidate in runs:
            differences = _changed_config_keys(_gate_config(candidate.config, candidate.recorded), expected)
            if differences:
                raise InvalidComparison(
                    f"candidate config differs from recorded inputs: {', '.join(sorted(differences))}"
                )
        panel_defaults = {preset.held_out_tasks: current_config}
        for candidate in controls:
            baseline = _panel_defaults(preset, candidate.recorded, panel_defaults)
            _check_control_config(candidate.config, baseline, candidate.recorded, preset=preset)
        invariants, comparisons = _compare_evaluations(preset, runs, controls, inputs.requirements)
        verdict, effect = _verdict(invariants, comparisons)
    except (InvalidComparison, SkyRLReceiptLookupError, FileNotFoundError) as error:
        reason = str(error)
    report = GateReport(
        preset=preset_name,
        verdict=verdict,
        effect=effect,
        reason=reason,
        run=metadata,
        control=control,
        invariants=invariants,
        comparisons=comparisons,
        gaps=list(preset.gaps),
    )
    return _store_report(report, output)


def _store_report(report: GateReport, output: str) -> GateReport:
    for name, value in (
        (
            GATE_REPORT_FILENAME,
            json.dumps(report.model_dump(mode="json", exclude={"path"}), indent=2, allow_nan=False) + "\n",
        ),
        (GATE_MARKDOWN_FILENAME, _render_markdown(report)),
    ):
        path = StoragePath(output) / name
        path.parent.mkdirs()
        path.write_text(value)
    return report


def control_name(preset: str, commit: str, replicate: int, *, table: ModelPresets) -> str:
    return f"checkpoints/{table.namespace}/control/{preset}-{commit}-r{replicate}"


def _pin_error(preset: presets.Preset, pin: ControlPin | None) -> InvalidComparison:
    name, replicas = (pin.preset, pin.replicates) if pin else (preset.control_preset, 4)
    return InvalidComparison(
        f"no control pin for {preset.name} with preset revision {preset.revision}; "
        f"recorded control revision={pin.revision if pin else None}; "
        f"collect current controls with --preset {name} "
        f"--version dev --control --replicates {replicas} --run"
    )


def control_pin(preset: presets.Preset, *, table: ModelPresets, check_revision: bool = True) -> ControlPin:
    pin = table.control_pins.get(preset.control_preset)
    if pin is None or (check_revision and pin.revision != preset.revision):
        raise _pin_error(preset, pin)
    return pin


def control_path(preset: presets.Preset, pin: ControlPin, index: int, *, table: ModelPresets) -> str:
    return recipe.run_path(preset, control_name(pin.preset, pin.commit, index, table=table), pin.revision)


def _read_previous_control(preset: presets.Preset, path: str) -> RunMeasurements:
    receipt = read_stored_skyrl_attempt(path)
    if receipt is None:
        raise FileNotFoundError(f"no terminal attempt record at {path}")
    launch = read_recorded_launch(receipt)
    if launch.max_steps is not None:
        preset = preset.with_updates(launch.max_steps)
    return read_measurements(path, preset, attempt=receipt)


def _write_control_link(
    preset: presets.Preset,
    current: ControlPin,
    previous: ControlPin,
    run_paths: tuple[str, ...],
    control_paths: tuple[str, ...],
    output: str,
) -> GateReport:
    metadata = RunMetadata(preset_name=preset.name, base_commit=current.commit, replicates=len(run_paths))
    baseline = ControlMetadata(commit=previous.commit, revision=previous.revision, replicates=len(control_paths))
    invariants, comparisons = [], []
    verdict, effect, reason = Verdict.INVALID, None, None
    try:
        runs = [read_measurements(path, preset) for path in run_paths]
        metadata = metadata.model_copy(
            update={
                "held_out_tasks": runs[0].held_out_tasks,
                "attempt_ids": tuple(run.attempt_id for run in runs),
                "engines": tuple(run.engine for run in runs),
            }
        )
        for candidate in runs:
            _check_control(candidate, current)
        controls = [_read_previous_control(preset, path) for path in control_paths]
        for candidate in controls:
            _check_control(candidate, previous)
        baseline = baseline.model_copy(
            update={"held_out_tasks": controls[0].held_out_tasks, "engines": tuple(run.engine for run in controls)}
        )
        _check_protocols(runs, controls)
        changes = _changed_config_keys(
            _gate_config(controls[0].config, controls[0].recorded),
            _gate_config(runs[0].config, runs[0].recorded),
        )
        metadata = metadata.model_copy(update={"config_changes": tuple(sorted(changes))})
        invariants, comparisons = _compare_evaluations(preset, runs, controls)
        verdict, effect = _verdict(invariants, comparisons)
    except (InvalidComparison, SkyRLReceiptLookupError, FileNotFoundError) as error:
        reason = str(error)
    return _store_report(
        GateReport(
            preset=preset.name,
            verdict=verdict,
            effect=effect,
            reason=reason,
            run=metadata,
            control=baseline,
            invariants=invariants,
            comparisons=comparisons,
            gaps=preset.gaps,
        ),
        output,
    )


def control_link_step(
    preset: presets.Preset, current: ControlPin, previous: ControlPin, *, table: ModelPresets
) -> ArtifactStep[GateReport]:
    """Compare consecutive control cohorts for the publication history."""
    if any(pin.preset != preset.name or pin.replicates < 2 for pin in (current, previous)):
        raise ValueError(f"expected two complete control pins for {preset.name}")
    return apply(
        user_owned_name(f"checkpoints/{table.namespace}/control-links/{preset.name}-{previous.commit}-{current.commit}"),
        _write_control_link,
        version=GATE_REVISION,
        artifact_type=GateReport,
        preset=preset,
        current=current,
        previous=previous,
        run_paths=tuple(control_path(preset, current, index, table=table) for index in range(current.replicates)),
        control_paths=tuple(control_path(preset, previous, index, table=table) for index in range(previous.replicates)),
        output=OUT,
    )


def _gate_artifact(
    name: str,
    preset: presets.Preset,
    candidates: tuple[str, ...],
    inputs: RunInputs,
    leave_out: int | None = None,
    *,
    table: ModelPresets,
    run_paths: tuple[str, ...],
    pin: ControlPin,
) -> ArtifactStep[GateReport]:
    defaults = recipe.run_step(preset, recipe.run_spec(preset, name="current-defaults"))
    name += "-defaults-" + fingerprint_hash(
        canonical_json((defaults.fingerprint(), defaults.runtime_args, Provenance.capture(recipe.REPO_ROOT).tree_hash))
    )
    controls = tuple(
        (
            f"{control_name(pin.preset, pin.commit, index, table=table)}@{pin.revision}",
            control_path(preset, pin, index, table=table),
        )
        for index in range(pin.replicates)
        if index != leave_out
    )
    name += "-req-" + fingerprint_hash(canonical_json(inputs.requirements)) if inputs.requirements else ""
    step = apply(
        name,
        _write_gate,
        version=GATE_REVISION,
        artifact_type=GateReport,
        preset=preset,
        run_paths=candidates,
        control_paths=tuple(identity for identity, _ in controls),
        control_pin=pin,
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
                    "control_paths": tuple(path for _, path in controls),
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
            receipt = read_skyrl_attempt(path)
        except (FileNotFoundError, SkyRLReceiptLookupError):
            state, attempt = "unresolved", None
        else:
            state = receipt.native.result.state if receipt else None
            attempt = receipt.native.config["run"]["attempt_id"] if receipt else None
        attempts.append({"run": identity, "native_state": state, "attempt_id": attempt})
    return replace(step, name=step.runtime_args["name"] + "-evidence-" + fingerprint_hash(canonical_json(attempts)))


def gate_step(
    preset: presets.Preset, runs: tuple[ArtifactStep[SkyRLRun], ...], *, table: ModelPresets, inputs: RunInputs
) -> ArtifactStep[GateReport]:
    pin = control_pin(preset, table=table, check_revision=False)
    name = runs[0].name.rsplit("-r", 1)[0] + f"/gate-n{len(runs)}-{pin.commit}"
    paths = tuple(candidate.path(recipe.run_prefix(preset)) for candidate in runs)
    identities = tuple(f"{candidate.name}@{candidate.version}" for candidate in runs)
    return replace(_gate_artifact(name, preset, identities, inputs, run_paths=paths, table=table, pin=pin), deps=runs)


def command(table: ModelPresets) -> click.Command:
    @click.command(cls=RLCommand, help=__doc__)
    @click.option(
        "--preset",
        "preset_name",
        type=click.Choice(sorted(name for name, preset in table.by_name.items() if preset.gated)),
        required=True,
        help="Gated preset whose pinned controls to compare with.",
    )
    @click.option(
        "--candidate",
        "candidate_name",
        metavar="NAME[@VERSION]",
        help="Saved run to compare, by the name and version the status block prints for it; VERSION defaults "
        "to the preset's revision. Give this or --leave-out.",
    )
    @click.option(
        "--leave-out",
        "leave_out",
        metavar="rN",
        help="Compare control replicate N against the remaining controls. Each replicate should PASS; use it "
        "after pinning new controls. Give this or --candidate.",
    )
    @click.option(
        "--require",
        "requirements",
        multiple=True,
        metavar="KEY<=|>=BOUND",
        help="Add a condition, as in rl --require, to those the run recorded at launch. Repeatable.",
    )
    def main(
        preset_name: str,
        candidate_name: str | None,
        leave_out: str | None,
        requirements: tuple[str, ...],
    ) -> None:
        if (candidate_name is None) == (leave_out is None):
            raise click.UsageError("choose --candidate NAME[@VERSION] or --leave-out r<index>")
        preset = table.by_name[preset_name]
        pin = control_pin(preset, table=table)
        index = None
        if leave_out is not None:
            if not re.fullmatch(r"r\d+", leave_out) or int(leave_out[1:]) >= pin.replicates:
                raise click.BadParameter(f"choose r0 through r{pin.replicates - 1}", param_hint="--leave-out")
            index = int(leave_out[1:])
            candidate_name = control_name(pin.preset, pin.commit, index, table=table)
        name, separator, version = candidate_name.partition("@")
        source = recipe.run_path(preset, name, version if separator else preset.revision)
        namespace = user_owned_name(
            f"checkpoints/{table.namespace}/compare/"
            f"{preset_name}-{fingerprint_hash(canonical_json((source, pin, index)))}"
        )
        recorded = RunInputs(preset_name=pin.preset) if index is not None else read_run_inputs(source)
        if index is None and recorded.preset_name != preset_name:
            raise InvalidComparison(f"recorded preset is {recorded.preset_name}, not {preset_name}")
        inputs = recorded.model_copy(
            update={"requirements": tuple(dict.fromkeys((*recorded.requirements, *requirements)))}
        )
        for text in inputs.requirements:
            parse_requirement(text)
        step = _gate_artifact(
            f"{namespace}/gate-n1-{pin.commit}",
            preset,
            (f"{name}@{version if separator else preset.revision}",),
            inputs,
            index,
            run_paths=(source,),
            table=table,
            pin=pin,
        )
        report = run(step)[0]
        payload = {
            "report": str(StoragePath(report.path) / GATE_REPORT_FILENAME),
            **report.model_dump(mode="json", exclude={"path"}),
        }
        click.echo(json.dumps(payload))
        raise click.exceptions.Exit(EXIT_CODES[report.verdict])

    return main


def completed_gate(gate: ArtifactStep[GateReport] | None) -> ArtifactStep[GateReport] | None:
    """Bind current native attempt evidence to the gate identity fixed at submission."""
    return _gate_with_evidence(gate) if gate is not None else None
