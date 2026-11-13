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
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import click
from marin.execution.artifact import Artifact, read_artifact
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, materialized_config, run
from marin.experiment.namespacing import user_owned_name
from marin.rl.rl_progress import Change, Comparison, ComparisonRole, Invariant, Label, Verdict
from marin.rl.skyrl import (
    SkyRLReceiptLookupError,
    SkyRLRun,
    read_skyrl_attempt,
    skyrl_cpu_command,
)
from marin.rl.skyrl_records import RecordedLaunch, read_stored_skyrl_attempt, recorded_launch
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import Provenance
from scipy.stats import t
from skyrlrecipe.documents import MISSING, get_path, set_path
from tabulate import tabulate

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_recipe as recipe
from experiments.post_training.snowball_rl_command import EXIT_CODES, SnowballCommand
from experiments.post_training.snowball_rl_presets import RUN_PREFIX
from experiments.post_training.snowball_rl_scoring import GATE_INPUTS_FILENAME as GATE_INPUTS_FILENAME
from experiments.post_training.snowball_rl_scoring import (
    GateInputs,
    InvalidComparison,
    RunData,
    load_run,
    parse_requirement,
    read_gate_inputs,
    with_objectives,
    with_requirements,
)
from experiments.post_training.snowball_rl_scoring import training_invariants as training_invariants

OUTCOME_PREFIX = "generate/outcome/"
GATE_REVISION = "2026.11.13"
GATE_REPORT_FILENAME = "gate_report.json"
GATE_MARKDOWN_FILENAME = "gate_report.md"


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


REPORTED_PREFIXES = ("async/", "consumed/", "generate/", "loss/", "policy/", "reward/", "startup/", "timing/", "vllm/")


def validate_requirements(preset: presets.Preset, requirements: Sequence[str], *, control: bool) -> None:
    """Validate a candidate's control pin and user requirements before GPU submission."""
    gated = preset.gated
    if requirements and (control or not gated):
        raise click.BadParameter("--require needs a gated candidate preset, without --control")
    keys = [parse_requirement(text).key for text in requirements]
    pins = control_pins(preset) if not control and gated else ()
    if not requirements:
        return
    reference = with_objectives(load_run(control_path(preset, pins[0], 0), preset), preset)
    missing = [key for key in keys if key not in reference.values]
    if missing:
        raise click.BadParameter(f"requirement metrics that the gate does not compute for control r0: {missing}")


def validate_candidate(preset: presets.Preset, inputs: GateInputs, *, control: bool = False) -> None:
    """Compose the submitted inputs and check every gated control against current defaults."""
    defaults, _ = _current_configs(preset, inputs)
    if control or not preset.gated:
        return
    pins = control_pins(preset)
    current = comparison_config(defaults)
    for pin in pins:
        for index in range(pin.runs):
            path = control_path(preset, pin, index)
            receipt = read_stored_skyrl_attempt(path)
            if receipt is None:
                raise InvalidComparison(f"control receipt is unavailable: {path}")
            try:
                launch = recorded_launch(receipt)
            except ValueError as error:
                raise InvalidComparison(str(error)) from error
            _check_control_config(receipt.native.config, current, launch)


def gate_inputs(
    preset_name: str,
    settings: Sequence[str] = (),
    tasks: str | None = None,
    init: str | None = None,
    requirements: Sequence[str] = (),
) -> GateInputs:
    values = presets.override_document(tuple(settings))
    return GateInputs(
        preset_name=preset_name,
        settings=tuple(f"{key}={json.dumps(value, sort_keys=True)}" for key, value in sorted(values.items())),
        tasks=tasks.removeprefix(presets.SELECTION_PREFIX + "/") if tasks else None,
        init=init,
        requirements=tuple(dict.fromkeys(requirements)),
    )


class RunMetadata(GateInputs):
    tree: str | None = None
    base_commit: str | None = None
    dirty: bool | None = None
    replicates: int
    eval_lite: str | None = None
    engines: tuple[dict, ...] = ()
    attempt_ids: tuple[str, ...] = ()
    config_changes: tuple[str, ...] = ()


class ControlMetadata(BaseModel):
    commit: str
    revision: str
    runs: int
    gate_revision: str = GATE_REVISION
    engines: tuple[dict, ...] = ()
    eval_lite: str | None = None


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
    objectives = {metric.key for metric in preset.objectives}
    built_in = {(metric.key, metric.higher_is_better) for metric in preset.guards}
    declarations = [
        (
            metric,
            ComparisonRole.OBJECTIVE if metric.key in objectives else ComparisonRole.CONSTRAINT,
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
    tables = (
        (
            "Objectives",
            (ComparisonRole.OBJECTIVE,),
            {"metric", "control", "run", "delta_percent", "noise_percent", "passes", "change"},
        ),
        ("Constraints", (ComparisonRole.CONSTRAINT, ComparisonRole.REQUIREMENT), {"metric", "control", "run", "passes"}),
        ("Invariants", (), None),
        ("Reported", (ComparisonRole.REPORTED,), {"metric", "control", "run", "delta", "change"}),
        ("Breakdown", (ComparisonRole.SLICE,), {"metric", "tasks", "control", "control_min", "control_max", "run"}),
    )
    for title, roles, fields in tables:
        rows = [row.model_dump(include=fields) for row in (report.rows(*roles) if roles else report.invariants)]
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


def comparison_config(launch: dict, recorded: RecordedLaunch | None = None) -> dict[str, object]:
    """Compare complete native configuration with declared runtime aliases resolved."""
    recorded = recorded or RecordedLaunch.from_document(launch, attempt=launch["run"]["attempt_id"])
    config = deepcopy(launch)
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
                    init=(presets.published_artifact(candidate.init) if candidate.init else None),
                    settings=candidate.settings,
                )
                step = recipe.run_step(preset, spec)
                path = root / f"config-{index}.yaml"
                path.write_text(
                    materialized_config(step, presets.artifact_prefix(preset.recipe.layout.cluster)).launch_config_yaml
                )
                paths.append(str(path))
            except (ValueError, click.BadParameter) as error:
                raise InvalidComparison("recorded inputs cannot build the current preset") from error

        return tuple(compose_configs(paths))


def compose_configs(paths: Sequence[str]) -> list[dict]:
    """Load launch files with the native SkyRL composer in its CPU environment."""
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
        return json.loads(result.stdout.splitlines()[-1])
    except (subprocess.SubprocessError, ValueError, IndexError) as error:
        raise InvalidComparison("current preset recipe cannot be composed") from error


def _changed_config_keys(left: dict[str, object], right: dict[str, object]) -> set[str]:
    return {
        key
        for key in left.keys() | right.keys()
        if key not in left or key not in right or canonical_json(left[key]) != canonical_json(right[key])
    }


def _setting_keys(inputs: GateInputs, keys: set[str]) -> set[str]:
    values = presets.override_document(inputs.settings)
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


def _check_control_config(config: dict, current: dict[str, object], recorded: RecordedLaunch | None = None) -> None:
    differences = _changed_config_keys(comparison_config(config, recorded), current)
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
        receipts = [read_stored_skyrl_attempt(path) for path in run_paths]
        metadata = metadata.model_copy(
            update={"attempt_ids": tuple(receipt.attempt_id for receipt in receipts if receipt)}
        )
        runs = [
            with_requirements(with_objectives(load_run(path, preset, inputs), preset), inputs.requirements)
            for path in run_paths
        ]
        if runs:
            fields = {"tree": "tree_hash", "base_commit": "base_commit", "dirty": "dirty"}
            metadata = metadata.model_copy(
                update={
                    "eval_lite": runs[0].eval_lite,
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
            pin: [with_objectives(load_run(path, preset), preset) for path in paths]
            for pin, paths in zip(pins, control_paths, strict=True)
        }
        controls = [candidate for group in grouped.values() for candidate in group]
        if not runs or not controls:
            raise InvalidComparison("comparison needs recorded runs and control replicates")
        control = control.model_copy(
            update={"eval_lite": controls[0].eval_lite, "engines": tuple(run.engine for run in controls)}
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
        if any(candidate.task_set_ids != controls[0].task_set_ids for candidate in (*runs, *controls)):
            raise InvalidComparison("held-out evaluation task IDs differ")
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
            differences = _changed_config_keys(comparison_config(candidate.config, candidate.recorded), expected)
            if differences:
                raise InvalidComparison(
                    f"candidate config differs from recorded inputs: {', '.join(sorted(differences))}"
                )
        for candidate in controls:
            _check_control_config(candidate.config, current, candidate.recorded)
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
        path = StoragePath(output) / name
        path.parent.mkdirs()
        path.write_text(value)
    return report


def control_name(preset: str, commit: str, replicate: int) -> str:
    return f"{RUN_PREFIX}/control/{preset}-{commit}-r{replicate}"


def control_pins(preset: presets.Preset) -> tuple[ControlPin, ...]:
    control_preset = preset.shared_control or preset.name
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
    return presets.run_path(preset.recipe.layout.cluster, control_name(pin.preset, pin.commit, index), pin.revision)


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
                f"{control_name(pin.preset, pin.commit, index)}@{pin.revision}",
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
            receipt = read_skyrl_attempt(path)
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


@click.command(cls=SnowballCommand, help=__doc__)
@click.option(
    "--preset",
    "preset_name",
    type=click.Choice(sorted(name for name, preset in presets.PRESETS.items() if preset.gated)),
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
    pins = control_pins(preset)
    index = None
    if leave_out is not None:
        if not re.fullmatch(r"r\d+", leave_out) or int(leave_out[1:]) >= pins[0].runs:
            raise click.BadParameter(f"choose r0 through r{pins[0].runs - 1}", param_hint="--leave-out")
        index = int(leave_out[1:])
        candidate_name = control_name(pins[0].preset, pins[0].commit, index)
    name, separator, version = candidate_name.partition("@")
    source = presets.run_path(preset.recipe.layout.cluster, name, version if separator else preset.revision)
    namespace = user_owned_name(
        f"{presets.RUN_PREFIX}/rescore/{preset_name}-{fingerprint_hash(canonical_json((source, pins, index)))}"
    )
    recorded = GateInputs(preset_name=pins[0].preset) if index is not None else read_gate_inputs(source)
    if index is None and recorded.preset_name != preset_name:
        raise InvalidComparison(f"recorded preset is {recorded.preset_name}, not {preset_name}")
    inputs = recorded.model_copy(update={"requirements": tuple(dict.fromkeys((*recorded.requirements, *requirements)))})
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
