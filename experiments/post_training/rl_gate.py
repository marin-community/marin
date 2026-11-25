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
from pathlib import Path, PurePosixPath

import click
from finestore.rl.rl_progress import Change, Comparison, ComparisonRole, Effect, Invariant, Verdict
from finestore.rl.run_records import AttemptRecord
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
    skyrl_cpu_command,
    skyrl_cpu_environment,
    skyrl_execution,
    skyrl_step,
)
from marin.rl.skyrl_records import RecordedLaunch, read_stored_skyrl_attempt
from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.provenance import Provenance
from scipy.stats import t
from skyrlrecipe.documents import MISSING, get_path, set_path
from skyrlrecipe.model_staging import DEFAULT_TOKENIZER_REVISION, HuggingFaceModelCache, model_metadata_relative_path
from tabulate import tabulate

from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_command import EXIT_CODES, RLCommand
from experiments.post_training.rl_measurements import (
    InvalidComparison,
    RunInputs,
    RunMeasurements,
    read_measurements,
    read_recorded_launch,
    read_run_inputs,
)
from experiments.post_training.rl_presets import ControlPin, ModelPresets
from experiments.post_training.rl_scoring import AAII_WEIGHTS

GATE_REVISION = "2026.11.13"
GATE_REPORT_FILENAME = "gate_report.json"
GATE_MARKDOWN_FILENAME = "gate_report.md"


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
    for index in range(pin.replicates):
        path = control_path(preset, pin, index, table=table)
        receipt = read_stored_skyrl_attempt(path)
        if receipt is None:
            raise InvalidComparison(f"control receipt is unavailable: {path}")
        launch = read_recorded_launch(receipt)
        _check_control_config(receipt.native.config, defaults, launch, preset=preset)
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
    version: str | None = None
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


def _normalize_model_bindings(config: dict, recorded: RecordedLaunch) -> None:
    aliases = [recorded.model_uri, recorded.model_local_path]
    model_metadata = model_metadata_relative_path(recorded.model_uri, recorded.model_identity)
    model = config["inputs"]["model"]
    tokenizer_uri = model.get("tokenizer_uri")
    tokenizer_metadata = (
        model_metadata_relative_path(tokenizer_uri, model.get("tokenizer_revision") or DEFAULT_TOKENIZER_REVISION)
        if tokenizer_uri
        else model_metadata
    )
    model_staging = "<model-staging>"
    model["local_path"] = model_staging
    for path, metadata in (
        ("skyrl.trainer.policy.model.path", model_metadata),
        ("skyrl.trainer.ref.model.path", model_metadata),
        ("skyrl.trainer.policy.model.tokenizer_path", tokenizer_metadata),
        ("skyrl.trainer.ref.model.tokenizer_path", tokenizer_metadata),
        ("skyrl.generator.model_name", None),
        ("skyrl.generator.trajectory_retention.model_path", None),
    ):
        value = get_path(config, path)
        if value is MISSING:
            continue
        staged = (
            metadata is not None
            and isinstance(value, str)
            and PurePosixPath(value).is_absolute()
            and PurePosixPath(value).parts[-len(metadata.parts) :] == metadata.parts
        )
        if value not in aliases and not staged:
            raise InvalidComparison(f"native model binding differs from declared inputs: {path}")
        if staged and path.endswith(".tokenizer_path"):
            revision_path = path.removesuffix("tokenizer_path") + "tokenizer_revision"
            if get_path(config, revision_path) is not None:
                raise InvalidComparison(f"native tokenizer revision differs from the staged tokenizer: {revision_path}")
            # Staging resolves the revision into the path, so the worker no longer needs it.
            set_path(config, revision_path, recorded.model_identity)
        set_path(config, path, model_staging)


def _comparison_document(launch: dict, recorded: RecordedLaunch | None = None) -> dict:
    """Normalize native configuration and resolve its runtime aliases for comparison."""
    recorded = recorded or RecordedLaunch.from_document(launch, attempt=launch["run"]["attempt_id"])
    config = deepcopy(launch)
    config["run"].pop("index", None)
    if get_path(config, "skyrl.trainer.rollout_buffer.groups_per_family") == {}:
        del config["skyrl"]["trainer"]["rollout_buffer"]["groups_per_family"]
    data = config["skyrl"]["data"]
    # Saved controls still record unused task-cache and source-binding defaults.
    data.pop("task_cache_dir", None)
    if data.get("terminal_bench_data") == []:
        del data["terminal_bench_data"]
    task_sessions = config["skyrl"].get("environment", {}).get("task_sessions", {})
    if task_sessions.get("agent_harnesses") == {}:
        del task_sessions["agent_harnesses"]
    trajectory_runner = config["skyrl"].get("trajectory_runner", {})
    if trajectory_runner.get("max_concurrent_training_machine_tasks") is None:
        trajectory_runner.pop("max_concurrent_training_machine_tasks", None)
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

    _normalize_model_bindings(config, recorded)
    replace_alias(
        "skyrl.generator.engine_init_kwargs.served_model_name",
        [uri.rstrip("/").rsplit("/", 1)[-1] for uri in (recorded.model_uri, recorded.model_local_path)],
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
    for field in ("run_id", "model", "attempt_id", "coordinator_iris_job_id", "iris_job_id", "run_index"):
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
    # Grader reports are attempt diagnostics, like other launch-output destinations.
    trainer = config["skyrl"]["trainer"]
    trainer.pop("grader_report_dir", None)
    trainer.pop("dump_grader_report", None)
    for section in ("trainer", "checkpoint_export"):
        export = config["skyrl"].get(section, {})
        if "hf_hub_private" in export and not export.get("hf_hub_repo_id"):
            # Without an upload destination, compare visibility using the stored controls' False default.
            export["hf_hub_private"] = False
    return config


def _normalize_draft_mirror(config: dict, expected_launch: dict) -> None:
    """Normalize a draft mirror when its receipt matches the planned Hub source and revision."""
    path = "skyrl.generator.speculative_decoding.model"
    draft, expected = get_path(config, path), get_path(expected_launch, path)
    if not isinstance(draft, dict) or not isinstance(expected, dict):
        return
    expected_uri = expected["source_uri"]
    model_id = expected_uri.removeprefix("hf://")
    if model_id == expected_uri:
        return
    mirrors = config["inputs"].get("hugging_face_models", [])
    matches = [
        index
        for index, mirror in enumerate(HuggingFaceModelCache(**row) for row in mirrors)
        if mirror.uri == draft["source_uri"]
        and mirror.identity == draft["source_identity"]
        and mirror.model_id == model_id
        and mirror.revision == mirror.resolved_revision == expected["source_identity"]
    ]
    if len(matches) == 1:
        draft.update(source_uri=expected_uri, source_identity=expected["source_identity"])
        del mirrors[matches[0]]


def _gate_config(
    launch: dict, recorded: RecordedLaunch | None = None, *, expected_launch: dict | None = None
) -> dict[str, object]:
    config = _comparison_document(launch, recorded)
    if expected_launch is not None:
        _normalize_draft_mirror(config, expected_launch)
    config["iris"].pop("priority", None)
    return _config_values(config)


def config_version(launch: dict, recorded: RecordedLaunch | None = None) -> str:
    """Fingerprint the effective launch configuration used by the gate."""
    return fingerprint_hash(canonical_json(_gate_config(launch, recorded)))


def compose_preset(preset: presets.Preset) -> dict:
    """Compose the preset with its published inputs and defaults."""
    (config,) = _compose_plans(preset, (RunInputs(preset_name=preset.name),))
    return config


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


def run_native_json(*command: str, timeout: int = 120):
    """Run a SkyRL CPU command that returns JSON on stdout."""
    try:
        result = subprocess.run(
            skyrl_cpu_command(*command),
            capture_output=True,
            env=skyrl_cpu_environment(),
            text=True,
            timeout=timeout,
        )
        result.check_returncode()
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        cause = f"timed out after {error.timeout:g} s" if isinstance(error, subprocess.TimeoutExpired) else str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr and error.stderr.strip():
            cause = error.stderr.strip().splitlines()[-1]
        raise InvalidComparison(f"native configuration cannot be composed: {cause}") from error


def compose_configs(paths: Sequence[str], *, validate_driver: bool = False) -> list[dict]:
    """Compose launch files and optionally check the driver configuration in SkyRL's CPU environment."""
    options = ("--validate-driver",) if validate_driver else ()
    return run_native_json(
        "python",
        "-I",
        "-m",
        "cloud.iris.config_cli",
        "--checkout",
        str(recipe.REPO_ROOT / "vendor/skyrl"),
        "compose",
        *options,
        "--",
        *paths,
    )


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
    return run_native_json(
        "python",
        "-I",
        "-m",
        "cloud.iris.config_cli",
        "--checkout",
        native_project,
        "schema-defaults",
        json.dumps(dict(groups)),
    )


def _check_control_config(
    config: dict, current_config: dict, recorded: RecordedLaunch | None = None, *, preset: presets.Preset
) -> None:
    observed = _gate_config(config, recorded, expected_launch=current_config)
    current = _gate_config(current_config)
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


def _write_gate(
    preset: presets.Preset,
    run_paths: tuple[str, ...],
    control_paths: tuple[str, ...],
    control_pin: ControlPin,
    inputs: RunInputs,
    output: str,
    run_attempts: tuple[AttemptRecord | None, ...] = (),
    control_attempts: tuple[AttemptRecord | None, ...] = (),
) -> GateReport:
    preset_name = preset.name
    metadata = RunMetadata(**inputs.model_dump(), replicates=len(run_paths))
    control = ControlMetadata(commit=control_pin.commit, version=control_pin.version, replicates=len(control_paths))
    invariants, comparisons = [], []
    verdict, effect, reason = Verdict.INVALID, None, None
    try:
        metadata = metadata.model_copy(
            update={"attempt_ids": tuple(receipt.attempt_id for receipt in run_attempts if receipt is not None)}
        )
        for paths, attempts in ((run_paths, run_attempts), (control_paths, control_attempts)):
            for path, receipt in zip(paths, attempts, strict=True):
                if receipt is None:
                    raise InvalidComparison(f"no stored native attempt at {path}")
        runs = [
            with_requirements(
                with_objectives(read_measurements(path, preset, inputs, attempt=receipt), preset), inputs.requirements
            )
            for path, receipt in zip(run_paths, run_attempts, strict=True)
        ]
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
        controls = [
            with_objectives(read_measurements(path, preset, attempt=receipt), preset)
            for path, receipt in zip(control_paths, control_attempts, strict=True)
        ]
        if not runs or not controls:
            raise InvalidComparison("comparison needs recorded runs and control replicates")
        control = control.model_copy(
            update={"held_out_tasks": controls[0].held_out_tasks, "engines": tuple(run.engine for run in controls)}
        )
        for candidate in controls:
            _check_control(candidate, control_pin)
        if any(candidate.evaluation_identity != controls[0].evaluation_identity for candidate in (*runs, *controls)):
            raise InvalidComparison("evaluation protocol or held-out TaskSpec content differs")
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
            observed = _gate_config(candidate.config, candidate.recorded, expected_launch=expected_config)
            differences = _changed_config_keys(observed, expected)
            if differences:
                raise InvalidComparison(
                    f"candidate config differs from recorded inputs: {', '.join(sorted(differences))}"
                )
        for candidate in controls:
            _check_control_config(candidate.config, current_config, candidate.recorded, preset=preset)
        comparisons = _compare(preset, runs, controls, inputs.requirements)
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


def control_pin(preset: presets.Preset, *, table: ModelPresets) -> ControlPin:
    pin = table.control_pins.get(preset.control_preset)
    if pin is None:
        raise InvalidComparison(
            f"no control pin for {preset.name}; collect current controls with --preset {preset.control_preset} "
            "--version dev --control --replicates 4 --run"
        )
    return pin


def control_path(preset: presets.Preset, pin: ControlPin, index: int, *, table: ModelPresets) -> str:
    return recipe.run_path(preset, control_name(pin.preset, pin.commit, index, table=table), pin.version)


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
            f"{control_name(pin.preset, pin.commit, index, table=table)}@{pin.version}",
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
            config.update(ctx.runtime_arg("evidence"))
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


def _stored_attempt(path: str) -> AttemptRecord | None:
    try:
        return read_stored_skyrl_attempt(path)
    except (FileNotFoundError, SkyRLReceiptLookupError):
        return None


def _gate_with_evidence(
    step: ArtifactStep[GateReport], run_attempts: tuple[AttemptRecord | None, ...] | None = None
) -> ArtifactStep[GateReport]:
    target = gate_target(step)
    if run_attempts is None:
        run_attempts = tuple(map(_stored_attempt, target.outputs))
    control_attempts = tuple(map(_stored_attempt, target.control_outputs))
    return replace(
        step,
        name=target.evidence_name(run_attempts, control_attempts),
        runtime_args={
            **step.runtime_args,
            "evidence": {"run_attempts": run_attempts, "control_attempts": control_attempts},
        },
    )


class GateTarget(BaseModel):
    """The gate and ordered input addresses fixed at submission."""

    name: str
    version: str
    candidates: tuple[str, ...]
    outputs: tuple[str, ...]
    controls: tuple[str, ...]
    control_outputs: tuple[str, ...]

    def evidence_name(
        self, candidates: tuple[AttemptRecord | None, ...], controls: tuple[AttemptRecord | None, ...]
    ) -> str:
        evidence = [
            (identity, receipt.native.model_dump(mode="json") if receipt is not None else None)
            for identities, receipts in ((self.candidates, candidates), (self.controls, controls))
            for identity, receipt in zip(identities, receipts, strict=True)
        ]
        return self.name + "-evidence-" + fingerprint_hash(canonical_json(evidence))


def gate_target(step: ArtifactStep[GateReport]) -> GateTarget:
    config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
    paths = step.runtime_args["paths"]
    return GateTarget(
        name=step.runtime_args["name"],
        version=step.version,
        candidates=tuple(config["run_paths"]),
        outputs=tuple(paths["run_paths"]),
        controls=tuple(config["control_paths"]),
        control_outputs=tuple(paths["control_paths"]),
    )


def gate_step(
    preset: presets.Preset,
    runs: tuple[ArtifactStep[SkyRLRun], ...],
    *,
    group_name: str,
    table: ModelPresets,
    inputs: RunInputs,
) -> ArtifactStep[GateReport]:
    pin = control_pin(preset, table=table)
    name = f"{group_name}/gate-n{len(runs)}-{pin.commit}"
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
        help="Saved run to compare, by the name and artifact version the status block prints for it. "
        "Give this or --leave-out.",
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
        if not separator:
            version = pin.version if index is not None else recipe.RUN_ARTIFACT_VERSION
        source = recipe.run_path(preset, name, version)
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
            (f"{name}@{version}",),
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


def completed_gate(
    gate: ArtifactStep[GateReport] | None, attempts: tuple[AttemptRecord | None, ...] | None = None
) -> ArtifactStep[GateReport] | None:
    """Bind native attempt evidence to the gate identity fixed at submission."""
    return _gate_with_evidence(gate, attempts) if gate is not None else None
