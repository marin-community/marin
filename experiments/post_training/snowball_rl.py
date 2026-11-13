# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Snowball RL experiment and compare a gated preset with pinned controls.

\b
  speed-short      engine throughput and full EvalLite quality; 8 H100, about 15 min.
  speed-medium     asynchronous update time and mismatch; 40 H100, runtime unmeasured.
  speed-long       asynchronous update time and mismatch; 40 H100, 45-75 min.
  learning-medium  EvalLite score after 25 async updates; 40 H100, runtime unmeasured.
  data-short       mixed-answer fraction on --tasks; 8 H100, about 15 min.
  data-medium      learning-medium on --tasks; 40 H100, runtime unmeasured.
  train-long       async training, checkpoints and HF export; 40 H100 east; no verdict.

Guide: docs/recipes/snowball-rl.md. Without --run, print the plan and saved status.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl`:
  --preset speed-medium --version dev --run --wait
  --preset learning-medium --set trainer.algorithm.advantage_estimator=grpo \
    --set trainer.algorithm.group_advantage_min_size=null --version dev --run
  --preset speed-medium --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
  --preset learning-medium --require 'eval_lite/aaii_category/coding/mean>=control' --version dev --run
  --preset learning-medium --version dev --run
  --preset data-short --tasks my-selection@2026.11.02 --version dev --run
  --preset train-long --set trainer.max_steps=100 --version dev --run
"""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import partial, wraps
from pathlib import Path

import click
import yaml
from finestore.rl.run_records import AttemptRecord
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from marin.execution.artifact import is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.lazy import run as build_artifacts
from marin.execution.step_status import get_status_path
from marin.experiment.cli import build_options_with_runner
from marin.rl.cli import submit_or_run
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    IrisSkyRLExecution,
    SkyRLReceiptLookupError,
    SkyRLRun,
    SkyRLRunConfig,
    read_skyrl_attempt,
    run_skyrl,
)
from marin.rl.skyrl_records import (
    RecordedLaunch,
    attempt_tracking,
    read_skyrl_metrics,
    read_stored_skyrl_attempt,
    recorded_launch,
)
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance, git_output

from experiments.post_training import snowball_rl_command as command
from experiments.post_training import snowball_rl_gate, snowball_rl_publication
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_recipe as recipe
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
)

_WANDB_ENTITY_ENV = "WANDB_ENTITY"
_FAILED_RUN_STATES = {JobState.FAILED, JobState.KILLED, JobState.WORKER_FAILED, JobState.UNSCHEDULABLE}


@dataclass(frozen=True)
class SnowballRun:
    runs: tuple[ArtifactStep[SkyRLRun], ...]
    gate: ArtifactStep[snowball_rl_gate.GateReport] | None = None
    inputs: snowball_rl_gate.GateInputs | None = None


def _provenance(*, control: bool) -> Provenance:
    root = presets.REPO_ROOT
    provenance = Provenance.capture(root)
    if (root / ".git").exists():
        untracked = git_output(
            ["ls-files", "--others", "--exclude-standard", "experiments", "lib", "vendor"], cwd=root
        ).splitlines()
        if untracked:
            raise click.BadParameter(f"stage source files before naming a run: {', '.join(untracked)}")
    if not provenance.tree_hash or (control and provenance.dirty):
        raise click.BadParameter("controls require a clean checkout; runs require git or Iris source provenance")
    return provenance


def _run_snowball(config: SkyRLRunConfig, *, inputs: snowball_rl_gate.GateInputs) -> SkyRLRun:
    path = StoragePath(config.output.resolved_config_uri).parent / snowball_rl_gate.GATE_INPUTS_FILENAME
    if not path.exists():
        path.parent.mkdirs()
        path.write_text(inputs.model_dump_json(indent=2) + "\n")
    return run_skyrl(config)


def build_run(
    preset: presets.Preset,
    *,
    tasks: ArtifactStep | None = None,
    init: ArtifactStep | None = None,
    settings: tuple[str, ...] = (),
    replicates: int = 1,
    control: bool = False,
    requirements: tuple[str, ...] = (),
) -> SnowballRun:
    """Construct a Snowball experiment: one run step per replicate and, for a gated preset, its gate step."""
    wandb_entity = os.environ.get(_WANDB_ENTITY_ENV)
    if not 1 <= replicates <= 5:
        raise click.BadParameter("replicates must be between 1 and 5")
    if settings and not preset.accepts_settings:
        raise click.BadParameter("this preset accepts no --set")
    if preset.tasks_input == "required":
        if tasks is None and not control:
            raise click.BadParameter("data presets require --tasks and accept no --set")
    elif preset.tasks_input == "forbidden" and tasks is not None:
        raise click.BadParameter("use a data or train-long preset with --tasks")
    if not preset.accepts_init and init is not None:
        raise click.BadParameter("--init requires a learning or train-long preset")
    if (not preset.accepts_replicates and replicates != 1) or (not preset.accepts_control and control):
        raise click.BadParameter(f"{preset.name} does not accept this replicate or control selection")
    if control and (settings or tasks is not None or init is not None or preset.shared_control is not None):
        raise click.BadParameter("controls use only the preset's pinned default inputs")
    default, selection = presets.task_selections(preset, tasks)
    if is_mutable_version(selection.version):
        raise click.BadParameter("curated task selections must have an immutable version")
    summary = presets.selection_summary(selection)
    if (
        tasks is not None
        and preset.tasks_input == "required"
        and summary["recipe"]["heldout"] != presets.selection_summary(default)["recipe"]["heldout"]
    ):
        raise click.BadParameter(
            "candidate tasks use a different heldout input; curate against the default task selection"
        )
    if tasks is not None:
        task_set = presets.selection_artifact(preset.eval_lite)
        excluded = summary["recipe"].get("exclusions", ())
        expected = str(StoragePath(task_set.path()) / FULL_VALIDATION_FILENAME)
        if not any(row["uri"] == expected and row["version"] == task_set.version for row in excluded):
            raise click.BadParameter("candidate tasks must record the full EvalLite in recipe.exclusions")
    spec = recipe.run_spec(preset, name="snowball-config", tasks=tasks, init=init, settings=settings)
    evaluation_inputs = [
        (evaluation.name, source.name, source.version)
        for evaluation in preset.evaluations
        for source in (evaluation.tasks(selection, default, preset.eval_lite),)
    ]
    changes = {
        "tasks": (selection.name, selection.version),
        "init": None if init is None else (init.name, init.version),
        "layout": preset.recipe.layout.name,
        "settings": sorted(presets.override_document(settings).items()),
        "evaluations": sorted(evaluation_inputs),
    }
    snowball_rl_gate.validate_requirements(preset, requirements, control=control)
    provenance = _provenance(control=control)
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True, default=str))
    inputs = snowball_rl_gate.gate_inputs(
        preset.name,
        settings,
        f"{tasks.name}@{tasks.version}" if tasks is not None else None,
        f"{init.name}@{init.version}" if init is not None else None,
        requirements,
    )
    runs = []
    for replicate in range(replicates):
        name = presets.run_name(preset, provenance, inputs_hash, replicate, control=control)
        run = recipe.run_step(preset, replace(spec, name=name), wandb_entity=wandb_entity)
        runs.append(replace(run, run=partial(_run_snowball, inputs=inputs)))
    gate = (
        snowball_rl_gate.gate_step(
            preset,
            tuple(runs),
            inputs=inputs,
        )
        if preset.gated and not control
        else None
    )
    return SnowballRun(tuple(runs), gate, inputs)


def _changed_paths(ref: str, paths: tuple[str, ...]) -> tuple[list[dict[str, str]] | None, str | None]:
    """Return source changes, or the reason git comparison is unavailable."""
    if not (presets.REPO_ROOT / ".git").exists():
        return None, "git metadata is unavailable on this host"
    try:
        output = git_output(["diff", "--no-renames", "--name-status", "-z", ref, "--", *paths], cwd=presets.REPO_ROOT)
    except (RuntimeError, FileNotFoundError) as error:
        return None, f"cannot compare control commit {ref}: {error}; run git fetch origin to obtain its ref"
    fields = output.split("\0")[:-1] if output else []
    return [{"path": path, "status": status} for status, path in zip(fields[::2], fields[1::2], strict=True)], None


def _grafana_links(receipt: AttemptRecord | None, cluster: str, launch: RecordedLaunch | None) -> dict[str, str]:
    if receipt is None or not receipt.native.result.iris_job_id:
        return {}
    run_id = receipt.run_id
    if not run_id or receipt.native.result.started_at is None:
        return {}
    training_type = launch.training_type if launch else None
    if training_type not in ("sync", "async"):
        return {}
    return presets.grafana_links(
        run_id=run_id,
        cluster=cluster,
        training_type=training_type,
        started_ms=presets.utc_millis(receipt.native.result.started_at),
        finished_ms=presets.utc_millis(receipt.native.result.finished_at) if receipt.native.result.finished_at else None,
    )


def _completed_update(path: str, workload: presets.Training | presets.Generation) -> int | None:
    key = workload.completion_metric
    try:
        metrics = read_skyrl_metrics(path)
    except FileNotFoundError:
        return None
    for _, values in sorted(metrics.items(), reverse=True):
        if key in values and workload.completed_row(values):
            return values["step"]
    return None


def _comparison_line(row: snowball_rl_gate.Comparison) -> str:
    if row.run is None or row.control is None:
        return f"{row.metric}: unavailable"
    values = f"{row.metric}: {row.control:.6g} -> {row.run:.6g}"
    if row.role in (snowball_rl_gate.ComparisonRole.CONSTRAINT, snowball_rl_gate.ComparisonRole.REQUIREMENT):
        return values + ("; within bound" if row.passes else "; outside bound")
    delta = f"{row.delta_percent:+.1f}%" if row.delta_percent is not None else "unavailable"
    noise = f"{row.noise_percent:.1f}%" if row.noise_percent is not None else "unavailable"
    return f"{values}; {delta}; noise +/-{noise}; {row.change}"


def _status(
    run: SnowballRun,
    preset: presets.Preset,
    *,
    stored_only: bool = False,
    issue: int | None = None,
    publication_urls: dict[str, str] | None = None,
) -> dict:
    rows = []
    planned = []
    for step in run.runs:
        execution = next(value for value in step.runtime_args.values() if isinstance(value, IrisSkyRLExecution))
        path = step.path(presets.artifact_prefix(execution.cluster))
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        planned.append(yaml.safe_load(config.launch_config_yaml))
        try:
            receipt = read_stored_skyrl_attempt(path) if stored_only else read_skyrl_attempt(path)
        except SkyRLReceiptLookupError as error:
            raise click.BadParameter(str(error)) from error
        result = receipt.native.result if receipt is not None else None
        try:
            launch = recorded_launch(receipt) if receipt is not None else None
        except ValueError as error:
            raise snowball_rl_gate.InvalidComparison(str(error)) from error
        tracking = attempt_tracking(receipt, launch) if receipt is not None else {}
        team = launch.wandb_entity if launch else execution.wandb_entity
        grafana = _grafana_links(receipt, execution.cluster, launch)
        state = (
            result.state.value
            if result
            else (
                StoragePath(get_status_path(path)).read_text().strip()
                if StoragePath(get_status_path(path)).exists()
                else "not built"
            )
        )
        rows.append(
            {
                "name": step.name,
                "version": step.version,
                "state": state,
                "iris_job_id": result.iris_job_id if result else None,
                "wandb_entity": team or recipe.DEFAULT_WANDB_ENTITY,
                **{
                    key: tracking.get(key)
                    for key in ("attempt_id", "wandb_url", "started_at", "engine")
                    if tracking.get(key) is not None
                },
                **({"grafana_links": grafana} if grafana else {}),
                "latest_completed_update": _completed_update(path, preset.recipe.layout.workload),
                "cluster": execution.cluster,
                "allocation": {
                    **planned[-1]["iris"]["allocation"],
                    "cpu": execution.cpu,
                    "memory": execution.memory,
                    "disk": execution.disk,
                },
                "output": path,
                "issue": issue,
                "rl_progress_url": (publication_urls or {}).get(path),
            }
        )
    verdict = {}
    report = None
    if run.gate is not None:
        path = run.gate.path(prefix=presets.artifact_prefix(preset.recipe.layout.cluster))
        report = snowball_rl_gate.read_gate_report(path)
        if report is not None:
            verdict = {
                "report": str(StoragePath(path) / snowball_rl_gate.GATE_REPORT_FILENAME),
                **report.model_dump(mode="json", include={"verdict", "label", "reason"}),
            }
            changes, reason = _changed_paths(
                report.control.commit,
                ("experiments/post_training", "lib/marin/src/marin/rl", "lib/taskcompendium", "vendor/skyrl"),
            )
            verdict["rl_path_changes_since_control"] = changes
            if reason:
                verdict["rl_path_changes_reason"] = reason
    if "verdict" not in verdict and preset.gated:
        crashed = next((row for row in rows if str(row["state"]).lower() in _FAILED_RUN_STATES), None)
        if crashed is not None:
            verdict = {
                "verdict": snowball_rl_gate.Verdict.INVALID.value,
                "reason": f"native job {crashed['iris_job_id']} is {crashed['state']}",
                "iris_job_id": crashed["iris_job_id"],
            }
    objective_rows = report.rows(snowball_rl_gate.ComparisonRole.OBJECTIVE) if report else ()
    constraint_rows = (
        report.rows(snowball_rl_gate.ComparisonRole.CONSTRAINT, snowball_rl_gate.ComparisonRole.REQUIREMENT)
        if report
        else ()
    )
    objectives = [_comparison_line(row) for row in objective_rows] or [
        f"{metric.key}: unavailable" for metric in preset.objectives
    ]
    constraints = [_comparison_line(row) for row in constraint_rows] or [
        f"{metric.key}: unavailable"
        for metric in preset.guards
        if metric.key not in {item.key for item in preset.objectives}
    ]
    invariant_lines = [
        f"r{row.replicate} {row.name}: {row.value} {row.comparator or '(comparator unknown)'} {row.limit}; "
        + ("within bound" if row.holds else "outside bound")
        for row in (report.invariants if report else ())
    ]
    return {
        "preset": preset.name,
        "objectives": objectives,
        "constraints": [*constraints, *invariant_lines],
        "result": (
            f"{verdict['verdict']} ({verdict['label']})" if verdict.get("label") else verdict.get("verdict", "pending")
        ),
        "code": asdict(Provenance.capture(presets.REPO_ROOT)),
        "inputs": {
            "model": planned[0]["inputs"]["model"]["identity"],
            **{
                name: [source["identity"] for source in planned[0]["inputs"][name]]
                for name in ("train_data", "validation_data")
            },
        },
        "context_budget": planned[0]["skyrl"]["context_budget"],
        "updates": preset.recipe.layout.workload.update_count(planned[0]["skyrl"]["trainer"]["max_steps"]),
        "measured_updates": list(preset.measured),
        "config_changes": list(report.run.config_changes) if report else [],
        "budget_estimate": preset.budget,
        "guards": [metric.key for metric in preset.guards],
        "gaps": [gap.value for gap in preset.gaps],
        "runs": rows,
        **verdict,
    }


def _from_handles(handles: list[ArtifactStep]) -> SnowballRun:
    return SnowballRun(
        tuple(step for step in handles if step.artifact_type is SkyRLRun),
        next((step for step in handles if step.artifact_type is snowball_rl_gate.GateReport), None),
    )


@contextmanager
def _iris_client() -> Iterator[IrisClient]:
    context = get_iris_ctx()
    if context is not None and context.client is not None:
        yield context.client
    else:
        with open_iris_client(config_file=Path(IRIS_HUB_CLUSTER_CONFIG), workspace=presets.REPO_ROOT) as client:
            yield client


def _publish_records(
    run: SnowballRun, preset: presets.Preset, *, issue: int | None = None
) -> snowball_rl_publication.PublicationOutcome | None:
    if not preset.publishes:
        return None
    prefix = presets.artifact_prefix(preset.recipe.layout.cluster)
    gate_path = run.gate.path(prefix) if run.gate is not None else None
    outputs = [training.path(prefix) for training in run.runs]
    return snowball_rl_publication.publish_runs(outputs, preset, gate_path=gate_path, issue=issue)


def _wait_for_run(
    job: JobName, handles: list[ArtifactStep], preset: presets.Preset, *, issue: int | None = None
) -> None:
    with _iris_client() as client:
        parent = client.job(job).wait(recipe.RUN_TIMEOUT_HOURS * 3600, poll_interval=60, raise_on_failure=False)
    coordinator_error = (
        f"coordinator {job} is {parent.state}; required verdict unavailable"
        if parent.state != JobState.SUCCEEDED
        else None
    )
    run = _from_handles(handles)
    run = replace(run, gate=snowball_rl_gate.completed_gate(run.gate))
    try:
        publication = _publish_records(run, preset, issue=issue)
    except Exception as error:
        publication = snowball_rl_publication.PublicationOutcome(
            errors=[str(error)], retry="Repeat the original command with --wait."
        )
    snapshot = _status(
        run,
        preset,
        issue=issue,
        stored_only=True,
        publication_urls=publication.urls if publication is not None else None,
    )
    if publication is not None:
        snapshot["publication"] = publication.to_dict()
    if "verdict" not in snapshot and (coordinator_error is not None or run.gate is not None):
        snapshot["reason"] = coordinator_error or "required verdict unavailable"
    click.echo(yaml.safe_dump(snapshot, sort_keys=False, width=1000))
    if "report" in snapshot:
        click.echo((StoragePath(snapshot["report"]).parent / snowball_rl_gate.GATE_MARKDOWN_FILENAME).read_text())
    verdict = snapshot.get("verdict")
    if verdict is None and run.gate is None and coordinator_error is None:
        verdict = snowball_rl_gate.Verdict.PASS
    raise click.exceptions.Exit(command.EXIT_CODES[verdict])


def _run_command(function: Callable) -> click.Command:
    @wraps(function)
    def checked(*args, wait: bool = False, **kwargs):
        if wait and not click.get_current_context().params["do_run"]:
            raise click.BadParameter("--wait requires --run")
        return function(*args, **kwargs)

    def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
        options = click.get_current_context().params
        preset = presets.PRESETS[options["preset_name"]]
        try:
            if get_job_info() is None:
                job = submit_or_run("experiments.post_training.snowball_rl", handles, max_concurrent, os.environ)
            else:
                planned = _from_handles(handles)
                build_artifacts(*planned.runs, max_concurrent=max_concurrent)
                gate = snowball_rl_gate.completed_gate(planned.gate)
                if gate is not None:
                    build_artifacts(gate, max_concurrent=max_concurrent)
                job = None
            if options["wait"] and job is not None:
                _wait_for_run(job, handles, preset, issue=options["issue"])
        except click.exceptions.Exit:
            raise
        except Exception as error:
            if not options["wait"]:
                raise
            traceback.print_exc()
            click.echo(yaml.safe_dump({"preset": options["preset_name"], "reason": str(error)}))
            raise click.exceptions.Exit(command.EXIT_CODES[None]) from error

    return click.command(cls=command.SnowballCommand, help=__doc__)(build_options_with_runner(checked, runner))


@_run_command
@click.option("--issue", type=click.IntRange(min=1), help="GitHub experiment issue for launcher publication.")
@click.option(
    "--wait",
    is_flag=True,
    help="Wait after --run and publish training attempts using the caller Marina login; "
    "exit 0 for PASS, 1 for FAIL, 2 for INVALID, 3 without a required verdict.",
)
@click.option(
    "--preset",
    "preset_name",
    required=True,
    type=click.Choice(sorted(presets.PRESETS)),
    help="Experiment to run; the table above says which preset fits a change.",
)
@click.option(
    "--set",
    "settings",
    multiple=True,
    metavar="KEY=VALUE",
    help="Change one typed SkyRL setting: a dotted key and a JSON value, for example "
    "trainer.algorithm.advantage_estimator=grpo. Text values may be bare. "
    "Repeatable. Keys the preset owns (placement, data, evaluation, measurement) are refused with the "
    "reason; data presets accept none. Code changes need no flag: the run name includes the source "
    "tree hash.",
)
@click.option(
    "--require",
    "requirements",
    multiple=True,
    metavar="KEY<=|>=BOUND",
    help="Add a condition the gate checks on a metric key; the examples above show two. With a "
    "number, every candidate replicate must meet the bound. With control, the candidate mean must not "
    "be worse than the controls' beyond tolerance. Repeatable; gated presets only, not with --control. "
    "A training metric the controls never recorded is refused before launch.",
)
@click.option(
    "--tasks",
    "tasks_name",
    metavar="NAME@VERSION",
    help="Curated task selection at an immutable version, recording the full EvalLite in recipe.exclusions. "
    "Required by data presets, whose selection must hold out the same task set as the default; "
    "train-long defaults to the pinned snowball-100k.",
)
@click.option(
    "--init",
    "init_name",
    metavar="NAME@VERSION",
    help="Initial weights: a published model artifact, or a finished train-long run (its HF export). "
    "learning presets and train-long only. The initial weights are evaluated too, so the report shows "
    "the gain.",
)
@click.option(
    "--replicates",
    type=click.IntRange(1, 5),
    default=1,
    show_default=True,
    help="Identical copies of a gated preset, sharing one seed. The gate compares their "
    "mean with the controls; more copies narrow the tolerance. Not for train-long.",
)
@click.option(
    "--control",
    is_flag=True,
    help="Run the preset's control replicates from a clean checkout with only its pinned default "
    "inputs. Only for pinning new controls; refuses --set, --tasks, --init, --require "
    "and data-medium, which uses the corresponding learning preset's controls.",
)
def main(
    preset_name: str,
    settings: tuple[str, ...],
    tasks_name: str | None,
    init_name: str | None,
    replicates: int,
    control: bool,
    requirements: tuple[str, ...],
    issue: int | None,
) -> tuple[ArtifactStep, ...]:
    if init_name and "@" not in init_name:
        raise click.BadParameter("use NAME@VERSION for initializer models", param_hint="--init")
    preset = presets.PRESETS[preset_name]
    run = build_run(
        preset,
        settings=settings,
        tasks=presets.selection_artifact(tasks_name) if tasks_name else None,
        init=presets.published_artifact(init_name) if init_name else None,
        replicates=replicates,
        control=control,
        requirements=requirements,
    )
    do_run = click.get_current_context().params.get("do_run", False)
    if do_run and get_job_info() is None:
        assert run.inputs is not None
        snowball_rl_gate.validate_candidate(
            preset,
            run.inputs,
            control=control,
        )
    snapshot = _status(
        replace(run, gate=snowball_rl_gate.completed_gate(run.gate)), preset, stored_only=do_run, issue=issue
    )
    click.echo(yaml.safe_dump(snapshot, sort_keys=False, width=1000))
    if not do_run and "verdict" in snapshot:
        raise click.exceptions.Exit(command.EXIT_CODES[snapshot["verdict"]])
    return (*run.runs, *((run.gate,) if run.gate else ()))


if __name__ == "__main__":
    main()
