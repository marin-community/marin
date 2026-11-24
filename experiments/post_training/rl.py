# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a declared RL preset and inspect its saved status or pinned-control verdict.

Without --run, print the plan and saved status. Use --run --wait for a terminal result.
Use the selected model entry's compare command to compare existing results on CPU.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import partial, wraps
from pathlib import Path

import click
import yaml
from finestore.rl.rl_progress import RunIndexIdentity, Verdict
from finestore.rl.run_records import AttemptRecord, write_attempt
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState, is_job_finished
from marin.execution.artifact import is_mutable_version
from marin.execution.fingerprint import canonical_json, fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.lazy import run as build_artifacts
from marin.execution.step_status import get_status_path
from marin.experiment.cli import build_options_with_runner
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import BASE_PRESET_REVISION_ENV, RUN_INDEX_ENV, submit_or_run, validate_coordinator_graph
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    SkyRLModelSource,
    SkyRLReceiptLookupError,
    SkyRLRun,
    SkyRLRunConfig,
    read_skyrl_attempt,
    run_skyrl,
    skyrl_execution,
)
from marin.rl.skyrl_records import (
    EXPORTS_SUBDIRECTORY,
    attempt_tracking,
    read_skyrl_metrics,
    read_stored_skyrl_attempt,
)
from pydantic import BaseModel, TypeAdapter
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance, git_output, username_segment

from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate, rl_publication
from experiments.post_training import rl_presets as presets
from experiments.post_training.rl_command import (
    EXIT_CODES,
    REFUSAL_EXIT_CODE,
    RLCommand,
    attempt_grafana_url,
    grafana_url,
)
from experiments.post_training.rl_measurements import (
    RUN_INPUTS_FILENAME,
    RunInputs,
    evaluation_status,
    lineage_gpu_hours,
    read_recorded_launch,
    run_inputs,
)
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME

_WANDB_ENTITY_ENV = "WANDB_ENTITY"
_PUBLICATION_POLL_SECONDS = 10
_IRIS_DASHBOARD_URL = "https://iris.oa.dev"
_FAILED_RUN_STATES = {JobState.FAILED, JobState.KILLED, JobState.WORKER_FAILED, JobState.UNSCHEDULABLE}
logger = logging.getLogger(__name__)
_SUBMITTED_INDEX = TypeAdapter(dict[str, RunIndexIdentity])


@dataclass(frozen=True)
class RLExperiment:
    runs: tuple[ArtifactStep[SkyRLRun], ...]
    gate: ArtifactStep[rl_gate.GateReport] | None = None
    inputs: RunInputs | None = None


def _provenance(*, control: bool) -> Provenance:
    root = recipe.REPO_ROOT
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


def _run_rl(
    config: SkyRLRunConfig, *, inputs: RunInputs, issue: int | None, forward_env: tuple[str, ...] = ()
) -> SkyRLRun:
    path = StoragePath(config.output.resolved_config_uri).parent / RUN_INPUTS_FILENAME
    if not path.exists():
        path.parent.mkdirs()
        path.write_text(inputs.model_dump_json(indent=2) + "\n")
    metadata = os.environ.get(RUN_INDEX_ENV)
    index = _SUBMITTED_INDEX.validate_json(metadata).get(config.run_id) if metadata is not None else None
    if issue is not None or index is not None:
        issue = rl_publication.reserve_issue(config.output.export_root, issue)
        base_revision = os.environ.get(BASE_PRESET_REVISION_ENV) if index is not None else None
        write_attempt(
            config.output.export_root,
            AttemptRecord(
                run_id=config.run_id,
                attempt_id=config.attempt_id,
                issue=issue,
                index=index,
                base_preset_revision=base_revision or None,
            ),
        )
    return run_skyrl(config, forward_env=forward_env)


def _run_name(
    preset: presets.Preset,
    provenance: Provenance,
    inputs_hash: str,
    replicate: int,
    *,
    table: presets.ModelPresets,
    control: bool,
) -> str:
    if control:
        return rl_gate.control_name(preset.name, provenance.base_commit, replicate, table=table)
    name = f"checkpoints/{table.namespace}/{preset.name}-{provenance.tree_hash}-in-{inputs_hash}"
    return user_owned_name(name + (f"-r{replicate}" if preset.gated else ""))


def build_experiment(
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    tasks: ArtifactStep | None = None,
    init: SkyRLModelSource | None = None,
    settings: tuple[str, ...] = (),
    replicates: int = 1,
    control: bool = False,
    requirements: tuple[str, ...] = (),
    issue: int | None = None,
    forward_env: tuple[str, ...] = (),
    validate_plan: bool = False,
    provenance: Provenance | None = None,
) -> RLExperiment:
    """Construct an RL experiment: one run step per replicate and, for a gated preset, its gate step."""
    wandb_entity = os.environ.get(_WANDB_ENTITY_ENV)
    if not 1 <= replicates <= 5:
        raise click.BadParameter("replicates must be between 1 and 5")
    if settings and not preset.accepts_settings:
        raise click.BadParameter("this preset accepts no --set")
    if preset.tasks_input == "required":
        if tasks is None and not control:
            raise click.BadParameter("data presets require --tasks and accept no --set")
    elif preset.tasks_input == "forbidden" and tasks is not None:
        raise click.BadParameter("this preset does not accept --tasks")
    if not preset.accepts_init and init is not None:
        raise click.BadParameter("this preset does not accept --init")
    if (not preset.gated and replicates != 1) or (control and preset.control_preset != preset.name):
        raise click.BadParameter(f"{preset.name} does not accept this replicate or control selection")
    if control and (settings or tasks is not None or init is not None):
        raise click.BadParameter("controls use only the preset's pinned default inputs")
    default, selection = recipe.task_selections(preset, tasks)
    if is_mutable_version(selection.version):
        raise click.BadParameter("curated task selections must have an immutable version")
    summary = recipe.selection_summary(selection)
    if (
        tasks is not None
        and preset.tasks_input == "required"
        and summary["recipe"]["heldout"] != recipe.selection_summary(default)["recipe"]["heldout"]
    ):
        raise click.BadParameter(
            "candidate tasks use a different development panel (recipe.heldout). "
            "Curate against the default task selection."
        )
    if tasks is not None and preset.held_out_tasks is not None:
        task_set = recipe.selection_artifact(preset.held_out_tasks)
        excluded = summary["recipe"].get("exclusions", ())
        expected = str(StoragePath(task_set.path()) / FULL_VALIDATION_FILENAME)
        if not any(row["uri"] == expected and row["version"] == task_set.version for row in excluded):
            raise click.BadParameter("candidate tasks must record the full held-out selection in recipe.exclusions")
    initializer = recipe.stored_initializer(init, preset=preset) if init is not None else None
    spec = recipe.run_spec(preset, name="rl-config", tasks=tasks, init=init, settings=settings, model_name=table.model)
    evaluation_inputs = [
        (evaluation.name, source.name, source.version)
        for evaluation in preset.evaluations
        for source in (recipe.evaluation_source(evaluation, selection, default, preset.held_out_tasks),)
    ]
    changes = {
        "tasks": (selection.name, selection.version),
        "init": initializer.model_dump(mode="json") if initializer is not None else None,
        "layout": preset.recipe.layout.name,
        "settings": sorted(recipe.override_document(settings).items()),
        "evaluations": sorted(evaluation_inputs),
    }
    rl_gate.validate_requirements(preset, requirements, table=table, control=control)
    provenance = provenance or _provenance(control=control)
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True, default=str))
    inputs = run_inputs(
        preset.name,
        settings,
        f"{tasks.name}@{tasks.version}" if tasks is not None else None,
        initializer,
        requirements,
    )
    runs = []
    for replicate in range(replicates):
        name = _run_name(preset, provenance, inputs_hash, replicate, table=table, control=control)
        run = recipe.run_step(preset, replace(spec, name=name), wandb_entity=wandb_entity, forward_env=forward_env)
        runs.append(replace(run, run=partial(_run_rl, inputs=inputs, issue=issue, forward_env=forward_env)))
    gate = (
        rl_gate.gate_step(
            preset,
            tuple(runs),
            inputs=inputs,
            table=table,
        )
        if preset.gated and not control
        else None
    )
    if validate_plan:
        rl_gate.validate_plan(preset, replace(spec, name=runs[0].name), runs[0])
    return RLExperiment(tuple(runs), gate, inputs)


def _evaluation_failed(snapshot: dict, preset: presets.Preset) -> bool:
    return bool(preset.checkpoint_every) and any(
        not fact["holds"] for row in snapshot["runs"] for fact in row.get("evaluation_invariants", ())
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


def _status(
    run: RLExperiment,
    preset: presets.Preset,
    *,
    stored_only: bool = False,
    issue: int | None = None,
    publication_urls: dict[str, str] | None = None,
) -> dict:
    rows = []
    planned = []
    for step in run.runs:
        execution = skyrl_execution(step)
        assert execution is not None
        path = step.path(recipe.artifact_prefix(execution.cluster))
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        planned.append(yaml.safe_load(config.launch_config_yaml))
        try:
            receipt = read_stored_skyrl_attempt(path) if stored_only else read_skyrl_attempt(path)
        except SkyRLReceiptLookupError as error:
            raise click.BadParameter(str(error)) from error
        result = receipt.native.result if receipt is not None else None
        launch = read_recorded_launch(receipt) if receipt is not None else None
        tracking = attempt_tracking(receipt, launch) if receipt is not None else {}
        team = launch.wandb_entity if launch else execution.wandb_entity
        grafana = attempt_grafana_url(receipt, launch) if receipt is not None and launch is not None else None
        status_path = StoragePath(get_status_path(path))
        iris_job_id = result.iris_job_id if result else None
        gpu_hours = lineage_gpu_hours(receipt, launch) if receipt is not None and launch is not None else None
        state = (
            result.state.value if result else status_path.read_text().strip() if status_path.exists() else "not built"
        )
        best = {}
        if receipt is not None and launch is not None:
            best_path = StoragePath(launch.export_root) / "best_checkpoint.json"
            native = receipt.native.config
            if best_path.exists():
                best["best_checkpoint_step"] = json.loads(best_path.read_text())["step"]
            elif preset.checkpoint_every or (
                native["run"].get("export_hf") and native["skyrl"]["trainer"].get("hf_export_checkpoint") == "best"
            ):
                best["best_checkpoint_step"] = "none eligible"
        rows.append(
            {
                "name": step.name,
                "version": step.version,
                "state": state,
                "iris_job_id": iris_job_id,
                "run_id": receipt.index_run_id if receipt is not None else None,
                "iris_url": (
                    JobName.from_wire(iris_job_id).dashboard_url(_IRIS_DASHBOARD_URL)
                    if iris_job_id and iris_job_id.startswith("/")
                    else None
                ),
                "gpu_hours": gpu_hours,
                "wandb_entity": team or preset.wandb_entity,
                **{
                    key: tracking.get(key)
                    for key in ("attempt_id", "wandb_url", "started_at", "engine")
                    if tracking.get(key) is not None
                },
                **({"grafana_links": {"run": grafana}} if grafana else {}),
                **(
                    evaluation_status(path, preset, receipt, launch).model_dump(
                        mode="json", exclude_unset=True, exclude_none=True
                    )
                    if receipt is not None and launch is not None
                    else {}
                ),
                **best,
                "latest_completed_update": _completed_update(path, preset.recipe.layout.workload),
                "cluster": execution.cluster,
                "allocation": {
                    **planned[-1]["iris"]["allocation"],
                    "cpu": execution.cpu,
                    "memory": execution.memory,
                    "disk": execution.disk,
                },
                "output": path,
                "issue": receipt.issue if receipt is not None else issue,
                "rl_progress_url": (publication_urls or {}).get(path),
            }
        )
    verdict = {}
    report = None
    if run.gate is not None:
        path = run.gate.path(prefix=recipe.artifact_prefix(preset.recipe.layout.cluster))
        report = rl_gate.read_gate_report(path)
        if report is not None:
            verdict = {
                "report": str(StoragePath(path) / rl_gate.GATE_REPORT_FILENAME),
                **report.model_dump(mode="json", include={"verdict", "effect", "reason"}),
            }
    if "verdict" not in verdict and preset.gated:
        crashed = next((row for row in rows if str(row["state"]).lower() in _FAILED_RUN_STATES), None)
        if crashed is not None:
            verdict = {
                "verdict": Verdict.INVALID.value,
                "reason": f"native job {crashed['iris_job_id']} is {crashed['state']}",
                "iris_job_id": crashed["iris_job_id"],
            }
    preset = preset.with_updates(planned[0]["skyrl"]["trainer"]["max_steps"])
    return {
        "preset": preset.name,
        "code": asdict(Provenance.capture(recipe.REPO_ROOT)),
        "inputs": {
            "model": planned[0]["inputs"]["model"]["identity"],
            **{
                name: [source["identity"] for source in planned[0]["inputs"][name]]
                for name in ("train_data", "validation_data")
            },
        },
        "context_budget": planned[0]["skyrl"]["context_budget"],
        "updates": preset.recipe.layout.workload.update_count(preset.updates),
        "measured_updates": list(preset.measured),
        "config_changes": list(report.run.config_changes) if report else [],
        "budget_estimate": preset.budget,
        "guards": [metric.key for metric in preset.guards],
        "gaps": list(preset.gaps),
        "runs": rows,
        **verdict,
    }


def _print_status(snapshot: dict) -> None:
    click.echo(yaml.safe_dump(snapshot, sort_keys=False, width=1000))
    if "report" in snapshot:
        click.echo((StoragePath(snapshot["report"]).parent / rl_gate.GATE_MARKDOWN_FILENAME).read_text())


def _from_handles(handles: list[ArtifactStep]) -> RLExperiment:
    return RLExperiment(
        tuple(step for step in handles if step.artifact_type is SkyRLRun),
        next((step for step in handles if step.artifact_type is rl_gate.GateReport), None),
    )


def _branch_pr_number(provenance: Provenance) -> int | None:
    if provenance.branch is None:
        return None
    try:
        result = subprocess.run(
            ["gh", "pr", "view", "--json", "number", "--jq", ".number"],
            cwd=recipe.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=4,
        )
        number = int(result.stdout)
        return number if number > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        logger.debug("GitHub PR lookup unavailable (%s)", type(error).__name__)
        return None


def run_index_identity(
    launch: dict, preset: presets.Preset, table: presets.ModelPresets, provenance: Provenance, *, user: str
) -> RunIndexIdentity:
    """Name a composed launch by the gate's normalized comparison configuration."""
    digest = fingerprint_hash(canonical_json(rl_gate.comparison_config(launch)))
    return RunIndexIdentity(
        run_id=f"{table.model}-{preset.name}-{user}-{digest}",
        model=table.model,
        preset=preset.name,
        preset_revision=preset.revision,
        user=user,
        branch=provenance.branch,
        source_revision=provenance.base_commit or "Unknown",
        dirty=provenance.dirty,
        pr_number=_branch_pr_number(provenance),
    )


def _base_preset_revision(module: str, model: str, preset: str, source_revision: str) -> str | None:
    root = recipe.REPO_ROOT
    program = """
import sys
from importlib import import_module
from experiments.post_training.rl_presets import ModelPresets

module, model, preset = sys.argv[1:]
revisions = {value.by_name[preset].revision for value in vars(import_module(module)).values()
             if isinstance(value, ModelPresets) and value.model == model and preset in value.by_name}
if len(revisions) == 1:
    print(revisions.pop())
"""
    try:
        base = git_output(["merge-base", source_revision, "origin/main"], cwd=root, timeout=4)
    except (OSError, RuntimeError):
        logger.debug("Git could not resolve the main merge base for preset %s", preset)
        return None
    try:
        snapshot = subprocess.run(
            [
                "git",
                "archive",
                base,
                "experiments/post_training",
                "experiments/evaluation",
                "pyproject.toml",
                "infra/marina/apps/rl_progress/applet.toml",
                "lib",
            ],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=8,
        ).stdout
        with tempfile.TemporaryDirectory(prefix="rl-main-base-") as directory:
            path = Path(directory)
            with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
                archive.extractall(path, filter="data")
            result = subprocess.run(
                [sys.executable, "-c", program, module, model, preset],
                text=True,
                capture_output=True,
                check=True,
                cwd=path,
                env={
                    "PATH": os.defpath,
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": os.pathsep.join(str(source) for source in sorted(path.glob("lib/*/src"))),
                },
                timeout=30,
            )
            return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError, tarfile.TarError) as error:
        logger.debug("Could not load preset %s from main commit %s (%s)", preset, base, type(error).__name__)
        return None


class _PublicationWatch(BaseModel):
    job: str
    preset: str
    prefix: str
    outputs: tuple[str, ...]
    gate: rl_gate.GateTarget | None
    service_url: str | None


def _publication_watch(
    job: JobName, handles: list[ArtifactStep], preset: presets.Preset, service_url: str | None
) -> _PublicationWatch:
    run = _from_handles(handles)
    prefix = recipe.artifact_prefix(preset.recipe.layout.cluster)
    return _PublicationWatch(
        job=str(job),
        preset=preset.name,
        prefix=prefix,
        outputs=tuple(step.path(prefix) for step in run.runs),
        gate=rl_gate.gate_target(run.gate) if run.gate is not None else None,
        service_url=service_url,
    )


def _follow_detached(module: str, watch: _PublicationWatch) -> str:
    """Start the caller's publication watcher and return its local log path."""
    with tempfile.NamedTemporaryFile(prefix="marin-rl-publication-", suffix=".log", delete=False) as log:
        subprocess.Popen(
            [sys.executable, "-m", module, "_follow-publication", watch.model_dump_json()],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return log.name


def _watch_publication(
    watch: _PublicationWatch, table: presets.ModelPresets
) -> tuple[JobState, rl_publication.PublicationOutcome]:
    preset = table.by_name[watch.preset]
    published = set()
    deadline = time.monotonic() + preset.recipe.layout.coordinator_timeout_hours * 3600
    with open_iris_client(config_file=Path(IRIS_HUB_CLUSTER_CONFIG), workspace=recipe.REPO_ROOT) as client:
        parent = client.job(JobName.from_wire(watch.job))
        while True:
            state = parent.state_only()
            for output in watch.outputs:
                rl_publication.publish_running(output, preset, table, watch.job, watch.service_url, published)
            if is_job_finished(state):
                break
            if time.monotonic() >= deadline:
                raise click.ClickException(
                    f"Timed out waiting for {watch.job}. Repeat the original command with --wait."
                )
            time.sleep(_PUBLICATION_POLL_SECONDS)
    gate_path = watch.gate.stored_path(watch.prefix) if watch.gate is not None else None
    publication = rl_publication.publish_runs(
        watch.outputs,
        preset,
        table=table,
        gate_path=gate_path,
        service_url=watch.service_url,
        failed_coordinator=watch.job if state != JobState.SUCCEEDED else None,
    )
    return state, publication


def _follow_command(table: presets.ModelPresets) -> click.Command:
    @click.command(cls=RLCommand)
    @click.argument("request_json")
    def follow(request_json: str) -> None:
        state, result = _watch_publication(_PublicationWatch.model_validate_json(request_json), table)
        click.echo(json.dumps(result.to_dict()))
        if state != JobState.SUCCEEDED or result.unserved_records:
            raise click.exceptions.Exit(1)

    return follow


def _wait_for_run(
    job: JobName,
    handles: list[ArtifactStep],
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    service_url: str | None,
) -> None:
    watch = _publication_watch(job, handles, preset, service_url)
    parent_state, publication = _watch_publication(watch, table)
    coordinator_error = (
        f"coordinator {job} is {parent_state}; required verdict unavailable"
        if parent_state != JobState.SUCCEEDED
        else None
    )
    run = _from_handles(handles)
    run = replace(run, gate=rl_gate.completed_gate(run.gate))
    snapshot = _status(
        run,
        preset,
        stored_only=True,
        publication_urls=publication.urls,
    )
    snapshot["publication"] = publication.to_dict()
    if "verdict" not in snapshot and (coordinator_error is not None or run.gate is not None):
        snapshot["reason"] = coordinator_error or "required verdict unavailable"
    _print_status(snapshot)
    if _evaluation_failed(snapshot, preset):
        raise click.exceptions.Exit(EXIT_CODES[Verdict.FAIL])
    verdict = snapshot.get("verdict")
    if verdict is None and run.gate is None and coordinator_error is None:
        verdict = Verdict.PASS
    raise click.exceptions.Exit(EXIT_CODES[verdict])


def _submit_from_launcher(
    module: str | None,
    argv: tuple[str, ...],
    handles: list[ArtifactStep],
    max_concurrent: int,
    context: click.Context,
    preset: presets.Preset,
) -> JobName:
    if module is None:
        specification = sys.modules["__main__"].__spec__
        module = specification.name if specification is not None else None
    if not module:
        raise click.UsageError("submit from python -m MODULE, or pass entry_module to rl.main")
    submitted_at = datetime.now(UTC).isoformat()
    index: RunIndexIdentity = context.meta["rl_index"]
    base_preset_revision = _base_preset_revision(module, index.model, index.preset, index.source_revision)
    index = index.model_copy(update={"submitted_at": submitted_at})
    runs = _from_handles(handles).runs
    indexed_runs = {
        step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).run_id: index for step in runs
    }
    metadata = _SUBMITTED_INDEX.dump_json(indexed_runs).decode()
    environment = {**os.environ, RUN_INDEX_ENV: metadata, BASE_PRESET_REVISION_ENV: base_preset_revision or ""}
    job = submit_or_run(module, handles, max_concurrent, environment, argv=argv)
    submitted = [
        {
            "name": step.name,
            "grafana_links": {
                "run": grafana_url(index.run_id, execution.cluster, context.meta["rl_training_type"], submitted_at)
            },
        }
        for step in runs
        if (execution := skyrl_execution(step)) is not None
    ]
    result = {
        "coordinator_job_id": str(job),
        "iris_url": job.dashboard_url(_IRIS_DASHBOARD_URL),
        "submitted_at": submitted_at,
        "base_preset_revision": base_preset_revision,
        "runs": submitted,
    }
    _print_status(result)
    if not context.params["wait"]:
        try:
            watch = _publication_watch(job, handles, preset, context.params["url"])
            _print_status({"publication_log": _follow_detached(module, watch)})
        except OSError as error:
            logger.warning("Iris accepted %s. The publication watcher could not start (%s)", job, type(error).__name__)
    return job


def _run_command(
    table: presets.ModelPresets,
    argv: tuple[str, ...],
    entry_module: str | None,
    function: Callable,
) -> click.Command:
    @wraps(function)
    def checked(*args, wait: bool = False, url: str | None = None, **kwargs):
        if wait and not click.get_current_context().params["do_run"]:
            raise click.BadParameter("--wait requires --run")
        return function(*args, **kwargs)

    def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
        if get_job_info() is not None:
            validate_coordinator_graph(handles)
            _run_graph(handles, max_concurrent)
            return
        context = click.get_current_context()
        options = context.params
        preset = table.by_name[options["preset_name"]]
        job = _submit_from_launcher(entry_module, argv, handles, max_concurrent, context, preset)
        if options["wait"]:
            _wait_for_run(job, handles, preset, table=table, service_url=options["url"])

    return click.command(
        cls=RLCommand, help=__doc__ + "\n\n" + "\n".join(f"{preset.name}: {preset.budget}" for preset in table.presets)
    )(build_options_with_runner(checked, runner))


def _validate_launch_issue(run: RLExperiment, preset: presets.Preset, issue: int) -> None:
    prefix = recipe.artifact_prefix(preset.recipe.layout.cluster)
    for step in run.runs:
        export = str(StoragePath(step.path(prefix)) / EXPORTS_SUBDIRECTORY)
        try:
            rl_publication.lineage_issue(export, issue)
        except rl_publication.IssueConflict as error:
            raise click.BadParameter(str(error), param_hint="--issue") from None


def command(
    table: presets.ModelPresets,
    *,
    argv: tuple[str, ...],
    init: SkyRLModelSource | None = None,
    entry_module: str | None = None,
    forward_env: tuple[str, ...] = (),
) -> click.Command:
    @partial(_run_command, table, argv, entry_module)
    @click.option("--issue", type=click.IntRange(min=1), help="GitHub experiment issue for this launch.")
    @click.option("--plan", is_flag=True, help="Print the plan without submission (the default).")
    @click.option("--check-pin", is_flag=True, help="Check the composed candidate and its pinned controls on CPU.")
    @click.option("--marina-url", "url", help="Marina origin for run publication (defaults to the Marina service).")
    @click.option(
        "--wait",
        is_flag=True,
        help="Wait after --run and print the terminal report. "
        "Exit 0 for PASS, 1 for FAIL, 2 for INVALID, 3 without a required verdict.",
    )
    @click.option(
        "--preset",
        "preset_name",
        required=True,
        type=click.Choice(sorted(table.by_name)),
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
        "Other presets use their declared task selection.",
    )
    @click.option(
        "--init",
        "init_name",
        metavar="NAME@VERSION",
        help="Initial weights: a published model artifact. "
        "Only presets that accept initial weights. The initial weights are evaluated too, so the report shows "
        "the gain.",
    )
    @click.option(
        "--replicates",
        type=click.IntRange(1, 5),
        default=1,
        show_default=True,
        help="Identical copies of a gated preset, sharing one seed. The gate compares their "
        "mean with the controls; more copies narrow the tolerance. Only gated presets.",
    )
    @click.option(
        "--control",
        is_flag=True,
        help="Run the preset's control replicates from a clean checkout with only its pinned default "
        "inputs. Only for pinning new controls; refuses --set, --tasks, --init, --require "
        "and aliases of another preset's controls.",
    )
    def run_command(
        preset_name: str,
        settings: tuple[str, ...],
        tasks_name: str | None,
        init_name: str | None,
        replicates: int,
        control: bool,
        requirements: tuple[str, ...],
        issue: int | None,
        plan: bool,
        check_pin: bool,
    ) -> tuple[ArtifactStep, ...]:
        context = click.get_current_context()
        do_run = context.params["do_run"]
        if plan and do_run:
            raise click.UsageError("Choose either --plan or --run")
        preset = table.by_name[preset_name]
        if init is not None and init_name is not None:
            raise click.UsageError("Python init and --init cannot be used together")
        missing = [name for name in forward_env if name not in os.environ]
        if missing:
            raise click.UsageError(f"missing required environment variables: {', '.join(missing)}")
        provenance = _provenance(control=control)
        from_launcher = get_job_info() is None
        run = build_experiment(
            preset,
            table=table,
            forward_env=forward_env,
            settings=settings,
            tasks=recipe.selection_artifact(tasks_name) if tasks_name else None,
            init=init if init is not None else recipe.published_model(init_name, preset=preset) if init_name else None,
            replicates=replicates,
            control=control,
            requirements=requirements,
            issue=issue,
            validate_plan=from_launcher,
            provenance=provenance,
        )
        handles = (*run.runs, *((run.gate,) if run.gate else ()))
        if from_launcher and (do_run or check_pin):
            assert run.inputs is not None
            validated = rl_gate.validate_candidate(
                preset,
                run.inputs,
                control=control,
                table=table,
            )
            if do_run:
                if issue is not None:
                    _validate_launch_issue(run, preset, issue)
                context.meta["rl_training_type"] = validated["runtime"]["training_type"]
                context.meta["rl_index"] = run_index_identity(
                    validated, preset, table, provenance, user=username_segment()
                )
        snapshot = _status(replace(run, gate=rl_gate.completed_gate(run.gate)), preset, stored_only=do_run, issue=issue)
        if preset.gated and not control:
            pin = rl_gate.control_pin(preset, table=table, check_revision=False)
            snapshot["control_pin"] = {
                "commit": pin.commit,
                "revision": pin.revision,
                "current": pin.revision == table.by_name[pin.preset].revision,
            }
        _print_status(snapshot)
        if not do_run and _evaluation_failed(snapshot, preset):
            raise click.exceptions.Exit(EXIT_CODES[Verdict.FAIL])
        if not do_run and "verdict" in snapshot:
            raise click.exceptions.Exit(EXIT_CODES[snapshot["verdict"]])
        return handles

    return run_command


def _run_graph(handles: list[ArtifactStep], max_concurrent: int) -> None:
    planned = _from_handles(handles)
    build_artifacts(*planned.runs, max_concurrent=max_concurrent)
    gate = rl_gate.completed_gate(planned.gate)
    if gate is not None:
        build_artifacts(gate, max_concurrent=max_concurrent)


def main(
    table: presets.ModelPresets,
    *,
    argv: Sequence[str] | None = None,
    init: SkyRLModelSource | None = None,
    entry_module: str | None = None,
    forward_env: tuple[str, ...] = (),
) -> int:
    """Run the selected preset table's CLI, retaining Click's exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args and args[0] == "_follow-publication":
            selected, args = _follow_command(table), args[1:]
        elif args and args[0] in {"compare", "publish"}:
            if init is not None and args[0] == "compare":
                raise click.UsageError("Python init cannot be used with compare")
            selected = (rl_gate if args[0] == "compare" else rl_publication).command(table)
            args = args[1:]
        else:
            selected = command(table, argv=tuple(args), init=init, entry_module=entry_module, forward_env=forward_env)
        result = selected.main(args=args, standalone_mode=False)
        return result if isinstance(result, int) else 0
    except click.ClickException as error:
        if isinstance(error, click.UsageError):
            error.exit_code = REFUSAL_EXIT_CODE
        error.show()
        return error.exit_code
    except click.exceptions.Exit as error:
        return error.exit_code
