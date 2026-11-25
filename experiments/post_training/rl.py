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
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from functools import partial, wraps
from pathlib import Path
from typing import Protocol, cast

import click
import yaml
from finestore.rl.rl_progress import ControlReplicate, Verdict
from finestore.rl.run_records import AttemptRecord, RunIndexIdentity
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState, is_job_finished
from marin.execution.artifact import is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.lazy import run as build_artifacts
from marin.execution.step_runner import check_cache
from marin.execution.step_status import get_status_path
from marin.experiment.cli import build_options_with_runner
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import RUN_INDEX_ENV, submit_or_run, validate_coordinator_graph
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactHfModel,
    SkyRLModelSource,
    SkyRLReceiptLookupError,
    SkyRLRun,
    SkyRLRunConfig,
    read_skyrl_attempt,
    run_skyrl,
    skyrl_cpu_environment,
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
    InvalidComparison,
    RunInputs,
    RunOptions,
    evaluation_status,
    lineage_gpu_hours,
    read_recorded_launch,
    run_inputs,
)
from experiments.post_training.task_curation.selection import FULL_VALIDATION_FILENAME

_WANDB_ENTITY_ENV = "WANDB_ENTITY"
_PUBLICATION_POLL_SECONDS = 60
_IRIS_DASHBOARD_URL = "https://iris.oa.dev"
_FAILED_RUN_STATES = {JobState.FAILED, JobState.KILLED, JobState.WORKER_FAILED, JobState.UNSCHEDULABLE}
logger = logging.getLogger(__name__)
_SUBMITTED_INDEX = TypeAdapter(dict[str, RunIndexIdentity])


class SFTModelPreset(Protocol):
    """The recipe and exported model exposed by an SFT preset."""

    @property
    def name(self) -> str: ...

    @property
    def recipe(self) -> object: ...

    def model(
        self, *, coordinator_cluster: str | None = None, coordinator_prefix: str | None = None
    ) -> ArtifactHfModel: ...


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
        rl_publication.reserve_issue(config.output.export_root, issue)
    if index is not None:
        launch = yaml.safe_load(config.launch_config_yaml)
        if "trainer.project_name" in recipe.override_document(inputs.settings):
            index = index.model_copy(update={"wandb_project": launch["skyrl"]["trainer"]["project_name"]})
        launch["run"]["index"] = index.model_dump(mode="json")
        config = replace(config, launch_config_yaml=yaml.safe_dump(launch, sort_keys=False))
    return run_skyrl(config, forward_env=forward_env)


def _run_name(
    preset: presets.Preset,
    provenance: Provenance,
    inputs_hash: str,
    replicate: int | None,
    *,
    table: presets.ModelPresets,
    control: bool,
    resume_step: int | None,
) -> str:
    if control:
        if replicate is None:
            raise ValueError("a control output requires a replicate")
        if provenance.base_commit is None:
            raise click.BadParameter("controls require a recorded source commit")
        return rl_gate.control_name(preset.name, provenance.base_commit, replicate, table=table)
    name = f"checkpoints/{table.namespace}/{preset.name}-{provenance.tree_hash}-in-{inputs_hash}"
    if resume_step is not None:
        name += f"-resume-step-{resume_step}"
    return user_owned_name(name + (f"-r{replicate}" if preset.gated and replicate is not None else ""))


def _check_resume(preset: presets.Preset, *, has_init: bool) -> None:
    if has_init:
        raise click.BadParameter("--resume cannot be combined with --init")
    if preset.checkpoints == "none":
        raise click.BadParameter("--resume requires checkpointing")


def _with_run_options(
    preset: presets.Preset,
    *,
    checkpoints: presets.CheckpointPolicy | None,
    checkpoint_every: int | None,
    export: presets.CheckpointPolicy | None,
    priority: str | None,
) -> presets.Preset:
    if preset.gated and any(value is not None for value in (checkpoints, checkpoint_every, export)):
        raise click.BadParameter("gated presets do not accept --checkpoints, --checkpoint-every, or --export")
    retention = checkpoints if checkpoints is not None else preset.checkpoints
    export_policy = export if export is not None else preset.export
    cadence = preset.checkpoint_every if checkpoint_every is None else checkpoint_every or None
    layout = preset.recipe.layout
    try:
        return replace(
            preset,
            checkpoints=retention,
            checkpoint_every=cadence,
            export=export_policy,
            recipe=replace(preset.recipe, layout=replace(layout, priority=priority or layout.priority)),
        )
    except ValueError as error:
        raise click.BadParameter(str(error)) from error


def build_experiment(
    preset: presets.Preset,
    *,
    table: presets.ModelPresets,
    tasks: ArtifactStep | None = None,
    init: SkyRLModelSource | None = None,
    sft_preset: str | None = None,
    resume: presets.StoredResume | None = None,
    settings: tuple[str, ...] = (),
    replicates: int = 1,
    control: bool = False,
    requirements: tuple[str, ...] = (),
    issue: int | None = None,
    forward_env: tuple[str, ...] = (),
    validate_plan: bool = False,
    plan_only: bool = False,
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
            raise click.BadParameter(f"{preset.name} requires --tasks")
    elif preset.tasks_input == "forbidden" and tasks is not None:
        raise click.BadParameter("this preset does not accept --tasks")
    if not preset.accepts_init and init is not None:
        raise click.BadParameter("this preset does not accept --init")
    if (not preset.gated and replicates != 1) or (control and preset.control_preset != preset.name):
        raise click.BadParameter(f"{preset.name} does not accept this replicate or control selection")
    if control and (settings or tasks is not None or init is not None or resume is not None):
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
    spec = recipe.run_spec(
        preset, name="rl-config", tasks=tasks, init=init, resume=resume, settings=settings, model_name=table.model
    )
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
    if resume is not None:
        changes["resume"] = resume.model_dump(mode="json", exclude_none=True)
    if sft_preset is not None:
        changes["sft_preset"] = sft_preset
    options = RunOptions.from_preset(preset)
    if options != RunOptions.from_preset(table.by_name[preset.name]):
        changes["run_options"] = options.model_dump(mode="json")
    rl_gate.validate_requirements(preset, requirements, table=table, control=control)
    provenance = provenance or _provenance(control=control)
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True, default=str))
    inputs = run_inputs(
        preset.name,
        settings,
        f"{tasks.name}@{tasks.version}" if tasks is not None else None,
        initializer,
        requirements,
        resume=resume,
        options=options,
        sft_preset=sft_preset,
        preset=preset,
    )
    runs = []
    for replicate in range(replicates):
        name = _run_name(
            preset,
            provenance,
            inputs_hash,
            replicate,
            table=table,
            control=control,
            resume_step=resume.step if resume is not None else None,
        )
        run = recipe.run_step(preset, replace(spec, name=name), wandb_entity=wandb_entity, forward_env=forward_env)
        recorded_inputs = inputs
        if control:
            member = ControlReplicate(commit=cast(str, provenance.base_commit), version=run.version, replicate=replicate)
            recorded_inputs = inputs.model_copy(update={"control": member})
        runs.append(replace(run, run=partial(_run_rl, inputs=recorded_inputs, issue=issue, forward_env=forward_env)))
    gate = (
        rl_gate.gate_step(
            preset,
            tuple(runs),
            group_name=_run_name(
                preset,
                provenance,
                inputs_hash,
                None,
                table=table,
                control=False,
                resume_step=resume.step if resume is not None else None,
            ),
            inputs=inputs,
            table=table,
        )
        if preset.gated and not control and (not plan_only or preset.control_preset in table.control_pins)
        else None
    )
    if validate_plan:
        rl_gate.validate_plan(preset, replace(spec, name=runs[0].name), runs[0])
    return RLExperiment(tuple(runs), gate, inputs)


def _evaluation_failed(snapshot: dict, preset: presets.Preset) -> bool:
    return isinstance(preset.recipe.layout.workload, presets.Training) and any(
        not fact["holds"] for row in snapshot["runs"] for fact in row.get("evaluation_invariants", ())
    )


def _completed_update(
    path: str, workload: presets.Training | presets.Generation, receipt: AttemptRecord | None
) -> int | None:
    if receipt is None:
        return None
    key = workload.completion_metric
    try:
        metrics = read_skyrl_metrics(path, receipt.attempt_id)
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
    scoped_attempts: Mapping[str, AttemptRecord] | None = None,
) -> dict:
    prefix = recipe.run_prefix(preset)
    ttl = preset.output_ttl_days
    source = "cluster default"
    if ttl is not None:
        if os.environ.get("MARIN_PREFIX"):
            source = "MARIN_PREFIX"
        elif os.environ.get("MARIN_TEMP_PREFIX"):
            source = "MARIN_TEMP_PREFIX"
        else:
            source = f"preset's {ttl}-day storage"
    output = {"root": prefix, "source": source}
    if (
        ttl
        and (source != "MARIN_PREFIX" or f"/tmp/ttl={ttl}d" in prefix)
        and preset.recipe.layout.coordinator_timeout_hours > 24 * ttl
    ):
        output["warning"] = "The coordinator time limit exceeds storage lifetime. Set MARIN_PREFIX to a durable root."
    rows = []
    planned = []
    for step in run.runs:
        execution = skyrl_execution(step)
        assert execution is not None
        path = step.path(prefix)
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        planned.append(yaml.safe_load(config.launch_config_yaml))
        try:
            if scoped_attempts is not None:
                receipt = scoped_attempts.get(path)
            else:
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
        state = result.state.value if result is not None else "not built"
        if result is None and scoped_attempts is None and status_path.exists():
            state = status_path.read_text().strip()
        best = {}
        if receipt is not None and launch is not None:
            native = receipt.native.config
            exports_best = (
                native["run"].get("export_hf") and native["skyrl"]["trainer"].get("hf_export_checkpoint") == "best"
            )
            if preset.checkpoints == "best" or exports_best:
                best_path = StoragePath(launch.export_root) / "best_checkpoint.json"
                best["best_checkpoint_step"] = (
                    json.loads(best_path.read_text())["step"] if best_path.exists() else "none eligible"
                )
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
                "latest_completed_update": _completed_update(path, preset.recipe.layout.workload, receipt),
                "cluster": execution.cluster,
                "allocation": {
                    **planned[-1]["iris"]["allocation"],
                    "cpu": execution.cpu,
                    "memory": execution.memory,
                    "disk": execution.disk,
                },
                "output": path,
                "issue": rl_publication.lineage_issue(launch.export_root, issue) if launch is not None else issue,
                "rl_progress_url": (publication_urls or {}).get(path),
            }
        )
    verdict = {}
    report = None
    if run.gate is not None:
        path = run.gate.path(prefix=recipe.run_prefix(preset))
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
    launch_inputs = planned[0]["inputs"]
    inputs = {"model": launch_inputs["model"]["identity"]}
    for name in ("train_data", "validation_data"):
        inputs[name] = [source["identity"] for source in launch_inputs[name]]
    if "starting_checkpoint" in launch_inputs:
        inputs["starting_checkpoint"] = launch_inputs["starting_checkpoint"]["identity"]
    return {
        "preset": preset.name,
        "output": output,
        "run_options": RunOptions.from_preset(preset).model_dump(mode="json"),
        "code": asdict(Provenance.capture(recipe.REPO_ROOT)),
        "inputs": inputs,
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
    launch: dict,
    preset: presets.Preset,
    table: presets.ModelPresets,
    provenance: Provenance,
    *,
    user: str,
    control: bool = False,
) -> RunIndexIdentity:
    """Name a composed launch by the gate's normalized comparison configuration."""
    version = rl_gate.config_version(launch)
    return RunIndexIdentity(
        run_id=f"{table.model}-{preset.name}-{user}-{version}",
        model=table.model,
        preset=preset.name,
        preset_version=version,
        user=user,
        branch=provenance.branch,
        source_revision=provenance.base_commit or "Unknown",
        tree_hash=provenance.tree_hash,
        dirty=provenance.dirty,
        pr_number=_branch_pr_number(provenance),
        is_control=control,
    )


def _base_preset(module: str, index: RunIndexIdentity) -> tuple[str | None, str | None]:
    """Fingerprint the nominal preset at the source commit's main merge base."""
    root = recipe.REPO_ROOT
    preset = index.preset
    try:
        base = git_output(["merge-base", index.source_revision, "origin/main"], cwd=root, timeout=4)
    except (OSError, RuntimeError):
        logger.debug("Git could not resolve the main merge base for preset %s", preset)
        return None, None
    if index.is_control and index.branch == "main" and index.source_revision == base and not index.dirty:
        return base, index.preset_version
    try:
        snapshot = subprocess.run(
            [
                "git",
                "archive",
                base,
                "experiments/__init__.py",
                "experiments/post_training",
                "experiments/evaluation",
                "config",
                "pyproject.toml",
                "infra/buckets/pyproject.toml",
                "infra/deploy/pyproject.toml",
                "infra/marina/pyproject.toml",
                "infra/marina/apps/rl_progress/applet.toml",
                "infra/pulumi/pyproject.toml",
                "lib",
                "vendor/harbor",
                "vendor/skyrl",
            ],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        with tempfile.TemporaryDirectory(prefix="rl-main-base-") as directory:
            path = Path(directory)
            with tarfile.open(fileobj=io.BytesIO(snapshot)) as archive:
                archive.extractall(path, filter="data")
            # The launcher owns the JSON protocol, including for bases that predate it.
            for name in ("rl_preset_cli.py", "rl_preset_plan.py"):
                source = Path(__file__).with_name(name)
                destination = path / "experiments/post_training" / name
                if not destination.is_file():
                    shutil.copyfile(source, destination)
            native_cli = Path("vendor/skyrl/cloud/iris/config_cli.py")
            if not (path / native_cli).is_file():
                shutil.copyfile(root / native_cli, path / native_cli)
            sources = [path, *sorted(path.glob("lib/*/src"))]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.post_training.rl_preset_cli",
                    "describe",
                    module,
                    index.model,
                    preset,
                ],
                text=True,
                capture_output=True,
                check=True,
                cwd=path,
                env={
                    **skyrl_cpu_environment(),
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": os.pathsep.join(map(str, sources)),
                },
                timeout=180,
            )
            launch = json.loads(result.stdout)
            return base, rl_gate.config_version(launch) if isinstance(launch, dict) else None
    except (OSError, subprocess.SubprocessError, tarfile.TarError, ValueError, InvalidComparison) as error:
        logger.debug("Could not load preset %s from main commit %s (%s)", preset, base, type(error).__name__)
        return base, None


class _PublicationWatch(BaseModel):
    job: str
    preset: str
    outputs: tuple[str, ...]
    gate_required: bool
    timeout_seconds: float
    service_url: str | None
    issue: int | None = None


def _publication_watch(
    job: JobName, handles: list[ArtifactStep], preset: presets.Preset, service_url: str | None, issue: int | None
) -> _PublicationWatch:
    run = _from_handles(handles)
    prefix = recipe.run_prefix(preset)
    return _PublicationWatch(
        job=str(job),
        preset=preset.name,
        outputs=tuple(step.path(prefix) for step in run.runs),
        gate_required=run.gate is not None,
        timeout_seconds=preset.recipe.layout.coordinator_timeout_hours * 3600,
        service_url=service_url,
        issue=issue,
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
) -> tuple[JobState, rl_publication.PublicationOutcome, str | None]:
    preset = table.by_name[watch.preset]
    published = set()
    deadline = time.monotonic() + watch.timeout_seconds
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
                    f"Timed out waiting for {watch.job}. Check its Iris state before resubmitting with --wait."
                )
            time.sleep(_PUBLICATION_POLL_SECONDS)
    gate_path = None
    try:
        evidence = rl_publication.read_coordinator_evidence(watch.outputs[0], watch.job)
        gate_path = evidence.gate_path if evidence is not None else None
        publication = rl_publication.publish_runs(
            watch.outputs,
            preset,
            table=table,
            gate_path=gate_path,
            service_url=watch.service_url,
            coordinator=watch.job,
            evidence=evidence,
            interrupted=state != JobState.SUCCEEDED,
            gate_required=watch.gate_required,
            issue=watch.issue,
        )
    except Exception as error:
        publication = rl_publication.PublicationOutcome(
            unserved_records=list(watch.outputs),
            errors=[str(error)],
            retry="Repeat the original command with --wait.",
        )
    return state, publication, gate_path


def _follow_command(table: presets.ModelPresets) -> click.Command:
    @click.command(cls=RLCommand)
    @click.argument("request_json")
    def follow(request_json: str) -> None:
        state, result, _ = _watch_publication(_PublicationWatch.model_validate_json(request_json), table)
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
    has_following: bool,
    issue: int | None,
) -> None:
    watch = _publication_watch(job, handles, preset, service_url, issue)
    parent_state, publication, gate_path = _watch_publication(watch, table)
    coordinator_error = (
        f"coordinator {job} is {parent_state}; required verdict unavailable"
        if parent_state != JobState.SUCCEEDED
        else None
    )
    run = _from_handles(handles)
    gate_required = run.gate is not None
    gate_step = replace(run.gate, override_path=gate_path) if run.gate is not None and gate_path is not None else None
    run = replace(run, gate=gate_step)
    snapshot = _status(
        run,
        preset,
        stored_only=True,
        publication_urls=publication.urls,
        scoped_attempts=publication.attempts,
    )
    snapshot["publication"] = publication.to_dict()
    rows = snapshot.get("runs", ())
    if (
        has_following
        and coordinator_error is not None
        and rows
        and all(row["state"] == JobState.SUCCEEDED for row in rows)
    ):
        snapshot["post_run_error"] = f"Coordinator {job} ended in state {parent_state}. RL has a committed result."
    elif "verdict" not in snapshot and (coordinator_error is not None or gate_required):
        snapshot["reason"] = coordinator_error or "required verdict unavailable"
    _print_status(snapshot)
    if _evaluation_failed(snapshot, preset):
        raise click.exceptions.Exit(EXIT_CODES[Verdict.FAIL])
    verdict = snapshot.get("verdict")
    if verdict is None and not gate_required and coordinator_error is None:
        verdict = Verdict.PASS
    raise click.exceptions.Exit(EXIT_CODES[verdict])


def _submit_from_launcher(
    module: str | None,
    argv: tuple[str, ...],
    handles: list[ArtifactStep],
    rl_handles: list[ArtifactStep],
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
    base_commit, base_version = _base_preset(module, index)
    index = index.model_copy(
        update={"submitted_at": submitted_at, "base_commit": base_commit, "base_preset_version": base_version}
    )
    runs = _from_handles(rl_handles).runs
    indexed_runs = {
        step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps)).run_id: index for step in runs
    }
    metadata = _SUBMITTED_INDEX.dump_json(indexed_runs).decode()
    environment = {**os.environ, RUN_INDEX_ENV: metadata}
    if revision := context.meta.get("sft_source_revision"):
        environment["GIT_COMMIT"] = revision
    if preset.output_ttl_days is not None:
        environment["MARIN_PREFIX"] = recipe.run_prefix(preset)
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
        "base_preset_version": index.base_preset_version,
        "runs": submitted,
    }
    _print_status(result)
    if not context.params["wait"]:
        try:
            watch = _publication_watch(job, rl_handles, preset, context.params["url"], context.params["issue"])
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
        context = click.get_current_context()
        options = context.params
        preset = context.meta["rl_preset"]
        count = context.meta["rl_handle_count"]
        rl_handles = handles[:count]
        if (coordinator := get_job_info()) is not None:
            validate_coordinator_graph(handles)
            _run_graph(rl_handles, max_concurrent, preset, str(coordinator.job_id))
            if following := handles[count:]:
                build_artifacts(*following, max_concurrent=max_concurrent)
            return
        job = _submit_from_launcher(entry_module, argv, handles, rl_handles, max_concurrent, context, preset)
        if options["wait"]:
            _wait_for_run(
                job,
                rl_handles,
                preset,
                table=table,
                service_url=options["url"],
                has_following=len(handles) > count,
                issue=options["issue"],
            )

    return click.command(
        cls=RLCommand, help=__doc__ + "\n\n" + "\n".join(f"{preset.name}: {preset.budget}" for preset in table.presets)
    )(build_options_with_runner(checked, runner))


def _validate_launch_issue(run: RLExperiment, preset: presets.Preset, issue: int) -> None:
    prefix = recipe.run_prefix(preset)
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
    sft_presets: Callable[[], Sequence[SFTModelPreset]] | None = None,
    entry_module: str | None = None,
    forward_env: tuple[str, ...] = (),
) -> click.Command:
    @partial(_run_command, table, argv, entry_module)
    @click.option("--issue", type=click.IntRange(min=1), help="GitHub experiment issue for this launch.")
    @click.option("--plan", is_flag=True, help="Print the plan without submission (the default).")
    @click.option(
        "--checkpoints", type=click.Choice(("none", "latest", "best")), help="Select native checkpoint retention."
    )
    @click.option(
        "--checkpoint-every", type=click.IntRange(min=0), help="Save every N updates. Use 0 for no periodic saves."
    )
    @click.option("--export", type=click.Choice(("none", "latest", "best")), help="Select the model export at the end.")
    @click.option(
        "--priority", type=click.Choice(("batch", "interactive", "production")), help="Request an Iris priority band."
    )
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
        "--resume",
        "resume_name",
        metavar="NAME@VERSION",
        help="Continue from a stopped run's native checkpoint and its starting model. "
        "A running source may prune the checkpoint before the coordinator starts it.",
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
        resume_name: str | None,
        replicates: int,
        control: bool,
        requirements: tuple[str, ...],
        issue: int | None,
        plan: bool,
        checkpoints: presets.CheckpointPolicy | None,
        checkpoint_every: int | None,
        export: presets.CheckpointPolicy | None,
        priority: str | None,
        sft_name: str | None = None,
    ) -> tuple[ArtifactStep, ...]:
        context = click.get_current_context()
        do_run = context.params["do_run"]
        if plan and do_run:
            raise click.UsageError("Choose either --plan or --run")
        if control and any(value is not None for value in (checkpoints, checkpoint_every, export, priority)):
            raise click.BadParameter("controls use only the preset's pinned run options")
        preset = _with_run_options(
            table.by_name[preset_name],
            checkpoints=checkpoints,
            checkpoint_every=checkpoint_every,
            export=export,
            priority=priority,
        )
        if init is not None and init_name is not None:
            raise click.UsageError("Python init and --init cannot be used together")
        initializer = init
        if sft_name is not None:
            if init is not None or init_name is not None or resume_name is not None or control:
                raise click.UsageError("--sft requires a fresh candidate without --init or --resume")
            if not preset.accepts_init:
                raise click.BadParameter(f"{preset.name} does not accept --sft")
            if preset.gated:
                raise click.BadParameter(f"{preset.name} uses default-model controls and does not accept --sft")
            assert sft_presets is not None
            by_name = {source.name: source for source in sft_presets()}
            if sft_name not in by_name:
                raise click.BadParameter(
                    f"unknown SFT preset {sft_name!r}. Choose from {', '.join(by_name)}.", param_hint="--sft"
                )
            from experiments.sft.launcher import HFModel, SFTSpec  # noqa: PLC0415 - load only for --sft

            sft = by_name[sft_name]
            try:
                initializer = sft.model(
                    coordinator_cluster=preset.recipe.layout.cluster,
                    coordinator_prefix=recipe.run_prefix(preset) if preset.temporary_output else None,
                )
            except ValueError as error:
                raise click.BadParameter(str(error), param_hint="--sft") from error
            export = initializer.step
            export_config = export.build_config(StepContext.for_fingerprint(export.runtime_args, export.deps))
            if isinstance(export_config, dict) and (revision := export_config.get("source_revision")):
                click.get_current_context().meta["sft_source_revision"] = revision
            sft_recipe = sft.recipe
            sft_model = sft_recipe.model if isinstance(sft_recipe, SFTSpec) else None
            if isinstance(sft_model, HFModel) and sft_model.task_timeout_seconds is not None:
                layout = preset.recipe.layout
                hours = layout.coordinator_timeout_hours + math.ceil(sft_model.task_timeout_seconds / 3600)
                layout = replace(layout, coordinator_timeout_hours=hours)
                preset = replace(preset, recipe=replace(preset.recipe, layout=layout))
        click.get_current_context().meta["rl_preset"] = preset
        if resume_name is not None:
            _check_resume(preset, has_init=init is not None or init_name is not None)
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
            init=recipe.published_model(init_name, preset=preset) if init_name else initializer,
            sft_preset=sft_name,
            resume=recipe.recovery_checkpoint(resume_name, preset=preset) if resume_name is not None else None,
            replicates=replicates,
            control=control,
            requirements=requirements,
            issue=issue,
            validate_plan=from_launcher,
            plan_only=not do_run,
            provenance=provenance,
        )
        assert run.inputs is not None
        pin_error = None
        pin_checked = from_launcher and (do_run or plan)
        preset_version = None
        if pin_checked:
            try:
                validated = rl_gate.validate_candidate(preset, run.inputs, table=table, control=control)
            except InvalidComparison as error:
                if do_run:
                    raise
                pin_error = str(error)
            else:
                if do_run:
                    if issue is not None:
                        _validate_launch_issue(run, preset, issue)
                    context.meta["rl_training_type"] = validated["runtime"]["training_type"]
                    index = run_index_identity(
                        validated,
                        preset,
                        table,
                        provenance,
                        user=username_segment(),
                        control=control,
                    )
                    context.meta["rl_index"] = index
                    preset_version = index.preset_version
                else:
                    preset_version = rl_gate.config_version(validated)
        model = preset.model
        if (
            table.needs_completed_default_export
            and run.inputs.init is None
            and (run.inputs.resume is None or run.inputs.resume.model is None)
            and isinstance(model, ArtifactHfModel)
            and model.step.adopt_source is not None
            and not check_cache(model.step.path())
        ):
            sft_hint = "Build it with --sft PRESET or use" if sft_presets is not None else "Use"
            raise click.BadParameter(
                f"default model {model.step.name}@{model.step.version} has no completed HF export. "
                f"{sft_hint} --init NAME@VERSION."
            )
        snapshot = _status(replace(run, gate=rl_gate.completed_gate(run.gate)), preset, stored_only=do_run, issue=issue)
        if preset_version is not None:
            snapshot["preset_version"] = preset_version
        if preset.gated and not control:
            try:
                pin = rl_gate.control_pin(preset, table=table)
            except InvalidComparison as error:
                pin_error = pin_error or str(error)
            else:
                snapshot["control_pin"] = {
                    "commit": pin.commit,
                    "version": pin.version,
                    "current": pin_error is None if pin_checked else None,
                }
            if pin_error is not None:
                snapshot["control_pin_error"] = pin_error
        _print_status(snapshot)
        if not do_run and _evaluation_failed(snapshot, preset):
            raise click.exceptions.Exit(EXIT_CODES[Verdict.FAIL])
        if not do_run and "verdict" in snapshot:
            raise click.exceptions.Exit(EXIT_CODES[snapshot["verdict"]])
        after_run = preset.then
        following = tuple(after_run(step) for step in run.runs) if after_run is not None else ()
        rl_handles = (*run.runs, *((run.gate,) if run.gate else ()))
        click.get_current_context().meta["rl_handle_count"] = len(rl_handles)
        return (*rl_handles, *following)

    if sft_presets is not None:
        run_command.params.append(
            click.Option(
                ["--sft", "sft_name"], metavar="PRESET", help="Build an SFT preset and start RL from its export."
            )
        )
    return run_command


def _run_graph(handles: list[ArtifactStep], max_concurrent: int, preset: presets.Preset, coordinator: str) -> None:
    planned = _from_handles(handles)
    prefix = recipe.run_prefix(preset)
    outputs = tuple(step.path(prefix) for step in planned.runs)
    artifacts = build_artifacts(*planned.runs, max_concurrent=max_concurrent)
    attempts = tuple(
        rl_publication.completed_artifact_attempt(path, artifact.attempt_id)
        for path, artifact in zip(outputs, artifacts, strict=True)
    )
    gate = rl_gate.completed_gate(planned.gate, attempts)
    evidence = rl_publication.CoordinatorEvidence(
        attempts=dict(zip(outputs, (row.attempt_id if row is not None else None for row in attempts), strict=True)),
        gate_path=gate.path(prefix) if gate is not None else None,
    )
    rl_publication.store_coordinator_evidence(outputs[0], coordinator, evidence)
    if gate is not None:
        build_artifacts(gate, max_concurrent=max_concurrent)


def main(
    table: presets.ModelPresets,
    *,
    argv: Sequence[str] | None = None,
    init: SkyRLModelSource | None = None,
    sft_presets: Callable[[], Sequence[SFTModelPreset]] | None = None,
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
            selected = command(
                table,
                argv=tuple(args),
                init=init,
                sft_presets=sft_presets,
                entry_module=entry_module,
                forward_env=forward_env,
            )
        result = selected.main(args=args, standalone_mode=False)
        return result if isinstance(result, int) else 0
    except click.ClickException as error:
        if isinstance(error, click.UsageError):
            error.exit_code = REFUSAL_EXIT_CODE
        error.show()
        return error.exit_code
    except click.exceptions.Exit as error:
        return error.exit_code
