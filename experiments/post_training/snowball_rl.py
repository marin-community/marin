# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Snowball RL experiment and compare a gated preset with pinned controls.

\b
  speed-short      engine throughput and full-panel quality; 8 H100, about 15 min.
  speed-medium     synchronous update time and mismatch; 40 H100, about 40 min.
  speed-long       asynchronous update time and mismatch; 40 H100, 45-75 min.
  learning-medium  panel score after 25 sync updates; 40 H100, about 2 h.
  data-short       mixed-answer fraction on --tasks; 8 H100, about 15 min.
  data-medium      learning-medium on --tasks; 40 H100, about 2 h.
  train-long       async training, checkpoints and HF export; 40 H100 east; no verdict.

Guide: docs/recipes/snowball-rl.md. Without --run, print the plan and saved status.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl`:
  --preset speed-medium --version dev --run --wait
  --preset learning-medium --set trainer.algorithm.advantage_estimator=grpo \
    --set trainer.algorithm.group_advantage_min_size=null --version dev --run
  --preset speed-medium --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
  --preset learning-medium --require 'test_panel/aaii_category/coding/mean>=control' --version dev --run
  --preset learning-medium --version dev --run
  --preset data-short --tasks my-selection@2026.11.02 --version dev --run
  --preset train-long --set trainer.max_steps=100 --version dev --run
"""

from __future__ import annotations

import json
import math
import os
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import partial, wraps
from pathlib import Path

import click
import yaml
from iris.cli.connect import open_iris_client
from iris.client.client import IrisClient, get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from iris.rpc import job_pb2
from marin.execution.artifact import is_mutable_version, read_record
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.lazy import run as build_artifacts
from marin.execution.step_status import get_status_path
from marin.experiment.cli import build_options_with_runner
from marin.rl.cli import submit_or_run
from marin.rl.skyrl import (
    BEST_CHECKPOINT_CALLBACK,
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLModelSource,
    SkyRLPolicyModel,
    SkyRLReceiptLookupError,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLSpec,
    SkyRLTopology,
    read_skyrl_launch_receipt,
    run_skyrl,
    skyrl_step,
)
from marin.rl.skyrl_records import (
    SkyRLLaunchReceipt,
    read_skyrl_attempt_tracking,
    read_skyrl_metrics,
    read_stored_skyrl_receipt,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance, git_output

from experiments.post_training import snowball_rl_gate, snowball_rl_publication
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.task_curation.selection import (
    FULL_VALIDATION_FILENAME,
    TRAIN_FILENAME,
    GraderKind,
)

# Iris limit for one Snowball run, coordinator and job alike.
RUN_TIMEOUT_HOURS = 72


DEFAULT_WANDB_ENTITY = "marin-community"
_WANDB_ENTITY_ENV = "WANDB_ENTITY"
_FAILED_RUN_STATES = {JobState.FAILED, JobState.KILLED, JobState.WORKER_FAILED, JobState.UNSCHEDULABLE}


def training_config(
    preset: presets.Preset,
    settings: tuple[presets.Setting, ...] = (),
    evaluations: tuple[presets.Evaluation, ...] = (),
) -> dict:
    recipe = preset.recipe
    plan = recipe.layout.role_plan
    config = {
        "context_budget": {
            "request_window_tokens": recipe.request_window_tokens,
            "max_new_tokens_per_turn": (
                recipe.fixed_reply_tokens if recipe.fixed_reply_tokens is not None else recipe.max_new_tokens
            ),
            "max_turns": 1,
        },
        "environment": {
            "task_sessions": {
                "local": {},
            }
        },
        "data": {"kind": "parquet", "shuffle": True},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": False,
            "gradient_checkpointing": True,
            "offload_optimizer_during_rollouts": False,
            "epochs": 50,
            "max_steps": preset.updates or 1,
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
            "eval_batch_size": 256,
            "eval_interval": max(1, preset.updates) if evaluations else -1,
            "eval_before_train": False,
            "resume_mode": "latest" if preset.focus == presets.Focus.TRAIN else "none",
            "seed": presets.SEED,
            "logger": ["console", "files", "wandb"],
            "project_name": "marin-snowball-rl",
            "tracker_commit_each_step": True,
            "training_metrics": True,
            "rollout_spans": True,
            "policy_train_spans": True,
            "algorithm": {**deepcopy(dict(recipe.estimator)), "batch_invariant": False},
            "policy": {"optimizer_config": deepcopy(dict(recipe.optimizer))},
            "rollout_buffer": {
                "max_staleness_steps": recipe.staleness,
                "batch_policy": "full_batch",
                "max_in_flight": plan.train_batch_size,
                "object_store_root": None,
            },
            "ckpt_interval": 0,
            "hf_save_interval": -1,
            "callbacks": [],
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "weight_sync_pause": {"mode": "keep", "clear_cache": True},
            "weight_sync_transport": "expert_block",
            "expert_block_sync": {"timeout_seconds": 600, "verify": False},
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 1024,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": True,
            "trajectory_reward_shaping": {
                "enabled": True,
                "overlong": {"penalty_scale": 0.0},
                "reasoning_span": {"penalty": 0.1, "target": "marker"},
            },
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "enable_http_endpoint": False,
            "chat_template": None,
            "apply_overlong_filtering": False,
            "engine_init_kwargs": {"moe_backend": "triton", "enable_auto_tools": True, "tool_parser": "hermes"},
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "logprobs": 0,
                "ignore_eos": recipe.fixed_reply_tokens is not None,
            },
        },
        "trajectory_runner": {"max_concurrent_tasks": 1024},
        "extra_env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    }
    geometry = recipe.layout.megatron
    config["trainer"]["policy"]["megatron_config"] = asdict(geometry)
    config["trainer"]["ref"] = {"megatron_config": asdict(geometry)}
    config["extra_env"].update(NCCL_NET="Socket", NCCL_IB_DISABLE="1", NCCL_SOCKET_FAMILY="AF_INET")
    for setting in settings:
        presets.apply_setting(config, setting, preset)
    return config


@dataclass(frozen=True)
class SnowballRun:
    runs: tuple[ArtifactStep[SkyRLRun], ...]
    gate: ArtifactStep[snowball_rl_gate.GateReport] | None = None


def _model_input(init: ArtifactStep | None) -> SkyRLModelSource:
    if init is None:
        model = ArtifactStep.adopt(
            f"models/snowball-datakit-sft-{presets.MODEL_REVISION}",
            "2026.09.21",
            presets.MODEL_URI,
            kind=LevanterCheckpoint,
        )
        return ArtifactHfModel(model, presets.MODEL_ID, presets.MODEL_REVISION, relative_path="")
    if init.artifact_type is SkyRLRun:
        record = read_record(init.path())
        if record is None:
            raise click.BadParameter("the initializer requires a completed published artifact record")
        result = record.result or {}
        if not result.get("hf_model_uri"):
            raise click.BadParameter("the initializer run has no HF export")
        return SkyRLPolicyModel(init)
    return ArtifactHfModel(init, presets.MODEL_ID, presets.MODEL_REVISION)


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


def _callbacks(
    preset: presets.Preset,
    evaluation: presets.Evaluation | None,
    profiles: dict,
    expected_tasks: int,
) -> list[dict]:
    if preset.recipe.layout.role_plan.generate_only:
        return [{"type": "evaluation", "additional_evaluations": profiles}] if profiles else []
    callbacks = []
    if preset.checkpoint_every:
        callbacks.append({"type": "checkpoint", "save_steps": preset.checkpoint_every})
    if evaluation is not None:
        for callback in presets.evaluation_callbacks(evaluation, updates=preset.updates):
            if profiles and callback["type"] == "evaluation":
                callback["additional_evaluations"] = profiles
            callbacks.append(callback)
    if preset.focus == presets.Focus.TRAIN:
        callbacks.extend(
            [
                {
                    "type": BEST_CHECKPOINT_CALLBACK,
                    "expected_tasks": expected_tasks,
                    "minimum_scored_tasks": math.ceil(presets.MINIMUM_PANEL_COVERAGE * expected_tasks),
                },
                {
                    "type": "hf_model_save",
                    "save_steps": ((preset.updates // preset.checkpoint_every) + 1) * preset.checkpoint_every,
                    "save_on_train_end": True,
                },
            ]
        )
    if preset.keep_weights and preset.focus != presets.Focus.TRAIN:
        callbacks.extend(
            (
                {"type": "checkpoint", "save_steps": preset.updates + 1, "save_on_train_end": True},
                {"type": "hf_model_save", "save_steps": -1, "save_on_train_end": True},
            )
        )
    return callbacks


def _run_snowball(config: SkyRLRunConfig, *, inputs: snowball_rl_gate.GateInputs) -> SkyRLRun:
    path = StoragePath(config.output.resolved_config_uri).parent / snowball_rl_gate.GATE_INPUTS_FILENAME
    if not path.exists():
        path.parent.mkdirs()
        path.write_text(inputs.model_dump_json(indent=2) + "\n")
    return run_skyrl(config)


def _run_step(
    name: str,
    config: dict,
    model: SkyRLModelSource,
    selection: ArtifactStep,
    *,
    layout: presets.Layout,
    export_hf: bool,
    retention: int,
    validation_data: tuple[ArtifactDataSource, ...],
    wandb_entity: str | None,
    evaluation_data: tuple[tuple[str, tuple[ArtifactDataSource, ...]], ...] = (),
) -> ArtifactStep[SkyRLRun]:
    plan = layout.role_plan
    return skyrl_step(
        SkyRLSpec(
            name=name,
            version=presets.PRESET_REVISION,
            config_yaml=yaml.safe_dump(config, sort_keys=False),
            model=model,
            train_data=(() if plan.generate_only else (ArtifactDataSource(selection, relative_path=TRAIN_FILENAME),)),
            validation_data=validation_data,
            evaluation_data=evaluation_data,
            topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=retention),
            seed=config["trainer"]["seed"],
        ),
        IrisSkyRLExecution(
            cluster=layout.cluster,
            cluster_config=presets.cluster_config_path(layout.cluster),
            cpu=32,
            memory=layout.host_memory,
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=layout.cluster,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=RUN_TIMEOUT_HOURS,
            job_timeout_seconds=RUN_TIMEOUT_HOURS * 3600,
            wandb_entity=wandb_entity or DEFAULT_WANDB_ENTITY,
            required_env_vars=(("WANDB_API_KEY", _WANDB_ENTITY_ENV) if wandb_entity else ("WANDB_API_KEY",)),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=export_hf,
    )


def _refuse_judged(summary: dict, filename: str) -> None:
    if summary["grader_kind_counts"][filename].get(GraderKind.REMOTE_JUDGE, 0):
        raise click.BadParameter("Snowball selections must contain no AI-judge-graded tasks")


def build_run(
    preset: presets.Preset,
    *,
    tasks: ArtifactStep | None = None,
    init: ArtifactStep | None = None,
    settings: tuple[presets.Setting, ...] = (),
    replicates: int = 1,
    control: bool = False,
    requirements: tuple[str, ...] = (),
) -> SnowballRun:
    """Construct a Snowball experiment: one run step per replicate and, for a gated preset, its gate step."""
    wandb_entity = os.environ.get(_WANDB_ENTITY_ENV)
    export_hf = preset.keep_weights
    settings = tuple(leaf for setting in settings for leaf in presets.setting_leaves(setting))
    if not 1 <= replicates <= 5:
        raise click.BadParameter("replicates must be between 1 and 5")
    if preset.focus == presets.Focus.DATA:
        if settings or (tasks is None and not control):
            raise click.BadParameter("data presets require --tasks and accept no --set")
    elif preset.focus != presets.Focus.TRAIN and tasks is not None:
        raise click.BadParameter("use a data or train-long preset with --tasks")
    if preset.focus not in (presets.Focus.LEARNING, presets.Focus.TRAIN) and init is not None:
        raise click.BadParameter("--init requires a learning or train-long preset")
    if preset.focus == presets.Focus.TRAIN and (replicates != 1 or control):
        raise click.BadParameter("train-long accepts neither --replicates nor --control")
    if control and (
        settings or tasks is not None or init is not None or preset.name in snowball_rl_gate.SHARED_CONTROLS
    ):
        raise click.BadParameter("controls use only the preset's pinned default inputs")
    batch_changes = {field: setting.value for setting in settings if (field := presets.BATCH_SETTINGS.get(setting.key))}
    if batch_changes:
        plan = replace(preset.recipe.layout.role_plan, **batch_changes)
        preset = replace(preset, recipe=replace(preset.recipe, layout=replace(preset.recipe.layout, role_plan=plan)))
    default = presets.default_selection()
    selection = tasks or default
    if is_mutable_version(selection.version):
        raise click.BadParameter("curated task selections must have an immutable version")
    summary = presets.selection_summary(selection)
    if (
        tasks is not None
        and preset.focus == presets.Focus.DATA
        and summary["recipe"]["heldout"] != presets.selection_summary(default)["recipe"]["heldout"]
    ):
        raise click.BadParameter("candidate tasks use a different heldout input; curate against the default panel")
    if tasks is not None:
        panel = presets.selection_artifact(f"{presets.DEFAULT_TEST_PANEL_NAME}@{presets.DEFAULT_TEST_PANEL_VERSION}")
        excluded = summary["recipe"].get("exclusions", ())
        expected = str(StoragePath(panel.path()) / FULL_VALIDATION_FILENAME)
        if not any(row["uri"] == expected and row["version"] == panel.version for row in excluded):
            raise click.BadParameter("candidate tasks must record the full test panel in recipe.exclusions")
    chosen = preset.evaluations
    generating = preset.recipe.layout.role_plan.generate_only
    _refuse_judged(summary, "train")
    config = training_config(preset, settings, chosen)
    preset = replace(preset, updates=config["trainer"]["max_steps"])
    model = _model_input(init)
    validation_data = ()
    evaluation_data = []
    profiles = {}
    evaluation_inputs = []
    expected_tasks = 0
    for index, evaluation in enumerate(chosen):
        evaluation_tasks = evaluation.tasks(selection, default)
        evaluation_inputs.append((evaluation.name, evaluation_tasks.name, evaluation_tasks.version))
        _refuse_judged(presets.selection_summary(evaluation_tasks), evaluation.file)
        profile_config = config if index == 0 else deepcopy(config)
        sources = presets.configure_evaluation(evaluation, profile_config, evaluation_tasks)
        if index == 0:
            validation_data = sources
            answer_count = presets.evaluation_answer_count(evaluation, evaluation_tasks)
            expected_tasks = answer_count // evaluation.answers
            if generating:
                config["trajectory_runner"]["max_concurrent_tasks"] = answer_count
        else:
            evaluation_data.append((evaluation.name, sources))
            profiles[evaluation.name] = {
                "sampling_params": profile_config["generator"]["eval_sampling_params"],
                "n_samples_per_prompt": profile_config["generator"]["eval_n_samples_per_prompt"],
            }
            if not generating:
                profiles[evaluation.name]["schedule"] = "final"
    config["trainer"]["callbacks"] = _callbacks(preset, chosen[0] if chosen else None, profiles, expected_tasks)
    changes = {
        "tasks": (selection.name, selection.version),
        "init": None if init is None else (init.name, init.version),
        "layout": preset.recipe.layout.name,
        "settings": sorted({setting.key: setting.value for setting in settings}.items()),
        "evaluations": sorted(evaluation_inputs),
    }
    snowball_rl_gate.validate_candidate(preset, requirements, control=control)
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
        run = _run_step(
            name,
            config,
            model,
            selection,
            layout=preset.recipe.layout,
            export_hf=export_hf,
            retention=5 if preset.checkpoint_every else 2,
            validation_data=validation_data,
            evaluation_data=tuple(evaluation_data),
            wandb_entity=wandb_entity,
        )
        runs.append(replace(run, run=partial(_run_snowball, inputs=inputs)))
    gate = (
        snowball_rl_gate.gate_step(
            preset,
            tuple(runs),
            inputs=inputs,
        )
        if presets.gated(preset) and not control
        else None
    )
    return SnowballRun(tuple(runs), gate)


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


def _grafana_links(preset: presets.Preset, receipt: SkyRLLaunchReceipt | None, cluster: str) -> dict[str, str]:
    if receipt is None or not receipt.result.iris_job_id:
        return {}
    run_id = receipt.config.get("run", {}).get("id")
    if not run_id or receipt.result.started_at is None:
        return {}
    training_type = receipt.config.get("runtime", {}).get("training_type")
    return presets.grafana_links(
        run_id=run_id,
        cluster=cluster,
        training_type=(
            "async" if training_type == "async" or (training_type is None and preset.recipe.staleness) else "sync"
        ),
        started_ms=presets.utc_millis(receipt.result.started_at),
        finished_ms=presets.utc_millis(receipt.result.finished_at) if receipt.result.finished_at else None,
    )


def _completed_update(path: str, *, generating: bool) -> int | None:
    key = presets.GENERATE_WALL_SECONDS_METRIC if generating else "timing/step_wall/unaccounted"
    try:
        metrics = read_skyrl_metrics(path)
    except FileNotFoundError:
        return None
    for _, values in sorted(metrics.items(), reverse=True):
        if key in values and (generating or values.get("trainer/global_step") == values["step"]):
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
    run = replace(run, gate=snowball_rl_gate.completed_gate(preset, run.runs, run.gate))
    rows = []
    planned = []
    for step in run.runs:
        execution = next(value for value in step.runtime_args.values() if isinstance(value, IrisSkyRLExecution))
        path = step.path(presets.artifact_prefix(execution.cluster))
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        planned.append(yaml.safe_load(config.launch_config_yaml))
        try:
            receipt = read_stored_skyrl_receipt(path) if stored_only else read_skyrl_launch_receipt(path)
        except SkyRLReceiptLookupError as error:
            raise click.BadParameter(str(error)) from error
        result = receipt.result if receipt is not None else None
        attempt = receipt.config.get("run", {}).get("attempt_id") if receipt is not None else None
        tracking = read_skyrl_attempt_tracking(path, attempt) if attempt else {}
        team = receipt.config.get("iris", {}).get("wandb_entity") if receipt is not None else execution.wandb_entity
        grafana = _grafana_links(preset, receipt, execution.cluster)
        state = (
            result.state
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
                "wandb_entity": team or DEFAULT_WANDB_ENTITY,
                **{
                    key: tracking.get(key)
                    for key in ("attempt_id", "wandb_url", "started_at", "engine")
                    if tracking.get(key) is not None
                },
                **({"grafana_links": grafana} if grafana else {}),
                "latest_completed_update": _completed_update(
                    path, generating=preset.recipe.layout.role_plan.generate_only
                ),
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
        path = (
            StoragePath(run.gate.path(prefix=presets.artifact_prefix(preset.recipe.layout.cluster)))
            / snowball_rl_gate.GATE_REPORT_FILENAME
        )
        if path.exists():
            report = snowball_rl_gate.GateReport.model_validate_json(path.read_text())
            verdict = {
                "report": str(path),
                **report.model_dump(mode="json", include={"verdict", "label", "reason"}),
            }
            changes, reason = _changed_paths(
                report.control.commit,
                ("experiments/post_training", "lib/marin/src/marin/rl", "lib/taskcompendium", "vendor/skyrl"),
            )
            verdict["rl_path_changes_since_control"] = changes
            if reason:
                verdict["rl_path_changes_reason"] = reason
    if "verdict" not in verdict:
        crashed = next((row for row in rows if str(row["state"]).lower() in _FAILED_RUN_STATES), None)
        if crashed is not None:
            verdict = {
                "verdict": snowball_rl_gate.Verdict.INVALID.value,
                "reason": f"native job {crashed['iris_job_id']} is {crashed['state']}",
                "iris_job_id": crashed["iris_job_id"],
            }
    objective_rows = (
        [row for row in report.comparisons if row.role == snowball_rl_gate.ComparisonRole.OBJECTIVE] if report else ()
    )
    constraint_rows = (
        [
            row
            for row in report.comparisons
            if row.role in (snowball_rl_gate.ComparisonRole.CONSTRAINT, snowball_rl_gate.ComparisonRole.REQUIREMENT)
        ]
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
        "updates": 0 if preset.recipe.layout.role_plan.generate_only else planned[0]["skyrl"]["trainer"]["max_steps"],
        "measured_updates": list(preset.measured),
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
    if not snowball_rl_publication.publishes(preset):
        return None
    prefix = presets.artifact_prefix(preset.recipe.layout.cluster)
    completed_gate = snowball_rl_gate.completed_gate(preset, run.runs, run.gate)
    gate_path = completed_gate.path(prefix) if completed_gate is not None else None
    outputs = [training.path(prefix) for training in run.runs]
    return snowball_rl_publication.publish_runs(
        outputs, preset, gate_path=gate_path, gate_required=completed_gate is not None, issue=issue
    )


def _finish_run(
    handles: list[ArtifactStep],
    preset: presets.Preset,
    coordinator_error: str | None = None,
    *,
    issue: int | None = None,
) -> None:
    run = _from_handles(handles)
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
        verdict = "pass"
    raise click.exceptions.Exit(snowball_rl_gate.EXIT_CODES[verdict])


def _wait_for_run(
    job: JobName, handles: list[ArtifactStep], preset: presets.Preset, *, issue: int | None = None
) -> None:
    with _iris_client() as client:
        parent = client.job(job).wait(RUN_TIMEOUT_HOURS * 3600, poll_interval=60, raise_on_failure=False)
    error = (
        f"coordinator {job} is {parent.state}; required verdict unavailable"
        if parent.state != JobState.SUCCEEDED
        else None
    )
    _finish_run(handles, preset, error, issue=issue)


def _run_command(function: Callable) -> click.Command:
    @wraps(function)
    def checked(*args, wait: bool = False, **kwargs):
        if wait and not click.get_current_context().params["do_run"]:
            raise click.BadParameter("--wait requires --run")
        try:
            return function(*args, **kwargs)
        except scoring.InvalidComparison as error:
            click.echo(
                json.dumps(
                    {
                        "preset": kwargs["preset_name"],
                        "verdict": snowball_rl_gate.Verdict.INVALID.value,
                        "reason": str(error),
                    }
                )
            )
            raise click.exceptions.Exit(snowball_rl_gate.REFUSAL_EXIT_CODE) from error

    def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
        options = click.get_current_context().params
        preset = presets.PRESETS[options["preset_name"]]
        try:
            if get_job_info() is None:
                job = submit_or_run("experiments.post_training.snowball_rl", handles, max_concurrent, os.environ)
            else:
                planned = _from_handles(handles)
                build_artifacts(*planned.runs, max_concurrent=max_concurrent)
                gate = snowball_rl_gate.completed_gate(preset, planned.runs, planned.gate)
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
            raise click.exceptions.Exit(snowball_rl_gate.EXIT_CODES[None]) from error

    return click.command(cls=snowball_rl_gate.SnowballCommand, help=__doc__)(build_options_with_runner(checked, runner))


@_run_command
@click.option("--issue", type=click.IntRange(min=1), help="GitHub experiment issue for launcher publication.")
@click.option(
    "--wait",
    is_flag=True,
    help="Wait after --run and publish trained-panel attempts using the caller Marina login; "
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
    help="Change one key of the composed SkyRL config: a dotted key and a YAML value, for example "
    "trainer.algorithm.advantage_estimator=grpo. Use +KEY=VALUE for a key the base config lacks. "
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
    help="Curated task selection at an immutable version, recording the full test panel in recipe.exclusions. "
    "Required by data presets, whose selection must hold out the same panel as the default; "
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
        settings=tuple(presets.parse_setting(setting) for setting in settings),
        tasks=presets.selection_artifact(tasks_name) if tasks_name else None,
        init=presets.published_artifact(init_name) if init_name else None,
        replicates=replicates,
        control=control,
        requirements=requirements,
    )
    do_run = click.get_current_context().params.get("do_run", False)
    snapshot = _status(run, preset, stored_only=do_run, issue=issue)
    click.echo(yaml.safe_dump(snapshot, sort_keys=False, width=1000))
    if not do_run and "verdict" in snapshot:
        raise click.exceptions.Exit(snowball_rl_gate.EXIT_CODES[snapshot["verdict"]])
    return (*run.runs, *((run.gate,) if run.gate else ()))


if __name__ == "__main__":
    main()
