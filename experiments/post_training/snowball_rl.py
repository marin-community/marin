# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Snowball RL experiment from a preset; gated presets compare the run with pinned controls.

Calibrated presets report PASS, FAIL or INVALID against control runs of the same preset, with a
change label. Choose the preset by what the change can affect. Each entry
gives what the preset runs, its cost, what it measures and the changes it is for:

\b
  speed-short      vLLM engine only: one pass of 512 prompts x 4 replies of 4,096 tokens;
                   8 H100, about 10 min. Measures engine tokens/s per GPU. For engine
                   options, kernels and vLLM wheels.
  speed-medium     10 sync updates, every reply 4,096 tokens; 40 H100, about 40 min. Measures
                   median update time and trainer-vs-engine log-prob mismatch. For trainer
                   speed, weight sync and startup.
  speed-long       20 async updates with staleness 2 and TIS, replies up to 8,192 tokens;
                   40 H100, target 45-75 min. Measures update time and learner/engine
                   tokens/s per GPU. For the async loop, grading and admission speed.
  learning-medium  25 sync updates, then the 910-task panel; 40 H100, about 2 h. Measures the
                   panel score. For loss, estimator, optimizer, schedule, reward shaping,
                   batch shape and --init.
  data-short       4 replies to each of 256 tasks of a new --tasks selection; 8 H100, about
                   15 min. Measures the fraction of tasks with both right and wrong replies,
                   the only tasks that give RLOO-N a learning signal. For a new task selection.
  data-medium      learning-medium trained on --tasks instead of the default selection;
                   40 H100, about 2 h. Measures the panel score after training on the new
                   selection. For a new task selection that passed data-short.
  train-long       the async production loop with checkpoints, panel-100 every 10 updates and
                   an HF export; 112 H100, or 40 with --layout p40. For release training and
                   for models that later runs load with --init. No verdict.

An engine change runs speed-short, then speed-medium. A trainer kernel, parallelism or numerics
change runs speed-medium, then learning-medium. A new task selection runs data-short, then
data-medium. --require adds a condition of your own, such as a mismatch bound on a speed run or
a per-category score on a learning run. Without --run the command prints the plan and the status
block, including the verdict of a finished gated run; --run submits it. --wait polls every 60 s
and prints the final status and report, exiting 0 for PASS, 1 for FAIL, 2 for INVALID.
--baseline NAME@VERSION adds a comparison with earlier finished runs; the verdict uses the
pinned controls. Status includes each run's latest completed update and saved W&B URL.
Set WANDB_ENTITY before submission to choose the W&B team; unset uses dogml.
--version dev satisfies
the launcher; Snowball steps use PRESET_REVISION. Guide: docs/recipes/snowball-rl.md.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl`:
  --preset speed-medium --version dev --run
  --preset speed-medium --version dev --run --wait
  --preset speed-medium --baseline EARLIER_RUN@2026.10.27 --version dev --run --wait
  --preset learning-medium --set trainer.algorithm.advantage_estimator=grpo --version dev --run
  --preset speed-medium --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
  --preset learning-medium --require 'panel-910/aaii_category/coding/mean>=control' --version dev --run
  --preset data-short --tasks my-selection@2026.11.02 --version dev --run
  --preset train-long --layout p40 --set trainer.max_steps=100 --version dev --run
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import wraps
from pathlib import Path

import click
import yaml
from iris.cli.connect import open_iris_client
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES, JobState
from iris.rpc import job_pb2
from marin.execution.artifact import is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_status import get_status_path
from marin.experiment.cli import build_options_with_runner
from marin.rl.cli import submit_or_run
from marin.rl.skyrl import (
    BEST_CHECKPOINT_CALLBACK,
    IRIS_HUB_CLUSTER_CONFIG,
    SKYRL_TEMPORARY_STORAGE_TTL_DAYS,
    TRACKING_FILENAME,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLModelSource,
    SkyRLPolicyModel,
    SkyRLReceiptLookupError,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    artifact_identity,
    read_skyrl_launch_receipt,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance, git_output

from experiments.post_training import snowball_rl_gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.task_curation.inventory import (
    TRAIN_FILENAME,
    GraderKind,
)

# Iris limit for one Snowball run, coordinator and job alike.
RUN_TIMEOUT_HOURS = 72


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
            "max_turns": recipe.max_turns,
        },
        "environment": {
            "task_sessions": {
                "local": {},
                "remote_judge": {"base_url": "https://api.together.xyz/v1", "api_key_env": presets.JUDGE_KEY},
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
            "project_name": "snowball-rl",
            "tracker_commit_each_step": True,
            "training_metrics": True,
            "rollout_spans": True,
            "policy_train_spans": True,
            "algorithm": deepcopy(dict(recipe.estimator)),
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
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "enable_http_endpoint": False,
            "chat_template": None,
            "apply_overlong_filtering": False,
            "engine_init_kwargs": deepcopy(dict(recipe.engine_init_kwargs)),
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
            "VERIFYIT_JUDGE_MODEL": "openai/gpt-oss-120b",
        },
    }
    geometry = recipe.layout.megatron
    config["trainer"]["policy"]["megatron_config"] = asdict(geometry)
    config["trainer"]["ref"] = {"megatron_config": asdict(geometry)}
    config["trainer"]["policy"]["megatron_config"].update(deepcopy(dict(recipe.policy_megatron_overrides)))
    if recipe.layout.sockets:
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
        try:
            result = presets.artifact_record(init.path()).result or {}
        except FileNotFoundError as error:
            raise click.BadParameter("the initializer requires a completed published artifact record") from error
        if not result.get("hf_model_uri"):
            raise click.BadParameter("the initializer run has no HF export; gated runs cannot initialize another run")
        return SkyRLPolicyModel(init)
    return ArtifactHfModel(init, presets.MODEL_ID, presets.MODEL_REVISION)


def _provenance(*, control: bool) -> Provenance:
    root = presets.REPO_ROOT
    provenance = Provenance.capture(root)
    if (root / ".git").exists():
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "experiments", "lib", "vendor"], cwd=root, text=True
        ).splitlines()
        if untracked:
            raise click.BadParameter(f"stage source files before naming a run: {', '.join(untracked)}")
    if not provenance.tree_hash or (control and provenance.dirty):
        raise click.BadParameter("controls require a clean checkout; runs require git or Iris source provenance")
    return provenance


def _callbacks(preset: presets.Preset, evaluations: tuple[presets.Evaluation, ...], expected_tasks: int) -> list[dict]:
    callbacks = []
    if preset.checkpoint_every:
        callbacks.append({"type": "checkpoint", "save_steps": preset.checkpoint_every})
    for evaluation in evaluations:
        callbacks.extend(presets.EVALUATORS[evaluation.harness].callbacks(evaluation, updates=preset.updates))
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
    return callbacks


def _resume_checkpoint_root(source: ArtifactStep[SkyRLRun], model: SkyRLModelSource) -> str:
    if isinstance(model, ArtifactHfModel):
        tokenizer, revision = model.tokenizer_uri, model.tokenizer_revision
    else:
        result = presets.artifact_record(model.step.path()).result
        tokenizer, revision = result["tokenizer_uri"], result["tokenizer_revision"]
    try:
        receipt = read_skyrl_launch_receipt(source.path())
    except SkyRLReceiptLookupError as error:
        raise click.BadParameter(str(error)) from error
    if receipt is None:
        raise click.BadParameter(f"no launch metadata at {source.path()}")
    config, result = receipt.config, receipt.result
    if result.state not in ("succeeded", "failed"):
        raise click.BadParameter(f"source Iris job {result.iris_job_id} is {result.state}; wait for a terminal state")
    if (
        f"/{presets.TRAIN_LONG_RUN}" not in f"/{source.name}"
        or not config["run"]["export_hf"]
        or config["run"]["id"] != f"{source.name}-{source.version}"
    ):
        raise click.BadParameter("--resume-from requires a train-long run")
    initial = config["inputs"]["model"]
    expected = (artifact_identity(model.step), tokenizer, revision)
    if tuple(initial[key] for key in ("identity", "tokenizer_uri", "tokenizer_revision")) != expected:
        raise click.BadParameter("resume source uses another model or tokenizer; use its matching --init artifact")
    root = config["artifacts"]["checkpoint_root"]
    marker = StoragePath(root) / "latest_ckpt_global_step.txt"
    missing = (
        f"no checkpoint remains at {root}; temporary checkpoints expire after {SKYRL_TEMPORARY_STORAGE_TTL_DAYS} days"
    )
    if not presets.exists(marker) or not presets.exists(marker.parent / f"global_step_{int(presets.read_text(marker))}"):
        raise click.BadParameter(missing)
    return root


def _run_step(
    name: str,
    config: dict,
    model: SkyRLModelSource,
    selection: ArtifactStep,
    required: set[str],
    *,
    layout: presets.Layout,
    export_hf: bool,
    retention: int,
    validation_data: tuple[ArtifactDataSource, ...],
    wandb_entity: str | None,
    checkpoint_root: str | None = None,
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
            topology=SkyRLTopology(num_nodes=plan.num_nodes, gpus_per_node=8, gpu_variant="H100", role_plan=plan),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=retention),
            seed=config["trainer"]["seed"],
            resume_checkpoint_root=checkpoint_root,
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
            wandb_entity=wandb_entity,
            required_env_vars=tuple(sorted(required | {"WANDB_API_KEY"})),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=export_hf,
    )


def build_run(
    preset: presets.Preset,
    *,
    tasks: ArtifactStep | None = None,
    init: ArtifactStep | None = None,
    layout: presets.Layout | None = None,
    resume_from: ArtifactStep[SkyRLRun] | None = None,
    settings: tuple[presets.Setting, ...] = (),
    evaluations: tuple[str, ...] = (),
    replicates: int = 1,
    control: bool = False,
    requirements: tuple[str, ...] = (),
    baselines: tuple[str, ...] = (),
) -> SnowballRun:
    """Construct a Snowball experiment: one run step per replicate and, for a gated preset, its gate step."""
    snowball_rl_gate.check_requirements(preset, requirements, control=control)
    if baselines and (control or preset.name not in snowball_rl_gate.CONTROLS):
        raise click.BadParameter("--baseline needs a gated candidate preset, without --control")
    settings = tuple(leaf for setting in settings for leaf in presets.setting_leaves(setting))
    if not 1 <= replicates <= 5:
        raise click.BadParameter("replicates must be between 1 and 5")
    if preset.focus == presets.Focus.DATA:
        if settings or (tasks is None and not control):
            raise click.BadParameter("data presets require --tasks and accept no --set")
    elif preset.focus != presets.Focus.TRAIN and tasks is not None:
        raise click.BadParameter("use a data or train-long preset with --tasks")
    if preset.focus not in (presets.Focus.LEARNING, presets.Focus.TRAIN) and init is not None:
        raise click.BadParameter("--init requires learning-medium or train-long")
    if preset.focus != presets.Focus.TRAIN and (layout is not None or resume_from is not None or evaluations):
        raise click.BadParameter("--layout, --resume-from and --eval require train-long")
    if preset.focus == presets.Focus.TRAIN and (replicates != 1 or control):
        raise click.BadParameter("train-long accepts neither --replicates nor --control")
    if control and (settings or tasks is not None or init is not None or evaluations or preset.name == "data-medium"):
        raise click.BadParameter("controls use only the preset's pinned default inputs")
    provenance = _provenance(control=control)
    if layout is not None:
        preset = replace(preset, recipe=presets.TRAIN_LONG_RECIPES[layout.name])
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
    selected = {evaluation.name: evaluation for evaluation in preset.evaluations}
    for name in evaluations:
        selected.setdefault(name, presets.EVALUATIONS[name])
    if init is not None:
        selected = {name: replace(evaluation, at_start=True) for name, evaluation in selected.items()}
    chosen = tuple(selected.values())
    if sum(evaluation.harness == presets.EvalHarness.PANEL for evaluation in chosen) > 1:
        raise click.BadParameter("a run evaluates at most one TaskSpec panel")
    required = {key for evaluation in selected.values() for key in presets.EVALUATORS[evaluation.harness].required_env}
    counts = summary["grader_kind_counts"]["train"]
    if counts.get(GraderKind.REMOTE_JUDGE, 0) and preset.name != "speed-short":
        required.add(presets.JUDGE_KEY)
    missing = sorted(key for key in required if not os.environ.get(key))
    if missing:
        raise click.BadParameter(f"missing required credentials: {', '.join(missing)}")
    config = training_config(preset, settings, chosen)
    preset = replace(preset, updates=config["trainer"]["max_steps"])
    model = _model_input(init)
    checkpoint_root = _resume_checkpoint_root(resume_from, model) if resume_from is not None else None
    validation_data = ()
    for evaluation in chosen:
        evaluation_tasks = evaluation.tasks(selection, default)
        validation_data = presets.EVALUATORS[evaluation.harness].configure(evaluation, config, evaluation_tasks)
        if preset.recipe.layout.role_plan.generate_only:
            config["trajectory_runner"]["max_concurrent_tasks"] = presets.evaluation_answer_count(
                evaluation, evaluation_tasks
            )
    if not preset.recipe.layout.role_plan.generate_only:
        expected_tasks = presets.selection_summary(default)["validation_rows"]
        config["trainer"]["callbacks"] = _callbacks(preset, chosen, expected_tasks)
    changes = {
        "tasks": (selection.name, selection.version),
        "init": None if init is None else (init.name, init.version),
        "layout": preset.recipe.layout.name,
        "resume": None if resume_from is None else (resume_from.name, resume_from.version),
        "settings": sorted({setting.key: setting.value for setting in settings}.items()),
        "evaluations": sorted(selected),
    }
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True, default=str))
    wandb_entity = os.environ.get("WANDB_ENTITY") or None
    runs = []
    for replicate in range(replicates):
        name = presets.run_name(preset, provenance, inputs_hash, replicate, control=control)
        run = _run_step(
            name,
            config,
            model,
            selection,
            required,
            layout=preset.recipe.layout,
            export_hf=preset.focus == presets.Focus.TRAIN,
            retention=5 if preset.checkpoint_every else 2,
            validation_data=validation_data,
            wandb_entity=wandb_entity,
            checkpoint_root=checkpoint_root,
        )
        runs.append(run)
    gate = (
        snowball_rl_gate.gate_step(
            preset, tuple(runs), inputs=snowball_rl_gate.gate_inputs(settings, tasks, init, requirements, baselines)
        )
        if preset.name in snowball_rl_gate.CONTROLS and not control
        else None
    )
    return SnowballRun(tuple(runs), gate)


def _changed_paths(ref: str, paths: tuple[str, ...]) -> tuple[list[dict[str, str]] | None, str | None]:
    if not (presets.REPO_ROOT / ".git").exists():
        return None, "git metadata is unavailable on this host"
    try:
        output = git_output(["diff", "--no-renames", "--name-status", "-z", ref, "--", *paths], cwd=presets.REPO_ROOT)
    except (RuntimeError, FileNotFoundError) as error:
        return None, str(error)
    fields = output.split("\0")[:-1] if output else []
    return [{"path": path, "status": status} for status, path in zip(fields[::2], fields[1::2], strict=True)], None


def _completed_update(path: str, *, generating: bool) -> int | None:
    for metric in reversed(scoring.paths(StoragePath(path) / "exports/metrics/step-*.json")):
        try:
            values = json.loads(presets.read_text(metric))
            step = int(str(metric).rsplit("step-", 1)[-1].removesuffix(".json"))
            if not isinstance(values, dict):
                continue
            key = "generate/wall_seconds" if generating else "timing/step_wall/unaccounted"
            completed = step == 0 if generating else values.get("trainer/global_step") == step
            if completed and math.isfinite(values.get(key, math.nan)):
                return step
        except (ValueError, TypeError):
            continue
    return None


def _status(run: SnowballRun, preset: presets.Preset) -> dict:
    rows = []
    planned = []
    for step in run.runs:
        execution = next(value for value in step.runtime_args.values() if isinstance(value, IrisSkyRLExecution))
        path = step.path(presets.artifact_prefix(execution.cluster))
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        planned.append(yaml.safe_load(config.launch_config_yaml))
        try:
            receipt = read_skyrl_launch_receipt(path)
        except SkyRLReceiptLookupError as error:
            raise click.BadParameter(str(error)) from error
        result = receipt.result if receipt is not None else None
        tracking_path = StoragePath(path) / "exports" / TRACKING_FILENAME
        tracking = (
            {"wandb_url": json.loads(presets.read_text(tracking_path))["wandb_url"]}
            if presets.exists(tracking_path)
            else {}
        )
        team = receipt.config.get("iris", {}).get("wandb_entity") if receipt is not None else execution.wandb_entity
        state = (
            result.state
            if result
            else (
                presets.read_text(get_status_path(path)).strip()
                if presets.exists(get_status_path(path))
                else "not built"
            )
        )
        rows.append(
            {
                "name": step.name,
                "version": step.version,
                "state": state,
                "iris_job_id": result.iris_job_id if result else None,
                "wandb_entity": team or "dogml",
                **tracking,
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
            }
        )
    verdict = {}
    if run.gate is not None:
        path = (
            StoragePath(run.gate.path(prefix=presets.artifact_prefix(preset.recipe.layout.cluster)))
            / snowball_rl_gate.GATE_REPORT_FILENAME
        )
        if presets.exists(path):
            report = snowball_rl_gate.GateReport.model_validate_json(presets.read_text(path))
            verdict = {"report": str(path), **report.model_dump(mode="json", include={"verdict", "label", "reason"})}
            changes, reason = _changed_paths(
                report.control.main_commit,
                ("experiments/post_training", "lib/marin/src/marin/rl", "lib/taskcompendium", "vendor/skyrl"),
            )
            verdict["rl_path_changes_since_control"] = changes
            if reason:
                verdict["rl_path_changes_reason"] = reason
        else:
            crashed = next((row for row in rows if str(row["state"]).lower() in ("failed", "killed")), None)
            if crashed is not None:
                verdict = {"verdict": "fail", "reason": "run crashed", "iris_job_id": crashed["iris_job_id"]}
    return {
        "preset": preset.name,
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
        "reported_metrics": [{**asdict(metric), "window": metric.window.value} for metric in preset.reported],
        "gaps": [gap.value for gap in preset.gaps],
        "runs": rows,
        **verdict,
    }


def _print_status(run: SnowballRun, preset: presets.Preset) -> None:
    click.echo(yaml.safe_dump(_status(run, preset), sort_keys=False))


def _wait_for_run(job: JobName, handles: list[ArtifactStep], preset: presets.Preset) -> None:
    with open_iris_client(config_file=Path(IRIS_HUB_CLUSTER_CONFIG), workspace=presets.REPO_ROOT) as client:
        while (parent := client.job_status(job)).state not in TERMINAL_JOB_STATES:
            time.sleep(60)
    if parent.state != JobState.SUCCEEDED:
        click.echo(
            yaml.safe_dump({"preset": preset.name, "verdict": "fail", "reason": f"coordinator {job} is {parent.state}"})
        )
        raise click.exceptions.Exit(1)
    run = SnowballRun(
        tuple(step for step in handles if step.artifact_type is SkyRLRun),
        next((step for step in handles if step.artifact_type is snowball_rl_gate.GateReport), None),
    )
    snapshot = _status(run, preset)
    failed = any(
        str(row["state"]).lower() in {"failed", "killed", "worker_failed", "unschedulable"} for row in snapshot["runs"]
    )
    complete = bool(run.runs) and all(str(row["state"]).lower() in {"succeeded", "success"} for row in snapshot["runs"])
    markdown = None
    if run.gate is not None:
        gate_root = run.gate.path(prefix=presets.artifact_prefix(preset.recipe.layout.cluster))
        marker = get_status_path(gate_root)
        failed = failed or (presets.exists(marker) and presets.read_text(marker).strip() == "FAILED")
        markdown = str(snapshot.get("report", "")).removesuffix(".json") + ".md"
        complete = (
            complete
            and presets.exists(marker)
            and presets.read_text(marker).strip() == "SUCCESS"
            and presets.exists(markdown)
        )
    if failed:
        snapshot.update(verdict="fail", reason="run or gate step failed")
    elif not complete:
        snapshot.update(verdict="invalid", reason="coordinator finished without complete run and gate evidence")
    click.echo(yaml.safe_dump(snapshot, sort_keys=False))
    if complete and markdown is not None:
        click.echo(presets.read_text(markdown))
    verdict = snapshot.get("verdict", "pass" if run.gate is None else "invalid")
    raise click.exceptions.Exit({"pass": 0, "fail": 1, "invalid": 2}[verdict])


def _run_command(function: Callable) -> click.Command:
    command = presets.run_command(help_text=__doc__)(function)
    command.params.append(
        click.Option(
            ["--baseline", "baselines"],
            multiple=True,
            metavar="NAME@VERSION",
            help="Compare with earlier finished runs; the verdict still uses pinned controls. Repeatable.",
        )
    )
    command.params.append(
        click.Option(["--wait"], is_flag=True, help="Wait after --run; exit 0 for PASS, 1 for FAIL, 2 for INVALID.")
    )

    @wraps(function)
    def callback(*args, wait: bool = False, **kwargs):
        if wait and not kwargs.get("do_run"):
            raise click.BadParameter("--wait requires --run")

        def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
            job = submit_or_run("experiments.post_training.snowball_rl", handles, max_concurrent, os.environ)
            if wait and job is not None:
                _wait_for_run(job, handles, presets.PRESETS[kwargs["preset_name"]])

        return build_options_with_runner(function, runner)(*args, **kwargs)

    command.callback = callback
    return command


@_run_command
def main(
    preset_name: str,
    settings: tuple[str, ...],
    tasks_name: str | None,
    init_name: str | None,
    resume_name: str | None,
    layout_name: str | None,
    evaluations: tuple[str, ...],
    replicates: int,
    control: bool,
    requirements: tuple[str, ...],
    baselines: tuple[str, ...],
) -> tuple[ArtifactStep, ...]:
    if init_name and "@" not in init_name:
        raise click.BadParameter("use NAME@VERSION for initializer models", param_hint="--init")
    preset = presets.PRESETS[preset_name]
    layout = presets.LAYOUTS[layout_name] if layout_name else None
    run = build_run(
        preset,
        settings=tuple(presets.parse_setting(setting) for setting in settings),
        tasks=presets.selection_artifact(tasks_name) if tasks_name else None,
        init=presets.published_artifact(init_name) if init_name else None,
        resume_from=presets.published_artifact(resume_name) if resume_name else None,
        layout=layout,
        evaluations=evaluations,
        replicates=replicates,
        control=control,
        requirements=requirements,
        baselines=baselines,
    )
    if resume_name and (presets.REPO_ROOT / ".git").exists():
        source_tree = presets.run_tree_hash(resume_name)
        changes, reason = _changed_paths(source_tree, ("vendor/skyrl", "lib/marin/src/marin/rl"))
        if reason is not None:
            click.echo(
                f"Cannot compare this checkout with resume source tree {source_tree}: {reason}; "
                "verify optimizer and checkpoint compatibility before continuing."
            )
        elif changes:
            click.echo(
                "Resume crosses SkyRL runtime changes; verify optimizer and checkpoint compatibility before continuing."
            )
    _print_status(run, preset)
    return (*run.runs, *((run.gate,) if run.gate else ()))


if __name__ == "__main__":
    main()
