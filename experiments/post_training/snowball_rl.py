# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Snowball RL experiment from a preset; gated presets compare the run with pinned controls.

Calibrated presets report PASS, FAIL or INVALID against controls of the same preset.
Choose the preset by what the change can affect:

\b
  speed-short      engine throughput and full-panel quality; 8 H100, about 15 min.
  speed-medium     synchronous update time and mismatch; 40 H100, about 40 min.
  speed-long       asynchronous update time and tokens/s/GPU; 40 H100, 45-75 min.
  learning-medium  panel score after 25 sync updates; 40 H100, about 2 h.
  data-short       mixed-answer fraction on --tasks; 8 H100, about 15 min.
  data-medium      learning-medium on --tasks; 40 H100, about 2 h.
  learning-long    panel score after 100 async updates; 40 H100, about 4 h.
  data-long        learning-long on --tasks; 40 H100, about 4 h.
  train-long       async training, checkpoints and HF export; 40 H100 east; no verdict.

Preset workloads and choice table: docs/recipes/snowball-rl.md#which-experiment-to-run.

An engine change runs speed-short, then speed-medium. A trainer kernel, parallelism or numerics
change runs speed-medium, then learning-medium. A new task selection runs data-short, then
data-medium. --require adds a condition of your own, such as a mismatch bound on a speed run or
a per-category score on a learning run. Without --run the command prints the plan and the status
block, including the verdict of a finished gated run; --run submits it. --wait polls every 60 s
and prints the final status and report. It exits 0 for PASS or a finished ungated run,
1 for FAIL, 2 for INVALID, 3 when a required verdict is unavailable, and 4 for a
refusal before execution (including bad flags, refused --set and missing controls).
--baseline NAME@VERSION adds a comparison with earlier finished runs; the verdict uses the
pinned controls. Status includes each run's latest completed update and saved W&B URL.
Panel evaluation is greedy, with natural replies up to EVAL_REPLY_TOKENS (8,192).
--keep-weights keeps a learning or data candidate's final HF export for --init or
later evaluation; their controls keep weights automatically.
Selections must contain no AI-judged tasks.
Set WANDB_ENTITY before submission to choose the W&B team; unset uses marin-community.
--version dev satisfies
the launcher; Snowball steps use PRESET_REVISION. Guide: docs/recipes/snowball-rl.md.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl`:
  --preset speed-medium --issue 123 --version dev --run
  --preset speed-medium --version dev --run --wait
  --preset speed-medium --baseline EARLIER_RUN@2026.10.27 --version dev --run --wait
  --preset learning-medium --set trainer.algorithm.advantage_estimator=grpo \\
    --set trainer.algorithm.group_advantage_min_size=null --version dev --run
  --preset speed-medium --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
  --preset learning-medium --require 'test_panel/aaii_category/coding/mean>=control' --version dev --run
  --preset learning-medium --keep-weights --version dev --run
  --preset data-short --tasks my-selection@2026.11.02 --version dev --run
  --preset train-long --set trainer.max_steps=100 --version dev --run
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import wraps

import click
import yaml
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.resources.state import JobState
from iris.rpc import job_pb2
from marin.execution.artifact import is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.lazy import run as build_artifacts
from marin.execution.step_status import get_status_path
from marin.experiment.cli import build_options_with_runner
from marin.rl import cli as rl_cli
from marin.rl.cli import submit_or_run
from marin.rl.skyrl import (
    BEST_CHECKPOINT_CALLBACK,
    IRIS_HUB_CLUSTER_CONFIG,
    SKYRL_TEMPORARY_STORAGE_TTL_DAYS,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLLaunchReceipt,
    SkyRLModelSource,
    SkyRLPolicyModel,
    SkyRLReceiptLookupError,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    artifact_identity,
    read_skyrl_attempt_tracking,
    read_skyrl_launch_receipt,
    read_stored_skyrl_receipt,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance, git_output

from experiments.post_training import snowball_rl_gate, snowball_rl_publication
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.task_curation.inventory import (
    TRAIN_FILENAME,
    GraderKind,
)

# Iris limit for one Snowball run, coordinator and job alike.
RUN_TIMEOUT_HOURS = 72


DEFAULT_WANDB_ENTITY = "marin-community"
_WANDB_ENTITY_ENV = "WANDB_ENTITY"
_FAILED_RUN_STATES = {"failed", "killed", "worker_failed", "unschedulable"}


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
        try:
            result = presets.artifact_record(init.path()).result or {}
        except FileNotFoundError as error:
            raise click.BadParameter("the initializer requires a completed published artifact record") from error
        if not result.get("hf_model_uri"):
            raise click.BadParameter("the initializer run has no HF export")
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


def _callbacks(
    preset: presets.Preset,
    evaluation: presets.Evaluation | None,
    profiles: dict,
    expected_tasks: int,
    *,
    keep_weights: bool = False,
) -> list[dict]:
    if preset.recipe.layout.role_plan.generate_only:
        return [{"type": "evaluation", "additional_evaluations": profiles}] if profiles else []
    callbacks = []
    if preset.checkpoint_every:
        callbacks.append({"type": "checkpoint", "save_steps": preset.checkpoint_every})
    if evaluation is not None:
        for callback in presets.PanelEvaluator().callbacks(evaluation, updates=preset.updates):
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
    if keep_weights:
        callbacks.extend(
            (
                {"type": "checkpoint", "save_steps": preset.updates + 1, "save_on_train_end": True},
                {"type": "hf_model_save", "save_steps": -1, "save_on_train_end": True},
            )
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
    *,
    layout: presets.Layout,
    export_hf: bool,
    retention: int,
    validation_data: tuple[ArtifactDataSource, ...],
    wandb_entity: str | None,
    evaluation_data: tuple[tuple[str, tuple[ArtifactDataSource, ...]], ...] = (),
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
            evaluation_data=evaluation_data,
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
            required_env_vars=(
                ("WANDB_API_KEY", _WANDB_ENTITY_ENV) if os.environ.get(_WANDB_ENTITY_ENV) else ("WANDB_API_KEY",)
            ),
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
    resume_from: ArtifactStep[SkyRLRun] | None = None,
    settings: tuple[presets.Setting, ...] = (),
    evaluations: tuple[str, ...] = (),
    replicates: int = 1,
    control: bool = False,
    keep_weights: bool = False,
    requirements: tuple[str, ...] = (),
    baselines: tuple[str, ...] = (),
) -> SnowballRun:
    """Construct a Snowball experiment: one run step per replicate and, for a gated preset, its gate step."""
    wandb_entity = os.environ.get(_WANDB_ENTITY_ENV) or DEFAULT_WANDB_ENTITY
    learning = (
        preset.focus in (presets.Focus.LEARNING, presets.Focus.DATA) and not preset.recipe.layout.role_plan.generate_only
    )
    if keep_weights and (control or not learning):
        raise click.BadParameter("--keep-weights needs a learning/data candidate")
    export_hf = preset.focus == presets.Focus.TRAIN or (learning and (control or keep_weights))
    if baselines and (control or not presets.gated(preset)):
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
        raise click.BadParameter("--init requires a learning or train-long preset")
    if preset.focus != presets.Focus.TRAIN and (resume_from is not None or evaluations):
        raise click.BadParameter("--resume-from and --eval require train-long")
    if preset.focus == presets.Focus.TRAIN and (replicates != 1 or control):
        raise click.BadParameter("train-long accepts neither --replicates nor --control")
    if control and (
        settings
        or tasks is not None
        or init is not None
        or evaluations
        or preset.name in snowball_rl_gate.SHARED_CONTROLS
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
    selected = {evaluation.name: evaluation for evaluation in preset.evaluations}
    for name in evaluations:
        selected.setdefault(name, presets.EVALUATIONS[name])
    chosen = tuple(selected.values())
    generating = preset.recipe.layout.role_plan.generate_only
    if preset.focus == presets.Focus.TRAIN and len(chosen) > 1:
        raise click.BadParameter("train-long evaluates one TaskSpec panel; --eval cannot add a second")
    _refuse_judged(summary, "train")
    config = training_config(preset, settings, chosen)
    preset = replace(preset, updates=config["trainer"]["max_steps"])
    model = _model_input(init)
    checkpoint_root = _resume_checkpoint_root(resume_from, model) if resume_from is not None else None
    validation_data = ()
    evaluation_data = []
    profiles = {}
    evaluation_inputs = []
    expected_tasks = 0
    for index, evaluation in enumerate(chosen):
        evaluation_tasks = evaluation.tasks(selection, default)
        evaluation_inputs.append((evaluation.name, evaluation_tasks.name, evaluation_tasks.version))
        _refuse_judged(presets.selection_summary(evaluation_tasks), evaluation.file)
        evaluator = presets.PanelEvaluator()
        profile_config = config if index == 0 else deepcopy(config)
        sources = evaluator.configure(evaluation, profile_config, evaluation_tasks)
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
    config["trainer"]["callbacks"] = _callbacks(
        preset, chosen[0] if chosen else None, profiles, expected_tasks, keep_weights=learning and export_hf
    )
    changes = {
        "tasks": (selection.name, selection.version),
        "init": None if init is None else (init.name, init.version),
        "layout": preset.recipe.layout.name,
        "resume": None if resume_from is None else (resume_from.name, resume_from.version),
        "settings": sorted({setting.key: setting.value for setting in settings}.items()),
        "evaluations": sorted(evaluation_inputs),
        "keep_weights": keep_weights,
    }
    snowball_rl_gate.validate_candidate(preset, requirements, control=control)
    provenance = _provenance(control=control)
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True, default=str))
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
            checkpoint_root=checkpoint_root,
        )
        runs.append(run)
    gate = (
        snowball_rl_gate.gate_step(
            preset,
            tuple(runs),
            inputs=snowball_rl_gate.gate_inputs(
                settings,
                f"{tasks.name}@{tasks.version}" if tasks is not None else None,
                f"{init.name}@{init.version}" if init is not None else None,
                requirements,
                baselines,
            ),
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
        return None, str(error)
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
        preset,
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
    for metric in reversed(scoring.paths(StoragePath(path) / "exports/metrics/step-*.json")):
        values = json.loads(presets.read_text(metric))
        if key in values and (generating or values.get("trainer/global_step") == values["step"]):
            return values["step"]
    return None


def _comparison_line(row: dict, *, constraint: bool = False) -> str:
    if row.get("run") is None or row.get("control") is None:
        return f"{row['metric']}: unavailable"
    values = f"{row['metric']}: {row['control']:.6g} -> {row['run']:.6g}"
    if constraint:
        return values + ("; within bound" if row["passes"] else "; outside bound")
    delta = f"{row['delta_percent']:+.1f}%" if row["delta_percent"] is not None else "unavailable"
    noise = f"{row['noise_percent']:.1f}%" if row["noise_percent"] is not None else "unavailable"
    return f"{values}; {delta}; noise +/-{noise}; {row['change']}"


def _status(
    run: SnowballRun,
    preset: presets.Preset,
    *,
    issue: int | None = None,
    stored_only: bool = False,
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
                presets.read_text(get_status_path(path)).strip()
                if presets.exists(get_status_path(path))
                else "not built"
            )
        )
        progress_url = (publication_urls or {}).get(path)
        rows.append(
            {
                "name": step.name,
                "version": step.version,
                "state": state,
                "iris_job_id": result.iris_job_id if result else None,
                "wandb_entity": team or DEFAULT_WANDB_ENTITY,
                "issue": issue,
                **{
                    key: tracking.get(key)
                    for key in ("attempt_id", "wandb_run_id", "wandb_url", "started_at", "engine")
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
                **({"rl_progress_url": progress_url} if progress_url else {}),
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
            verdict = {
                "report": str(path),
                **report.model_dump(
                    mode="json", include={"verdict", "label", "reason", "objectives", "constraints", "invariants"}
                ),
            }
            changes, reason = _changed_paths(
                report.control.main_commit,
                ("experiments/post_training", "lib/marin/src/marin/rl", "lib/taskcompendium", "vendor/skyrl"),
            )
            verdict["rl_path_changes_since_control"] = changes
            if reason:
                verdict["rl_path_changes_reason"] = reason
    if "verdict" not in verdict:
        crashed = next((row for row in rows if str(row["state"]).lower() in _FAILED_RUN_STATES), None)
        if crashed is not None:
            verdict = {
                "verdict": "invalid",
                "reason": snowball_rl_gate.native_failure_reason(crashed["state"], crashed["iris_job_id"]),
                "iris_job_id": crashed["iris_job_id"],
            }
    objectives = verdict.pop("objectives", None) or [
        {"metric": metric.key} for metric in snowball_rl_gate.objective_metrics(preset)
    ]
    constraints = verdict.pop("constraints", None) or [
        {"metric": metric.key} for metric in preset.guards if metric not in snowball_rl_gate.objective_metrics(preset)
    ]
    invariant_lines = [
        f"r{row['replicate']} {row['name']}: {row['value']} (required {row['limit']}); "
        + ("within bound" if row["holds"] else "outside bound")
        for row in verdict.pop("invariants", [])
    ]
    return {
        "preset": preset.name,
        "objectives": [_comparison_line(row) for row in objectives],
        "constraints": [*(_comparison_line(row, constraint=True) for row in constraints), *invariant_lines],
        "result": snowball_rl_gate.result_text(verdict.get("verdict"), verdict.get("label")),
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


def _from_handles(handles: list[ArtifactStep]) -> SnowballRun:
    return SnowballRun(
        tuple(step for step in handles if step.artifact_type is SkyRLRun),
        next((step for step in handles if step.artifact_type is snowball_rl_gate.GateReport), None),
    )


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
    publication = _publish_records(run, preset, issue=issue)
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
        click.echo(presets.read_text(snapshot["report"].removesuffix(".json") + ".md"))
    verdict = snapshot.get("verdict")
    if verdict is None and run.gate is None and coordinator_error is None:
        verdict = "pass"
    raise click.exceptions.Exit(presets.EXIT_CODES[verdict])


def _wait_for_run(
    job: JobName, handles: list[ArtifactStep], preset: presets.Preset, *, issue: int | None = None
) -> None:
    with rl_cli.iris_client() as client:
        parent = client.job(job).wait(RUN_TIMEOUT_HOURS * 3600, poll_interval=60, raise_on_failure=False)
    error = (
        f"coordinator {job} is {parent.state}; required verdict unavailable"
        if parent.state != JobState.SUCCEEDED
        else None
    )
    _finish_run(handles, preset, error, issue=issue)


def _run_command(function: Callable) -> click.Command:
    command = click.option(
        "--wait",
        is_flag=True,
        help="Wait after --run and publish retained trained-panel attempts using the caller Marina login; "
        "exit 0 for PASS or a finished ungated run, "
        "1 for FAIL, 2 for INVALID, 3 without a required verdict.",
    )(presets.run_command(help_text=__doc__)(function))

    @wraps(function)
    def callback(*args, wait: bool = False, **kwargs):
        if wait and not kwargs.get("do_run"):
            raise click.BadParameter("--wait requires --run")

        def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
            try:
                if get_job_info() is None:
                    preset = presets.PRESETS[kwargs["preset_name"]]
                    job = submit_or_run(
                        "experiments.post_training.snowball_rl",
                        handles,
                        max_concurrent,
                        os.environ,
                    )
                else:
                    planned = _from_handles(handles)
                    build_artifacts(*planned.runs, max_concurrent=max_concurrent)
                    preset = presets.PRESETS[kwargs["preset_name"]]
                    completed = replace(
                        planned, gate=snowball_rl_gate.completed_gate(preset, planned.runs, planned.gate)
                    )
                    if completed.gate is not None:
                        build_artifacts(completed.gate, max_concurrent=max_concurrent)
                    job = None
                if wait and job is not None:
                    preset = presets.PRESETS[kwargs["preset_name"]]
                    _wait_for_run(job, handles, preset, issue=kwargs.get("issue"))
            except click.exceptions.Exit:
                raise
            except Exception as error:
                if not wait:
                    raise
                click.echo(yaml.safe_dump({"preset": kwargs["preset_name"], "reason": str(error)}))
                raise click.exceptions.Exit(3) from error

        try:
            return build_options_with_runner(function, runner)(*args, **kwargs)
        except scoring.InvalidComparison as error:
            click.echo(json.dumps({"preset": kwargs["preset_name"], "verdict": "invalid", "reason": str(error)}))
            raise click.exceptions.Exit(presets.REFUSAL_EXIT_CODE) from error

    command.callback = callback
    return command


@_run_command
@click.option(
    "--issue", type=click.IntRange(min=1), help="GitHub experiment issue number for launcher-side publication."
)
def main(
    preset_name: str,
    settings: tuple[str, ...],
    tasks_name: str | None,
    init_name: str | None,
    resume_name: str | None,
    evaluations: tuple[str, ...],
    replicates: int,
    control: bool,
    keep_weights: bool,
    requirements: tuple[str, ...],
    baselines: tuple[str, ...],
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
        resume_from=presets.published_artifact(resume_name) if resume_name else None,
        evaluations=evaluations,
        replicates=replicates,
        control=control,
        keep_weights=keep_weights,
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
    do_run = click.get_current_context().params.get("do_run", False)
    snapshot = _status(run, preset, issue=issue, stored_only=do_run)
    click.echo(yaml.safe_dump(snapshot, sort_keys=False, width=1000))
    if not do_run and "verdict" in snapshot:
        raise click.exceptions.Exit(presets.EXIT_CODES[snapshot["verdict"]])
    return (*run.runs, *((run.gate,) if run.gate else ()))


if __name__ == "__main__":
    main()
