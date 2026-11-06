# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Snowball RL experiment from a preset; gated presets compare the run with pinned controls.

Calibrated presets report PASS, FAIL or INVALID against control runs of the same preset, with a
change label. Choose the preset by what the change can affect. Each entry
gives what the preset runs, its cost, what it measures and the changes it is for:

\b
  speed-short      vLLM engine only: one pass of 512 prompts x 4 replies of 4,096 tokens;
                   then the 1,500-task greedy test panel. 8 H100, about 15 min. Measures
                   engine tokens/s per GPU and panel quality. For engine options,
                   kernels and vLLM wheels. Quality keeps the throughput engine settings.
  speed-medium     10 sync updates, every reply 4,096 tokens; 40 H100, about 40 min. Measures
                   median update time and trainer-vs-engine log-prob mismatch. For trainer
                   speed, weight sync and startup.
  speed-long       20 async updates with staleness 2 and TIS, replies up to 8,192 tokens;
                   40 H100, target 45-75 min. Measures update time and learner/engine
                   tokens/s per GPU. For the async loop, grading and admission speed.
  learning-medium  25 sync updates, then the 1,500-task test panel; 40 H100, about 2 h. Measures the
                   panel score. For loss, estimator, optimizer, schedule, reward shaping,
                   batch shape and --init.
  data-short       4 replies to each of 256 tasks of a new --tasks selection; 8 H100, about
                   15 min. Measures the fraction of tasks with both right and wrong replies,
                   to screen a new task selection.
  data-medium      learning-medium trained on --tasks instead of the default selection;
                   40 H100, about 2 h. Measures the panel score after training on the new
                   selection. For a new task selection that passed data-short.
  train-long       the async production loop with checkpoints, test_panel_100 every 10 updates and
                   an HF export; 112 H100, or 40 with --layout p40. For release training and
                   for models that later runs load with --init. No verdict.

An engine change runs speed-short, then speed-medium. A trainer kernel, parallelism or numerics
change runs speed-medium, then learning-medium. A new task selection runs data-short, then
data-medium. --require adds a condition of your own, such as a mismatch bound on a speed run or
a per-category score on a learning run. Without --run the command prints the plan and the status
block, including the verdict of a finished gated run; --run submits it. --wait polls every 60 s
and prints the final status and report. It exits 0 for PASS or a finished ungated run,
1 for FAIL, 2 for INVALID, and 3 when a required verdict is unavailable.
--baseline NAME@VERSION adds a comparison with earlier finished runs; the verdict uses the
pinned controls. Status includes each run's latest completed update and saved W&B URL.
Panel evaluation is seeded and greedy, with natural replies up to EVAL_REPLY_TOKENS (8,192).
--keep-weights retains a learning-medium/data-medium candidate's final HF export for --init or
later evaluation; their controls retain weights automatically. --control --rescore-panel NAME@VERSION
scores every pinned control on that panel using its retained weights and recorded engine settings.
Selections must contain no AI-judged tasks.
Set WANDB_ENTITY before submission to choose the W&B team; unset uses marin-community.
--version dev satisfies
the launcher; Snowball steps use PRESET_REVISION. Guide: docs/recipes/snowball-rl.md.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl`:
  --preset speed-medium --version dev --run
  --preset speed-medium --version dev --run --wait
  --preset speed-medium --baseline EARLIER_RUN@2026.10.27 --version dev --run --wait
  --preset learning-medium --set trainer.algorithm.advantage_estimator=grpo \\
    --set trainer.algorithm.group_advantage_min_size=null --version dev --run
  --preset speed-medium --require 'policy/mismatch/staleness0/log_ratio_abs_max<=0' --version dev --run
  --preset learning-medium --require 'test_panel/aaii_category/coding/mean>=control' --version dev --run
  --preset learning-medium --keep-weights --version dev --run
  --preset data-short --tasks my-selection@2026.11.02 --version dev --run
  --preset train-long --layout p40 --set trainer.max_steps=100 --version dev --run
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from collections.abc import Callable
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from functools import wraps

import click
import yaml
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_JOB_STATES, JobState
from iris.rpc import job_pb2
from marin.execution.artifact import is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext, apply
from marin.execution.lazy import run as build_artifacts
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
    SkyRLLaunchReceipt,
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

from experiments.post_training import snowball_rl_gate, snowball_rl_publication
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring
from experiments.post_training.inventory import loop_result
from experiments.post_training.inventory.train_quick_loop import ATTENTION_HELPER_SHA256
from experiments.post_training.task_curation.inventory import (
    TRAIN_FILENAME,
    GraderKind,
)

logger = logging.getLogger(__name__)

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
    deterministic = recipe.engine_profile == presets.EngineProfile.DETERMINISTIC
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
            "algorithm": {**deepcopy(dict(recipe.estimator)), "batch_invariant": deterministic},
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
            "vllm_attention_backend": "TRITON_ATTN" if deterministic else "FLASH_ATTN",
            "required_vllm_attention_helper_sha256": ATTENTION_HELPER_SHA256 if deterministic else None,
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "weight_sync_pause": {"mode": "keep", "clear_cache": True},
            "weight_sync_transport": "expert_block",
            "expert_block_sync": {"timeout_seconds": 600, "verify": False},
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 1024,
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": not deterministic,
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
    evaluations: tuple[presets.Evaluation, ...],
    expected_tasks: int,
    *,
    keep_weights: bool = False,
) -> list[dict]:
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
    required: set[str],
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
            required_env_vars=tuple(sorted(required | {"WANDB_API_KEY"})),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=export_hf,
    )


def _recorded_step(
    step: ArtifactStep[SkyRLRun], preset: presets.Preset, layout: presets.Layout
) -> ArtifactStep[SkyRLRun]:
    metadata = loop_result.SnowballAttempt(step.fingerprint(), layout.name)

    def launch(config):
        updates = (
            0
            if layout.role_plan.generate_only
            else yaml.safe_load(config.launch_config_yaml)["skyrl"]["trainer"]["max_steps"]
        )
        return loop_result.run_loop(loop_result.LoopRunConfig(config, preset.name, updates, snowball=metadata))

    return replace(step, run=launch)


def _panel_rescore_runs(preset: presets.Preset, panel: str) -> SnowballRun:
    evaluation_preset = presets.panel_rescore_preset(preset, panel)
    selection = presets.selection_artifact(panel)
    evaluation = evaluation_preset.evaluations[0]
    if presets.selection_summary(selection)["grader_kind_counts"][evaluation.file].get(GraderKind.REMOTE_JUDGE, 0):
        raise click.BadParameter("Snowball evaluation selections must contain no AI-judge-graded tasks")
    pins = snowball_rl_gate.control_pins(preset.name)
    steps = []
    wandb_entity = os.environ.get(_WANDB_ENTITY_ENV) or DEFAULT_WANDB_ENTITY
    for pin in pins:
        original_preset = presets.PRESETS[pin.preset]
        for index in range(pin.runs):
            name = presets.control_name(pin.preset, pin.main_commit, index)
            path = snowball_rl_gate._control_path(preset, pin, index)
            record = presets.artifact_record(path)
            receipt = read_skyrl_launch_receipt(path)
            if receipt is None or receipt.result.state != "succeeded":
                raise click.BadParameter(f"control {name}@{pin.revision} did not complete successfully")
            original = receipt.config
            protocol = snowball_rl_gate.evaluation_protocol(original_preset, original)
            authored = yaml.safe_load(record.config["launch_config_yaml"])
            config = deepcopy(authored["skyrl"])
            binding = snowball_rl_gate.control_model_binding(path, original_preset, original)
            if original_preset.recipe.layout.role_plan.generate_only:
                model_name, _, model_version = binding["identity"].partition(":")[0].rpartition("@")
                model = ArtifactHfModel(
                    ArtifactStep.adopt(model_name, model_version, binding["uri"], kind=LevanterCheckpoint),
                    binding["tokenizer_uri"],
                    binding["tokenizer_revision"],
                    relative_path="",
                )
            else:
                if not binding["uri"] or not presets.exists(StoragePath(binding["uri"]) / "config.json"):
                    raise click.BadParameter(f"control {name}@{pin.revision} has no kept HF weights")
                model = SkyRLPolicyModel(
                    apply(name, SkyRLRun.raw_load, version=pin.revision, artifact_type=SkyRLRun, pin=path, source=path)
                )
            config["trainer"].update(
                max_steps=1,
                eval_interval=1,
                dump_eval_results=True,
                callbacks=[],
                ckpt_interval=0,
                hf_save_interval=-1,
                resume_mode="none",
            )
            config["generator"].update(
                eval_sampling_params=protocol["generator"]["eval_sampling_params"],
                eval_n_samples_per_prompt=protocol["generator"]["eval_n_samples_per_prompt"],
            )
            step = _run_step(
                presets.panel_rescore_name(name, pin.revision, panel),
                config,
                model,
                selection,
                set(),
                layout=presets.E8,
                export_hf=False,
                retention=2,
                validation_data=(ArtifactDataSource(selection, relative_path=f"{evaluation.file}.parquet"),),
                wandb_entity=wandb_entity,
            )
            steps.append(_recorded_step(step, preset, presets.E8))
    return SnowballRun(tuple(steps))


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
    keep_weights: bool = False,
    requirements: tuple[str, ...] = (),
    baselines: tuple[str, ...] = (),
) -> SnowballRun:
    """Construct a Snowball experiment: one run step per replicate and, for a gated preset, its gate step."""
    wandb_entity = os.environ.get(_WANDB_ENTITY_ENV) or DEFAULT_WANDB_ENTITY
    snowball_rl_gate.validate_candidate(preset, requirements, control=control)
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
    generating = preset.recipe.layout.role_plan.generate_only
    if not generating and len(chosen) > 1:
        raise click.BadParameter("a run evaluates at most one TaskSpec panel")
    required = {key for evaluation in selected.values() for key in presets.EVALUATORS[evaluation.harness].required_env}
    counts = summary["grader_kind_counts"]["train"]
    if counts.get(GraderKind.REMOTE_JUDGE, 0):
        raise click.BadParameter("Snowball training selections must contain no AI-judge-graded tasks")
    missing = sorted(key for key in required if not os.environ.get(key))
    if missing:
        raise click.BadParameter(f"missing required credentials: {', '.join(missing)}")
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
        grader_counts = presets.selection_summary(evaluation_tasks)["grader_kind_counts"][evaluation.file]
        if grader_counts.get(GraderKind.REMOTE_JUDGE, 0):
            raise click.BadParameter("Snowball evaluation selections must contain no AI-judge-graded tasks")
        evaluator = presets.EVALUATORS[evaluation.harness]
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
    if generating and profiles:
        config["trainer"]["callbacks"] = [{"type": "evaluation", "additional_evaluations": profiles}]
    if not generating:
        config["trainer"]["callbacks"] = _callbacks(preset, chosen, expected_tasks, keep_weights=learning and export_hf)
    changes = {
        "tasks": (selection.name, selection.version),
        "init": None if init is None else (init.name, init.version),
        "layout": preset.recipe.layout.name,
        "resume": None if resume_from is None else (resume_from.name, resume_from.version),
        "settings": sorted({setting.key: setting.value for setting in settings}.items()),
        "evaluations": sorted(evaluation_inputs),
        "keep_weights": keep_weights,
    }
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True, default=str))
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
            export_hf=export_hf,
            retention=5 if preset.checkpoint_every else 2,
            validation_data=validation_data,
            evaluation_data=tuple(evaluation_data),
            wandb_entity=wandb_entity,
            checkpoint_root=checkpoint_root,
        )
        runs.append(_recorded_step(run, preset, preset.recipe.layout))
    gate = (
        snowball_rl_gate.gate_step(
            preset, tuple(runs), inputs=snowball_rl_gate.gate_inputs(settings, tasks, init, requirements, baselines)
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


def _grafana_links(
    client: IrisClient | None, preset: presets.Preset, receipt: SkyRLLaunchReceipt | None, cluster: str
) -> dict[str, str]:
    if client is None or receipt is None or not receipt.result.iris_job_id:
        return {}
    run_id = receipt.config.get("run", {}).get("id")
    if not run_id:
        return {}
    try:
        job = client.job_status(JobName.from_wire(receipt.result.iris_job_id))
    except Exception as error:
        logger.warning("Could not read Iris timing for %s: %s", receipt.result.iris_job_id, error)
        return {}
    if job.started_at is None or (job.state in TERMINAL_JOB_STATES and job.finished_at is None):
        return {}
    training_type = receipt.config.get("runtime", {}).get("training_type")
    return presets.grafana_links(
        preset,
        run_id=run_id,
        cluster=cluster,
        training_type=(
            "async" if training_type == "async" or (training_type is None and preset.recipe.staleness) else "sync"
        ),
        started_ms=job.started_at.epoch_ms(),
        finished_ms=job.finished_at.epoch_ms() if job.finished_at is not None else None,
    )


def _completed_update(path: str, *, generating: bool) -> int | None:
    key = presets.GENERATE_WALL_SECONDS_METRIC if generating else "timing/step_wall/unaccounted"
    for metric in reversed(scoring.paths(StoragePath(path) / "exports/metrics/step-*.json")):
        values = json.loads(presets.read_text(metric))
        if key in values and (generating or values.get("trainer/global_step") == values["step"]):
            return values["step"]
    return None


def _completed_gate(run: SnowballRun, preset: presets.Preset) -> ArtifactStep | None:
    if run.gate is None:
        return None
    config = run.gate.build_config(StepContext.for_fingerprint(run.gate.runtime_args, run.gate.deps))
    return snowball_rl_gate.gate_step(preset, run.runs, inputs=config["inputs"])


def _publications(run: SnowballRun, preset: presets.Preset) -> tuple[ArtifactStep, ...]:
    if not presets.published(preset):
        return ()
    publications = []
    for training in run.runs:
        execution = next(value for value in training.runtime_args.values() if isinstance(value, IrisSkyRLExecution))
        prefix = presets.artifact_prefix(execution.cluster)
        publications.append(
            snowball_rl_publication.publication_step(
                replace(training, override_path=training.path(prefix)),
                gate=replace(run.gate, override_path=run.gate.path(prefix)) if run.gate is not None else None,
            )
        )
    return tuple(publications)


def _status(run: SnowballRun, preset: presets.Preset) -> dict:
    run = replace(run, gate=_completed_gate(run, preset))
    rows = []
    native_attempts = []
    planned = []
    with ExitStack() as stack:
        client, lookup_started = None, False
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
            if (
                receipt is not None
                and receipt.result.iris_job_id
                and receipt.config.get("run", {}).get("id")
                and not lookup_started
            ):
                lookup_started = True
                try:
                    client = stack.enter_context(presets.iris_client())
                except Exception as error:
                    logger.warning("Could not connect to Iris for status links: %s", error)
                    client = None
            grafana = _grafana_links(client, preset, receipt, execution.cluster)
            state = (
                result.state
                if result
                else (
                    presets.read_text(get_status_path(path)).strip()
                    if presets.exists(get_status_path(path))
                    else "not built"
                )
            )
            native_attempts.append(receipt.config.get("run", {}).get("attempt_id") if receipt is not None else None)
            rows.append(
                {
                    "name": step.name,
                    "version": step.version,
                    "state": state,
                    "iris_job_id": result.iris_job_id if result else None,
                    "wandb_entity": team or DEFAULT_WANDB_ENTITY,
                    **tracking,
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
            crashed = next((row for row in rows if str(row["state"]).lower() in _FAILED_RUN_STATES), None)
            if crashed is not None:
                verdict = {"verdict": "fail", "reason": "run crashed", "iris_job_id": crashed["iris_job_id"]}
    publications = _publications(run, preset)
    for row, publication, attempt in zip(rows, publications, native_attempts, strict=False):
        receipt_path = StoragePath(publication.path(presets.artifact_prefix(row["cluster"]))) / "publication.json"
        if presets.exists(receipt_path):
            published = json.loads(presets.read_text(receipt_path))
            identity = published["attempt_id"]
            if attempt is not None and (identity == attempt or identity.startswith(attempt + "-gate-")):
                row["rl_progress_url"] = published["url"]
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


def _from_handles(handles: list[ArtifactStep]) -> SnowballRun:
    return SnowballRun(
        tuple(step for step in handles if step.artifact_type is SkyRLRun),
        next((step for step in handles if step.artifact_type is snowball_rl_gate.GateReport), None),
    )


def _wait_for_run(job: JobName, handles: list[ArtifactStep], preset: presets.Preset) -> None:
    with presets.iris_client() as client:
        parent = client.job(job).wait(RUN_TIMEOUT_HOURS * 3600, poll_interval=60, raise_on_failure=False)
    if parent.state != JobState.SUCCEEDED:
        click.echo(yaml.safe_dump({"preset": preset.name, "reason": f"coordinator {job} is {parent.state}"}))
        raise click.exceptions.Exit(3)
    run = _from_handles(handles)
    snapshot = _status(run, preset)
    if run.gate is not None and "report" not in snapshot:
        snapshot.pop("verdict", None)
        snapshot["reason"] = "coordinator finished without a gate report"
    click.echo(yaml.safe_dump(snapshot, sort_keys=False))
    if "report" in snapshot:
        click.echo(presets.read_text(snapshot["report"].removesuffix(".json") + ".md"))
    if run.gate is not None and "report" not in snapshot:
        raise click.exceptions.Exit(3)
    verdict = snapshot.get("verdict", "pass")
    raise click.exceptions.Exit({"pass": 0, "fail": 1, "invalid": 2}[verdict])


def _run_command(function: Callable) -> click.Command:
    command = click.option(
        "--wait",
        is_flag=True,
        help="Wait after --run; exit 0 for PASS or a finished ungated run, "
        "1 for FAIL, 2 for INVALID, 3 without a required verdict.",
    )(presets.run_command(help_text=__doc__)(function))

    @wraps(function)
    def callback(*args, wait: bool = False, **kwargs):
        if wait and not kwargs.get("do_run"):
            raise click.BadParameter("--wait requires --run")

        def runner(handles: list[ArtifactStep], max_concurrent: int) -> None:
            try:
                if get_job_info() is None:
                    job = submit_or_run(
                        "experiments.post_training.snowball_rl",
                        handles,
                        max_concurrent,
                        os.environ,
                        extra_env_vars=(_WANDB_ENTITY_ENV,),
                    )
                else:
                    planned = _from_handles(handles)
                    build_artifacts(*planned.runs, max_concurrent=max_concurrent)
                    preset = presets.PRESETS[kwargs["preset_name"]]
                    completed = replace(planned, gate=_completed_gate(planned, preset))
                    if completed.gate is not None:
                        build_artifacts(completed.gate, max_concurrent=max_concurrent)
                    publications = _publications(completed, preset)
                    if publications:
                        try:
                            build_artifacts(*publications, max_concurrent=max_concurrent)
                        except Exception as error:
                            logger.warning(
                                "Native run completed; publication is unavailable and can be retried: %s", error
                            )
                    job = None
                if wait and job is not None:
                    preset = presets.PRESETS[kwargs["preset_name"]]
                    if kwargs.get("rescore_panel"):
                        preset = presets.panel_rescore_preset(preset, kwargs["rescore_panel"])
                    _wait_for_run(job, handles, preset)
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
            raise click.exceptions.Exit(2) from error

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
    keep_weights: bool,
    requirements: tuple[str, ...],
    baselines: tuple[str, ...],
    rescore_panel: str | None = None,
) -> tuple[ArtifactStep, ...]:
    if init_name and "@" not in init_name:
        raise click.BadParameter("use NAME@VERSION for initializer models", param_hint="--init")
    preset = presets.PRESETS[preset_name]
    layout = presets.LAYOUTS[layout_name] if layout_name else None
    if rescore_panel is not None:
        if not control or any(
            (
                settings,
                tasks_name,
                init_name,
                layout_name,
                resume_name,
                evaluations,
                requirements,
                baselines,
                keep_weights,
            )
        ):
            raise click.BadParameter("--rescore-panel requires --control with no experiment overrides")
        if replicates != 1:
            raise click.BadParameter("--rescore-panel scores every pinned control; omit --replicates")
        _provenance(control=True)
        run = _panel_rescore_runs(preset, rescore_panel)
        preset = presets.panel_rescore_preset(preset, rescore_panel)
    else:
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
    click.echo(yaml.safe_dump(_status(run, preset), sort_keys=False))
    return (*run.runs, *((run.gate,) if run.gate else ()))


if __name__ == "__main__":
    main()
