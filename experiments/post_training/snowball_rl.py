# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a Snowball RL experiment from a preset.

Choose the preset by what the change can affect:

\b
  speed-short      vLLM engine only; 8 H100, about 10 min. Engine options, kernels, wheels.
  speed-medium     10 sync updates, every reply 4,096 tokens; 40 H100, about 40 min. Trainer
                   speed, weight sync, startup and trainer-vs-engine log-prob mismatch.
  learning-medium  25 sync updates, then the 910-task panel; 40 H100, about 2 h. Loss,
                   estimator, optimizer, schedule, reward shaping, batch shape, --init.
  data-short       256 tasks of a new --tasks selection, 4 answers each; 8 H100, about 15 min.
  data-medium      learning-medium trained on --tasks; 40 H100, about 2 h.
  train-long       the async production loop with checkpoints, HF export and a final
                   910-task evaluation; 112 H100, or 40 with --layout p40.

An engine change runs speed-short, then speed-medium. A trainer kernel, parallelism or numerics
change runs speed-medium, then learning-medium. A new task selection runs data-short, then
data-medium. Without --run the command prints the plan and the status block; --run submits
it. --version dev satisfies the launcher; Snowball steps use PRESET_REVISION. Guide:
docs/recipes/snowball-rl.md.

\b
Examples, each after `uv run python -m experiments.post_training.snowball_rl`:
  --preset speed-medium --version dev --run
  --preset learning-medium --set trainer.algorithm.advantage_estimator=grpo --version dev --run
  --preset data-short --tasks my-selection@2026.11.02 --version dev --run
  --preset train-long --layout p40 --set trainer.max_steps=100 --version dev --run
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import click
import yaml
from iris.rpc import job_pb2
from marin.execution.artifact import is_mutable_version
from marin.execution.fingerprint import fingerprint_hash
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_status import get_status_path
from marin.rl.cli import rl_build_options
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
    SkyRLSpec,
    SkyRLTopology,
    artifact_identity,
    read_skyrl_launch_receipt,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.task_curation.inventory import (
    TRAIN_FILENAME,
    GraderKind,
)


def training_config(
    preset: presets.Preset, settings: tuple[presets.Setting, ...] = (), evaluations: tuple[presets.Evaluation, ...] = ()
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
    post_run_evaluations: tuple[ArtifactStep[SkyRLRun], ...] = ()


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
    missing = f"no checkpoint remains at {root}; temporary checkpoints expire after 14 days"
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
            coordinator_timeout_hours=72,
            job_timeout_seconds=72 * 3600,
            wandb_entity="dogml",
            required_env_vars=tuple(sorted(required | {"WANDB_API_KEY"})),
            container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
        ),
        export_hf=export_hf,
    )


def _post_evaluation(
    run: ArtifactStep[SkyRLRun],
    evaluation: presets.Evaluation,
    selection: ArtifactStep,
    parent: presets.Preset,
    parent_config: dict,
) -> ArtifactStep[SkyRLRun]:
    parent_layout = parent.recipe.layout
    engine = presets.E8.role_plan
    plan = replace(
        parent_layout.role_plan,
        generate_only=True,
        num_inference_engines=engine.num_inference_engines,
        inference_engine_tensor_parallel_size=engine.inference_engine_tensor_parallel_size,
        inference_engine_pipeline_parallel_size=engine.inference_engine_pipeline_parallel_size,
        inference_engine_data_parallel_size=engine.inference_engine_data_parallel_size,
        inference_engine_expert_parallel_size=engine.inference_engine_expert_parallel_size,
    )
    layout = replace(parent_layout, name=presets.E8.name, role_plan=plan)
    config = deepcopy(parent_config)
    config["trainer"].update(
        callbacks=[], eval_interval=1, eval_before_train=False, resume_mode="none", ckpt_interval=0, hf_save_interval=-1
    )
    evaluator = presets.EVALUATORS[evaluation.harness]
    validation = evaluator.configure(evaluation, config, selection)
    config["trajectory_runner"]["max_concurrent_tasks"] = presets.evaluation_answer_count(evaluation, selection)
    return _run_step(
        f"{run.name}/eval/{evaluation.name}",
        config,
        SkyRLPolicyModel(run),
        selection,
        set(evaluator.required_env),
        layout=layout,
        export_hf=False,
        retention=2,
        validation_data=validation,
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
) -> SnowballRun:
    """Construct the complete Snowball experiment and its terminal evaluation dependencies."""
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
    post = tuple(evaluation for evaluation in selected.values() if evaluation.post_run)
    in_run = tuple(evaluation for evaluation in selected.values() if evaluation not in post)
    if sum(evaluation.harness == presets.EvalHarness.PANEL for evaluation in in_run) > 1:
        raise click.BadParameter("one in-run TaskSpec panel is supported; panel-910 runs on the final export")
    required = {key for evaluation in selected.values() for key in presets.EVALUATORS[evaluation.harness].required_env}
    counts = summary["grader_kind_counts"]["train"]
    if counts.get(GraderKind.REMOTE_JUDGE, 0) and preset.name != "speed-short":
        required.add(presets.JUDGE_KEY)
    missing = sorted(key for key in required if not os.environ.get(key))
    if missing:
        raise click.BadParameter(f"missing required credentials: {', '.join(missing)}")
    config = training_config(preset, settings, in_run)
    preset = replace(preset, updates=config["trainer"]["max_steps"])
    model = _model_input(init)
    checkpoint_root = _resume_checkpoint_root(resume_from, model) if resume_from is not None else None
    validation_data = ()
    for evaluation in in_run:
        evaluation_tasks = evaluation.tasks(selection, default)
        validation_data = presets.EVALUATORS[evaluation.harness].configure(evaluation, config, evaluation_tasks)
        if preset.recipe.layout.role_plan.generate_only:
            config["trajectory_runner"]["max_concurrent_tasks"] = presets.evaluation_answer_count(
                evaluation, evaluation_tasks
            )
    if not preset.recipe.layout.role_plan.generate_only:
        expected_tasks = presets.selection_summary(default)["validation_rows"]
        config["trainer"]["callbacks"] = _callbacks(preset, in_run, expected_tasks)
    changes = {
        "tasks": (selection.name, selection.version),
        "init": None if init is None else (init.name, init.version),
        "layout": preset.recipe.layout.name,
        "resume": None if resume_from is None else (resume_from.name, resume_from.version),
        "settings": sorted({setting.key: setting.value for setting in settings}.items()),
        "evaluations": sorted(selected),
    }
    inputs_hash = fingerprint_hash(json.dumps(changes, sort_keys=True, default=str))
    runs = []
    post_runs = []
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
            checkpoint_root=checkpoint_root,
        )
        runs.append(run)
        for evaluation in post:
            post_runs.append(_post_evaluation(run, evaluation, default, preset, config))
    return SnowballRun(tuple(runs), tuple(post_runs))


def _print_status(run: SnowballRun, preset: presets.Preset) -> None:
    rows = []
    planned = []
    for step in (*run.runs, *run.post_run_evaluations):
        execution = next(value for value in step.runtime_args.values() if isinstance(value, IrisSkyRLExecution))
        path = step.path(presets.artifact_prefix(execution.cluster))
        config = step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))
        planned.append(yaml.safe_load(config.launch_config_yaml))
        try:
            receipt = read_skyrl_launch_receipt(path)
        except SkyRLReceiptLookupError as error:
            raise click.BadParameter(str(error)) from error
        result = receipt.result if receipt is not None else None
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
    click.echo(
        yaml.safe_dump(
            {
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
                "updates": (
                    0 if preset.recipe.layout.role_plan.generate_only else planned[0]["skyrl"]["trainer"]["max_steps"]
                ),
                "measured_updates": list(preset.measured),
                "budget_estimate": preset.budget,
                "guards": [metric.key for metric in preset.guards],
                "reported_metrics": [{**asdict(metric), "window": metric.window.value} for metric in preset.reported],
                "gaps": [gap.value for gap in preset.gaps],
                "runs": rows,
            },
            sort_keys=False,
        )
    )


@click.command(help=__doc__)
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
    "reason; data presets accept none. Code changes need no flag: the run name includes the source tree hash.",
)
@click.option(
    "--tasks",
    "tasks_name",
    metavar="NAME@VERSION",
    help="Curated task selection to train on, at an immutable version. Required by data-short and "
    "data-medium, whose selection must hold out the same panel as the default; train-long defaults to "
    "the pinned snowball-100k.",
)
@click.option(
    "--init",
    "init_name",
    metavar="NAME@VERSION",
    help="Initial weights: a published model artifact, or a finished train-long run (its HF export). "
    "learning-medium and train-long only. The initial weights are evaluated too, so the report shows the gain.",
)
@click.option(
    "--resume-from",
    "resume_name",
    metavar="NAME[@VERSION]",
    help="Continue an earlier train-long run's checkpoints and optimizer state under a new run name. "
    "train-long only.",
)
@click.option(
    "--layout",
    "layout_name",
    type=click.Choice(sorted(presets.TRAIN_LONG_RECIPES)),
    help="train-long GPU layout: l112 (default; 64 trainer + 48 engine H100 on cw-rno2a, the production "
    "recipe) or p40 (32 + 8 H100 on cw-us-east-02a, learning-medium's batch shape).",
)
@click.option(
    "--eval",
    "evaluations",
    multiple=True,
    type=click.Choice(sorted(presets.EVALUATIONS)),
    help="Add an evaluation to train-long, which already runs panel-100 during training and panel-910 on its "
    "final export. Repeatable.",
)
@click.option(
    "--replicates",
    type=click.IntRange(1, 5),
    default=1,
    show_default=True,
    help="Identical copies of a short or medium preset, sharing one seed. Not for train-long.",
)
@click.option(
    "--control",
    is_flag=True,
    help="Run the preset's control replicates from a clean checkout with only its pinned default inputs. "
    "Only for pinning new controls; refuses --set, --tasks, --init, --eval and data-medium, which uses "
    "learning-medium's controls.",
)
@rl_build_options
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
    )
    if resume_name and (presets.REPO_ROOT / ".git").exists():
        source_tree = presets.run_tree_hash(resume_name)
        changes = subprocess.run(
            ["git", "diff", "--name-only", source_tree, "--", "vendor/skyrl", "lib/marin/src/marin/rl"],
            cwd=presets.REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if changes.returncode != 0:
            click.echo(
                f"Cannot compare this checkout with resume source tree {source_tree}: {changes.stderr.strip()}; "
                "verify optimizer and checkpoint compatibility before continuing."
            )
        elif changes.stdout.strip():
            click.echo(
                "Resume crosses SkyRL runtime changes; verify optimizer and checkpoint compatibility before continuing."
            )
    _print_status(run, preset)
    return (*run.runs, *run.post_run_evaluations)


if __name__ == "__main__":
    main()
