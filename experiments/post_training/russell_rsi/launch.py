# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Russell code-repair RL, checkpoint export, and evaluation artifacts."""

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import cast

import click
import yaml
from fray.types import ResourceConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity, resolve, run
from marin.execution.remote import remote
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLRuntimeProfile,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import EvaluationResult, eval_step
from experiments.post_training.russell_rsi.adaptive_tasks import AdaptiveTasksConfig, run_adaptive_tasks_in_project
from experiments.post_training.russell_rsi.bootstrap_loop import (
    CALIBRATION_TEMPERATURE,
    CheckpointScore,
    FrozenRoundConfig,
    LoopState,
    QualifiedTask,
    RoundResult,
    StopReason,
    advance,
    calibration_measurements,
    freeze_round_dataset,
    load_round,
    qualified_bank,
    round_inputs,
    round_plan,
    seal_round,
    write_once,
)
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CODING_ANALYSIS_CONTEXT_PROTOCOL,
    CodingAnalysisConfig,
    CodingEvidenceConfig,
    CodingPanel,
    analyze_coding_eval_failures,
    collect_coding_eval_evidence,
)
from experiments.post_training.russell_rsi.heldout_evaluation import heldout_comparison_step, heldout_panel_ids
from experiments.post_training.russell_rsi.repair_tasks import (
    QualifiedUnionConfig,
    RepairTasksConfig,
    pinned_bytes,
    run_qualified_union_in_project,
    run_repair_tasks_in_project,
)
from experiments.post_training.russell_rsi.rollout_eval import DevelopmentEvaluationConfig, run_development_evaluation
from experiments.post_training.russell_rsi.settings import (
    CHAT_TEMPLATE_KWARGS,
    CONTEXT_TOKENS,
    GLM_TOKEN_ENV,
    PROMPT_TOKENS,
    RESPONSE_TOKENS,
    ROLLOUT_CONCURRENCY,
    STOP_TOKEN_IDS,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.skyrl_evaluation import SKYRL_POLICY_LOCATION, resolve_skyrl_model

MODEL = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_REVISION = "b8c07f7df1df65525abbfdbcd1572318ba11c42f"
CLUSTER = "cw-us-east-02a"
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=4,
    policy_num_gpus_per_node=8,
    num_inference_engines=1,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=8,
    inference_engine_expert_parallel_size=8,
    train_batch_size=16,
    policy_mini_batch_size=16,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=4,
)


@dataclass(frozen=True)
class Scale:
    updates: int
    ttl_days: int


SCALES = {"smoke": Scale(1, 1), "pilot": Scale(4, 7)}


def recipe(scale: Scale, machine_config: dict | None = None) -> str:
    """Configure shared task rollouts and bounded GRPO training."""
    return yaml.safe_dump(
        {
            "entrypoint": "taskcompendium",
            "context_budget": {
                "request_window_tokens": CONTEXT_TOKENS,
                "max_new_tokens_per_turn": RESPONSE_TOKENS,
                "max_turns": 16,
                "max_prompt_tokens": PROMPT_TOKENS,
            },
            "data": {"kind": "tasks", "train_data": [], "val_data": []},
            "trainer": {
                "strategy": "megatron",
                "flash_attn": False,
                "use_sample_packing": False,
                "offload_optimizer_during_rollouts": True,
                "gradient_checkpointing": True,
                "algorithm": {"advantage_estimator": "grpo", "use_kl_loss": False},
                # SkyRL schedules floor(rows / batch) * epochs; sixteen rows need one epoch per update.
                "epochs": scale.updates,
                "max_steps": scale.updates,
                "update_epochs_per_batch": 1,
                "eval_batch_size": 64,
                "micro_forward_batch_size_per_gpu": 1,
                "eval_before_train": False,
                "eval_interval": scale.updates if scale.updates > 1 else -1,
                "ckpt_interval": scale.updates,
                "resume_mode": "none",
                "logger": "console",
                "project_name": "marin-russell-rsi",
                "hf_hub_repo_id": None,
                "policy": {
                    "optimizer_config": {"lr": 5.0e-7, "max_grad_norm": 1.0},
                    "megatron_config": {
                        "tensor_model_parallel_size": 1,
                        "pipeline_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "expert_model_parallel_size": 8,
                        "expert_tensor_parallel_size": 1,
                        "optimizer_checkpoint_sharding_type": "dp_reshardable",
                        "ddp_config": {
                            "overlap_grad_reduce": True,
                            "overlap_param_gather": True,
                            "grad_reduce_in_fp32": False,
                        },
                    },
                },
            },
            "generator": {
                "backend": "vllm",
                "model_dtype": "bfloat16",
                "vllm_attention_backend": "FLASH_ATTN",
                "gpu_memory_utilization": 0.75,
                "max_num_batched_tokens": PROMPT_TOKENS,
                "run_engines_locally": True,
                "weight_sync_backend": "nccl",
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "engine_init_kwargs": {
                    "moe_backend": "triton",
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                },
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "stop_token_ids": list(STOP_TOKEN_IDS),
                },
                "error_handling": {"default_error_treatment": "mask", "preserve_logprobs_on_timeout": True},
                "trajectory_retention": {
                    "enabled": True,
                    "phases": ["train", "eval"],
                    "sample_count_per_step": 256,
                    "always_retain_failures": True,
                    "required": True,
                    "max_bytes_per_run": 1073741824,
                },
            },
            "trajectory_runner": {
                **({"machine": machine_config} if machine_config is not None else {}),
                "command_timeout": 120,
                "max_concurrent_tasks": ROLLOUT_CONCURRENCY,
                "rollout_workers": {"num_workers": 4, "cpus_per_worker": 8, "executor_threads": 32},
            },
            "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        }
    )


def train_step(
    data: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint],
    scale: str,
    version: str,
    development: ArtifactStep[Artifact] | None = None,
    machine_config: dict | None = None,
    name_component: str | None = None,
) -> ArtifactStep[SkyRLRun]:
    selected = SCALES[scale]
    label = f"{name_component}-{scale}" if name_component else scale
    # The HF export is at the artifact root. A dot is a literal S3 key component.
    return skyrl_step(
        SkyRLSpec(
            name=f"checkpoints/russell-rsi-{label}",
            version=version,
            config_yaml=recipe(selected, machine_config),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(model, MODEL, MODEL_REVISION, relative_path=""),
            train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
            validation_data=(ArtifactDataSource(development or data, relative_path="development.parquet"),),
            topology=SkyRLTopology(num_nodes=5, gpus_per_node=8, gpu_variant="H100", role_plan=ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=selected.ttl_days),
            seed=9528,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="512GB",
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=12,
            wandb_entity=None,
        ),
        export_hf=True,
    )


def evaluation_model(name: str, location: str, revision: str | None) -> ModelConfig:
    return ModelConfig(
        name=name,
        location=location,
        revision=revision,
        tokenizer=MODEL,
        tokenizer_revision=MODEL_REVISION,
        apply_chat_template=True,
        resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g", disk="300g"),
        serve=ServeConfig(
            tensor_parallel_size=1,
            data_parallel_size=8,
            max_model_len=CONTEXT_TOKENS,
            max_num_batched_tokens=8192,
            max_num_seqs=32,
            auto_overrides=False,
            vllm_extra_args=SNOWBALL_VLLM_ARGS,
        ),
        generation=GenerationConfig(max_gen_toks=8192, chat_template_kwargs=CHAT_TEMPLATE_KWARGS),
    )


def development_step(
    data: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint] | ArtifactStep[SkyRLRun],
    version: str,
    runtime_bundle: RuntimeBundle,
    label: str,
    relative_path: str = "development.parquet",
    samples_per_task: int = 1,
    temperature: float = 0.0,
    require_reward_variation: bool = False,
    limit: int = 32,
) -> ArtifactStep[Artifact]:
    def build_config(ctx: StepContext) -> DevelopmentEvaluationConfig:
        if model.artifact_type is SkyRLRun:
            resolved = resolve_skyrl_model(
                ctx,
                cast(ArtifactStep[SkyRLRun], model),
                evaluation_model(f"russell-rsi-{label}", SKYRL_POLICY_LOCATION, None),
            )
            location = resolved.location
            tokenizer = resolved.tokenizer
            tokenizer_revision = resolved.tokenizer_revision
            assert tokenizer is not None and tokenizer_revision is not None
        else:
            location = ctx.artifact_path(model)
            tokenizer, tokenizer_revision = MODEL, MODEL_REVISION

        return DevelopmentEvaluationConfig(
            model_uri=location,
            model_identity=artifact_identity(model),
            tokenizer=tokenizer,
            tokenizer_revision=tokenizer_revision,
            tasks_identity=artifact_identity(data),
            tasks_path=prefix_join(ctx.artifact_path(data), relative_path),
            output_path=ctx.output_path,
            runtime_bundle=runtime_bundle,
            limit=limit,
            samples_per_task=samples_per_task,
            temperature=temperature,
            require_reward_variation=require_reward_variation,
        )

    return ArtifactStep(
        name=f"evals/russell-rsi-{label}-development",
        version=version,
        artifact_type=Artifact,
        deps=(data, model),
        build_config=build_config,
        run=remote(
            run_development_evaluation,
            # Remote workers inherit the required CW02 root coordinator; SkyRL and public eval submit separate roots.
            resources=ResourceConfig.with_gpu("H100", 8, cpu=32, ram="512GB", disk="2TB"),
            pip_packages=[MARIN_SKYRL.requirement()],
            env_vars={"UV_PRERELEASE": "allow"},
        ),
    )


def run_adaptive_tasks(config: AdaptiveTasksConfig) -> None:
    # This remote worker inherits the required CW02 root coordinator.
    remote(
        run_adaptive_tasks_in_project,
        resources=ResourceConfig.with_cpu(cpu=32, ram="128GB", disk="64GB"),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


def run_repair_tasks(config: RepairTasksConfig) -> None:
    remote(
        run_repair_tasks_in_project,
        resources=ResourceConfig.with_cpu(cpu=32, ram="128GB", disk="64GB"),
        env_vars={GLM_TOKEN_ENV: os.environ[GLM_TOKEN_ENV]},
    )(config)


def parent_public_step(model: ArtifactStep[LevanterCheckpoint], version: str) -> ArtifactStep:
    """Score the frozen parent on the same bounded public cohort as its candidate."""
    parent_model = evaluation_model("russell-rsi-parent", MODEL, None)
    return eval_step(
        parent_model,
        "humanevalplus,mbppplus",
        version=version,
        deps=(model,),
        resolve_model=lambda ctx: replace(
            parent_model, location=ctx.artifact_path(model), identity=artifact_identity(model)
        ),
        limit=32,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )


def repair_spike_workflow(
    seed: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint],
    scale: str,
    version: str,
    *,
    relay_job: str,
    image: str,
    runtime_bundle: RuntimeBundle,
    machine_config: dict,
    wheels: ArtifactStep[Artifact],
    manifest_uri: str,
    manifest_sha256: str,
) -> dict[str, ArtifactStep]:
    """Repair sealed round-one candidates and qualify their union before training."""
    evidence = ArtifactStep.adopt(
        "documents/russell-rsi-round-1-evidence",
        version,
        manifest_uri,
        config={"manifest_sha256": manifest_sha256},
    )
    baseline = development_step(seed, model, version, runtime_bundle, "parent")
    repaired = ArtifactStep(
        name="documents/russell-rsi-repair-1",
        version=version,
        artifact_type=Artifact,
        deps=(evidence, baseline, wheels),
        build_config=lambda ctx: RepairTasksConfig(
            manifest_uri=ctx.artifact_path(evidence),
            manifest_sha256=manifest_sha256,
            output_path=ctx.output_path,
            relay_job=relay_job,
            image=image,
            runtime_bundle=runtime_bundle,
            dependency_wheels_uri=ctx.artifact_path(wheels),
            admission_concurrency=8,
            parent_development_identity=artifact_identity(baseline),
        ),
        run=run_repair_tasks,
    )
    qualified = ArtifactStep(
        name="documents/russell-rsi-qualified-union-1",
        version=version,
        artifact_type=Artifact,
        deps=(evidence, repaired),
        build_config=lambda ctx: QualifiedUnionConfig(
            original_manifest_uri=ctx.artifact_path(evidence),
            original_manifest_sha256=manifest_sha256,
            repair_output_uri=ctx.artifact_path(repaired),
            output_path=ctx.output_path,
            minimum_train_rows=ROLE_PLAN.train_batch_size,
            parent_development_identity=artifact_identity(baseline),
        ),
        run=remote(
            run_qualified_union_in_project,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="64GB"),
        ),
    )
    return post_admission_workflow(
        qualified, seed, model, scale, version, runtime_bundle, machine_config, name_component="repair-1"
    )


def spike_workflow(
    seed: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint],
    scale: str,
    version: str,
    relay_job: str,
    image: str,
    runtime_bundle: RuntimeBundle,
    machine_config: dict,
    wheels: ArtifactStep[Artifact],
    training_sources: ArtifactStep[Artifact],
    snapshots_sha256: str,
) -> dict[str, ArtifactStep]:
    baseline = development_step(seed, model, version, runtime_bundle, "parent")
    adaptive = ArtifactStep(
        name="documents/russell-rsi-adaptive-round-1",
        version=version,
        artifact_type=Artifact,
        deps=(seed, baseline, wheels, training_sources),
        build_config=lambda ctx: AdaptiveTasksConfig(
            snapshots_uri=prefix_join(ctx.artifact_path(training_sources), "snapshots.jsonl"),
            snapshots_sha256=snapshots_sha256,
            inventory_uri=prefix_join(ctx.artifact_path(seed), "inventory.jsonl"),
            traces_uri=prefix_join(ctx.artifact_path(baseline), "traces.jsonl"),
            output_path=ctx.output_path,
            relay_job=relay_job,
            image=image,
            runtime_bundle=runtime_bundle,
            max_candidates=80,
            admission_concurrency=8,
            minimum_train_rows=ROLE_PLAN.train_batch_size,
            dependency_wheels_uri=ctx.artifact_path(wheels),
            source_identities=(
                artifact_identity(seed),
                artifact_identity(baseline),
                artifact_identity(wheels),
                artifact_identity(training_sources),
            ),
        ),
        run=run_adaptive_tasks,
    )
    return post_admission_workflow(adaptive, seed, model, scale, version, runtime_bundle, machine_config)


def coding_feedback_steps(
    coding_eval: ArtifactStep[EvaluationResult],
    model: ArtifactStep,
    panel: CodingPanel,
    version: str,
    label: str,
    relay_job: str,
) -> dict[str, ArtifactStep]:
    """Validate coding eval archives before the private capability analyst."""

    def evidence_config(ctx: StepContext) -> CodingEvidenceConfig | dict:
        if ctx.is_fingerprint:
            return {"coding_eval": artifact_identity(coding_eval), "model": artifact_identity(model), "panel": panel}
        result = ctx.resolved(coding_eval)
        return CodingEvidenceConfig(
            records_prefix=result.records_prefix,
            run_ids=result.run_ids,
            results_paths=result.results_paths,
            model_identity=artifact_identity(model),
            panel=panel,
            output_path=ctx.output_path,
        )

    evidence = ArtifactStep(
        name=f"documents/russell-rsi-{label}-coding-evidence-{CODING_ANALYSIS_CONTEXT_PROTOCOL}",
        version=version,
        artifact_type=Artifact,
        deps=(coding_eval, model),
        build_config=evidence_config,
        run=collect_coding_eval_evidence,
    )
    analysis = ArtifactStep(
        name=f"documents/russell-rsi-{label}-capabilities-{CODING_ANALYSIS_CONTEXT_PROTOCOL}",
        version=version,
        artifact_type=Artifact,
        deps=(evidence,),
        build_config=lambda ctx: CodingAnalysisConfig(
            evidence_path=ctx.artifact_path(evidence),
            evidence_identity=artifact_identity(evidence),
            relay_job=relay_job,
            output_path=ctx.output_path,
        ),
        run=analyze_coding_eval_failures,
    )
    return {"coding-evidence": evidence, "capabilities": analysis}


@dataclass(frozen=True)
class OptimizerStepConfig:
    expected_updates: int
    actual_updates: int | None
    export_uri: str | None


def require_optimizer_updates(config: OptimizerStepConfig) -> None:
    if config.actual_updates != config.expected_updates or not config.export_uri:
        raise ValueError("Training did not publish the exact bounded optimizer updates and HF export")


def bootstrap_round_workflow(
    training: ArtifactStep[Artifact],
    retention: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint],
    version: str,
    runtime_bundle: RuntimeBundle,
    machine_config: dict,
    *,
    round_number: int,
    panel: CodingPanel,
    relay_job: str,
    calibration: ArtifactStep[Artifact],
    scale: str = "pilot",
) -> dict[str, ArtifactStep]:
    """Bind one new protocol round to coding eval feedback, without a parent rerun."""
    label = "bootstrap-initial" if scale == "smoke" else f"bootstrap-round-{round_number}"
    trained = train_step(training, model, scale, version, retention, machine_config, label)
    trained = replace(trained, deps=(*trained.deps, calibration))

    def optimizer_config(ctx: StepContext) -> OptimizerStepConfig | dict:
        if ctx.is_fingerprint:
            return {"producer": artifact_identity(trained), "expected_updates": SCALES[scale].updates}
        result = ctx.resolved(trained)
        return OptimizerStepConfig(SCALES[scale].updates, result.global_step, result.hf_model_uri)

    optimizer_gate = ArtifactStep(
        name=f"documents/russell-rsi-{label}-optimizer-gate",
        version=version,
        artifact_type=Artifact,
        deps=(trained,),
        build_config=optimizer_config,
        run=require_optimizer_updates,
    )
    selected_model = evaluation_model(f"russell-rsi-{label}", SKYRL_POLICY_LOCATION, None)
    reload = eval_step(
        selected_model,
        "mmlu-smoke",
        version=version,
        deps=(trained, optimizer_gate),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, selected_model),
        limit=1,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    if scale == "smoke":
        return {"rl": trained, "reload": reload}
    coding = eval_step(
        selected_model,
        "humanevalplus,mbppplus",
        version=version,
        deps=(trained, reload),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, selected_model),
        limit=32,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    outputs = coding_feedback_steps(coding, trained, panel, version, label, relay_job)
    outputs.update({"rl": trained, "reload": reload, "coding-development": coding})
    return outputs


def post_admission_workflow(
    training: ArtifactStep[Artifact],
    seed: ArtifactStep[Artifact],
    model: ArtifactStep[LevanterCheckpoint],
    scale: str,
    version: str,
    runtime_bundle: RuntimeBundle,
    machine_config: dict,
    name_component: str | None = None,
) -> dict[str, ArtifactStep]:
    """Calibrate qualified tasks before policy allocation, then export and reload."""
    label = f"{name_component}-{scale}" if name_component else scale
    calibration_label = f"{name_component}-train-calibration" if name_component else "train-calibration"
    trained = train_step(
        training, model, scale, version, development=seed, machine_config=machine_config, name_component=name_component
    )
    calibration = development_step(
        training,
        model,
        version,
        runtime_bundle,
        calibration_label,
        relative_path="train.parquet",
        samples_per_task=4,
        temperature=1.0,
        require_reward_variation=True,
    )
    trained = replace(trained, deps=(*trained.deps, calibration))
    selected_model = evaluation_model(f"russell-rsi-{label}", SKYRL_POLICY_LOCATION, None)
    reload = eval_step(
        selected_model,
        "mmlu-smoke",
        version=version,
        deps=(trained,),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, selected_model),
        limit=1,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    if scale == "smoke":
        return {"reload": reload}
    candidate = development_step(seed, trained, version, runtime_bundle, f"candidate-{label}")
    candidate = replace(candidate, deps=(*candidate.deps, reload))
    parent_benchmarks = parent_public_step(model, version)
    benchmarks = eval_step(
        selected_model,
        "humanevalplus,mbppplus",
        version=version,
        deps=(trained, reload, parent_benchmarks),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, selected_model),
        limit=32,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )
    return {"development": candidate, "parent-coding-subset": parent_benchmarks, "candidate-coding-subset": benchmarks}


@click.command(help=__doc__)
@click.option(
    "--stage",
    type=click.Choice(("rl", "reload", "evaluation", "spike", "repair-spike", "parent-development")),
    required=True,
)
@click.option("--scale", type=click.Choice(tuple(SCALES)), required=True)
@click.option("--data-name", required=True)
@click.option("--data-version", required=True)
@click.option("--data-uri", required=True)
@click.option("--model-uri", required=True, help="Complete HF export of the pinned September 21 SFT model.")
@click.option("--evals", help="Evaluation suites; required for the evaluation stage.")
@click.option("--relay-job", help="Pinned GLM relay used for the adaptive task round.")
@click.option("--task-image", help="Digest-addressed image used for task admission.")
@click.option("--machine-config-json", help="Explicit SkyRL machine backend and QEMU asset settings.")
@click.option("--dependency-wheels-uri", help="Pinned wheel artifact for offline task environments.")
@click.option("--training-sources-uri", help="Frozen training-source artifact, separate from the development seed.")
@click.option("--training-snapshots-sha256", help="SHA256 of the frozen training snapshots.jsonl file.")
@click.option("--repair-manifest-uri", help="Sealed original round-one evidence and repair eligibility manifest.")
@click.option("--repair-manifest-sha256", help="SHA256 of the sealed repair manifest.")
@rl_build_options
def main(
    stage: str,
    scale: str,
    data_name: str,
    data_version: str,
    data_uri: str,
    model_uri: str,
    evals: str | None,
    relay_job: str | None,
    task_image: str | None,
    machine_config_json: str | None,
    dependency_wheels_uri: str | None,
    training_sources_uri: str | None,
    training_snapshots_sha256: str | None,
    repair_manifest_uri: str | None,
    repair_manifest_sha256: str | None,
) -> ArtifactStep | dict[str, ArtifactStep]:
    if stage == "evaluation" and evals is None:
        raise click.UsageError("--evals is required for --stage evaluation")
    if stage == "spike" and (repair_manifest_uri is not None or repair_manifest_sha256 is not None):
        raise click.UsageError("Repair manifest options require --stage repair-spike")
    if stage == "repair-spike" and (training_sources_uri is not None or training_snapshots_sha256 is not None):
        raise click.UsageError("Training source options require --stage spike")
    version = resolve_version("russell-rsi", None)
    data = ArtifactStep.adopt(data_name, data_version, data_uri)
    model = ArtifactStep.adopt(
        "checkpoints/russell-sft-parent",
        "2026.09.21",
        model_uri,
        kind=LevanterCheckpoint,
        config={"repository": MODEL, "revision": MODEL_REVISION},
    )
    if stage in ("spike", "repair-spike"):
        if click.get_current_context().params.get("do_run") and not os.environ.get(GLM_TOKEN_ENV):
            raise click.UsageError(f"--stage {stage} --run requires {GLM_TOKEN_ENV} before any GPU work")
        if relay_job is None or task_image is None or machine_config_json is None or dependency_wheels_uri is None:
            raise click.UsageError(
                f"--stage {stage} requires --relay-job, --task-image, --machine-config-json, and --dependency-wheels-uri"
            )
    if stage == "repair-spike" and (repair_manifest_uri is None or repair_manifest_sha256 is None):
        raise click.UsageError("--stage repair-spike requires --repair-manifest-uri and --repair-manifest-sha256")
    if stage == "spike":
        if training_sources_uri is None or training_snapshots_sha256 is None:
            raise click.UsageError("--stage spike requires --training-sources-uri and --training-snapshots-sha256")
    if stage in ("spike", "repair-spike", "parent-development"):
        if machine_config_json is None:
            raise click.UsageError(f"--stage {stage} requires --machine-config-json")
        machine_config = json.loads(machine_config_json)
        if machine_config["backend"] != "qemu":
            raise click.UsageError("The current Python repair tasks require the network-disabled QEMU task backend")
        runtime_bundle = RuntimeBundle(**machine_config["runtime_bundle"])
        if stage == "parent-development":
            return development_step(data, model, version, runtime_bundle, "parent")
        assert relay_job is not None and task_image is not None and dependency_wheels_uri is not None
        wheels = ArtifactStep.adopt("documents/russell-rsi-dependency-wheels", version, dependency_wheels_uri)
        if stage == "repair-spike":
            assert repair_manifest_uri is not None and repair_manifest_sha256 is not None
            manifest = json.loads(pinned_bytes(repair_manifest_uri, repair_manifest_sha256))
            baseline = development_step(data, model, version, runtime_bundle, "parent")
            if manifest["inputs"]["parent_development_identity"] != artifact_identity(baseline):
                raise click.UsageError("Sealed evidence parent-development identity does not match this launch")
            return repair_spike_workflow(
                data,
                model,
                scale,
                version,
                relay_job=relay_job,
                image=task_image,
                runtime_bundle=runtime_bundle,
                machine_config=machine_config,
                wheels=wheels,
                manifest_uri=repair_manifest_uri,
                manifest_sha256=repair_manifest_sha256,
            )
        assert training_sources_uri is not None and training_snapshots_sha256 is not None
        training_sources = ArtifactStep.adopt(
            "documents/russell-rsi-training-sources",
            version,
            training_sources_uri,
            config={"snapshots_sha256": training_snapshots_sha256},
        )
        return spike_workflow(
            data,
            model,
            scale,
            version,
            relay_job,
            task_image,
            runtime_bundle,
            machine_config,
            wheels,
            training_sources,
            training_snapshots_sha256,
        )
    trained = train_step(data, model, scale, version)
    if stage == "rl":
        return trained
    selected_model = evaluation_model(f"russell-rsi-{scale}", SKYRL_POLICY_LOCATION, None)
    if stage == "reload":
        evals = "mmlu-smoke"
    assert evals is not None
    return eval_step(
        selected_model,
        evals,
        version=version,
        deps=(trained,),
        resolve_model=lambda ctx: resolve_skyrl_model(ctx, trained, selected_model),
        limit=1 if stage == "reload" else None,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )


def final_coding_evaluation(
    checkpoint: ArtifactStep[LevanterCheckpoint],
    label: str,
    version: str,
) -> ArtifactStep[EvaluationResult]:
    """Build final-only evaluation of the working and held-out coding panels."""
    model = evaluation_model(f"russell-rsi-final-{label}", MODEL, None)
    return eval_step(
        model,
        "humanevalplus,mbppplus",
        version=version,
        deps=(checkpoint,),
        resolve_model=lambda ctx: replace(
            model, location=ctx.artifact_path(checkpoint), identity=artifact_identity(checkpoint)
        ),
        limit=64,
        accelerator="H100x8",
        submission_cluster=CLUSTER,
        federated_cluster=CLUSTER,
    )


def run_bootstrap_loop(
    seed_bank: ArtifactStep[Artifact],
    parent: ArtifactStep[LevanterCheckpoint],
    retention: ArtifactStep[Artifact],
    panel: CodingPanel,
    heldout_manifest_uri: str,
    heldout_manifest_sha256: str,
    parent_coding_evidence_uri: str,
    parent_coding_evidence_sha256: str,
    parent_retention_evidence_uri: str,
    parent_retention_evidence_sha256: str,
    version: str,
    runtime_bundle: RuntimeBundle,
    machine_config: dict,
    relay_job: str,
    manifest_directory: StoragePath,
    build_next_bank: Callable[[ArtifactStep[Artifact], LoopState, int], ArtifactStep[Artifact] | None],
    initial_calibration: ArtifactStep[Artifact] | None = None,
) -> LoopState:
    """Execute bounded artifact rounds with an explicit independent source builder."""
    heldout_manifest = json.loads(pinned_bytes(heldout_manifest_uri, heldout_manifest_sha256))
    heldout_panel_ids(panel, heldout_manifest)
    heldout = {"manifest_sha256": heldout_manifest_sha256, "development": asdict(panel)}
    write_once(manifest_directory / "panels.json", heldout)
    panel_identity = compact_json_sha256(asdict(panel))
    runtime_identity = compact_json_sha256(
        {"qemu": runtime_bundle.archive_sha256, "skyrl": MARIN_SKYRL.commit, "machine": machine_config}
    )
    parent_coding = json.loads(pinned_bytes(parent_coding_evidence_uri, parent_coding_evidence_sha256))
    parent_retention = json.loads(pinned_bytes(parent_retention_evidence_uri, parent_retention_evidence_sha256))
    if (
        parent_coding["model_identity"] != artifact_identity(parent)
        or parent_coding["panel_sha256"] != panel_identity
        or parent_retention["model_identity"] != artifact_identity(parent)
        or parent_retention["tasks_identity"] != artifact_identity(retention)
    ):
        raise ValueError("Parent baseline evidence does not identify the pinned model and panels")
    baseline_rewards = [reward for group in parent_retention["task_rewards"].values() for reward in group]
    if (
        not baseline_rewards
        or len(baseline_rewards) != parent_retention["count"]
        or any(len(group) != 1 for group in parent_retention["task_rewards"].values())
    ):
        raise ValueError("Parent retention baseline requires one graded reward per task")
    parent_score = CheckpointScore(
        artifact_identity(parent),
        (parent_coding["scores"]["humanevalplus"], parent_coding["scores"]["mbppplus"]),
        sum(baseline_rewards) / len(baseline_rewards),
    )
    bank_artifact = resolve(seed_bank)
    bank_record = json.loads(StoragePath(prefix_join(bank_artifact.path, "bank.json")).read_text())
    initial_bank = qualified_bank(tuple(QualifiedTask(**item) for item in bank_record["tasks"]))
    state = LoopState(parent_score, parent_score, parent_score, initial_bank)
    checkpoint_handles = {artifact_identity(parent): parent}
    bank_handle = seed_bank
    feedback_identity = bank_record["feedback_identity"]
    previous_sha256 = compact_json_sha256(
        {
            **heldout,
            "source_bank": artifact_identity(seed_bank),
            "parent_coding_sha256": parent_coding_evidence_sha256,
            "parent_retention_sha256": parent_retention_evidence_sha256,
        }
    )
    smoke_complete = False
    feedback_labels: set[str] = set()
    while state.stop_reason is None:
        number = state.completed_pilots + 1
        current = checkpoint_handles[state.working.checkpoint_identity]
        bank_artifact = resolve(bank_handle)
        bank_record = json.loads(StoragePath(prefix_join(bank_artifact.path, "bank.json")).read_text())
        bank = qualified_bank(tuple(QualifiedTask(**item) for item in bank_record["tasks"]))
        if len(bank) < 16:
            raise ValueError("Fewer than sixteen qualified tasks. Do not allocate calibration GPUs")
        retained_hashes = {task.task_sha256 for task in state.bank}
        fresh = tuple(task for task in bank if task.task_sha256 not in retained_hashes)
        if {task.task_sha256 for task in state.bank} - {task.task_sha256 for task in bank}:
            raise ValueError("A new bank discarded qualified retained tasks")
        if state.completed_pilots and not any(
            task.contract_id not in {retained.contract_id for retained in state.bank}
            and task.relation not in {"variant", "replacement", "alias"}
            and feedback_labels.intersection(task.capability.split(","))
            for task in fresh
        ):
            state = replace(state, stop_reason=StopReason.TASK_SUPPLY)
            break
        difficulty = (
            initial_calibration
            if number == 1 and initial_calibration is not None
            else development_step(
                bank_handle,
                current,
                version,
                runtime_bundle,
                f"bootstrap-round-{number}-bank-difficulty",
                relative_path="train.parquet",
                samples_per_task=8,
                temperature=CALIBRATION_TEMPERATURE,
                require_reward_variation=True,
                limit=len(bank),
            )
        )
        inputs = round_inputs(
            state,
            bank,
            bank_identity=artifact_identity(bank_handle),
            calibration_identity=artifact_identity(difficulty),
            feedback_labels=tuple(sorted(feedback_labels)),
            development_identity=panel_identity,
            retention_identity=artifact_identity(retention),
            feedback_identity=feedback_identity,
            runtime_identity=runtime_identity,
            seed=9528,
        )
        path = manifest_directory / f"bootstrap-{version}-pilot-{number}.json"
        resumed = load_round(path, inputs, previous_sha256) if path.exists() else None
        if resumed is not None:
            plan = resumed.plan
        else:
            measured = resolve(difficulty)
            summary = json.loads(StoragePath(prefix_join(measured.path, "failure_summary.json")).read_text())
            measurements = calibration_measurements(
                summary, bank, artifact_identity(current), artifact_identity(bank_handle)
            )
            plan = round_plan(
                state,
                fresh,
                measurements,
                run_id=f"bootstrap-{version}",
                **{
                    key: value
                    for key, value in inputs.items()
                    if key
                    not in {"current_checkpoint", "champion_checkpoint", "task_bank", "updates", "max_glm_responses"}
                },
            )
        frozen = ArtifactStep(
            name=f"documents/russell-rsi-bootstrap-round-{number}-train",
            version=version,
            artifact_type=Artifact,
            deps=(bank_handle, difficulty),
            build_config=lambda ctx, selected=plan, source=bank_handle: FrozenRoundConfig(
                bank_path=ctx.artifact_path(source),
                plan=selected,
                output_path=ctx.output_path,
            ),
            run=remote(
                freeze_round_dataset,
                resources=ResourceConfig.with_cpu(cpu=4, ram="16GB", disk="64GB"),
                pip_packages=["./lib/taskcompendium"],
            ),
        )
        outputs = bootstrap_round_workflow(
            frozen,
            retention,
            current,
            version,
            runtime_bundle,
            machine_config,
            round_number=number,
            panel=panel,
            relay_job=relay_job,
            calibration=difficulty,
        )
        if not smoke_complete:
            smoke = bootstrap_round_workflow(
                frozen,
                retention,
                parent,
                version,
                runtime_bundle,
                machine_config,
                round_number=1,
                panel=panel,
                relay_job=relay_job,
                calibration=difficulty,
                scale="smoke",
            )
            expected_smoke = {
                "rl": artifact_identity(smoke["rl"]),
                "reload": artifact_identity(smoke["reload"]),
                "coding_baseline_sha256": parent_coding_evidence_sha256,
                "retention_baseline_sha256": parent_retention_evidence_sha256,
            }
            smoke_path = manifest_directory / "smoke.json"
            if not smoke_path.exists():
                if resumed is not None:
                    raise ValueError("Sealed pilot is missing its smoke completion record")
                run(smoke["rl"], smoke["reload"])
            write_once(smoke_path, expected_smoke)
            smoke_complete = True
        if resumed is not None:
            state = resumed.state
            value = asdict(resumed.result)
            candidate_identity = resumed.result.candidate.checkpoint_identity
            export_uri = resumed.result.checkpoint_uri
            previous_sha256 = resumed.sha256
            adopted = ArtifactStep.adopt(
                f"checkpoints/russell-rsi-bootstrap-round-{number}-export",
                version,
                export_uri,
                kind=LevanterCheckpoint,
                config={"producer": value["reload_identity"]},
            )
            if artifact_identity(adopted) != candidate_identity:
                raise ValueError("Resumed export identity changed")
            checkpoint_handles[candidate_identity] = adopted
            # The independent builder consumes only the canonical capability artifact.
            capabilities = outputs["capabilities"]
            if artifact_identity(capabilities) != value["feedback_identity"]:
                raise ValueError("Resumed capability artifact identity changed")
        else:
            values = run(outputs["rl"], outputs["reload"], outputs["coding-evidence"], outputs["capabilities"])
            trained = cast(SkyRLRun, values[0])
            evidence = json.loads(StoragePath(prefix_join(values[2].path, "coding-evidence.json")).read_text())
            export_uri = cast(str, trained.hf_model_uri)
            adopted = ArtifactStep.adopt(
                f"checkpoints/russell-rsi-bootstrap-round-{number}-export",
                version,
                export_uri,
                kind=LevanterCheckpoint,
                config={"producer": artifact_identity(outputs["reload"])},
            )
            candidate_identity = artifact_identity(adopted)
            checkpoint_handles[candidate_identity] = adopted
            retention_eval = development_step(
                retention,
                outputs["rl"],
                version,
                runtime_bundle,
                f"bootstrap-round-{number}-retention",
                relative_path="development.parquet",
                limit=len(parent_retention["task_rewards"]),
            )
            retention_result = resolve(retention_eval)
            retention_summary = json.loads(
                StoragePath(prefix_join(retention_result.path, "failure_summary.json")).read_text()
            )
            retention_rewards = [reward for group in retention_summary["task_rewards"].values() for reward in group]
            if (
                retention_summary["model_identity"] != artifact_identity(outputs["rl"])
                or retention_summary["tasks_identity"] != artifact_identity(retention)
                or retention_summary["task_rewards"].keys() != parent_retention["task_rewards"].keys()
                or retention_summary["count"] != parent_retention["count"]
                or any(len(group) != 1 for group in retention_summary["task_rewards"].values())
            ):
                raise ValueError("Retention panel does not have a graded reward for each task")
            result = RoundResult(
                candidate=CheckpointScore(
                    candidate_identity,
                    tuple(evidence["scores"][suite] for suite in ("humanevalplus", "mbppplus")),
                    sum(retention_rewards) / len(retention_rewards),
                ),
                reload_identity=artifact_identity(outputs["reload"]),
                feedback_identity=artifact_identity(outputs["capabilities"]),
                optimizer_steps=cast(int, trained.global_step),
                checkpoint_uri=export_uri,
            )
            state = advance(state, plan, result)
            previous_sha256 = seal_round(manifest_directory, state, plan, result, previous_sha256)
            capabilities = outputs["capabilities"]
        if state.stop_reason is None:
            capability_artifact = resolve(capabilities)
            capabilities_bytes = StoragePath(prefix_join(capability_artifact.path, "capabilities.json")).read_bytes()
            capability_record = json.loads(capabilities_bytes)
            feedback_labels = {skill["label"] for skill in capability_record["skills"]}
            if not feedback_labels:
                write_once(
                    manifest_directory / f"feedback-insufficient-after-{state.completed_pilots}.json",
                    {
                        "state": asdict(state),
                        "last_round_sha256": previous_sha256,
                        "feedback_identity": artifact_identity(capabilities),
                        "capabilities_sha256": hashlib.sha256(capabilities_bytes).hexdigest(),
                        "reason": "empty_capability_feedback",
                    },
                )
                return state
            next_bank = build_next_bank(capabilities, state, 24)
            if next_bank is None:
                write_once(
                    manifest_directory / f"construction-required-after-{state.completed_pilots}.json",
                    {
                        "state": asdict(state),
                        "last_round_sha256": previous_sha256,
                        "prior_bank_sha256": compact_json_sha256({"tasks": [asdict(task) for task in state.bank]}),
                        "feedback_identity": artifact_identity(capabilities),
                        "capabilities_uri": prefix_join(capability_artifact.path, "capabilities.json"),
                        "capabilities_sha256": hashlib.sha256(capabilities_bytes).hexdigest(),
                        "response_cap": 24,
                    },
                )
                return state
            bank_handle = next_bank
            feedback_identity = artifact_identity(capabilities)
    terminal = {
        "state": asdict(state),
        "last_round_sha256": previous_sha256,
        "heldout_manifest_sha256": heldout_manifest_sha256,
    }
    write_once(manifest_directory / "terminal-state.json", terminal)
    champion = checkpoint_handles[state.champion.checkpoint_identity]
    parent_final = final_coding_evaluation(parent, "parent", version)
    champion_final = (
        parent_final
        if artifact_identity(champion) == artifact_identity(parent)
        else final_coding_evaluation(champion, "champion", version)
    )
    comparison = heldout_comparison_step(
        parent_final,
        champion_final,
        parent_identity=artifact_identity(parent),
        champion_identity=artifact_identity(champion),
        working_panel=panel,
        manifest_uri=heldout_manifest_uri,
        manifest_sha256=heldout_manifest_sha256,
        terminal_state=terminal,
        version=version,
    )
    resolve(comparison)
    return state


if __name__ == "__main__":
    main()
