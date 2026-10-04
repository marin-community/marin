# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build Russell code-repair RL, checkpoint export, and evaluation artifacts."""

import json
import os
from dataclasses import dataclass, replace
from typing import cast

import click
import yaml
from fray.types import ResourceConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
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
from rigging.filesystem.storage_path import prefix_join
from rigging.runtime_bundle import RuntimeBundle

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS
from experiments.evaluation.pipeline import eval_step
from experiments.post_training.russell_rsi.adaptive_tasks import AdaptiveTasksConfig, run_adaptive_tasks_in_project
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
            "data": {"kind": "tasks", "train_data": [], "val_data": []},
            "trainer": {
                "strategy": "megatron",
                "flash_attn": False,
                "use_sample_packing": False,
                "offload_optimizer_during_rollouts": True,
                "gradient_checkpointing": True,
                "algorithm": {"advantage_estimator": "grpo", "use_kl_loss": False},
                "epochs": 1,
                "max_steps": scale.updates,
                "update_epochs_per_batch": 1,
                "max_prompt_length": PROMPT_TOKENS,
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
                "max_turns": 16,
                "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
                "engine_init_kwargs": {
                    "moe_backend": "triton",
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                    "max_model_len": CONTEXT_TOKENS,
                },
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_generate_length": RESPONSE_TOKENS,
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
) -> ArtifactStep[SkyRLRun]:
    selected = SCALES[scale]
    return skyrl_step(
        SkyRLSpec(
            name=f"checkpoints/russell-rsi-{scale}",
            version=version,
            config_yaml=recipe(selected, machine_config),
            runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
            model=ArtifactHfModel(model, MODEL, MODEL_REVISION, relative_path="."),
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
            resources=ResourceConfig.with_gpu("H100", 8, cpu=32, ram="512GB", disk="2TB", target_cluster=CLUSTER),
            pip_packages=[MARIN_SKYRL.requirement()],
        ),
    )


def run_adaptive_tasks(config: AdaptiveTasksConfig) -> None:
    remote(
        run_adaptive_tasks_in_project,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32GB", disk="64GB", target_cluster=CLUSTER),
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
) -> dict[str, ArtifactStep]:
    baseline = development_step(seed, model, version, runtime_bundle, "parent")
    parent_benchmarks = parent_public_step(model, version)
    adaptive = ArtifactStep(
        name="documents/russell-rsi-adaptive-round-1",
        version=version,
        artifact_type=Artifact,
        deps=(seed, baseline, wheels),
        build_config=lambda ctx: AdaptiveTasksConfig(
            snapshots_uri=prefix_join(ctx.artifact_path(seed), "snapshots.jsonl"),
            inventory_uri=prefix_join(ctx.artifact_path(seed), "inventory.jsonl"),
            traces_uri=prefix_join(ctx.artifact_path(baseline), "traces.jsonl"),
            output_path=ctx.output_path,
            relay_job=relay_job,
            image=image,
            runtime_bundle=runtime_bundle,
            max_candidates=80,
            minimum_train_rows=ROLE_PLAN.train_batch_size,
            dependency_wheels_uri=ctx.artifact_path(wheels),
            source_identities=(artifact_identity(seed), artifact_identity(baseline), artifact_identity(wheels)),
        ),
        run=run_adaptive_tasks,
    )
    trained = train_step(adaptive, model, scale, version, development=seed, machine_config=machine_config)
    calibration = development_step(
        adaptive,
        model,
        version,
        runtime_bundle,
        "train-calibration",
        relative_path="train.parquet",
        samples_per_task=4,
        temperature=1.0,
        require_reward_variation=True,
    )
    # Calibration must show reward variation before Iris allocates policy workers.
    trained = replace(trained, deps=(*trained.deps, calibration))
    candidate = development_step(seed, trained, version, runtime_bundle, "candidate")
    selected_model = evaluation_model(f"russell-rsi-{scale}", SKYRL_POLICY_LOCATION, None)
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
@click.option("--stage", type=click.Choice(("rl", "reload", "evaluation", "spike", "parent-development")), required=True)
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
) -> ArtifactStep | dict[str, ArtifactStep]:
    if stage == "evaluation" and evals is None:
        raise click.UsageError("--evals is required for --stage evaluation")
    version = resolve_version("russell-rsi", None)
    data = ArtifactStep.adopt(data_name, data_version, data_uri)
    model = ArtifactStep.adopt(
        "checkpoints/russell-sft-parent",
        "2026.09.21",
        model_uri,
        kind=LevanterCheckpoint,
        config={"repository": MODEL, "revision": MODEL_REVISION},
    )
    if stage == "spike":
        if click.get_current_context().params.get("do_run") and not os.environ.get(GLM_TOKEN_ENV):
            raise click.UsageError(f"--stage spike --run requires {GLM_TOKEN_ENV} before any GPU work")
        if relay_job is None or task_image is None or machine_config_json is None or dependency_wheels_uri is None:
            raise click.UsageError(
                "--stage spike requires --relay-job, --task-image, " "--machine-config-json, and --dependency-wheels-uri"
            )
    if stage in ("spike", "parent-development"):
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


if __name__ == "__main__":
    main()
