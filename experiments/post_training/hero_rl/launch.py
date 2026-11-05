# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a 4K Hero numerical probe or four-update RL qualification."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

import click
import yaml
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    InferenceEngineBackend,
    IrisSkyRLExecution,
    MegatronGeometry,
    SkyRLRetentionPolicy,
    SkyRLRolePlan,
    SkyRLRun,
    SkyRLSpec,
    SkyRLTopology,
    _materialize_role_plan_config,
    skyrl_cpu_command,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.hero_rl.piecewise_runtime import MANIFEST_SHA256
from experiments.post_training.iceball_micro import (
    GSM8K_TRAIN_FILENAME,
    GSM8K_VALIDATION_FILENAME,
    Gsm8kParquetConfig,
    write_gsm8k_parquet,
)
from experiments.post_training.mismatch_probe.launch import ProbeSettings, probe_block

MODEL_URI = "s3://marin-us-east-02a/marin/users/romain/hero-vllm-rl/hero-535b-step144000-bf16-split-v1"
MODEL_VERSION = "2026.09.24"
EXPORT_MANIFEST_SHA256 = "b20fdd303eb6ff8993ae5b5d7add14d65787e70981c056fea6de9dea62acfae1"
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
CLUSTER = "cw-us-east-02a"
GPUS_PER_NODE = 8
SEED = 42
REQUEST_WINDOW = 4096
MAX_NEW_TOKENS = 3072
UPDATES = 4
TRAIN_ROWS = 1024
VALIDATION_ROWS = 128
PROBE_PROMPTS = 16
EOT_TOKEN_ID = 128009
STREAMER_MEMORY_LIMIT_BYTES = 16 * 1024**3
STREAMER_CONCURRENCY = 4
GEOMETRY = MegatronGeometry(1, 16, 2, 8, 1)
ROLE_PLAN = SkyRLRolePlan(
    colocate_all=False,
    policy_num_nodes=16,
    policy_num_gpus_per_node=GPUS_PER_NODE,
    num_inference_engines=1,
    inference_engine_backend=InferenceEngineBackend.MULTIPROCESSING,
    inference_engine_tensor_parallel_size=1,
    inference_engine_pipeline_parallel_size=1,
    inference_engine_data_parallel_size=24,
    inference_engine_expert_parallel_size=24,
    train_batch_size=16,
    policy_mini_batch_size=16,
    micro_train_batch_size_per_gpu=1,
    n_samples_per_prompt=8,
)


class Hardware(StrEnum):
    H100 = "h100"
    GB200_48 = "gb200-48"


@dataclass(frozen=True)
class HardwareProfile:
    cluster: str
    gpu_variant: str
    geometry: MegatronGeometry
    role_plan: SkyRLRolePlan
    memory: str
    gpu_memory_utilization: float
    checkpoint_cpu_offload: bool


HARDWARE_PROFILES = {
    Hardware.H100: HardwareProfile(CLUSTER, "H100", GEOMETRY, ROLE_PLAN, "1800GB", 0.92, False),
    Hardware.GB200_48: HardwareProfile(
        cluster="cw-us-east-08a",
        gpu_variant="GB200",
        geometry=MegatronGeometry(1, 12, 1, 3, 1),
        role_plan=SkyRLRolePlan(
            colocate_all=False,
            policy_num_nodes=9,
            policy_num_gpus_per_node=4,
            num_inference_engines=1,
            inference_engine_backend=InferenceEngineBackend.MULTIPROCESSING,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=12,
            inference_engine_expert_parallel_size=12,
            train_batch_size=15,
            policy_mini_batch_size=15,
            micro_train_batch_size_per_gpu=1,
            n_samples_per_prompt=8,
        ),
        memory="940GiB",
        gpu_memory_utilization=0.90,
        checkpoint_cpu_offload=True,
    ),
}


@dataclass(frozen=True)
class ExportCheckConfig:
    output_path: str
    model_uri: str
    manifest_sha256: str
    tokenizer_uri: str
    tokenizer_revision: str


def check_export(config: ExportCheckConfig) -> None:
    """Check the completed export's metadata without reading model shards."""
    configure_coreweave_s3()
    with StoragePath(prefix_join(config.model_uri, "export-manifest.json")).open("rb") as source:
        manifest = source.read()
    digest = hashlib.sha256(manifest).hexdigest()
    if digest != config.manifest_sha256:
        raise ValueError(f"Hero export manifest changed: expected {config.manifest_sha256}, found {digest}")
    with StoragePath(prefix_join(config.model_uri, "config.json")).open("r") as source:
        model_config = json.load(source)
    with StoragePath(prefix_join(config.model_uri, "model.safetensors.index.json")).open("r") as source:
        index = json.load(source)
    if not index.get("weight_map"):
        raise ValueError("Hero HF export has no indexed model weights")
    if (
        model_config.get("model_type") != "grug_moe"
        or model_config.get("hidden_size") != 6144
        or model_config.get("num_hidden_layers") != 48
        or model_config.get("num_experts") != 384
    ):
        raise ValueError("Hero qualification requires Grug MoE with 48 layers, hidden size 6144 and 384 experts")
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    prepared = subprocess.run(
        skyrl_cpu_command(
            "python",
            "-m",
            "cloud.iris.model_preflight",
            "--runtime-profile",
            "megatron",
            "--model-uri",
            config.model_uri,
            "--model-identity",
            f"hero-export:{digest}",
            "--tokenizer-uri",
            config.tokenizer_uri,
            "--tokenizer-revision",
            config.tokenizer_revision,
        ),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=600,
    )
    staging = json.loads(prepared.stdout)
    with StoragePath(prefix_join(config.output_path, "validation.json")).open("w") as destination:
        json.dump(
            {
                "model_uri": config.model_uri,
                "export_manifest_sha256": digest,
                "indexed_tensors": len(index["weight_map"]),
                "runtime_staging": staging,
                "scope": "export metadata only; model shard contents are not rehashed",
            },
            destination,
            sort_keys=True,
        )


def training_config(hardware: Hardware = Hardware.H100) -> dict:
    """Render the selected hardware geometry with a four-update, 4K budget."""
    profile = HARDWARE_PROFILES[hardware]
    plan = profile.role_plan
    megatron = {
        **asdict(profile.geometry),
        "moe_router_replay": True,
        "optimizer_checkpoint_sharding_type": "dp_reshardable",
        "ddp_config": {
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": True,
            "overlap_param_gather": False,
            "average_in_collective": True,
        },
        "transformer_config_kwargs": {
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
            "seq_length": 4224,
            "bias_activation_fusion": True,
        },
    }
    if profile.checkpoint_cpu_offload:
        megatron["transformer_config_kwargs"]["grug_checkpoint_cpu_offload"] = True
    config = {
        "entrypoint": "standard",
        "context_budget": {
            "request_window_tokens": REQUEST_WINDOW,
            "max_new_tokens_per_turn": MAX_NEW_TOKENS,
            "max_turns": 1,
        },
        "environment": {"env_class": "gsm8k"},
        "data": {"kind": "parquet", "shuffle": True},
        "trainer": {
            "strategy": "megatron",
            "debug_mode": "distributed",
            "bf16": True,
            "flash_attn": False,
            "gradient_checkpointing": True,
            "use_sample_packing": True,
            "offload_optimizer_during_rollouts": True,
            "epochs": 1,
            "max_steps": UPDATES,
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": plan.micro_train_batch_size_per_gpu,
            "eval_batch_size": VALIDATION_ROWS,
            "eval_before_train": True,
            "eval_interval": UPDATES,
            "ckpt_interval": -1,
            "hf_save_interval": -1,
            "hf_hub_repo_id": None,
            "resume_mode": "none",
            "seed": SEED,
            "logger": "wandb",
            "project_name": "marin-hero-rl-qualification",
            "tracker_commit_each_step": True,
            "algorithm": {
                "advantage_estimator": "grpo",
                "grpo_norm_by_std": True,
                "policy_loss_type": "behavior_clip",
                "loss_reduction": "token_mean",
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "use_kl_loss": False,
                "use_kl_in_reward": False,
                "use_entropy_loss": False,
                "off_policy_correction": "none",
                "batch_invariant": True,
            },
            "policy": {
                "grug_query_bias_update_mode": "frozen",
                "optimizer_config": {
                    "optimizer": "MuonH",
                    "lr": 1e-6,
                    "adam_betas": [0.9, 0.95],
                    "weight_decay": 0.0,
                    "max_grad_norm": 0.0,
                    "num_warmup_steps": 0,
                    "scheduler": "constant_with_warmup",
                    "optimizer_kwargs": {"adam_lr": 1e-6, "offload_momentum": True},
                },
                "megatron_config": megatron,
            },
            "ref": {"megatron_config": asdict(profile.geometry)},
            "rollout_buffer": {
                "max_staleness_steps": 1,
                "batch_policy": "full_batch",
                "max_in_flight": 32,
                "object_store_root": None,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "vllm_attention_backend": "FLASH_ATTN",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "weight_sync_transport": "broadcast",
            "weight_sync_pause": {"mode": "keep", "clear_cache": True},
            "r3_transport": "decentral",
            "gpu_memory_utilization": profile.gpu_memory_utilization,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 16,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "chat_template": None,
            "engine_init_kwargs": {
                "model_loader_extra_config": {
                    "distributed": True,
                    "memory_limit": STREAMER_MEMORY_LIMIT_BYTES,
                    "concurrency": STREAMER_CONCURRENCY,
                },
                "kernel_config": {"moe_backend": "triton"},
                "enable_return_routed_experts": True,
                "enable_flashinfer_autotune": False,
                "logprobs_mode": "processed_logprobs",
                "generation_config": "vllm",
            },
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "logprobs": 0,
                "stop_token_ids": [EOT_TOKEN_ID],
            },
            "eval_sampling_params": {"temperature": 0.0, "logprobs": None, "stop_token_ids": [EOT_TOKEN_ID]},
        },
        "extra_env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    }
    _materialize_role_plan_config(config, plan)
    if hardware is Hardware.GB200_48:
        # The native cache cap also prevents large-block rounding. Unrounded
        # blocks must bypass Torch's power-of-two cache buckets to avoid reuse
        # of a smaller block for a larger request in the same bucket.
        config["extra_env"]["PYTORCH_CUDA_ALLOC_CONF"] += ",pinned_max_cached_size_mb:1"
        config["generator"]["engine_init_kwargs"]["cudagraph_metrics"] = True
        config["generator"]["engine_init_kwargs"]["compilation_config"] = {
            "mode": 3,  # Exact-39e CompilationMode.VLLM_COMPILE.
            "cudagraph_mode": "PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        }
    return config


@dataclass(frozen=True)
class HeroQualification:
    numerics: ArtifactStep[SkyRLRun]
    updates: ArtifactStep[SkyRLRun]


def build_qualification(
    version: str,
    *,
    model_uri: str = MODEL_URI,
    model_version: str = MODEL_VERSION,
    export_manifest_sha256: str = EXPORT_MANIFEST_SHA256,
    probe_prompts: int = PROBE_PROMPTS,
    hardware: Hardware = Hardware.H100,
    priority: str = "batch",
) -> HeroQualification:
    """Bind a completed HF export to separate fixed-weight and update stages."""
    if not 1 <= probe_prompts <= VALIDATION_ROWS:
        raise ValueError(f"Hero probe must select between one and {VALIDATION_ROWS} validation prompts")
    profile = HARDWARE_PROFILES[hardware]
    plan = profile.role_plan
    stage_root = "checkpoints/hero-rl"
    if hardware is not Hardware.H100:
        stage_root = f"{stage_root}/{hardware.value}"
    if hardware is Hardware.GB200_48:
        stage_root = f"{stage_root}/piecewise-{MANIFEST_SHA256[:16]}"
    model = ArtifactStep.adopt(
        "models/hero-535b-rl-initializer",
        model_version,
        model_uri,
        kind=LevanterCheckpoint,
        config={"export_manifest_sha256": export_manifest_sha256},
    )
    metadata = ArtifactStep(
        name=user_owned_name("documents/hero-rl-export-check"),
        version=version,
        artifact_type=Artifact,
        run=check_export,
        build_config=lambda ctx: ExportCheckConfig(
            output_path=ctx.output_path,
            model_uri=ctx.artifact_path(model),
            manifest_sha256=export_manifest_sha256,
            tokenizer_uri=TOKENIZER,
            tokenizer_revision=TOKENIZER_REVISION,
        ),
        deps=(model,),
    )
    data = ArtifactStep(
        name=user_owned_name("documents/hero-rl-gsm8k"),
        version=version,
        artifact_type=Artifact,
        run=remote(write_gsm8k_parquet, resources=ResourceConfig.with_cpu(cpu=2, ram="8GB", disk="16GB")),
        build_config=lambda ctx: Gsm8kParquetConfig(
            output_path=ctx.output_path,
            train_rows=TRAIN_ROWS,
            validation_rows=VALIDATION_ROWS,
        ),
    )
    execution = IrisSkyRLExecution(
        cluster=profile.cluster,
        cluster_config=f"lib/iris/config/{profile.cluster}.yaml",
        cpu=128,
        memory=profile.memory,
        disk="4TB",
        priority=priority,
        max_retries=0,
        target_cluster=profile.cluster,
        parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
        coordinator_timeout_hours=8,
        job_timeout_seconds=6 * 60 * 60,
        wandb_entity="marin-community",
    )
    base = SkyRLSpec(
        name=user_owned_name(f"{stage_root}/updates-4k"),
        version=version,
        config_yaml=yaml.safe_dump(training_config(hardware), sort_keys=False),
        model=ArtifactHfModel(model, TOKENIZER, TOKENIZER_REVISION, relative_path=""),
        train_data=(ArtifactDataSource(data, GSM8K_TRAIN_FILENAME),),
        validation_data=(ArtifactDataSource(data, GSM8K_VALIDATION_FILENAME),),
        topology=SkyRLTopology(plan.num_nodes, plan.policy_num_gpus_per_node, profile.gpu_variant, plan),
        retention=SkyRLRetentionPolicy(),
        seed=SEED,
    )
    config = training_config(hardware)
    config["trainer"].update(eval_before_train=False, eval_interval=-1)
    config["trainer"]["rollout_buffer"]["max_staleness_steps"] = 0
    config["trainer"]["rollout_buffer"]["max_in_flight"] = plan.train_batch_size
    settings = ProbeSettings(
        SEED, probe_prompts, plan.n_samples_per_prompt, 0, 0.5, "off", None, None, ("router_replay",)
    )
    block = probe_block(settings)
    config["trainer"]["mismatch_probe"] = block["trainer"]["mismatch_probe"]
    numerics_spec = replace(
        base,
        name=user_owned_name(f"{stage_root}/numerics-4k"),
        config_yaml=yaml.safe_dump(config, sort_keys=False),
    )
    numerics = skyrl_step(numerics_spec, execution, export_hf=False)
    updates = skyrl_step(base, execution, export_hf=False)
    return HeroQualification(
        numerics=replace(numerics, deps=(*numerics.deps, metadata)),
        updates=replace(updates, deps=(*updates.deps, metadata)),
    )


@click.command(help=__doc__)
@click.option("--stage", type=click.Choice(("numerics", "updates")), default="numerics", show_default=True)
@click.option(
    "--hardware",
    type=click.Choice([hardware.value for hardware in Hardware]),
    default=Hardware.H100.value,
    show_default=True,
)
@click.option("--priority", default="batch", show_default=True, help="Iris scheduling priority, such as interactive.")
@click.option("--model-uri", default=MODEL_URI, show_default=True, help="Exact completed HF export directory.")
@click.option("--model-version", default=MODEL_VERSION, show_default=True)
@click.option("--export-manifest-sha256", default=EXPORT_MANIFEST_SHA256, show_default=True)
@click.option("--probe-prompts", type=click.IntRange(1, VALIDATION_ROWS), default=PROBE_PROMPTS, show_default=True)
@rl_build_options
def main(
    stage: str,
    hardware: str,
    priority: str,
    model_uri: str,
    model_version: str,
    export_manifest_sha256: str,
    probe_prompts: int,
) -> ArtifactStep[SkyRLRun]:
    workflow = build_qualification(
        resolve_version("hero-rl-qualification", None),
        model_uri=model_uri,
        model_version=model_version,
        export_manifest_sha256=export_manifest_sha256,
        probe_prompts=probe_prompts,
        hardware=Hardware(hardware),
        priority=priority,
    )
    return {"numerics": workflow.numerics, "updates": workflow.updates}[stage]


if __name__ == "__main__":
    main()
