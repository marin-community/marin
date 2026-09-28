# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch frozen-token RL mismatch probes through Marin's SkyRL artifact path."""

from __future__ import annotations

from dataclasses import dataclass

import click
import yaml
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_owned_name
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
from rigging.provenance import Provenance

CLUSTER = "cw-rno2a"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@dataclass(frozen=True)
class ArmSpec:
    name: str
    use_sample_packing: bool


@dataclass(frozen=True)
class ProbeSettings:
    seed: int
    prompt_count: int
    samples_per_prompt: int
    updates: tuple[int, ...]
    keep_fraction: float
    cache_mode: str
    reuse_probe: str | None
    resume_path: str | None


ARMS = {
    arm.name: arm
    for arm in (
        ArmSpec("native-layout", use_sample_packing=False),
        ArmSpec("packed-layout", use_sample_packing=True),
    )
}


def mismatch_probe_config(settings: ProbeSettings, *, marin_commit: str, skyrl_commit: str, replay_modes: bool) -> dict:
    """Configure frozen-token scoring shared by the fixture and model launchers."""
    return {
        "enabled": True,
        "prompts": {"count": settings.prompt_count, "samples_per_prompt": settings.samples_per_prompt},
        "seed": settings.seed,
        "archive_uri": None,
        "reuse_probe": settings.reuse_probe,
        "score_after_updates": list(settings.updates),
        "extra_trainer_modes": ["router_replay", "router_replay_filtered"] if replay_modes else [],
        "filtered_replay": {"keep_fraction": settings.keep_fraction},
        "rescore_prefix_cache": settings.cache_mode,
        "layer_tokens": 0,
        "marin_commit": marin_commit,
        "skyrl_commit": skyrl_commit,
    }


def probe_recipe(
    arm: ArmSpec,
    settings: ProbeSettings,
    *,
    warmup: bool,
    marin_commit: str,
    skyrl_commit: str,
) -> str:
    """Render one role-independent training recipe for the selected arm."""
    config = {
        "entrypoint": "standard",
        "context_budget": {"request_window_tokens": 128, "max_new_tokens_per_turn": 8, "max_turns": 1},
        "environment": {"env_class": "mismatch_fixture"},
        "trainer": {
            "strategy": "megatron",
            "flash_attn": False,
            "use_sample_packing": arm.use_sample_packing,
            "epochs": 8,
            "max_steps": 1 if warmup else max(settings.updates),
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": 2,
            "ckpt_interval": 1,
            "eval_before_train": False,
            "eval_interval": -1,
            "resume_mode": "from_path" if settings.resume_path else "latest",
            "resume_path": settings.resume_path,
            "reset_global_step_on_resume": bool(settings.resume_path),
            "logger": "console",
            "project_name": "marin-mismatch-probe",
            "algorithm": {"advantage_estimator": "grpo", "use_kl_loss": False, "use_kl_in_reward": False},
            "policy": {
                "optimizer_config": {"lr": 0.02, "max_grad_norm": 0.0},
                "megatron_config": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 2,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "moe_router_replay": True,
                },
            },
            "mismatch_probe": {
                **mismatch_probe_config(
                    settings, marin_commit=marin_commit, skyrl_commit=skyrl_commit, replay_modes=True
                ),
                "enabled": not warmup,
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "async_engine": True,
            "batched": False,
            "gpu_memory_utilization": 0.35,
            "enable_prefix_caching": settings.cache_mode != "off",
            "require_exact_chat_transport": True,
            "chat_template": {"source": "name", "name_or_path": "qwen2_5_with_generation_tag_simplified"},
            "engine_init_kwargs": {
                "enable_return_routed_experts": True,
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
            },
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": []},
    }
    return yaml.safe_dump(config, sort_keys=False)


def build_arms(
    *,
    arms: tuple[ArmSpec, ...],
    settings: ProbeSettings,
    model_uri: str,
    data_uri: str,
    fixture_version: str,
    runtime_commit: str,
    warmup: bool,
) -> dict[str, ArtifactStep[SkyRLRun]]:
    """Build separate training artifacts for arms sharing model and data inputs."""
    model = ArtifactStep.adopt(
        user_owned_name("models/mismatch-probe-tiny-grug"),
        fixture_version,
        model_uri,
        kind=LevanterCheckpoint,
    )
    data: ArtifactStep[Artifact] = ArtifactStep.adopt(
        user_owned_name("documents/mismatch-probe-tiny-grug"), fixture_version, data_uri
    )
    role_plan = SkyRLRolePlan(
        colocate_all=True,
        policy_num_nodes=1,
        policy_num_gpus_per_node=2,
        num_inference_engines=1,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=2,
        inference_engine_expert_parallel_size=2,
        train_batch_size=4,
        policy_mini_batch_size=4,
        micro_train_batch_size_per_gpu=2,
        n_samples_per_prompt=2,
    )
    topology = SkyRLTopology(num_nodes=1, gpus_per_node=2, gpu_variant="H100", role_plan=role_plan)
    execution = IrisSkyRLExecution(
        cluster=CLUSTER,
        cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
        cpu=16,
        memory="128GB",
        disk="256GB",
        priority="interactive",
        max_retries=1,
        target_cluster=CLUSTER,
        parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
        coordinator_timeout_hours=12,
        wandb_entity="marin-community",
    )
    result = {}
    marin_commit = Provenance.capture().base_commit
    for arm in arms:
        name = user_owned_name(f"checkpoints/mismatch-probe/{arm.name}{'-warmup' if warmup else ''}")
        result[arm.name] = skyrl_step(
            SkyRLSpec(
                name=name,
                version=resolve_version(name, None),
                config_yaml=probe_recipe(
                    arm,
                    settings,
                    warmup=warmup,
                    marin_commit=marin_commit,
                    skyrl_commit=runtime_commit,
                ),
                runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON, commit=runtime_commit),
                model=ArtifactHfModel(
                    step=model, tokenizer_uri=TOKENIZER, tokenizer_revision=TOKENIZER_REVISION, relative_path=""
                ),
                train_data=(ArtifactDataSource(data, relative_path="train.parquet"),),
                validation_data=(ArtifactDataSource(data, relative_path="validation.parquet"),),
                topology=topology,
                retention=SkyRLRetentionPolicy(resume_checkpoint_count=2, temporary_storage_ttl_days=30),
                seed=settings.seed,
            ),
            execution,
            export_hf=False,
        )
    return result


@click.command(help=__doc__)
@click.option("--arm", "arm_names", multiple=True, type=click.Choice(sorted(ARMS)), default=("native-layout",))
@click.option("--model-uri", required=True)
@click.option("--data-uri", required=True)
@click.option("--fixture-version", required=True)
@click.option("--runtime-commit", required=True, help="Full SHA of the reviewed MarinSkyRL probe runtime.")
@click.option("--resume-path")
@click.option("--reuse-probe")
@click.option("--warmup", is_flag=True)
@click.option("--seed", type=int, default=17, show_default=True)
@click.option("--prompt-count", type=int, default=2, show_default=True)
@click.option("--samples-per-prompt", type=int, default=2, show_default=True)
@click.option("--score-after-update", "updates", multiple=True, type=int, default=(0, 1, 2))
@click.option("--keep-fraction", type=float, default=0.5, show_default=True)
@click.option("--rescore-prefix-cache", "cache_mode", type=click.Choice(("off", "on", "both")), default="off")
@rl_build_options
def main(
    arm_names: tuple[str, ...],
    model_uri: str,
    data_uri: str,
    fixture_version: str,
    runtime_commit: str,
    resume_path: str | None,
    reuse_probe: str | None,
    warmup: bool,
    seed: int,
    prompt_count: int,
    samples_per_prompt: int,
    updates: tuple[int, ...],
    keep_fraction: float,
    cache_mode: str,
) -> dict[str, ArtifactStep[SkyRLRun]]:
    settings = ProbeSettings(
        seed=seed,
        prompt_count=prompt_count,
        samples_per_prompt=samples_per_prompt,
        updates=updates,
        keep_fraction=keep_fraction,
        cache_mode=cache_mode,
        reuse_probe=reuse_probe,
        resume_path=resume_path,
    )
    return build_arms(
        arms=tuple(ARMS[name] for name in arm_names),
        settings=settings,
        model_uri=model_uri,
        data_uri=data_uri,
        fixture_version=fixture_version,
        runtime_commit=runtime_commit,
        warmup=warmup,
    )


if __name__ == "__main__":
    main()
