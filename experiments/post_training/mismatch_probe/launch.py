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
from mergedeep import merge

from experiments.post_training.curriculum_rl.launch import PolicySpec

TINY_GRUG_POLICY = PolicySpec(
    label="tiny-grug",
    cluster="cw-us-east-02a",
    tokenizer_uri="Qwen/Qwen2.5-0.5B-Instruct",
    tokenizer_revision="7ae557604adf67be50417f59c2c2f167def9a775",
    model_relative_path="",
    enable_thinking=None,
    task_memory="128GB",
    serve_gpus=2,
)

WARMUP_UPDATES = 3
REPLAY_MODES = ("router_replay", "router_replay_response", "router_replay_filtered")
# Numerics campaign: determinism and batch-layout references, generation-route replay (full and
# response-only), and replay of the prefill re-read's routes (the prefill metric and its placebo).
NUMERICS_MODES = (
    "native_again",
    "router_replay",
    "router_replay_response",
    "repeat_replay",
    "reread_replay",
    "repeat_reread_replay",
)


@dataclass(frozen=True)
class ProbeLayout:
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


PROBE_LAYOUTS = {
    arm.name: arm
    for arm in (
        ProbeLayout("native-layout", use_sample_packing=False),
        ProbeLayout("packed-layout", use_sample_packing=True),
    )
}


def probe_block(
    settings: ProbeSettings,
    *,
    router_replay: bool = True,
    trainer_modes: tuple[str, ...] = REPLAY_MODES,
    capture_layers: tuple[int, ...] = (),
    timing_modes: tuple[str, ...] = (),
) -> dict:
    """Render fixed-token collection settings for a synchronous Megatron recipe.

    Capture layers add the native_capture mode. Sampling is the full distribution at temperature
    one; MarinSkyRL applies the behavior-logprob sampling program to probe runs, which sets
    ``min_tokens`` to zero, so vLLM reports the probability of the distribution the trainer scores.
    """
    modes = (*trainer_modes, "native_capture") if capture_layers else trainer_modes
    return {
        "trainer": {
            "policy": {"megatron_config": {"moe_router_replay": router_replay}},
            "mismatch_probe": {
                "enabled": True,
                "prompts": {"count": settings.prompt_count, "samples_per_prompt": settings.samples_per_prompt},
                "seed": settings.seed,
                "archive_uri": None,
                "reuse_probe": settings.reuse_probe,
                "score_after_updates": list(settings.updates),
                "extra_trainer_modes": list(modes),
                "filtered_replay": {"keep_fraction": settings.keep_fraction},
                "rescore_prefix_cache": settings.cache_mode,
                "reread_again": settings.cache_mode != "on",
                "capture_layers": list(capture_layers),
                "timing_modes": list(timing_modes),
            },
        },
        "generator": {
            "require_exact_chat_transport": True,
            "enable_prefix_caching": settings.cache_mode != "off",
            "engine_init_kwargs": {
                "enable_return_routed_experts": router_replay,
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
    }


def tiny_grug_recipe(
    arm: ProbeLayout,
    settings: ProbeSettings,
    *,
    warmup: bool,
    trainer_modes: tuple[str, ...] = REPLAY_MODES,
    capture_layers: tuple[int, ...] = (),
    timing_modes: tuple[str, ...] = (),
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
            "epochs": WARMUP_UPDATES + max(settings.updates),
            "max_steps": WARMUP_UPDATES if warmup else max(settings.updates),
            "update_epochs_per_batch": 1,
            "micro_forward_batch_size_per_gpu": 2,
            "ckpt_interval": 1,
            "eval_before_train": False,
            "eval_interval": -1,
            "resume_mode": "from_path" if settings.resume_path else "none",
            "resume_path": settings.resume_path,
            "reset_global_step_on_resume": False,
            "logger": "console",
            "project_name": "marin-mismatch-probe",
            "algorithm": {"advantage_estimator": "grpo", "use_kl_loss": False, "use_kl_in_reward": False},
            "policy": {
                "optimizer_config": {"lr": 0.02, "max_grad_norm": 0.0},
                "megatron_config": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                },
            },
        },
        "generator": {
            "backend": "vllm",
            "model_dtype": "bfloat16",
            "run_engines_locally": True,
            "weight_sync_backend": "nccl",
            "gpu_memory_utilization": 0.35,
            "chat_template": {"source": "name", "name_or_path": "qwen2_5_with_generation_tag_simplified"},
        },
        "data": {"kind": "parquet", "train_data": [], "val_data": []},
    }
    probe = probe_block(settings, trainer_modes=trainer_modes, capture_layers=capture_layers, timing_modes=timing_modes)
    probe["trainer"]["mismatch_probe"]["enabled"] = not warmup
    config = merge({}, config, probe)
    return yaml.safe_dump(config, sort_keys=False)


def build_arms(
    *,
    arms: tuple[ProbeLayout, ...],
    settings: ProbeSettings,
    model_uri: str,
    data_uri: str,
    fixture_version: str,
    warmup: bool,
    trainer_modes: tuple[str, ...] = REPLAY_MODES,
    capture_layers: tuple[int, ...] = (),
    timing_modes: tuple[str, ...] = (),
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
        cluster=TINY_GRUG_POLICY.cluster,
        cluster_config=f"lib/iris/config/{TINY_GRUG_POLICY.cluster}.yaml",
        cpu=16,
        memory=TINY_GRUG_POLICY.task_memory,
        disk="256GB",
        priority="interactive",
        max_retries=1,
        target_cluster=TINY_GRUG_POLICY.cluster,
        parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
        coordinator_timeout_hours=12,
        wandb_entity="marin-community",
    )
    result = {}
    for arm in arms:
        name = user_owned_name(f"checkpoints/mismatch-probe/{arm.name}{'-warmup' if warmup else ''}")
        result[arm.name] = skyrl_step(
            SkyRLSpec(
                name=name,
                version=resolve_version(name, None),
                config_yaml=tiny_grug_recipe(
                    arm,
                    settings,
                    warmup=warmup,
                    trainer_modes=trainer_modes,
                    capture_layers=capture_layers,
                    timing_modes=timing_modes,
                ),
                runtime=SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON),
                model=ArtifactHfModel(
                    step=model,
                    tokenizer_uri=TINY_GRUG_POLICY.tokenizer_uri,
                    tokenizer_revision=TINY_GRUG_POLICY.tokenizer_revision,
                    relative_path=TINY_GRUG_POLICY.model_relative_path,
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
@click.option("--arm", "arm_names", multiple=True, type=click.Choice(sorted(PROBE_LAYOUTS)), default=("native-layout",))
@click.option("--model-uri", required=True)
@click.option("--data-uri", required=True)
@click.option("--fixture-version", required=True)
@click.option("--resume-path")
@click.option("--reuse-probe")
@click.option("--warmup", is_flag=True)
@click.option("--seed", type=int, default=17, show_default=True)
@click.option("--prompt-count", type=int, default=2, show_default=True)
@click.option("--samples-per-prompt", type=int, default=2, show_default=True)
@click.option("--score-after-update", "updates", multiple=True, type=int, default=(0, 1, 2))
@click.option("--keep-fraction", type=float, default=0.5, show_default=True)
@click.option("--rescore-prefix-cache", "cache_mode", type=click.Choice(("off", "on", "both")), default="off")
@click.option("--trainer-mode", "trainer_modes", multiple=True, default=REPLAY_MODES, show_default=True)
@click.option("--capture-layer", "capture_layers", multiple=True, type=int)
@click.option("--timing-mode", "timing_modes", multiple=True)
@rl_build_options
def main(
    arm_names: tuple[str, ...],
    model_uri: str,
    data_uri: str,
    fixture_version: str,
    resume_path: str | None,
    reuse_probe: str | None,
    warmup: bool,
    seed: int,
    prompt_count: int,
    samples_per_prompt: int,
    updates: tuple[int, ...],
    keep_fraction: float,
    cache_mode: str,
    trainer_modes: tuple[str, ...],
    capture_layers: tuple[int, ...],
    timing_modes: tuple[str, ...],
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
        arms=tuple(PROBE_LAYOUTS[name] for name in arm_names),
        settings=settings,
        model_uri=model_uri,
        data_uri=data_uri,
        fixture_version=fixture_version,
        warmup=warmup,
        trainer_modes=trainer_modes,
        capture_layers=capture_layers,
        timing_modes=timing_modes,
    )


if __name__ == "__main__":
    main()
