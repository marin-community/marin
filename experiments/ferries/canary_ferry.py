# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary ferry: Grug MoE daily accelerator smoke canary.

Supports TPU (v6e-4, FineWeb-Edu 10M, ~0.25B tokens) and GPU (8x H100, FineWeb-Edu 10M, ~50 steps).
Config is driven by env vars set in the GH Actions workflow env: block and forwarded
to the Iris container. workflow_dispatch inputs override CANARY_TARGET_TOKENS.

    CANARY_ACCELERATOR   tpu | gpu
    CANARY_ATTENTION_IMPLEMENTATION gpu-only attention backend, e.g. gpu_fa4_cute
    CANARY_TPU_TYPE      tpu-only comma-separated slice types, primary first
                         (default v6e-4)
    CANARY_BATCH_SIZE    per-device batch size
    CANARY_CACHE_COPY_MAX_WORKERS gpu-only cache-copy worker cap
    CANARY_GPU_TYPE      gpu-only accelerator type, e.g. H100, GH200, B200
    CANARY_GPU_COUNT     gpu-only accelerator count per replica
    CANARY_GPU_REPLICAS  gpu-only replica count
    CANARY_PROFILER_ENABLED true | false
    CANARY_PROFILER_NUM_STEPS profiler duration in steps
    CANARY_PROFILER_START_STEP profiler start step
    CANARY_STEPS         explicit training step count; overrides CANARY_TARGET_TOKENS
    CANARY_TARGET_TOKENS total training tokens
    CANARY_TRACKER       wandb | json_logger
    RUN_ID               unique run identifier
"""

import dataclasses
import datetime
import os
from typing import cast

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.grug.attention import GrugAttentionImplementation
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.processing.tokenize.data_configs import with_pack
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path
from rigging.filesystem.cluster_config import marin_prefix, marin_temp_bucket

from experiments.datasets.prebuilt_caches import fineweb_edu_10M_dataset
from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    env_int,
    run_grug_moe_trial,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugTrainerConfig

CANARY_OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=48,
)

CANARY_TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

# This fixed 9.9M-parameter MoE exercises routing, attention, and optimizer state
# without turning an accelerator smoke test into a multi-billion-parameter run.
CANARY_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=32,
    intermediate_dim=128,
    shared_expert_intermediate_dim=128,
    num_experts=32,
    num_experts_per_token=2,
    num_layers=4,
    num_heads=1,
    num_kv_heads=1,
    head_dim=32,
    max_seq_len=8192,
    sliding_window=2048,
)

_GPU_FA4_CUTE_ATTENTION: GrugAttentionImplementation = "gpu_fa4_cute"
_GPU_FA4_THD_ATTENTION: GrugAttentionImplementation = "gpu_fa4_thd"
_GPU_ATTENTION_IMPLEMENTATIONS: tuple[GrugAttentionImplementation, ...] = (
    "reference",
    _GPU_FA4_CUTE_ATTENTION,
    _GPU_FA4_THD_ATTENTION,
)

CANARY_OUTPUT_SUBDIR = "canary"

CANARY_OUTPUT_TTL_DAYS = 1


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    return raw.lower() in ("1", "true")


# The scheduled canary exercises v6e while v5p capacity is unavailable. Manual
# dispatches may still provide comma-separated alternatives with the same VM and
# chip topology (enforced by ResourceConfig).
_DEFAULT_CANARY_TPU_TYPES = ("v6e-4",)


def _tpu_types_from_env() -> list[str]:
    raw = os.environ.get("CANARY_TPU_TYPE", "")
    types = [t.strip() for t in raw.split(",") if t.strip()]
    return types or list(_DEFAULT_CANARY_TPU_TYPES)


def build() -> ArtifactStep[LevanterCheckpoint]:
    """The Grug MoE canary as a lazy checkpoint, configured from the env.

    The data mixture and the WandB ``replicate_path`` depend on the run context, so
    they are assembled inside ``build_config``; everything else is resolved from the
    env at call time. The TPU/GPU slice is a run-arg, so it never bears on identity.
    """
    accelerator = os.environ.get("CANARY_ACCELERATOR", "tpu")
    if accelerator not in ("tpu", "gpu"):
        raise ValueError(f"Unknown CANARY_ACCELERATOR={accelerator!r}, expected 'tpu' or 'gpu'")

    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")

    if accelerator == "tpu":
        model = CANARY_MODEL
        # The dominant train_step HBM allocation is the MoE expert grouped-matmul
        # over batch_size * max_seq_len tokens, so per-device HBM scales with the
        # global batch. The default 128 fits the v6e-4 canary profile.
        batch_size = env_int("CANARY_BATCH_SIZE", 128)
        # Keep wall-clock bounded via a fixed token budget: tokens = batch_size *
        # max_seq_len * steps. At 250M tokens with batch 128 and the heuristic
        # model's max_seq_len=8192 this is ~238 steps (the regression gate's
        # CANARY_MIN_STEPS floor is set accordingly).
        target_tokens = env_int("CANARY_TARGET_TOKENS", 250_000_000)
        name = "canary-ferry-moe"
        resources = ResourceConfig.with_tpu(_tpu_types_from_env())
        wandb_group = "canary-ferry-moe"
        wandb_tags = ["canary", "ferry", "grug", "moe"]

        # This hardware smoke canary uses a ~21 MB prebuilt cache; restart sampling
        # repeats it for the configured token budget.
        # The launcher and TPU may be in different regions, so make the training
        # process localize that small cache through the mirrored filesystem.
        train = fineweb_edu_10M_dataset()
        deps = (train,)

        def build_data(ctx: StepContext):
            data = mixture(ctx, {train: 1.0})
            component = data.components[train.name]
            assert isinstance(component, DatasetComponent)
            return dataclasses.replace(
                data,
                components={train.name: dataclasses.replace(component, cache_dir=train.path("mirror://"))},
            )

    else:
        gpu_type = os.environ.get("CANARY_GPU_TYPE", "H100")
        gpu_count = env_int("CANARY_GPU_COUNT", 8)
        gpu_replicas = env_int("CANARY_GPU_REPLICAS", 1)
        model = CANARY_MODEL

        attention_implementation = os.environ.get("CANARY_ATTENTION_IMPLEMENTATION", _GPU_FA4_CUTE_ATTENTION)
        if attention_implementation not in _GPU_ATTENTION_IMPLEMENTATIONS:
            raise ValueError(
                f"Unknown CANARY_ATTENTION_IMPLEMENTATION={attention_implementation!r}, expected one of "
                f"{_GPU_ATTENTION_IMPLEMENTATIONS}"
            )
        attention_implementation = cast(GrugAttentionImplementation, attention_implementation)
        model = dataclasses.replace(
            model,
            attention_implementation=attention_implementation,
            # The THD backend only handles full causal windows. Setting the model
            # window to 2x seq_len makes Grug's short-window mask a full window.
            sliding_window=(
                model.max_seq_len * 2 if attention_implementation == _GPU_FA4_THD_ATTENTION else model.sliding_window
            ),
        )

        batch_size = env_int("CANARY_BATCH_SIZE", 32)
        target_tokens = env_int("CANARY_TARGET_TOKENS", batch_size * model.max_seq_len * 50)

        resources = ResourceConfig.with_gpu(
            gpu_type,
            count=gpu_count,
            cpu=32,
            ram="256g",
            disk="256g",
            replicas=gpu_replicas,
        )
        attention_tag = attention_implementation.removeprefix("gpu_")
        name = f"canary-ferry-cw-{gpu_type.lower()}x{gpu_count}-r{gpu_replicas}-d{model.hidden_dim}-{attention_tag}"
        wandb_group = f"canary-ferry-moe-gpu-{gpu_type.lower()}-r{gpu_replicas}-{attention_tag}"
        wandb_tags = [
            "canary",
            "ferry",
            "grug",
            "moe",
            "gpu",
            gpu_type.lower(),
            f"d{model.hidden_dim}",
            attention_tag,
        ]
        train = fineweb_edu_10M_dataset()
        deps = (train,)

        def build_data(ctx: StepContext):
            data = mixture(ctx, {train: 1.0})
            if attention_implementation == _GPU_FA4_THD_ATTENTION:
                # THD attention only handles full causal windows; pack so each example is one.
                data = with_pack(data, 1)
            return data

    num_steps = env_int("CANARY_STEPS", target_tokens // (batch_size * model.max_seq_len))
    if num_steps <= 0:
        raise ValueError(
            f"CANARY_STEPS={num_steps} invalid; set CANARY_STEPS or CANARY_TARGET_TOKENS high enough for "
            f"batch_size={batch_size} x seq_len={model.max_seq_len}"
        )

    use_json_logger = os.environ.get("CANARY_TRACKER", "wandb").lower() == "json_logger"
    json_logger_name = os.environ.get("CANARY_JSON_LOGGER", "canary_ferry.metrics")
    wandb_entity = os.environ.get("WANDB_ENTITY") or None
    wandb_project = os.environ.get("WANDB_PROJECT", "marin")
    wandb_mode = os.environ.get("CANARY_WANDB_MODE") or os.environ.get("WANDB_MODE") or None

    profiler_enabled = _env_bool("CANARY_PROFILER_ENABLED", True)
    profiler_start_step = env_int("CANARY_PROFILER_START_STEP", 5)
    profiler_num_steps = env_int("CANARY_PROFILER_NUM_STEPS", 25)

    step_name = f"{CANARY_OUTPUT_SUBDIR}/{name}-{run_id}"
    override_output_path = marin_temp_bucket(
        ttl_days=CANARY_OUTPUT_TTL_DAYS,
        prefix=step_name,
        source_prefix=marin_prefix(),
    )

    def build_tracker(ctx: StepContext):
        if use_json_logger:
            return JsonLoggerConfig(logger_name=json_logger_name)
        return WandbConfig(
            entity=wandb_entity,
            project=wandb_project,
            tags=wandb_tags,
            group=wandb_group,
            mode=wandb_mode,
            name=None,
            replicate_path=ctx.output_path,
        )

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        return GrugMoeLaunchConfig(
            model=model,
            data=build_data(ctx),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=num_steps,
            batch_size=batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=build_tracker(ctx),
            optimizer=CANARY_OPTIMIZER,
            grug_trainer=CANARY_TRAINER,
            eval=None,
            checkpointer=dataclasses.replace(
                resolve_checkpointer_output_path(
                    CheckpointerConfig(save_interval=None, keep=None),
                    ctx.output_path,
                ),
                temporary_base_path=None,
            ),
            profiler=ProfilerConfig(
                enabled=profiler_enabled,
                start_step=profiler_start_step,
                num_steps=profiler_num_steps,
            ),
        )

    return ArtifactStep(
        name=step_name,
        version="2026.06.28",
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=deps,
        runtime_args={"train_resources": resources},
        override_path=override_output_path,
    )


if __name__ == "__main__":
    StepRunner().run([build().lower()])
