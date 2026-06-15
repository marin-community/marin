# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary ferry: Grug MoE daily pretraining canary.

Supports TPU (v5p-8, Nemotron, ~0.25B tokens) and GPU (8x H100, SlimPajama, ~50 steps).
Config is driven by env vars set in the GH Actions workflow env: block and forwarded
to the Iris container. workflow_dispatch inputs override CANARY_TARGET_TOKENS.

    CANARY_ACCELERATOR   tpu | gpu
    CANARY_ATTENTION_IMPLEMENTATION gpu-only attention backend, e.g. gpu_fa4_cute
    CANARY_TPU_TYPE      tpu-only comma-separated slice types, primary first (default v5p-8,v4-8)
    CANARY_BATCH_SIZE    per-device batch size
    CANARY_CACHE_COPY_MAX_WORKERS gpu-only cache-copy worker cap
    CANARY_GPU_TYPE      gpu-only accelerator type, e.g. H100, GH200, B200
    CANARY_GPU_COUNT     gpu-only accelerator count per replica
    CANARY_GPU_REPLICAS  gpu-only replica count
    CANARY_HIDDEN_DIM    gpu-only model hidden dim; scales the MoE via the heuristic
                         (1024 trial default -> 2048 -> 3072 -> 4096)
    CANARY_PROFILER_ENABLED true | false
    CANARY_PROFILER_NUM_STEPS profiler duration in steps
    CANARY_PROFILER_START_STEP profiler start step
    CANARY_STEPS         explicit training step count; overrides CANARY_TARGET_TOKENS
    CANARY_CACHE_COPY_MAX_WORKERS gpu-only cache-copy worker cap
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
from levanter.data.text import DatasetComponent
from levanter.grug.attention import GrugAttentionImplementation
from levanter.optim import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    env_int,
    run_grug_moe_trial,
    slimpajama_6b_data,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

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
_GPU_FA4_CUTE_ATTENTION: GrugAttentionImplementation = "gpu_fa4_cute"
_GPU_FA4_THD_ATTENTION: GrugAttentionImplementation = "gpu_fa4_thd"
_GPU_ATTENTION_IMPLEMENTATIONS: tuple[GrugAttentionImplementation, ...] = (
    "reference",
    _GPU_FA4_CUTE_ATTENTION,
    _GPU_FA4_THD_ATTENTION,
)

# Compute budget passed to the heuristic when CANARY_HIDDEN_DIM scales the model.
# Only the model *shape* (from hidden_dim) is used here; the budget-derived batch
# size, step count, and optimizer are all overridden by CANARY_* settings below.
_HEURISTIC_BUDGET = 1e18

# Canary MoE hidden dim. Deliberately smaller than the d1024 trial model so the
# canary is a *representative* MoE that fits comfortably rather than one sized to
# the HBM/VMEM ceiling. The binding constraint is the MoE grouped-matmul (gmm)
# Pallas kernel's 16M VMEM scratchpad: it holds a double-buffered per-expert
# weight window of shape [1, hidden_dim, 2*intermediate_dim]. Eval runs the gmm in
# float32 (the eval loss fn does not cast to the bf16 compute dtype), so that
# window is twice the size of the train step's. At d1024 the f32 window is 8.4M
# and the whole kernel needs 16.44M -- 452K over budget, which crashed the first
# eval deterministically. At d768 the f32 window is 4.7M, leaving the eval gmm a
# comfortable VMEM margin on both v5p and v4 (VMEM is a fixed 16M on both). 768
# stays divisible by the heuristic's hidden_head_ratio (128).
_CANARY_TPU_HIDDEN_DIM = 768

# Subdirectory of MARIN_PREFIX the canary writes per-run output dirs into, so
# they stay out of the root. scripts/canary/prune_canary_outputs.py imports this
# to sweep the same subdir.
CANARY_OUTPUT_SUBDIR = "canary"


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    return raw.lower() in ("1", "true")


# Primary first; the run lands on whichever pool has capacity. v5p has only two
# zones (us-central1-a, us-east5-a) and is preemptible-only, so a single-pool
# stockout otherwise strands the canary indefinitely. v4-8 is a topology-compatible
# fallback (both are single-VM 4-chip slices, so training shape is unchanged) and
# adds us-central2-b plus the v4 reserved pool, which is not subject to preemptible
# capacity churn. v4 has only ~1/3 the per-chip HBM of v5p (~30.75 vs 95 GiB), so the
# canary's batch size is sized to fit v4 (see the TPU branch below); keep any new
# entry's per-chip HBM at or above v4's. All entries must share vm_count and
# chips_per_vm (ResourceConfig enforces this).
_DEFAULT_CANARY_TPU_TYPES = ("v5p-8", "v4-8")


def _tpu_types_from_env() -> list[str]:
    raw = os.environ.get("CANARY_TPU_TYPE", "")
    types = [t.strip() for t in raw.split(",") if t.strip()]
    return types or list(_DEFAULT_CANARY_TPU_TYPES)


def _build_step_from_env() -> ExecutorStep:
    accelerator = os.environ.get("CANARY_ACCELERATOR", "tpu")
    if accelerator not in ("tpu", "gpu"):
        raise ValueError(f"Unknown CANARY_ACCELERATOR={accelerator!r}, expected 'tpu' or 'gpu'")

    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    if accelerator == "tpu":
        # Representative MoE shape sized to fit the f32 eval gmm in VMEM (see
        # _CANARY_TPU_HIDDEN_DIM). Only the model *shape* is taken from the
        # heuristic; batch size, step count, and optimizer are set below.
        model, _, _, _ = build_from_heuristic(budget=_HEURISTIC_BUDGET, hidden_dim=_CANARY_TPU_HIDDEN_DIM)
        # Global batch is sized to fit the smallest pool in the fallback list. The
        # dominant train_step HBM allocation is the MoE expert grouped-matmul over
        # batch_size * max_seq_len tokens, so per-device HBM scales with the global
        # batch. 128 leaves comfortable headroom on the v4-8 fallback (~30.75 GiB
        # usable, ~1/3 of v5p) while staying valid on v5p, giving one config across
        # both pools. With the smaller representative model above this is well
        # within v4's budget.
        batch_size = env_int("CANARY_BATCH_SIZE", 128)
        # Hold the step count steady (~476) so wall-clock stays bounded after the
        # batch shrink: tokens = batch_size * max_seq_len * steps.
        target_tokens = env_int("CANARY_TARGET_TOKENS", 250_000_000)
        name = "canary-ferry-moe"
        data = NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
        resources = ResourceConfig.with_tpu(_tpu_types_from_env())
        eval_config: GrugEvalConfig | None = GrugEvalConfig(
            eval_batch_size=batch_size,
            steps_per_eval=240,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        )
        wandb_group = "canary-ferry-moe"
        wandb_tags = ["canary", "ferry", "grug", "moe"]
    else:
        gpu_type = os.environ.get("CANARY_GPU_TYPE", "H100")
        gpu_count = env_int("CANARY_GPU_COUNT", 8)
        gpu_replicas = env_int("CANARY_GPU_REPLICAS", 1)

        # Model-size knob: scale the MoE by hidden_dim; the heuristic auto-scales
        # depth/heads/intermediate. Defaults to the ~1.1B trial model. Step up
        # CANARY_HIDDEN_DIM (1024 -> 2048 -> 3072 -> 4096) to grow the model across
        # more H100 nodes (pair with CANARY_GPU_REPLICAS).
        hidden_dim = env_int("CANARY_HIDDEN_DIM", GRUG_MOE_TRIAL_MODEL.hidden_dim)
        if hidden_dim == GRUG_MOE_TRIAL_MODEL.hidden_dim:
            model = GRUG_MOE_TRIAL_MODEL
        else:
            model, _, _, _ = build_from_heuristic(budget=_HEURISTIC_BUDGET, hidden_dim=hidden_dim)

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

        data = slimpajama_6b_data()
        if attention_implementation == _GPU_FA4_THD_ATTENTION:
            data = dataclasses.replace(
                data,
                components={
                    name: (
                        dataclasses.replace(component, pack=1) if isinstance(component, DatasetComponent) else component
                    )
                    for name, component in data.components.items()
                },
            )
        resources = ResourceConfig.with_gpu(
            gpu_type,
            count=gpu_count,
            cpu=32,
            ram="256g",
            disk="256g",
            replicas=gpu_replicas,
        )
        attention_tag = attention_implementation.removeprefix("gpu_")
        name = f"canary-ferry-cw-{gpu_type.lower()}x{gpu_count}-r{gpu_replicas}-d{hidden_dim}-{attention_tag}"
        wandb_group = f"canary-ferry-moe-gpu-{gpu_type.lower()}-r{gpu_replicas}-{attention_tag}"
        wandb_tags = ["canary", "ferry", "grug", "moe", "gpu", gpu_type.lower(), f"d{hidden_dim}", attention_tag]
        eval_config = None

    num_steps = env_int("CANARY_STEPS", target_tokens // (batch_size * model.max_seq_len))
    if num_steps <= 0:
        raise ValueError(
            f"CANARY_STEPS={num_steps} invalid; set CANARY_STEPS or CANARY_TARGET_TOKENS high enough for "
            f"batch_size={batch_size} x seq_len={model.max_seq_len}"
        )
    if os.environ.get("CANARY_TRACKER", "wandb").lower() == "json_logger":
        tracker = JsonLoggerConfig(logger_name=os.environ.get("CANARY_JSON_LOGGER", "canary_ferry.metrics"))
    else:
        tracker = WandbConfig(
            entity=os.environ.get("WANDB_ENTITY") or None,
            project=os.environ.get("WANDB_PROJECT", "marin"),
            tags=wandb_tags,
            group=wandb_group,
            mode=os.environ.get("CANARY_WANDB_MODE") or os.environ.get("WANDB_MODE") or None,
            name=None,
            replicate_path=this_output_path(),
        )

    profiler_enabled = _env_bool("CANARY_PROFILER_ENABLED", True)
    profiler_start_step = env_int("CANARY_PROFILER_START_STEP", 5)
    profiler_num_steps = env_int("CANARY_PROFILER_NUM_STEPS", 25)

    return ExecutorStep(
        name=f"{CANARY_OUTPUT_SUBDIR}/{name}-{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=data,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(resources),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=tracker,
            optimizer=versioned(CANARY_OPTIMIZER),
            grug_trainer=versioned(CANARY_TRAINER),
            eval=versioned(eval_config) if eval_config is not None else None,
            profiler=ProfilerConfig(
                enabled=profiler_enabled,
                start_step=profiler_start_step,
                num_steps=profiler_num_steps,
            ),
        ),
    )


canary_moe_step = _build_step_from_env()


def main():
    executor_main(steps=[canary_moe_step])


if __name__ == "__main__":
    main()
