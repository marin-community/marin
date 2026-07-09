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
from levanter.grug.attention import GrugAttentionImplementation
from levanter.optim.config import AdamConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.processing.tokenize.data_configs import with_pack
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import marin_prefix, marin_temp_bucket

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    GrugMoeLaunchConfig,
    env_int,
    run_grug_moe_trial,
    slimpajama_6b_dataset,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer

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

# Subdirectory the canary writes per-run output dirs into, so they stay out of
# the storage root. On R2 these land under the TTL temp bucket (lifecycle-managed
# cleanup); on GCS they stay under MARIN_PREFIX.
CANARY_OUTPUT_SUBDIR = "canary"

# TTL for R2 canary outputs. Lifecycle rules on the bucket delete them after this
# many days; must be one of config/marin.yaml (data.temp.ttl_days).
CANARY_OUTPUT_TTL_DAYS = 7

# Nemotron CC mixture weights: the corpus's TiB proportions, plus starcoder and
# proof-pile at their published weights. Policy lives here, in the experiment.
_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055


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
        # Keep wall-clock bounded via a fixed token budget: tokens = batch_size *
        # max_seq_len * steps. At 250M tokens with batch 128 and the heuristic
        # model's max_seq_len=8192 this is ~238 steps (the regression gate's
        # CANARY_MIN_STEPS floor is set accordingly).
        target_tokens = env_int("CANARY_TARGET_TOKENS", 250_000_000)
        name = "canary-ferry-moe"
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

        nem = nemotron_datasets(tokenizer=llama3_tokenizer)
        train = {nem[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
        train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
        train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
        validation = [
            *paloma_datasets(tokenizer=llama3_tokenizer).values(),
            *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
        ]
        deps = (*train, *validation)

        def build_data(ctx: StepContext):
            return mixture(ctx, train, validation=validation)

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

        slimpajama = slimpajama_6b_dataset()
        deps = (slimpajama,)

        def build_data(ctx: StepContext):
            data = mixture(ctx, {slimpajama: 1.0})
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
    # On R2 (the CoreWeave canary), the per-run training output goes to the
    # TTL temp bucket so bucket lifecycle rules sweep it after the TTL. R2 is a
    # single, region-stable bucket, so the absolute pin resolves identically in
    # the job and in validate_canary_metrics.py (which imports this same build()).
    # The GCS canary keeps outputs under MARIN_PREFIX: it lands in a variable
    # region and validate relies on mirror:// to find them across buckets, which
    # an absolute pin would defeat. The SlimPajama tokenize cache is a separate
    # dependency, so it stays under MARIN_PREFIX either way.
    override_output_path = (
        marin_temp_bucket(ttl_days=CANARY_OUTPUT_TTL_DAYS, prefix=step_name)
        if marin_prefix().startswith("s3://")
        else None
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
            eval=eval_config,
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
