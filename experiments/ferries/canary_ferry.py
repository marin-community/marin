# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary ferry: Grug MoE daily pretraining canary.

Supports TPU (v5p-8, Nemotron, ~1B tokens) and GPU (8x H100, SlimPajama, ~50 steps).
Config is driven by env vars set in the GH Actions workflow env: block and forwarded
to the Iris container. workflow_dispatch inputs override CANARY_TARGET_TOKENS.

    CANARY_ACCELERATOR   tpu | gpu
    CANARY_BATCH_SIZE    per-device batch size
    CANARY_CACHE_COPY_MAX_WORKERS gpu-only cache-copy worker cap
    CANARY_TARGET_TOKENS total training tokens
    RUN_ID               unique run identifier
"""

import dataclasses
import datetime
import os

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data.text import BlockShuffleConfig, TextLmDatasetFormat
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, this_output_path, versioned, executor_main
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.defaults import default_tokenize
from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
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


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


def _build_step_from_env() -> ExecutorStep:
    accelerator = os.environ.get("CANARY_ACCELERATOR", "tpu")
    if accelerator not in ("tpu", "gpu"):
        raise ValueError(f"Unknown CANARY_ACCELERATOR={accelerator!r}, expected 'tpu' or 'gpu'")

    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")

    if accelerator == "tpu":
        batch_size = _env_int("CANARY_BATCH_SIZE", 512)
        target_tokens = _env_int("CANARY_TARGET_TOKENS", 1_000_000_000)
        name = "canary-ferry-moe"
        data = NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
        resources = ResourceConfig.with_tpu("v5p-8")
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
        multi_host = os.environ.get("CANARY_MULTI_HOST", "").lower() in ("1", "true")
        batch_size = _env_int("CANARY_BATCH_SIZE", 32)
        cache_copy_max_workers = _env_int("CANARY_CACHE_COPY_MAX_WORKERS", 12)
        target_tokens = _env_int("CANARY_TARGET_TOKENS", batch_size * GRUG_MOE_TRIAL_MODEL.max_seq_len * 50)
        # SlimPajama-6B with block-shuffle — small dataset, re-tokenized on first run.
        tokenize_step = default_tokenize(
            name="slimpajama-6b-cw",
            dataset="DKYoon/SlimPajama-6B",
            tokenizer=llama3_tokenizer,
            format=TextLmDatasetFormat(),
        )
        tokenize_step = dataclasses.replace(
            tokenize_step,
            config=dataclasses.replace(
                tokenize_step.config,
                # SlimPajama-6B tokenization OOMs at the default 10g worker_resources.
                worker_resources=ResourceConfig(ram="64g", disk="64g"),
                cache_copy_max_workers=cache_copy_max_workers,
            ),
        )
        data = lm_data_config(
            training_set=tokenize_step,
            shuffle=BlockShuffleConfig(io_block_size=256, window_blocks=256, perm_type="feistel"),
        )
        if multi_host:
            name = "canary-ferry-cw-multihost"
            resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g", replicas=2)
            wandb_group = "canary-ferry-moe-gpu-multihost"
            wandb_tags = ["canary", "ferry", "grug", "moe", "gpu", "multihost"]
        else:
            name = "canary-ferry-cw"
            resources = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g")
            wandb_group = "canary-ferry-moe-gpu"
            wandb_tags = ["canary", "ferry", "grug", "moe", "gpu"]
        eval_config = None

    num_steps = target_tokens // (batch_size * GRUG_MOE_TRIAL_MODEL.max_seq_len)
    if num_steps <= 0:
        raise ValueError(
            f"CANARY_TARGET_TOKENS={target_tokens} too small for batch_size={batch_size} "
            f"x seq_len={GRUG_MOE_TRIAL_MODEL.max_seq_len} -- would produce 0 steps"
        )

    return ExecutorStep(
        name=f"{name}-{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(GRUG_MOE_TRIAL_MODEL),
            data=data,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(resources),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=wandb_tags,
                group=wandb_group,
                name=None,
                replicate_path=this_output_path(),
            ),
            optimizer=versioned(CANARY_OPTIMIZER),
            grug_trainer=versioned(CANARY_TRAINER),
            eval=versioned(eval_config) if eval_config is not None else None,
            profiler=ProfilerConfig(enabled=True),
        ),
    )


canary_moe_step = _build_step_from_env()


def main():
    executor_main(steps=[canary_moe_step])


if __name__ == "__main__":
    main()
