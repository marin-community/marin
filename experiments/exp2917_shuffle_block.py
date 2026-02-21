# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #2917: three-arm shuffle ablation on Nemotron mix at 150M scale.

Compares:
1. Full shuffle
2. Era shuffle
3. Block shuffle

Each arm is launched in a single `executor_main` call.
"""

import dataclasses
import datetime as dt
import os

from levanter.data.text import BlockShuffleConfig
from levanter.store.jagged_array import DEFAULT_CHUNK_SIZE

from experiments.defaults import default_train
from experiments.llama import compute_num_parameters, llama3_tokenizer_vocab_size, llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main

ISSUE_NUMBER = 2917
EXP_TAG = f"exp{ISSUE_NUMBER}"
EXP_NAME = "shuffle_block"

DEFAULT_MODEL_FLOPS_TARGET = int(1e18)
TRAIN_BATCH_SIZE = 512
TRAIN_SEQ_LEN = 1024
LEARNING_RATE = 3e-3
ERA_LENGTH = 1024
BLOCK_WINDOW_BLOCKS = 4
READ_STATS_LOG_EVERY = 50
LIBTPU_SCOPED_VMEM_LIMIT = "--xla_tpu_scoped_vmem_limit_kib=50000"


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _with_env(step: ExecutorStep, env_vars: dict[str, str]) -> ExecutorStep:
    merged = dict(getattr(step.config, "env_vars", None) or {})
    merged.update(env_vars)
    return dataclasses.replace(step, config=dataclasses.replace(step.config, env_vars=merged))


def _build_arm_mix(shuffle_policy: bool | int | BlockShuffleConfig):
    return dataclasses.replace(
        nemotron_mix,
        shuffle=shuffle_policy,
        permutation_type="feistel",
    )


def _build_arm(
    *,
    arm_name: str,
    shuffle_policy: bool | int | BlockShuffleConfig,
    run_prefix: str,
    train_config: SimpleTrainConfig,
    model_config,
) -> ExecutorStep:
    step = default_train(
        name=f"{run_prefix}-{arm_name}",
        tokenized=_build_arm_mix(shuffle_policy),
        model_config=model_config,
        train_config=train_config,
        tags=["ablation", "shuffle-block", EXP_TAG, "150m", "nemotron", arm_name],
        eval_harness_tasks=[],
        use_default_validation=False,
        wandb_group=run_prefix,
        wandb_name=f"{run_prefix}-{arm_name}",
    )

    return _with_env(
        step,
        {
            "LEVANTER_LOG_DATA_READ_STATS_EVERY": str(READ_STATS_LOG_EVERY),
            "LEVANTER_SHUFFLE_ARM": arm_name,
            "LIBTPU_INIT_ARGS": LIBTPU_SCOPED_VMEM_LIMIT,
        },
    )


def main() -> None:
    model_config = dataclasses.replace(llama_150m, max_seq_len=TRAIN_SEQ_LEN)

    model_flops_target = _int_env("SHUFFLE_ABLATION_MODEL_FLOPS_TARGET", DEFAULT_MODEL_FLOPS_TARGET)
    num_model_params = compute_num_parameters(model_config, llama3_tokenizer_vocab_size)
    num_train_tokens = model_flops_target // (6 * num_model_params)
    default_num_steps = max(1, num_train_tokens // (TRAIN_BATCH_SIZE * TRAIN_SEQ_LEN))
    num_train_steps = _int_env("SHUFFLE_ABLATION_NUM_TRAIN_STEPS", default_num_steps)

    run_date = os.environ.get("RUN_DATE", dt.date.today().isoformat())
    run_prefix = f"{EXP_TAG}_{EXP_NAME}_150m_{run_date}"

    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=TRAIN_BATCH_SIZE,
        train_seq_len=TRAIN_SEQ_LEN,
        num_train_steps=num_train_steps,
        learning_rate=LEARNING_RATE,
        lr_schedule="linear",
        decay=0.2,
        warmup=0.1,
        min_lr_ratio=0.1,
        weight_decay=0.1,
        z_loss_weight=1e-4,
        steps_per_eval=500,
        data_seed=42,
    )

    io_block_size = max(1, DEFAULT_CHUNK_SIZE // TRAIN_SEQ_LEN)
    block_shuffle = BlockShuffleConfig(
        io_block_size=io_block_size,
        window_blocks=BLOCK_WINDOW_BLOCKS,
        perm_type="feistel",
    )

    arms: list[tuple[str, bool | int | BlockShuffleConfig]] = [
        ("full_shuffle", True),
        ("era_shuffle", ERA_LENGTH),
        ("block_shuffle", block_shuffle),
    ]

    steps = [
        _build_arm(
            arm_name=arm_name,
            shuffle_policy=shuffle_policy,
            run_prefix=run_prefix,
            train_config=train_config,
            model_config=model_config,
        )
        for arm_name, shuffle_policy in arms
    ]

    description = (
        f"Issue #{ISSUE_NUMBER}: 150M Nemotron shuffle ablation (full vs era vs block) "
        "with TensorStore read-stats instrumentation."
    )
    executor_main(steps=steps, description=description)


if __name__ == "__main__":
    main()
