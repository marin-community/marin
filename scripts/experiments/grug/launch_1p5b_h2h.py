# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch a ~1.5B grug trial for base or array-stacked variants via executor."""

from __future__ import annotations

import argparse
from datetime import datetime

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.array_stacked.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION as ARRAY_DATA,
)
from experiments.grug.array_stacked.launch import (
    GrugArrayStackedLaunchConfig,
    run_grug_array_stacked_trial,
)
from experiments.grug.array_stacked.model import GrugModelConfig as ArrayModelConfig
from experiments.grug.array_stacked.train import GrugEvalConfig as ArrayEvalConfig
from experiments.grug.array_stacked.train import GrugTrainerConfig as ArrayTrainerConfig
from experiments.grug.base.launch import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION as BASE_DATA
from experiments.grug.base.launch import GrugBaseLaunchConfig, run_grug_base_trial
from experiments.grug.base.model import GrugModelConfig as BaseModelConfig
from experiments.grug.base.train import GrugEvalConfig as BaseEvalConfig
from experiments.grug.base.train import GrugTrainerConfig as BaseTrainerConfig


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _default_run_id(variant: str) -> str:
    return f"grug-{variant}-1p5b-h2h-{_timestamp()}"


def _optimizer() -> AdamConfig:
    return AdamConfig(
        learning_rate=3e-3,
        weight_decay=0.1,
        lr_schedule="cosine",
        decay=0.2,
        min_lr_ratio=0.1,
        warmup=1000,
    )


def _base_model() -> BaseModelConfig:
    # ~1.5B params target:
    # - double depth vs 130M template (6 -> 12)
    # - widen hidden/intermediate (512/1792 -> 2304/7680)
    return BaseModelConfig(
        vocab_size=128_256,
        hidden_dim=2304,
        intermediate_dim=7680,
        num_layers=12,
        num_heads=18,
        num_kv_heads=18,
        max_seq_len=4096,
        head_dim=None,  # resolved to 128 (= hidden_dim / num_heads)
    )


def _array_model() -> ArrayModelConfig:
    return ArrayModelConfig(
        vocab_size=128_256,
        hidden_dim=2304,
        intermediate_dim=7680,
        num_layers=12,
        num_heads=18,
        num_kv_heads=18,
        max_seq_len=4096,
        head_dim=None,
    )


def _build_base_step(*, run_id: str, tpu: str, steps: int, batch_size: int) -> ExecutorStep:
    output_path = this_output_path()
    return ExecutorStep(
        name=f"grug/h2h-1p5b/base-{run_id}",
        fn=run_grug_base_trial,
        config=GrugBaseLaunchConfig(
            model=versioned(_base_model()),
            data=BASE_DATA,
            output_path=output_path,
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(tpu)),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "h2h", "1p5b", "base"],
                group=f"{run_id}-group",
                name=run_id,
                replicate_path=output_path,
            ),
            optimizer=versioned(_optimizer()),
            grug_trainer=versioned(
                BaseTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                BaseEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


def _build_array_step(*, run_id: str, tpu: str, steps: int, batch_size: int) -> ExecutorStep:
    output_path = this_output_path()
    return ExecutorStep(
        name=f"grug/h2h-1p5b/array-{run_id}",
        fn=run_grug_array_stacked_trial,
        config=GrugArrayStackedLaunchConfig(
            model=versioned(_array_model()),
            data=ARRAY_DATA,
            output_path=output_path,
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu(tpu)),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "h2h", "1p5b", "array-stacked"],
                group=f"{run_id}-group",
                name=run_id,
                replicate_path=output_path,
            ),
            optimizer=versioned(_optimizer()),
            grug_trainer=versioned(
                ArrayTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                ArrayEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grug 1.5B head-to-head variant.")
    parser.add_argument("--variant", choices=["base", "array"], required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tpu", default="v5p-8")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prefix", default=None, help="Optional executor prefix override.")
    args = parser.parse_args()

    if args.variant == "base":
        run_id = args.run_id or _default_run_id("base")
        step = _build_base_step(run_id=run_id, tpu=args.tpu, steps=args.steps, batch_size=args.batch_size)
        executor_main.__wrapped__(
            ExecutorMainConfig(prefix=args.prefix),
            steps=[step],
            description="Grug 1.5B base head-to-head run.",
        )
        return

    run_id = args.run_id or _default_run_id("array")
    step = _build_array_step(run_id=run_id, tpu=args.tpu, steps=args.steps, batch_size=args.batch_size)
    executor_main.__wrapped__(
        ExecutorMainConfig(prefix=args.prefix),
        steps=[step],
        description="Grug 1.5B array-stacked head-to-head run.",
    )


if __name__ == "__main__":
    main()
