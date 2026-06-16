# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proportional-mix baseline at the swarm's training shape.

A single grug-MoE training run using exactly the swarm-fisher-dsp model,
optimizer, step count, phase split, batch, block size, resources, and tokenizer
— but with both phases pinned to ``PROPORTIONAL_WEIGHTS`` (per-bucket
token-share). That gives us a direct apples-to-apples baseline against any
swarm candidate at the same compute budget.
"""

from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.types import this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.datakit_moe_mix import COMPONENTS, PROPORTIONAL_WEIGHTS, TARGET_BUDGET
from experiments.grug.moe.grug_moe_mix import GrugEvalConfig, GrugMoeLaunchConfig, GrugTrainerConfig, run_grug_moe_mix
from experiments.grug.moe.launch_swarm import (
    _BATCH,
    _BUDGET,
    _EXPERIMENT_BUDGET,
    _HIDDEN_DIM,
    _MODEL,
    _OPTIMIZER,
    _PHASE1_START_STEP,
    _STEPS,
    _SWARM_BLOCK_SIZE,
)
from experiments.marin_models import marin_tokenizer


def _build_step() -> ExecutorStep:
    # Two-element schedule with the same weights — matches the swarm's
    # 80/20 phase structure exactly; just both phases pinned to proportional.
    mixture_schedule: list[tuple[int, dict[str, float]]] = [
        (0, PROPORTIONAL_WEIGHTS),
        (_PHASE1_START_STEP, PROPORTIONAL_WEIGHTS),
    ]

    base_mixture = LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=COMPONENTS,
        train_weights=mixture_schedule,
        auto_build_caches=False,
        mixture_block_size=_SWARM_BLOCK_SIZE,
        target_budget=TARGET_BUDGET,
        experiment_budget=_EXPERIMENT_BUDGET,
    )
    data = add_validation_sets_to_mixture(base_mixture, default_validation_sets(tokenizer=marin_tokenizer))

    slug = f"d{_HIDDEN_DIM}_proportional"
    return ExecutorStep(
        name=f"grug/proportional_baseline_{slug}",
        fn=run_grug_moe_mix,
        config=GrugMoeLaunchConfig(
            model=versioned(_MODEL),
            data=data,
            output_path=this_output_path(),
            run_id=f"proportional_baseline_{slug}",
            resources=versioned(ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=False)),
            steps=versioned(_STEPS),
            batch_size=versioned(_BATCH),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "proportional_baseline", "uscentral2", slug],
                group="proportional_baseline_uscentral2",
                name=None,
            ),
            optimizer=versioned(_OPTIMIZER),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


proportional_baseline_steps: list[ExecutorStep] = [_build_step()]


if __name__ == "__main__":
    executor_main(
        steps=proportional_baseline_steps,
        description=(
            f"Proportional-mix grug-MoE baseline at swarm shape: d={_HIDDEN_DIM}, "
            f"budget={_BUDGET:.2e} FLOPs, steps={_STEPS}, phase split at "
            f"{_PHASE1_START_STEP}/{_STEPS}. PROPORTIONAL_WEIGHTS in both phases."
        ),
    )
