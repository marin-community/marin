# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequence length sweep: 1k, 2k, 8k vs baseline 4k at fixed step count."""


from fray.v2.types import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

SEQ_LENS: list[int] = [1024, 2048, 8192]


def _make_step(seq_len: int, dim: int, budget: float, batch_override: int | None = None) -> ExecutorStep:
    _, _, _, baseline_steps = build_from_heuristic(budget=budget, hidden_dim=dim, seq_len=4096)
    model, optimizer, batch, _ = build_from_heuristic(budget=budget, hidden_dim=dim, seq_len=seq_len)
    if batch_override is not None:
        batch = batch_override
    sl_tag = f"sl{seq_len // 1024}k"
    run_id = f"seqlen-{sl_tag}-d{dim}-{budget:.2e}"
    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            enable_cross_region_ckpt_read=True,
            steps=versioned(baseline_steps),
            batch_size=versioned(batch),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="dial_moe",
                tags=["seqlen", sl_tag, f"d={dim}"],
                group="seqlen",
                name=run_id,
            ),
            optimizer=versioned(optimizer),
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


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    # for seq_len in SEQ_LENS:
    #     for dim, budget in GATE1_CONFIGS:
    #         steps.append(_make_step(seq_len, dim, budget))
    # Re-run d512 8k with batch_size=16 to keep tokens_per_batch=131k
    steps.append(_make_step(8192, 512, 2.19e17, batch_override=16))
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(steps=all_steps, description="Sequence length sweep: 1k, 2k, 8k vs baseline 4k.")
