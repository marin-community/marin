# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""50-step TPS test: 1pct-noclip baseline + unused aurorah branch in multi_transform.

The 1pct-noclip baseline (``GNMuonHConfig``) lands in the ~332K TPS
"slow" cluster at d512. Five recent variants all using
``AuroraTargetConfig`` (even with all aurora flags False, e.g.
``split-wupgate``) land at ~352-357K TPS — a +7% MFU delta with no
quality change.

The only code difference between the two configs (when all aurora
flags are False) is that ``AuroraTargetConfig`` registers an unused
``"aurorah"`` branch in ``optax.multi_transform``. This launcher tests
the hypothesis that the branch-registration *itself* — through XLA
graph shape / JIT cache key — is what causes the speedup.

We modify ``GNMuonHConfig`` to register a no-op ``"aurorah_unused"``
branch and run a 50-step d512 trial. If TPS lands at ~355K, the
hypothesis is confirmed and the baseline recipe can be retrofitted for
free MFU. If it stays at ~332K, the difference is elsewhere
(e.g. specific aurora computation, sharding spec, etc.).

Submit on us-east5-a (same zone as the original fast-cluster runs):

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_aurorah_branch_test
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig, GrugMoeMuonHMayArchGNMuonHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_BUDGET: float = 2.19e17  # d512 1pct-noclip budget (irrelevant at 50 steps)
_HIDDEN_DIM: int = 512
_NUM_STEPS_OVERRIDE: int = 50

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14  # used by heuristic for batch sizing only
_GROUP_NAME: str = "muonh-may-arch-1pct-aurorah-branch-test"
_RUN_SUFFIX: str = "v1"
_RUN_ID: str = f"muonh-may-arch-1pct-aurorah-branch-test-{_RUN_SUFFIX}-d{_HIDDEN_DIM}"


def _muonh_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeMuonHMayArchGNMuonHConfig:
    return GrugMoeMuonHMayArchGNMuonHConfig(
        learning_rate=base_optimizer.learning_rate,
        adam_lr=base_optimizer.adam_lr,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=_WARMUP_FRACTION,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=None,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
    )


def _build_step() -> ExecutorStep:
    model, base_optimizer, batch_size, _num_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    return ExecutorStep(
        name=f"grug/muonh_may_arch_1pct_aurorah_branch_test/{_RUN_ID}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=_RUN_ID,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(_NUM_STEPS_OVERRIDE),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=["moe", "muonh_may_arch_1pct_aurorah_branch_test", f"d{_HIDDEN_DIM}", "50step"],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=0.0,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=10000,  # don't eval during this 50-step TPS probe
                    max_eval_batches=8,
                    eval_current=False,
                    eval_ema=False,
                )
            ),
            enable_cross_region_ckpt_read=False,
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_step()],
        description=(
            f"50-step TPS test: 1pct-noclip baseline + unused aurorah branch (d{_HIDDEN_DIM}, "
            f"run_suffix={_RUN_SUFFIX!r})."
        ),
    )
