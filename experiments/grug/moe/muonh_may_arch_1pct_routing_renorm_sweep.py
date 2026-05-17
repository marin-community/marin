# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip + renormalize sigmoid combine-weight sum to X.

Builds on the 1pct-noclip recipe. Selection of the K=2 routed experts is
unchanged (top-K on **biased** logits), but the per-token sigmoid combine
weights on the **unbiased** logits are rescaled so they sum to ``X``
across the K selected experts. The current baseline applies the raw
sigmoid weights (sum varies per token in roughly ``[0, K]``).

Sweep ``X`` ∈ {1, 2, 2.5, 3, 4} at d512 and d768 to learn the right
total magnitude.

10 runs total. Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/examples/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_routing_renorm_sweep
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

_POINTS: tuple[tuple[int, float], ...] = (
    (512, 2.19e17),
    (768, 1.70e18),
)

_X_VALUES: tuple[float, ...] = (1.0, 2.0, 2.5, 3.0, 4.0)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-routing-renorm-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _format_x(x: float) -> str:
    # 1.0 -> "1p0", 2.5 -> "2p5"
    s = f"{x:.1f}"
    return s.replace(".", "p")


def _format_run_id(hidden_dim: int, budget: float, x: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-routing-renorm-x{_format_x(x)}-{suffix}d{hidden_dim}-{budget_label}"


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


def _build_step(hidden_dim: int, budget: float, x: float, run_suffix: str = "") -> ExecutorStep:
    model, base_optimizer, batch_size, num_steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=_BASELINE_TARGET_STEPS,
    )
    model = dataclasses.replace(
        model,
        num_experts=_NUM_EXPERTS,
        partial_key_offset="every_4th",
        use_partial_rope=True,
        last_layer_pko=True,
        router_z_loss_coef=0.0,
        routing_renorm_sum=x,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    run_id = _format_run_id(hidden_dim, budget, x, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_1pct_routing_renorm_sweep/{run_id}"

    return ExecutorStep(
        name=step_name,
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(num_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="marin_moe",
                tags=[
                    "moe",
                    "muonh_may_arch_1pct_routing_renorm_sweep",
                    f"d{hidden_dim}",
                    f"x{_format_x(x)}",
                ],
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
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
            enable_cross_region_ckpt_read=True,
        ),
    )


_RUN_SUFFIX: str = "v1"


if __name__ == "__main__":
    steps = [_build_step(hidden_dim=d, budget=c, x=x, run_suffix=_RUN_SUFFIX) for d, c in _POINTS for x in _X_VALUES]
    executor_main(
        steps=steps,
        description=(
            "MoE may_arch + 1pct-noclip + sigmoid-combine renormalized to fixed sum X "
            f"(X ∈ {list(_X_VALUES)}, d512/d768, run_suffix={_RUN_SUFFIX!r})."
        ),
    )
