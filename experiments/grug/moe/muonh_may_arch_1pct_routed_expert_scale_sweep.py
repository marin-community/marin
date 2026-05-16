# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch + 1pct-noclip baseline, ablate the routed-expert output scale.

Built on the current best d512 recipe (``muonh-may-arch-gn-muonh-1pct-noclip``,
d512=3.6427, d768=3.3040). Inside each Block:

    mlp_out = self.mlp(mlp_in)                 # routed top-k expert mixture
    mlp_out = mlp_out * routed_expert_scale    # NEW: up-weight routed branch
    if shared is not None:
        mlp_out = mlp_out + shared(mlp_in)
    x = x + mlp_out

Tests ``routed_expert_scale in {2.0, 3.0}`` at d512 and d768 (4 runs).

Submit on us-east5-a:

    .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait \\
      --zone us-east5-a \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -- python -m experiments.grug.moe.muonh_may_arch_1pct_routed_expert_scale_sweep
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

_SCALES: tuple[float, ...] = (2.0, 3.0)

_WARMUP_FRACTION: float = 0.01
_NUM_EXPERTS: int = 256
_BASELINE_TARGET_STEPS: int = 2**14
_GROUP_NAME: str = "muonh-may-arch-1pct-routed-expert-scale-sweep"


def _format_budget(budget: float) -> str:
    return f"{budget:.2e}".replace("+", "")


def _scale_label(scale: float) -> str:
    """Format scale as e.g. ``2p0`` or ``3p0``."""
    return str(scale).replace(".", "p")


def _format_run_id(scale: float, hidden_dim: int, budget: float, run_suffix: str = "") -> str:
    budget_label = _format_budget(budget)
    normalized_suffix = run_suffix.strip()
    if normalized_suffix and not normalized_suffix.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_suffix may only contain letters, numbers, hyphens, and underscores")
    suffix = f"{normalized_suffix}-" if normalized_suffix else ""
    return f"muonh-may-arch-1pct-routedscale-{_scale_label(scale)}-{suffix}d{hidden_dim}-{budget_label}"


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


def _build_step(hidden_dim: int, budget: float, scale: float, run_suffix: str = "") -> ExecutorStep:
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
        routed_expert_scale=scale,
    )
    optimizer = _muonh_optimizer(base_optimizer)

    run_id = _format_run_id(scale, hidden_dim, budget, run_suffix=run_suffix)
    step_name = f"grug/muonh_may_arch_1pct_routed_expert_scale_sweep/{run_id}"

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
                    "muonh_may_arch_1pct_routed_expert_scale_sweep",
                    f"d{hidden_dim}",
                    f"scale-{_scale_label(scale)}",
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
    steps = [
        _build_step(hidden_dim=dim, budget=budget, scale=scale, run_suffix=_RUN_SUFFIX)
        for dim, budget in _POINTS
        for scale in _SCALES
    ]
    executor_main(
        steps=steps,
        description=(
            f"MoE may_arch + 1pct-noclip + routed-expert output scale "
            f"({len(_SCALES)} scales x {len(_POINTS)} sizes = {len(_SCALES) * len(_POINTS)} runs, "
            f"run_suffix={_RUN_SUFFIX!r})."
        ),
    )
