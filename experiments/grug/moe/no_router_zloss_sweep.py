# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove MoE router z-loss ablation.

Two variants:
1. Remove router z-loss only (router_z_loss_coef=0, router stays on Adam)
2. Remove router z-loss + move router to AdamH

GitHub issue: https://github.com/marin-community/marin/issues/TBD
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
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

# (label, remove_router_zloss, router_to_adamh)
CONFIGS: list[tuple[str, bool, bool]] = [
    ("no-router-zloss", True, False),
    ("no-router-zloss-adamh-router", True, True),
]


def _make_adamh_router_optimizer(base_optimizer: GrugMoeAdamHConfig) -> GrugMoeAdamHConfig:
    """Return a copy of the optimizer that routes .router to adamh instead of adam."""
    # We override create_mask by subclassing inline — but that's complex.
    # Simpler: use a flag. But we don't have one on the base branch.
    # Instead, create a subclass here.
    import jax
    from levanter.utils.jax_utils import leaf_key_paths

    class AdamHRouterConfig(GrugMoeAdamHConfig):
        def create_mask(self, params):
            paths = leaf_key_paths(params)

            def mask_fn(param, path):
                path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
                path_lower = path_str.lower()
                if "token_embed" in path_lower:
                    return "adam"
                if "router_bias" in path_lower or "attn_gate" in path_lower:
                    return "adam"
                # .router goes to adamh (not adam)
                if ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                    return "adamh_expert"
                if hasattr(param, "ndim") and param.ndim >= 2:
                    return "adamh"
                return "adam"

            return jax.tree.map(mask_fn, params, paths)

    return AdamHRouterConfig(
        learning_rate=base_optimizer.learning_rate,
        min_lr_ratio=base_optimizer.min_lr_ratio,
        warmup=base_optimizer.warmup,
        lr_schedule=base_optimizer.lr_schedule,
        decay=base_optimizer.decay,
        beta1=base_optimizer.beta1,
        beta2=base_optimizer.beta2,
        epsilon=base_optimizer.epsilon,
        max_grad_norm=base_optimizer.max_grad_norm,
        adam_lr=base_optimizer.adam_lr,
        expert_lr=base_optimizer.expert_lr,
    )


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for label, remove_zloss, router_adamh in CONFIGS:
        for dim, budget in GATE1_CONFIGS:
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            if remove_zloss:
                model = dataclasses.replace(model, router_z_loss_coef=0.0)
            if router_adamh:
                optimizer = _make_adamh_router_optimizer(optimizer)
            run_id = f"{label}-d{dim}-{budget:.2e}"

            steps.append(
                ExecutorStep(
                    name=f"grug/{run_id}",
                    fn=run_grug_moe_trial,
                    config=GrugMoeLaunchConfig(
                        model=versioned(model),
                        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                        output_path=this_output_path(),
                        run_id=run_id,
                        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                        enable_cross_region_ckpt_read=True,
                        steps=versioned(num_steps),
                        batch_size=versioned(batch),
                        seed=versioned(0),
                        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                        tracker=WandbConfig(
                            project="dial_moe",
                            tags=["no-router-zloss", f"d={dim}", f"budget={budget:.2e}", label],
                            group="no-router-zloss",
                            name=run_id,
                        ),
                        optimizer=versioned(optimizer),
                        grug_trainer=versioned(
                            GrugTrainerConfig(
                                z_loss_weight=1e-4,
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
                    ),
                )
            )
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Remove router z-loss: with and without AdamH router, gate 1.",
    )
