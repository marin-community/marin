# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GrugMuon sweep: pick 2 from 32, expert width=hidden_dim, separate gate/up, Muon on expert MLPs."""

import dataclasses
from dataclasses import dataclass
from functools import partial

import jax
import optax
from fray.cluster import ResourceConfig
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon, _match_update_sharding
from levanter.tracker.wandb import WandbConfig
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.adamh import scale_by_adamh
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

MUON_LRS: list[float] = [0.01, 0.02, 0.03, 0.04, 0.05]


@OptimizerConfig.register_subclass("grug_moe_adamh_muon")
@dataclass(frozen=True)
class GrugMoeAdamHMuonConfig(OptimizerConfig):
    """AdamH for attention/norms, Muon for expert MLP weights (w_gate, w_up, w_down)."""

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    muon_lr: float = 0.02
    muon_momentum: float = 0.95

    def build(self, num_train_steps):
        adamh_lr_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        muon_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.muon_lr)

        def optimizer(learning_rate, adam_lr, muon_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            def muon_transform():
                components = []
                components.append(_grug_scale_with_muon(self.muon_momentum, True, 5, 1e-8, False, "quintic"))
                components.append(optax.scale(-muon_lr))
                components.append(_match_update_sharding())
                return optax.chain(*components)

            return optax.multi_transform(
                {"adamh": adamh_transform(), "adam": adam_transform(), "muon": muon_transform()},
                partial(self._create_mask),
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=adamh_lr_schedule, adam_lr=adam_lr_schedule, muon_lr=muon_lr_schedule
        )

    def _create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            # Expert MLP weights -> muon
            if ".mlp.w_gate" in path_lower or ".mlp.w_up" in path_lower or ".mlp.w_down" in path_lower:
                return "muon"
            # Shared expert MLP -> muon
            if ".shared.w_" in path_lower:
                return "muon"
            # Embeddings, router, biases, norms -> adam
            if "token_embed" in path_lower or "router" in path_lower or "attn_gate" in path_lower:
                return "adam"
            # 2D attention weights -> adamh
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for muon_lr in MUON_LRS:
        for dim, budget in GATE1_CONFIGS:
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            # Override model: pick 2 from 32, expert width = hidden_dim, separate gate/up, no z-loss
            model = dataclasses.replace(
                model,
                num_experts=32,
                num_experts_per_token=2,
                intermediate_dim=dim,
                separate_gate_up=True,
                router_z_loss_coef=0.0,
            )
            # Override optimizer: AdamH + Muon
            opt = GrugMoeAdamHMuonConfig(
                learning_rate=optimizer.learning_rate,
                adam_lr=optimizer.adam_lr if hasattr(optimizer, "adam_lr") else 6e-4,
                muon_lr=muon_lr,
                beta1=optimizer.beta1,
                beta2=optimizer.beta2,
                epsilon=optimizer.epsilon,
                max_grad_norm=optimizer.max_grad_norm,
                warmup=optimizer.warmup,
                lr_schedule=optimizer.lr_schedule,
                min_lr_ratio=optimizer.min_lr_ratio,
            )
            lr_str = f"{muon_lr:.2f}".replace("0.", ".")
            run_id = f"grugmuon-lr{lr_str}-d{dim}-{budget:.2e}"

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
                            tags=["grugmuon", f"muon_lr={muon_lr}", f"d={dim}"],
                            group="grugmuon",
                            name=run_id,
                        ),
                        optimizer=versioned(opt),
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
            )
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(steps=all_steps, description="GrugMuon sweep: Muon on expert MLPs, AdamH on attention.")
