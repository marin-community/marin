# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import optax

from levanter.optim import GrugMuonConfig, OptimizerConfig
from levanter.optim.util import CoefficientType
from experiments.grug.moe.adamh import scale_by_adamh
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.w_gate_up, mlp.w_down, shared.w_*)
    - adam: norms, biases, router, embeddings, attention gates (1D / small params)
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    expert_lr: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "adamh_expert": adamh_expert_transform(),
                    "adam": adam_transform(),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "token_embed" in path_lower:
                return "adam"
            if "router_bias" in path_lower or "attn_gate" in path_lower or ".router" in path_lower:
                return "adam"
            if ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muon")
@dataclass(frozen=True)
class GrugMoeMuonConfig(GrugMuonConfig):
    """Muon for Grug MoE.

    - muon: attention, gated-norm, shared expert, and expert MLP weight matrices
    - adamw: embeddings, output projection, router path, attention gates, norms, biases
    """

    def create_mask(self, params, use_kimi_scaling=True):
        del use_kimi_scaling
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if (
                "token_embed" in path_lower
                or "output_proj" in path_lower
                or "router_bias" in path_lower
                or "attn_gate" in path_lower
                or ".router" in path_lower
            ):
                return "adamw"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "muon"
            return "adamw"

        return jax.tree.map(mask_fn, params, paths)


@dataclass(frozen=True)
class MuonPreset:
    learning_rate: float
    adam_lr: float
    momentum: float
    decay: float
    max_grad_norm: float


_MUON_PRESETS: tuple[tuple[int, MuonPreset], ...] = (
    (
        512,
        MuonPreset(
            learning_rate=0.016,
            adam_lr=0.0032,
            momentum=0.95,
            decay=0.8,
            max_grad_norm=1.0,
        ),
    ),
    (
        1024,
        MuonPreset(
            learning_rate=0.008,
            adam_lr=0.0024,
            momentum=0.98,
            decay=1.0,
            max_grad_norm=1.0,
        ),
    ),
    (
        1 << 30,
        MuonPreset(
            learning_rate=0.004,
            adam_lr=0.0012,
            momentum=0.98,
            decay=1.0,
            max_grad_norm=2.0,
        ),
    ),
)

_COEFFICIENT_STEPS: dict[CoefficientType, int] = {
    "simple": 1,
    "quintic": 5,
    "polar_express": 8,
    "aol": 4,
}


def build_grug_moe_muon_config(
    *,
    hidden_dim: int,
    coefficient_type: CoefficientType = "aol",
) -> GrugMoeMuonConfig:
    """Return the size-matched Muon preset for the current MoE recipe."""
    preset = next(config for max_hidden_dim, config in _MUON_PRESETS if hidden_dim <= max_hidden_dim)
    return GrugMoeMuonConfig(
        learning_rate=preset.learning_rate,
        adam_lr=preset.adam_lr,
        weight_decay=0.0,
        min_lr_ratio=0.0,
        warmup=0.0,
        momentum=preset.momentum,
        beta1=0.8,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=preset.max_grad_norm,
        lr_schedule="linear",
        decay=preset.decay,
        coefficient_type=coefficient_type,
        backend_steps=_COEFFICIENT_STEPS[coefficient_type],
    )


__all__ = ["GrugMoeAdamHConfig", "GrugMoeMuonConfig", "build_grug_moe_muon_config"]
