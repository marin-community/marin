# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ISOFlop sweep for grug MoE core architecture.

Sweeps over a grid of FLOP budgets and hidden dims, scaling all other
architecture dimensions from hidden_dim per the moe_core defaults:
- num_heads = hidden_dim // 128
- expert intermediate_dim = hidden_dim // 2
- shared expert intermediate_dim = hidden_dim
- dense intermediate_dim = 3 * hidden_dim
- num_layers = round(hidden_dim / (64 + 4*log2(hidden_dim) - 9))
- 2 leading dense layers, MoE (E=64, K=4) in all subsequent layers
"""

import math
from dataclasses import dataclass
from functools import partial

import jax
import optax
from fray.cluster import ResourceConfig
from levanter.optim import OptimizerConfig
from levanter.optim import GrugMuonConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.tracker.wandb import WandbConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_core_muon_auxfree.model import GrugModelConfig
from experiments.grug.moe_core_muon_auxfree.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle

# Re-use the launch config from launch.py for wiring.
from experiments.grug.moe_core_muon_auxfree.launch import GrugMoeLaunchConfig, run_grug_moe_trial

@OptimizerConfig.register_subclass("moe_core_muon_auxfree")
@dataclass(frozen=True)
class MoeCoreMusonConfig(GrugMuonConfig):
    """Muon + per-group Adam for moe_core isoflop sweep."""

    expert_lr_mul: float = 0.25

    def build(self, num_train_steps):
        lr_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        cfg = self

        def optimizer(learning_rate, adam_lr):
            def muon_tx(lr_mul=1.0):
                components = [
                    _grug_scale_with_muon(
                        cfg.momentum, cfg.nesterov, cfg.backend_steps,
                        cfg.muon_epsilon, cfg.use_kimi_scaling, cfg.coefficient_type,
                    ),
                ]
                if cfg.weight_decay > 0:
                    components.append(optax.add_decayed_weights(cfg.weight_decay))
                components.append(optax.scale(-learning_rate * lr_mul))
                return optax.chain(*components)

            adam = optax.chain(
                optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2, eps=cfg.epsilon),
                optax.scale(-adam_lr),
            )

            transforms = {
                "muon": muon_tx(),
                "expert_muon": muon_tx(cfg.expert_lr_mul),
                "embed": adam,
                "adam": adam,
            }

            grouped = optax.multi_transform(transforms, partial(self._create_mask))
            if cfg.max_grad_norm:
                return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), grouped)
            return grouped

        return optax.inject_hyperparams(optimizer)(learning_rate=lr_schedule, adam_lr=adam_lr_schedule)

    def _create_mask(self, params):
        paths = leaf_key_paths(params)

        def classify(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            p = path_str.lower()
            if "router" in p:
                return "muon"
            if "token_embed" in p or "output_proj" in p:
                return "embed"
            if "gate" in p:
                return "adam"
            if hasattr(param, "ndim") and param.ndim == 3:
                return "expert_muon"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "muon"
            return "adam"

        return jax.tree.map(classify, params, paths)


BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19)
HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1536, 2048)
SEQ_LEN: int = 4096
VOCAB_SIZE: int = 128_256

# Muon LR = 0.02 at d=512, bs=128: C = 0.02 * 512 / sqrt(128) ≈ 0.9051
MUON_LR_CONSTANT: float = 0.02 * 512 / 128**0.5
ADAM_LR_CONSTANT: float = 0.33
MIN_BATCH_SIZE: int = 8

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_num_layers(hidden_dim: int) -> int:
    """Compute number of layers from hidden dim using the depth-width formula from Marin2025Recipe."""
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


def _build_model_config(hidden_dim: int) -> GrugModelConfig:
    """Build a GrugModelConfig scaling all dims from hidden_dim."""
    num_heads = hidden_dim // 128
    num_layers = _compute_num_layers(hidden_dim)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim // 2,
        shared_expert_intermediate_dim=hidden_dim,
        num_experts=64,
        num_experts_per_token=4,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=SEQ_LEN,
        num_dense_layers=2,
        dense_intermediate_dim=3 * hidden_dim,
        bias_update_rate=0.0,
        load_balancing_loss_coef=0.01,
    )


def _compute_flops_per_token(cfg: GrugModelConfig) -> float:
    return lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        num_heads=cfg.num_heads,
        seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        glu=True,
        num_experts=cfg.num_experts,
        num_shared_experts=1 if cfg.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=cfg.num_experts_per_token,
    )


def _compute_tokens_and_batch(budget: float, flops_per_token: float) -> tuple[float, int, int]:
    """Compute tokens, batch_size, and train_steps for a FLOP budget.

    Uses 3x multiplier (forward + backward) and targets ~2^16 steps.
    """
    tokens = budget / (3 * flops_per_token)
    target_steps = 2**14
    batch_exact = tokens / (target_steps * SEQ_LEN)
    batch_size = max(8, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
    return tokens, batch_size, train_steps


def create_moe_isoflop_steps() -> list[ExecutorStep]:
    """Create ExecutorSteps for the MoE isoflop grid."""
    steps: list[ExecutorStep] = []

    for budget in BUDGETS:
        for hidden_dim in HIDDEN_DIMS:
            model_cfg = _build_model_config(hidden_dim)
            fpt = _compute_flops_per_token(model_cfg)
            _tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, fpt)
            muon_lr = min(0.03, MUON_LR_CONSTANT * math.sqrt(batch_size) / hidden_dim)
            adam_lr = min(0.01, ADAM_LR_CONSTANT * math.sqrt(batch_size) / hidden_dim)

            run_id = f"moe-isoflop-muon-lbl-{budget:.0e}-d{hidden_dim}"

            config = GrugMoeLaunchConfig(
                model=versioned(model_cfg),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=run_id,
                resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                steps=versioned(train_steps),
                batch_size=versioned(batch_size),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="dial_moe",
                    tags=["grug", "moe-core", "isoflop", "muon", "lbl", f"budget={budget:.0e}", f"d={hidden_dim}"],
                    group="moe-core-isoflop-muon-auxfree",
                    name=run_id,
                ),
                optimizer=versioned(
                    MoeCoreMusonConfig(
                        learning_rate=muon_lr,
                        adam_lr=adam_lr,
                        weight_decay=0.1,
                        min_lr_ratio=0.0,
                        warmup=0.1,
                        momentum=0.95,
                        beta1=0.8,
                        beta2=0.95,
                        epsilon=1e-15,
                        muon_epsilon=1e-8,
                        max_grad_norm=1,
                        lr_schedule="linear",
                        decay=0.2,
                        expert_lr_mul=0.25,
                    )
                ),
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
            )

            step = ExecutorStep(
                name=f"grug/moe-isoflop-muon-lbl-{budget:.0e}-d{hidden_dim}",
                fn=run_grug_moe_trial,
                config=config,
            )
            steps.append(step)

    return steps


moe_isoflop_steps = create_moe_isoflop_steps()

if __name__ == "__main__":
    executor_main(
        steps=moe_isoflop_steps,
        description="MoE core isoflop sweep (Muon + auxfree): 3 budgets x 5 hidden dims on Nemotron mix.",
    )
