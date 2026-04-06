# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare AdamW vs Muon at 1e18 FLOPs, d=1024, bs=128 with aux-free load balancing.

Fixed config:
- budget = 1e18 FLOPs
- hidden_dim = 1024
- batch_size = 128
- steps = 889 (to match 1e18 FLOPs)
- aux-free: bias_update_rate=0.01, load_balancing_loss_coef=0.0
"""

import math
from dataclasses import dataclass
from functools import partial

import jax
import optax
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.optim import GrugMuonConfig
from levanter.optim.grugmuon import _grug_scale_with_muon
from levanter.tracker.wandb import WandbConfig
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_scaling.model import GrugModelConfig
from experiments.grug.moe_scaling.train import GrugEvalConfig, GrugTrainerConfig
from experiments.grug.moe_scaling.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


# ============================================================
# Muon optimizer (copied from moe_core_muon_auxfree)
# ============================================================

@OptimizerConfig.register_subclass("batchsize_compare_muon")
@dataclass(frozen=True)
class MoeCoreMusonConfig(GrugMuonConfig):
    """Muon + per-group Adam for batchsize comparison."""

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


@OptimizerConfig.register_subclass("batchsize_compare_adam_expert_lr")
@dataclass(frozen=True)
class AdamExpertLrConfig(AdamConfig):
    """Adam with reduced expert learning rate."""

    expert_lr_mul: float = 0.25

    def build(self, num_train_steps):
        lr_schedule = self.lr_scheduler(num_train_steps)
        cfg = self

        def optimizer(learning_rate):
            def adam_tx(lr_mul=1.0):
                components = [
                    optax.scale_by_adam(b1=cfg.beta1, b2=cfg.beta2, eps=cfg.epsilon),
                ]
                if cfg.weight_decay > 0:
                    components.append(optax.add_decayed_weights(cfg.weight_decay))
                components.append(optax.scale(-learning_rate * lr_mul))
                return optax.chain(*components)

            transforms = {
                "default": adam_tx(),
                "expert": adam_tx(cfg.expert_lr_mul),
            }

            grouped = optax.multi_transform(transforms, partial(self._create_mask))
            if cfg.max_grad_norm:
                return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), grouped)
            return grouped

        return optax.inject_hyperparams(optimizer)(learning_rate=lr_schedule)

    def _create_mask(self, params):
        paths = leaf_key_paths(params)

        def classify(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if hasattr(param, "ndim") and param.ndim == 3:
                return "expert"
            return "default"

        return jax.tree.map(classify, params, paths)


# ============================================================
# Fixed config
# ============================================================

HIDDEN_DIM = 1024
BATCH_SIZE = 128
SEQ_LEN = 2048
VOCAB_SIZE = 128_256
BUDGET = 1e18

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _compute_num_layers(hidden_dim: int) -> int:
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


MODEL_CONFIG = GrugModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_dim=HIDDEN_DIM,
    intermediate_dim=HIDDEN_DIM // 2,
    shared_expert_intermediate_dim=HIDDEN_DIM,
    num_experts=64,
    num_experts_per_token=4,
    num_layers=_compute_num_layers(HIDDEN_DIM),
    num_heads=HIDDEN_DIM // 128,
    num_kv_heads=HIDDEN_DIM // 128,
    max_seq_len=SEQ_LEN,
    num_dense_layers=2,
    dense_intermediate_dim=3 * HIDDEN_DIM,
    bias_update_rate=0.01,
    load_balancing_loss_coef=0.0,
)

# steps = tokens / (batch_size * seq_len), tokens = budget / (3 * fpt)
from levanter.utils.flop_utils import lm_flops_per_token

FPT = lm_flops_per_token(
    hidden_dim=MODEL_CONFIG.hidden_dim,
    intermediate_dim=MODEL_CONFIG.intermediate_dim,
    num_layers=MODEL_CONFIG.num_layers,
    num_kv_heads=MODEL_CONFIG.num_kv_heads,
    num_heads=MODEL_CONFIG.num_heads,
    seq_len=MODEL_CONFIG.max_seq_len,
    vocab_size=MODEL_CONFIG.vocab_size,
    glu=True,
    num_experts=MODEL_CONFIG.num_experts,
    num_shared_experts=1,
    num_experts_per_tok=MODEL_CONFIG.num_experts_per_token,
)
TOKENS = BUDGET / (3 * FPT)
TRAIN_STEPS = max(1, round(TOKENS / (BATCH_SIZE * SEQ_LEN)))

# LR scaling
ADAM_LR = min(0.01, 0.33 * math.sqrt(BATCH_SIZE) / HIDDEN_DIM)
ADAM_BETA2 = max(0.95, 0.98 ** (BATCH_SIZE / 128))

MUON_LR_CONSTANT = 0.02 * 512 / 128**0.5
MUON_LR = min(0.03, MUON_LR_CONSTANT * math.sqrt(BATCH_SIZE) / HIDDEN_DIM)
MUON_ADAM_LR = min(0.01, 0.33 * math.sqrt(BATCH_SIZE) / HIDDEN_DIM)


# ============================================================
# Shared trainer/eval config
# ============================================================

GRUG_TRAINER = versioned(
    GrugTrainerConfig(
        z_loss_weight=1e-4,
        ema_beta=None,
        log_every=1,
    )
)

EVAL_CONFIG = versioned(
    GrugEvalConfig(
        eval_batch_size=512,
        steps_per_eval=TRAIN_STEPS,
        max_eval_batches=8,
        eval_current=True,
        eval_ema=False,
    )
)


# ============================================================
# Two runs: AdamW vs Muon
# ============================================================

adamw_run = ExecutorStep(
    name="grug/bs-compare-adamw-expertlr-auxfree-1e18-d1024-seq2048-v3",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(MODEL_CONFIG),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id="bs-compare-adamw-expertlr-auxfree-1e18-d1024-seq2048-v3",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(TRAIN_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["grug", "moe-core", "bs-compare", "adamw", "auxfree", "1e18", "d1024", "bs128"],
            group="batchsize-compare-opt-seq2048",
            name="bs-compare-adamw-expertlr-auxfree-1e18-d1024-seq2048-v3",
        ),
        optimizer=versioned(
            AdamExpertLrConfig(
                learning_rate=ADAM_LR,
                beta1=0.95,
                beta2=ADAM_BETA2,
                epsilon=1e-15,
                weight_decay=0.1,
                lr_schedule="linear",
                decay=0.2,
                min_lr_ratio=0.0,
                warmup=0.1,
                max_grad_norm=1,
                expert_lr_mul=0.25,
            )
        ),
        grug_trainer=GRUG_TRAINER,
        eval=EVAL_CONFIG,
    ),
)

muon_run = ExecutorStep(
    name="grug/bs-compare-muon-auxfree-1e18-d1024-seq2048-nowarmup",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(MODEL_CONFIG),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id="bs-compare-muon-auxfree-1e18-d1024-seq2048-nowarmup",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(TRAIN_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["grug", "moe-core", "bs-compare", "muon", "auxfree", "1e18", "d1024", "bs128"],
            group="batchsize-compare-opt-seq2048",
            name="bs-compare-muon-auxfree-1e18-d1024-seq2048-nowarmup",
        ),
        optimizer=versioned(
            MoeCoreMusonConfig(
                learning_rate=MUON_LR,
                adam_lr=MUON_ADAM_LR,
                weight_decay=0.1,
                min_lr_ratio=0.0,
                warmup=0.0,
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
        grug_trainer=GRUG_TRAINER,
        eval=EVAL_CONFIG,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[adamw_run],
        description="AdamW vs Muon at 1e18 FLOPs, d=1024, bs=128, aux-free load balancing.",
    )
