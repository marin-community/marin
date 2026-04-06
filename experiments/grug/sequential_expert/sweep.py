# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sequential expert ablation: shared expert feeds residual before routed experts.

Matches v2 1e18-d768 config (lbl=0.001, bias=0.01, expert_lr_mul=1.0, min_batch=32).
"""

import math
from dataclasses import dataclass
from functools import partial

import jax
import optax
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.sequential_expert.model import GrugModelConfig
from experiments.grug.sequential_expert.train import GrugEvalConfig, GrugTrainerConfig
from experiments.grug.sequential_expert.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@OptimizerConfig.register_subclass("sequential_expert_adam_expert_lr")
@dataclass(frozen=True)
class AdamExpertLrConfig(AdamConfig):
    """Adam with per-group expert LR."""

    expert_lr_mul: float = 1.0

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
            if hasattr(param, "ndim") and param.ndim == 3:
                return "expert"
            return "default"

        return jax.tree.map(classify, params, paths)


SEQ_LEN = 4096
VOCAB_SIZE = 128_256

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_num_layers(hidden_dim: int) -> int:
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


# Match v2 1e18-d768
HIDDEN_DIM = 768
BUDGET = 1e18

model_cfg = GrugModelConfig(
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
    load_balancing_loss_coef=0.001,
)

fpt = lm_flops_per_token(
    hidden_dim=model_cfg.hidden_dim,
    intermediate_dim=model_cfg.intermediate_dim,
    num_layers=model_cfg.num_layers,
    num_kv_heads=model_cfg.num_kv_heads,
    num_heads=model_cfg.num_heads,
    seq_len=model_cfg.max_seq_len,
    vocab_size=model_cfg.vocab_size,
    glu=True,
    num_experts=model_cfg.num_experts,
    num_shared_experts=1,
    num_experts_per_tok=model_cfg.num_experts_per_token,
)

tokens = BUDGET / (3 * fpt)
batch_exact = tokens / (2**14 * SEQ_LEN)
batch_size = max(32, _round_to_power_of_two(batch_exact))
train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
lr = min(0.01, (0.33 * math.sqrt(batch_size)) / HIDDEN_DIM)
beta2 = max(0.95, 0.98 ** (batch_size / 128))

import dataclasses

# Unified: both route and compute on post-shared input
unified_cfg = dataclasses.replace(model_cfg, split_route=False)
run_id = "sequential-expert-unified-1e18-d768"

sequential_expert_step = ExecutorStep(
    name=f"grug/{run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(unified_cfg),
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
            tags=["grug", "moe-core", "sequential-expert", "1e18", "d768"],
            group="sequential-expert",
            name=run_id,
        ),
        optimizer=versioned(
            AdamExpertLrConfig(
                learning_rate=lr,
                beta1=0.95,
                beta2=beta2,
                epsilon=1e-15,
                weight_decay=0.1,
                lr_schedule="linear",
                decay=0.2,
                min_lr_ratio=0.0,
                warmup=0.1,
                max_grad_norm=1,
                expert_lr_mul=1.0,
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
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[sequential_expert_step],
        description="Sequential expert ablation: 1e18, d768",
    )
