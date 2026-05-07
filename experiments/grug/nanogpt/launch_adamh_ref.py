# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NanoGPT with AdamH optimizer, matching nanogpt_adamh_ref.py exactly.

Same architecture as model.py but with:
- AdamH (lr=0.018, betas=(0.9,0.95), eps=1e-8) for all 2D block params
- AdamW for embed/head/1D (same LRs as Muon ref)
- Kaiming-uniform init with per-module multipliers
- 250-step warmup for AdamH, cooldown_frac=1.0; no warmup for AdamW, cooldown_frac=0.4
- 4875 steps (not 3600)
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial

import jax
import jax.numpy as jnp
import jmp
import optax
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.grug.nanogpt.model import BATCH_SIZE, NanoGPTConfig
from experiments.grug.nanogpt.train_adamh_ref import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# ---- Constants from nanogpt_adamh_ref.py ----
ADAMH_REF_TRAIN_STEPS = 4875
ADAMH_REF_MATRIX_LR = 0.018
ADAMH_REF_WARMUP_STEPS = 250


# ---- Schedule matching nanogpt_adamh_ref.py ----
# Two separate schedules:
# - "h" (AdamH): linear warmup over 250 steps, then full linear cooldown (cooldown_frac=1.0)
# - "aux" (AdamW): no warmup, cooldown_frac=0.4


def _adamh_ref_h_schedule(num_steps: int, warmup_steps: int = ADAMH_REF_WARMUP_STEPS):
    """AdamH schedule: linear warmup then full linear cooldown."""

    def schedule(step):
        progress = step / num_steps
        warmup_eta = step / warmup_steps
        cooldown_eta = (1 - progress) / 1.0  # cooldown_frac=1.0
        return jnp.where(step < warmup_steps, warmup_eta, cooldown_eta)

    return schedule


def _adamh_ref_aux_schedule(num_steps: int, cooldown_frac: float = 0.4):
    """AdamW auxiliary schedule: no warmup, cooldown_frac=0.4."""

    def schedule(step):
        progress = step / num_steps
        return jnp.where(progress < 1 - cooldown_frac, 1.0, (1 - progress) / cooldown_frac)

    return schedule


# ---- Optimizer ----


@OptimizerConfig.register_subclass("nanogpt_adamh_ref")
@dataclass(frozen=True)
class NanoGPTAdamHRefConfig(OptimizerConfig):
    """AdamH + AdamW optimizer matching nanogpt_adamh_ref.py exactly."""

    # AdamH for 2D block params
    h_lr: float = ADAMH_REF_MATRIX_LR
    h_beta1: float = 0.9
    h_beta2: float = 0.95
    h_eps: float = 1e-8

    # AdamW for embed/head/1D
    embed_lr: float = 0.3
    head_lr: float = 1.0 / 320.0
    norm_lr: float = 0.01
    aux_beta1: float = 0.8
    aux_beta2: float = 0.95
    aux_eps: float = 1e-10

    warmup_steps: int = ADAMH_REF_WARMUP_STEPS

    def build(self, num_train_steps: int):
        h_schedule = _adamh_ref_h_schedule(num_train_steps, self.warmup_steps)
        aux_schedule = _adamh_ref_aux_schedule(num_train_steps)

        def optimizer(h_lr_scale, aux_lr_scale):
            def adamh_transform():
                return scale_by_adamh(self.h_beta1, self.h_beta2, self.h_eps, self.h_lr * h_lr_scale)

            def adamw_group(base_lr):
                return optax.chain(
                    optax.scale_by_adam(b1=self.aux_beta1, b2=self.aux_beta2, eps=self.aux_eps),
                    optax.scale(-base_lr * aux_lr_scale),
                )

            return optax.multi_transform(
                {
                    "adamh": adamh_transform(),
                    "embed": adamw_group(self.embed_lr),
                    "head": adamw_group(self.head_lr),
                    "norm_bias": adamw_group(self.norm_lr),
                },
                partial(self._create_mask),
            )

        return optax.inject_hyperparams(optimizer)(h_lr_scale=h_schedule, aux_lr_scale=aux_schedule)

    def _create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "embed" in path_lower and "norm" not in path_lower:
                return "embed"
            if path_lower.startswith("proj.") or path_lower == "proj":
                return "head"
            if "gated_norm" in path_lower or "attn_gate" in path_lower:
                return "norm_bias"
            if "blocks" in path_lower and hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "norm_bias"

        return jax.tree.map(mask_fn, params, paths)


# ---- Data config ----


def _fineweb_gpt2_data() -> LmDataConfig:
    from rigging.filesystem import marin_prefix

    base = os.path.join(marin_prefix(), "data", "fineweb10B-gpt2")
    return LmDataConfig(
        tokenizer="gpt2",
        block_cross_document_attention=False,
        auto_build_caches=False,
        components={
            "fineweb_train": DatasetComponent(cache_dir=base, split="train"),
            "fineweb_val": DatasetComponent(cache_dir=base, split="validation"),
        },
        train_weights={"fineweb_train": 1.0},
    )


# ---- Launch config ----
# We reuse NanoGPTLaunchConfig from launch.py but import would create circular deps,
# so we define a minimal one here.


@dataclass(frozen=True)
class NanoGPTAdamHRefLaunchConfig:
    model: NanoGPTConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: WandbConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def _resolve_tracker(tracker, run_id):
    return dataclasses.replace(tracker, name=run_id)


def run_nanogpt_adamh_ref_trial(config: NanoGPTAdamHRefLaunchConfig) -> None:
    """Run nanogpt with AdamH ref init (Kaiming + multipliers) and optimizer."""
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 500}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


# ---- Launch step ----

NANOGPT_MODEL = NanoGPTConfig()
ADAMH_REF_OPTIMIZER = NanoGPTAdamHRefConfig()

nanogpt_adamh_ref_trial = ExecutorStep(
    name="grug/nanogpt-adamh-ref-trial-v2",
    fn=run_nanogpt_adamh_ref_trial,
    config=NanoGPTAdamHRefLaunchConfig(
        model=versioned(NANOGPT_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id="nanogpt-adamh-ref-trial-v2",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(ADAMH_REF_TRAIN_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["nanogpt", "adamh-ref"],
            group="nanogpt",
            name="nanogpt-adamh-ref-trial-v2",
        ),
        optimizer=versioned(ADAMH_REF_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=BATCH_SIZE,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[nanogpt_adamh_ref_trial], description="NanoGPT with AdamH ref optimizer.")
