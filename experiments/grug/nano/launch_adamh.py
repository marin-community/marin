# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nano (modded-nanogpt) trial run with the AdamH-ref optimizer.

Mirrors `experiments/grug/nanogpt_adamh_ref.py` exactly:
- Architecture identical to `nano/launch.py` (12L, 768d, head_dim=128, ReLU^2,
  half-truncated RoPE, attn_scale=0.12, logit_cap=15).
- Init switched to Kaiming-uniform with per-module multipliers
  (`init_scheme="adamh_ref"`) so AdamH has non-zero hidden matrices to operate on.
- Optimizer: AdamH (matrix_lr=0.018, betas=(0.9, 0.95), eps=1e-8) for every
  >=2D parameter inside the transformer blocks; AdamW with the same three
  groups as `nano/launch.py` for embed (lr=0.3), lm head (lr=1/320), and 1D
  scalars/biases (lr=0.01), betas=(0.8, 0.95), eps=1e-10, weight_decay=0.
- AdamH uses the **hardened** `scale_by_adamh_safe` from
  `experiments/grug/nano/optimizer.py` so both `||u||` and `||new_p||` are
  clamped (the levanter copy is missing the second clamp on the 2-D path).
- Schedule: AdamH group has 250-step linear warmup then full linear cooldown
  (cooldown_frac=1.0, no plateau). AdamW group has no warmup and cooldown_frac=0.4.
- 4875 train steps; data is the same fineweb10B-gpt2 sequential read as the
  Muon variant.
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
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.optimizer import scale_by_adamh_safe
from experiments.grug.nano.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# ---- Constants from nanogpt_adamh_ref.py ----
ADAMH_REF_TRAIN_STEPS = 4875
ADAMH_REF_MATRIX_LR = 0.018
ADAMH_REF_WARMUP_STEPS = 250


# ---- Schedules ----
# Two separate schedules for the two parameter group families:
# - "h" (AdamH on block 2D): linear warmup over 250 steps, then full linear
#   cooldown (cooldown_frac=1.0, no plateau).
# - "aux" (AdamW on embed/head/1D): no warmup, cooldown_frac=0.4.
# Both schedules clamp `eta >= 0` so a stray step beyond `num_train_steps`
# doesn't drive the LR negative.


def _adamh_h_schedule(num_steps: int, warmup_steps: int):
    def schedule(step):
        progress = step / num_steps
        warmup_eta = step / warmup_steps
        cooldown_eta = jnp.maximum(0.0, 1.0 - progress)  # cooldown_frac=1.0
        return jnp.where(step < warmup_steps, warmup_eta, cooldown_eta)

    return schedule


def _adamh_aux_schedule(num_steps: int, cooldown_frac: float = 0.4):
    def schedule(step):
        progress = step / num_steps
        decayed = jnp.maximum(0.0, (1 - progress) / cooldown_frac)
        return jnp.where(progress < 1 - cooldown_frac, 1.0, decayed)

    return schedule


# ---- Optimizer ----


@OptimizerConfig.register_subclass("nano_adamh_ref")
@dataclass(frozen=True)
class NanoAdamHRefConfig(OptimizerConfig):
    """AdamH (block 2D) + AdamW (embed/head/1D) optimizer for the nano transformer.

    Mirrors `experiments/grug/nanogpt_adamh_ref.py` exactly. The AdamH update
    is `scale_by_adamh_safe` (this directory's hardened copy that clamps both
    `||u||` and `||new_p||` against divide-by-zero).
    """

    # AdamH (the "h" group): preserves Frobenius norm of every hidden matrix.
    h_lr: float = ADAMH_REF_MATRIX_LR
    h_beta1: float = 0.9
    h_beta2: float = 0.95
    h_eps: float = 1e-8
    h_norm_eps: float = 1e-10  # clamp floor for ||u|| and ||new_p||

    # AdamW (the "aux" group): three sub-groups with independent base LRs.
    embed_lr: float = 0.3
    head_lr: float = 1.0 / 320.0
    norm_lr: float = 0.01
    aux_beta1: float = 0.8
    aux_beta2: float = 0.95
    aux_eps: float = 1e-10

    warmup_steps: int = ADAMH_REF_WARMUP_STEPS

    def build(self, num_train_steps: int):
        h_schedule = _adamh_h_schedule(num_train_steps, self.warmup_steps)
        aux_schedule = _adamh_aux_schedule(num_train_steps)

        def adamw_group(base_lr: float, aux_lr_scale) -> optax.GradientTransformation:
            return optax.chain(
                optax.scale_by_adam(b1=self.aux_beta1, b2=self.aux_beta2, eps=self.aux_eps),
                optax.scale(-base_lr * aux_lr_scale),
            )

        def adamh_group(h_lr_scale) -> optax.GradientTransformation:
            return scale_by_adamh_safe(
                b1=self.h_beta1,
                b2=self.h_beta2,
                eps=self.h_eps,
                learning_rate=self.h_lr * h_lr_scale,
                norm_eps=self.h_norm_eps,
            )

        def optimizer(h_lr_scale, aux_lr_scale):
            return optax.multi_transform(
                {
                    "adam_embed": adamw_group(self.embed_lr, aux_lr_scale),
                    "adam_head": adamw_group(self.head_lr, aux_lr_scale),
                    "adam_norm": adamw_group(self.norm_lr, aux_lr_scale),
                    "adamh": adamh_group(h_lr_scale),
                },
                partial(self._create_mask),
            )

        return optax.inject_hyperparams(optimizer)(h_lr_scale=h_schedule, aux_lr_scale=aux_schedule)

    def _create_mask(self, params):
        """Route every leaf of `params` to one of four optimizer groups.

        Mirrors the ref's manual partitioning: 1D params (norms, biases) ->
        adam_norm; embedding -> adam_embed; top-level lm head -> adam_head;
        every >=2D param inside `blocks` -> adamh.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()

            ndim = getattr(param, "ndim", None)
            if ndim is not None and ndim < 2:
                return "adam_norm"

            # Top-level embedding matrix.
            if "embed" in path_lower:
                return "adam_embed"

            # 2D+ params inside transformer blocks -> AdamH.
            if "blocks" in path_lower:
                return "adamh"

            # Top-level lm head (`Transformer.proj`).
            return "adam_head"

        return jax.tree.map(mask_fn, params, paths)


# ---- Launch config ----


@dataclass(frozen=True)
class NanoAdamHLaunchConfig:
    model: NanoModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


# Default model: identical architecture to nano/launch.py, but with AdamH-ref
# init (Kaiming-uniform with per-module multipliers).
NANO_124M_ADAMH_MODEL = NanoModelConfig(
    vocab_size=50304,
    hidden_dim=768,
    intermediate_dim=3072,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    max_seq_len=1024,
    zero_init_proj=False,  # required by adamh_ref init
    init_scheme="adamh_ref",
)


def _fineweb_gpt2_data() -> LmDataConfig:
    """Fineweb-10B tokenized with gpt2, sequential read, no intra-doc masking."""
    from rigging.filesystem import marin_prefix

    base = os.path.join(marin_prefix(), "data", "fineweb10B-gpt2")
    return LmDataConfig(
        tokenizer="gpt2",
        block_cross_document_attention=False,
        auto_build_caches=False,
        shuffle=False,
        components={
            "fineweb_train": DatasetComponent(cache_dir=base, split="train"),
            "fineweb_val": DatasetComponent(cache_dir=base, split="validation"),
        },
        train_weights={"fineweb_train": 1.0},
    )


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_nano_adamh_trial(config: NanoAdamHLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
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


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-rsqrt_cap-v2")


nano_adamh_trial = ExecutorStep(
    name="grug/nano-adamh-trial",
    fn=run_nano_adamh_trial,
    config=NanoAdamHLaunchConfig(
        model=versioned(NANO_124M_ADAMH_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(ADAMH_REF_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2"],
            group="nano-trial",
            name=None,  # filled from run_id in _resolve_tracker
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NanoAdamHRefConfig()),
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
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_adamh_trial],
        description="Nano (modded-nanogpt) 124M, AdamH+AdamW, 4875 steps on fineweb10B-gpt2.",
    )
