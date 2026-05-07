# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch config for the nanogpt grug variant.

Matches modded-nanogpt exactly:
- AdamW for embed (lr=0.3), lm_head (lr=1/320), 1D params (lr=0.01)
- Muon for all 2D block params (lr=0.02, wd=0.01)
- Schedule: no warmup, constant 30%, linear decay 70%
- batch=512 seqs, seq_len=1024, 3600 steps
"""

import dataclasses
import os
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial

import jax
import jmp
import optax
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon, _match_update_sharding
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.nanogpt.model import BATCH_SIZE, NanoGPTConfig
from experiments.grug.nanogpt.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# ---- Optimizer matching nanogpt_ref.py exactly ----
# The ref uses a "stable then decay" schedule: constant for first 30%, linear decay over last 70%.
# No warmup at all.


def _nanogpt_lr_schedule(num_steps: int, cooldown_frac: float = 0.7):
    """Stable then linear decay. No warmup."""

    def schedule(step):
        progress = step / num_steps
        eta = jax.lax.cond(
            progress < 1 - cooldown_frac,
            lambda: 1.0,
            lambda: (1 - progress) / cooldown_frac,
        )
        return eta

    return schedule


@OptimizerConfig.register_subclass("nanogpt_adamw_muon")
@dataclass(frozen=True)
class NanoGPTOptimizerConfig(OptimizerConfig):
    """AdamW + Muon optimizer matching nanogpt_ref.py exactly.

    AdamW groups:
    - embed: lr=0.3, betas=(0.8, 0.95), eps=1e-10, wd=0
    - lm_head (proj.weight): lr=1/320, same betas/eps/wd
    - 1D params (norms, biases): lr=0.01, same betas/eps/wd

    Muon group:
    - All 2D params in blocks: lr=0.02, wd=0.01, mu=0.95
    """

    embed_lr: float = 0.3
    head_lr: float = 1.0 / 320.0
    norm_lr: float = 0.01
    muon_lr: float = 0.02
    muon_wd: float = 0.01
    muon_mu: float = 0.95
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    adam_eps: float = 1e-10
    cooldown_frac: float = 0.7

    # NS iterations: ref uses 12 with simple coefficients (2, -1.5, 0.5)
    ns_steps: int = 12
    ns_coeff_type: str = "simple"

    def build(self, num_train_steps: int):
        schedule = _nanogpt_lr_schedule(num_train_steps, self.cooldown_frac)

        def optimizer(lr_scale):
            def adamw_group(base_lr):
                return optax.chain(
                    optax.scale_by_adam(b1=self.adam_beta1, b2=self.adam_beta2, eps=self.adam_eps),
                    optax.scale(-base_lr * lr_scale),
                )

            def muon_transform():
                return optax.chain(
                    _grug_scale_with_muon(self.muon_mu, True, self.ns_steps, 1e-8, False, self.ns_coeff_type),
                    optax.add_decayed_weights(self.muon_wd),
                    optax.scale(-self.muon_lr * lr_scale),
                    _match_update_sharding(),
                )

            return optax.multi_transform(
                {
                    "embed": adamw_group(self.embed_lr),
                    "head": adamw_group(self.head_lr),
                    "norm_bias": adamw_group(self.norm_lr),
                    "muon": muon_transform(),
                },
                partial(self._create_mask),
            )

        return optax.inject_hyperparams(optimizer)(lr_scale=schedule)

    def _create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            # Embedding
            if "embed" in path_lower and "norm" not in path_lower:
                return "embed"
            # LM head (proj at top level of Transformer)
            if path_lower.startswith("proj.") or path_lower == "proj":
                return "head"
            # Block 2D params -> Muon
            if "blocks" in path_lower and hasattr(param, "ndim") and param.ndim >= 2:
                return "muon"
            # Everything else (norms, biases) -> norm_bias
            return "norm_bias"

        return jax.tree.map(mask_fn, params, paths)


# ---- Launch config ----


@dataclass(frozen=True)
class NanoGPTLaunchConfig:
    model: NanoGPTConfig
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


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_nanogpt_trial(config: NanoGPTLaunchConfig) -> None:
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


# ---- Data config ----
# Points at the FineWeb10B-GPT2 TreeCache. The cache path will be set
# at launch time depending on which region the data was written to.

FINEWEB_GPT2_DATA = LmDataConfig(
    tokenizer="gpt2",
    vocab_size=BATCH_SIZE,  # placeholder, overridden by model
    block_cross_document_attention=False,  # match ref: plain causal, no intradoc masking
    components={
        "fineweb_train": dataclasses.field(default_factory=dict),  # type: ignore[call-overload]
    },
)

# For now, the data config is a placeholder. The actual cache paths
# need to be set once the convert_fineweb_gpt2 job completes.
# TODO: Wire up the fineweb10B-gpt2 TreeCache paths here.


if __name__ == "__main__":
    # Placeholder — will be wired up once data conversion is done.
    print("NanoGPT launch config ready. Data paths need to be configured.")
