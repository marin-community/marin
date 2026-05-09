# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: nano (modded-nanogpt) trial run.

Mirrors `experiments/grug/base/launch.py` but swaps in:
- The nano transformer architecture from `experiments/grug/nano/model.py`
  (RMSNorm + bias-on-linear + half-truncated RoPE + QK-norm + ReLU^2 MLP +
  logit soft-cap), matching `experiments/grug/nanogpt_ref.py`.
- A dual AdamW + Muon optimizer matching the ref's `optimizer1`/`optimizer2`,
  routed via `optax.multi_transform` (see `experiments/grug/modular_opt` for
  the multi-optimizer pattern). Muon iterations reuse Levanter's grug helpers
  from `levanter.optim.grugmuon`.
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
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import _grug_scale_with_muon, _match_update_sharding
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import leaf_key_paths
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.training.training import temporary_checkpoint_base_path

from experiments.grug.nano.model import NanoModelConfig
from experiments.grug.nano.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

# ---- Optimizer ----
# Matches `experiments/grug/nanogpt_ref.py`:
#   optimizer1 = AdamW with three groups
#                  - embed.weight  -> lr=0.3
#                  - proj.weight (top-level lm head) -> lr=1/320
#                  - all p.ndim < 2 (norms, biases) -> lr=0.01
#                betas=(0.8, 0.95), eps=1e-10, weight_decay=0
#   optimizer2 = Muon for block params with p.ndim >= 2
#                  lr=0.02, weight_decay=0.01, mu=0.95, nesterov=True
#                  Newton-Schulz: 12 iterations
# Schedule: stable then linear decay (no warmup), ref's `set_hparams` with cooldown_frac=0.7.
#
# The ref hardcodes Newton-Schulz coefficients (a, b, c) = (2, -1.5, 0.5).
# Levanter's "simple" preset is (3.4445, -4.7750, 2.0315), which is the modern
# nanogpt-modded upgrade. We default to "simple" but expose `ns_coeff_type` so
# experiments can pin to a specific table.


def _stable_then_decay_schedule(num_steps: int, cooldown_frac: float):
    """Stable-then-decay LR multiplier matching `set_hparams` in nanogpt_ref.py."""

    def schedule(step):
        progress = step / num_steps
        eta = jax.lax.cond(
            progress < 1 - cooldown_frac,
            lambda: 1.0,
            lambda: jax.numpy.maximum(0.0, (1 - progress) / cooldown_frac),
        )
        return eta

    return schedule


@OptimizerConfig.register_subclass("nano_adamw_muon")
@dataclass(frozen=True)
class NanoAdamWMuonConfig(OptimizerConfig):
    """Dual AdamW + Muon optimizer for the nano transformer.

    AdamW has three parameter groups (embed, lm head, 1D scalars/biases) with
    independent base LRs but shared betas/eps/wd. Muon handles every >=2D
    parameter inside the transformer blocks. All groups share a single LR
    schedule multiplier so changing total steps rescales every group at once.
    """

    # AdamW group base LRs.
    embed_lr: float = 0.3
    head_lr: float = 1.0 / 320.0
    norm_lr: float = 0.01
    # AdamW shared knobs.
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95
    adam_eps: float = 1e-10
    adam_weight_decay: float = 0.0
    # Muon group.
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 12
    muon_ns_coeff_type: str = "simple"
    muon_eps: float = 1e-8
    muon_use_kimi_scaling: bool = False
    # Optional global gradient clip (the ref does not clip; default off).
    max_grad_norm: float | None = None
    # Schedule: ref uses 0% warmup and 70% linear cooldown.
    cooldown_frac: float = 0.7

    def build(self, num_train_steps: int):
        schedule = _stable_then_decay_schedule(num_train_steps, self.cooldown_frac)

        def adamw_group(base_lr: float) -> optax.GradientTransformation:
            components: list[optax.GradientTransformation] = [
                optax.scale_by_adam(b1=self.adam_beta1, b2=self.adam_beta2, eps=self.adam_eps),
            ]
            if self.adam_weight_decay > 0:
                components.append(optax.add_decayed_weights(self.adam_weight_decay))
            components.append(optax.scale(-base_lr))
            return optax.chain(*components)

        def muon_group() -> optax.GradientTransformation:
            # Order matters: `_match_update_sharding` must run *before*
            # `add_decayed_weights`, not after.
            #
            # `_grug_scale_with_muon` defaults to the `STACK_BATCH_SHARDED`
            # orthogonalization layout, which for a 3-D expert tensor
            # ``(E, D, I)`` reshapes the leading axis across all non-trivial
            # mesh axes — outputting updates with spec
            # ``P(("data", "model"), None, None)`` rather than the params'
            # ``P("expert", "data", "model")``. `add_decayed_weights` then
            # computes ``updates + wd * params`` and fails the broadcast under
            # explicit-mesh-axes (`ShardingTypeError: incompatible shardings`).
            # Resharding updates back to the param spec right after Muon
            # makes every downstream op (decay, scale) see matching shardings.
            components = [
                _grug_scale_with_muon(
                    self.muon_momentum,
                    self.muon_nesterov,
                    self.muon_ns_steps,
                    self.muon_eps,
                    self.muon_use_kimi_scaling,
                    self.muon_ns_coeff_type,
                ),
                _match_update_sharding(),
            ]
            if self.muon_weight_decay > 0:
                components.append(optax.add_decayed_weights(self.muon_weight_decay))
            components.append(optax.scale(-self.muon_lr))
            return optax.chain(*components)

        def optimizer(lr_scale):
            transformations = {
                "adam_embed": optax.chain(adamw_group(self.embed_lr), optax.scale(lr_scale)),
                "adam_head": optax.chain(adamw_group(self.head_lr), optax.scale(lr_scale)),
                "adam_norm": optax.chain(adamw_group(self.norm_lr), optax.scale(lr_scale)),
                "muon": optax.chain(muon_group(), optax.scale(lr_scale)),
            }
            grouped = optax.multi_transform(transformations, partial(self._create_mask))
            if self.max_grad_norm is None:
                return grouped
            return optax.chain(optax.clip_by_global_norm(self.max_grad_norm), grouped)

        return optax.inject_hyperparams(optimizer)(lr_scale=schedule)

    def _create_mask(self, params):
        """Route every leaf of `params` to one of the four optimizer groups.

        Order mirrors the ref's manual partitioning: 1D params (norms, biases)
        always land in `adam_norm`; the embedding goes to `adam_embed`; the
        top-level lm-head projection goes to `adam_head`; everything else,
        which is the >=2D params inside `blocks`, goes to `muon`.

        Under MoE: the router weight (2-D ``(D, E)``) goes to AdamW per the
        user's spec — kept off Muon because the router is a small classifier
        whose semantics rely on logit *magnitudes*, which Muon's orthogonal
        update would erase. ``router_bias`` is 1-D and already lands in
        ``adam_norm``. The 3-D expert tensors (``w_gate`` / ``w_up`` / ``w_down``)
        go through Muon, which vmaps over the leading expert axis and
        orthogonalizes the trailing two dims (see
        ``levanter.optim.grugmuon._grug_scale_with_muon``).
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()

            ndim = getattr(param, "ndim", None)
            if ndim is not None and ndim < 2:
                return "adam_norm"

            # `attn_gate` and `gated_norm` go to AdamW (norm-bias group), not Muon.
            # They are 2D matrices but want gentle, scale-invariant updates.
            if "attn_gate" in path_lower or "gated_norm" in path_lower:
                return "adam_norm"

            # MoE router weight -> AdamW (matches user's spec).
            if ".router" in path_lower or path_lower.endswith(".router"):
                return "adam_norm"

            # Top-level embedding matrix.
            if "embed" in path_lower:
                return "adam_embed"

            # 2D+ and 3D MoE expert tensors inside transformer blocks -> Muon.
            if "blocks" in path_lower:
                return "muon"

            # Top-level lm head (`Transformer.proj`).
            if path_lower == "proj.weight" or path_lower.startswith("proj."):
                return "adam_head"

            # Defensive default: anything 2D+ that wasn't caught above gets
            # AdamW with the head LR. Should not trigger for the current model.
            return "adam_head"

        return jax.tree.map(mask_fn, params, paths)


@dataclass(frozen=True)
class NanoLaunchConfig:
    """Last-mile run config for the nano template."""

    model: NanoModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str  # jmp policy string, e.g. "params=float32,compute=bfloat16,output=bfloat16".
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


# Default model: matches the modded-nanogpt reference exactly
# (12L, 768d, head_dim=128, 6 heads, ReLU^2 MLP at 4x, vocab=50304 / gpt2).
NANO_124M_MODEL = NanoModelConfig(
    vocab_size=50304,
    hidden_dim=768,
    intermediate_dim=3072,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    max_seq_len=1024,
)


def _fineweb_gpt2_data(*, block_cross_document_attention: bool = False) -> LmDataConfig:
    """Fineweb-10B tokenized with gpt2, sequential read.

    Mirrors `experiments/grug/nanogpt/launch.py:_fineweb_gpt2_data` so the
    nano run reads the same tokens in the same order as the modded-nanogpt
    reference. Resolved at runtime via `marin_prefix()` so it picks the
    local region's bucket.

    By default, intra-doc masking is OFF (matches the modded-nanogpt
    reference). Setting ``block_cross_document_attention=True`` flips the
    flag on `CausalLmDataset`, which derives ``segment_ids`` from the EOS
    positions in the cached token stream — every `<|endoftext|>` (50256)
    starts a new segment and the attention mask blocks any query-key pair
    whose segments differ. The token cache itself is unchanged.
    """
    from rigging.filesystem import marin_prefix

    base = os.path.join(marin_prefix(), "data", "fineweb10B-gpt2")
    return LmDataConfig(
        tokenizer="gpt2",
        block_cross_document_attention=block_cross_document_attention,
        auto_build_caches=False,
        shuffle=False,  # sequential data reading, like the ref
        components={
            "fineweb_train": DatasetComponent(cache_dir=base, split="train"),
            "fineweb_val": DatasetComponent(cache_dir=base, split="validation"),
        },
        train_weights={"fineweb_train": 1.0},
    )


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id and append `FERRY_DATE` when launching from ferry workflows."""
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_nano_trial(config: NanoLaunchConfig) -> None:
    # Map template launch knobs onto full Levanter TrainerConfig.
    # When the model is MoE, add an `expert` mesh axis (size 1, like moe's
    # default) so `MoEMLP` can shard expert tensors and `_apply_qb_betas`'s
    # shard_map sees the right batch axes. Dense models inherit the previous
    # axis-less behavior to keep their cache hashes stable.
    extra_kwargs = {"mesh": MeshConfig(axes={"expert": 1})} if config.model.use_moe else {}
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
            # Only the final force=True checkpoint is kept permanently — pick a
            # step interval larger than any walk's total steps so the pinning
            # rule never fires. Intermediate checkpoints rotate via save_interval.
            keep=[{"every": 1_000_000}],
        ),
        **extra_kwargs,
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


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-rsqrt_cap")


nano_trial = ExecutorStep(
    name="grug/nano-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(NANO_124M_MODEL),
        data=_fineweb_gpt2_data(),
        # this_output_path() resolves to this step's output root (e.g. gs://.../grug/nano-trial-<version>).
        output_path=this_output_path(),
        # Keep run id out of versioning so changing job metadata doesn't create a new output path.
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        # Ref runs 3600 steps; default here matches.
        steps=versioned(3600),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2"],
            group="nano-trial",
            name=None,  # filled from run_id in _resolve_tracker
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NanoAdamWMuonConfig()),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,  # ref does not use z-loss
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=125,  # match ref: validate every 125 steps
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_trial],
        description="Nano (modded-nanogpt) 124M, exact-ref data: fineweb10B-gpt2, no shuffle, AdamW+Muon, 3600 steps.",
    )
